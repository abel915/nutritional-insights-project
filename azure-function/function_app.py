"""
Phase 2 - Cloud Dashboard Development
Azure Functions (Python v2 programming model)

Three HTTP-triggered endpoints that back the nutrition dashboard:

  GET /api/nutritional-insights  -> per-diet macro averages + a macro correlation matrix
  GET /api/recipes                -> cleaned recipe rows (optional ?diet=<type> filter)
  GET /api/clusters                -> lightweight k-means grouping of recipes by macros

All three read All_Diets.csv from Azure Blob Storage (connection string + container +
blob name come from Function App Application Settings / environment variables, never
hard-coded), reuse the exact Phase 1 cleaning logic from data_analysis.py, and return
JSON with CORS left to the Function App's CORS configuration (set in the Portal).
"""
from __future__ import annotations

import io
import json
import logging
import os
import time

import azure.functions as func
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

from data_analysis import (
    MACROS,
    add_ratio_metrics,
    average_macros,
    clean_data,
    ratio_summary,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

_CACHE: dict = {"df": None, "loaded_at": 0.0}
_CACHE_TTL_SECONDS = 300  # re-read the blob at most every 5 minutes per instance


def _json_response(payload: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload, default=_to_native),
        status_code=status_code,
        mimetype="application/json",
    )


def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if pd.isna(o):
        return None
    return str(o)


def _load_dataframe() -> pd.DataFrame:
    """Download All_Diets.csv from Blob Storage and clean it, caching per warm instance."""
    now = time.time()
    if _CACHE["df"] is not None and (now - _CACHE["loaded_at"]) < _CACHE_TTL_SECONDS:
        return _CACHE["df"]

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container = os.environ.get("BLOB_CONTAINER", "diet-data")
    blob_name = os.environ.get("BLOB_NAME", "All_Diets.csv")

    service = BlobServiceClient.from_connection_string(conn_str)
    blob = service.get_blob_client(container=container, blob=blob_name)
    raw_bytes = blob.download_blob().readall()

    raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    df, report = clean_data(raw_df)
    df = add_ratio_metrics(df)
    logging.info("Loaded + cleaned %s rows from blob (%s)", len(df), report)

    _CACHE["df"] = df
    _CACHE["loaded_at"] = now
    return df


@app.route(route="nutritional-insights", methods=["GET"])
def nutritional_insights(req: func.HttpRequest) -> func.HttpResponse:
    start = time.time()
    try:
        df = _load_dataframe()
        avg = average_macros(df)  # index: diet_type, cols: Protein(g) Carbs(g) Fat(g)
        ratios = ratio_summary(df)

        diets = []
        for diet in avg.index:
            diets.append({
                "diet": diet,
                "protein": float(avg.loc[diet, "Protein(g)"]),
                "carbs": float(avg.loc[diet, "Carbs(g)"]),
                "fat": float(avg.loc[diet, "Fat(g)"]),
                "proteinToCarbsRatio": float(ratios.loc[diet, "Protein_to_Carbs_ratio"]),
                "carbsToFatRatio": float(ratios.loc[diet, "Carbs_to_Fat_ratio"]),
                "recipeCount": int((df["Diet_type"] == diet).sum()),
            })

        # Correlation matrix across macros + a derived calorie estimate (4-4-9 rule)
        corr_df = df[MACROS].copy()
        corr_df["Calories"] = corr_df["Protein(g)"] * 4 + corr_df["Carbs(g)"] * 4 + corr_df["Fat(g)"] * 9
        labels = ["Protein", "Carbs", "Fat", "Calories"]
        matrix = corr_df.corr().round(3).values.tolist()

        return _json_response({
            "diets": diets,
            "correlations": {"labels": labels, "matrix": matrix},
            "source": "azure-blob-storage",
            "executionTimeMs": round((time.time() - start) * 1000, 1),
        })
    except Exception as exc:  # noqa: BLE001
        logging.exception("nutritional-insights failed")
        return _json_response({"error": str(exc)}, status_code=500)


@app.route(route="recipes", methods=["GET"])
def recipes(req: func.HttpRequest) -> func.HttpResponse:
    start = time.time()
    try:
        df = _load_dataframe()
        diet = req.params.get("diet")
        if diet:
            df = df[df["Diet_type"] == diet.strip().lower()]

        page = max(int(req.params.get("page", 1)), 1)
        page_size = min(max(int(req.params.get("pageSize", 500)), 1), 2000)
        total = len(df)
        start_idx = (page - 1) * page_size
        page_df = df.iloc[start_idx:start_idx + page_size]

        rows = [
            {
                "diet": r["Diet_type"],
                "name": r["Recipe_name"],
                "cuisine": r["Cuisine_type"],
                "protein": float(r["Protein(g)"]),
                "carbs": float(r["Carbs(g)"]),
                "fat": float(r["Fat(g)"]),
            }
            for _, r in page_df.iterrows()
        ]

        return _json_response({
            "recipes": rows,
            "page": page,
            "pageSize": page_size,
            "total": int(total),
            "source": "azure-blob-storage",
            "executionTimeMs": round((time.time() - start) * 1000, 1),
        })
    except Exception as exc:  # noqa: BLE001
        logging.exception("recipes failed")
        return _json_response({"error": str(exc)}, status_code=500)


@app.route(route="clusters", methods=["GET"])
def clusters(req: func.HttpRequest) -> func.HttpResponse:
    """Lightweight k-means (pure numpy, no scikit-learn dependency) on standardized macros."""
    start = time.time()
    try:
        df = _load_dataframe()
        k = min(max(int(req.params.get("k", 4)), 2), 8)

        X = df[MACROS].to_numpy(dtype=float)
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xs = (X - mu) / sigma

        rng = np.random.default_rng(42)
        centroids = Xs[rng.choice(len(Xs), size=k, replace=False)]
        for _ in range(25):
            dists = np.linalg.norm(Xs[:, None, :] - centroids[None, :, :], axis=2)
            assignments = dists.argmin(axis=1)
            new_centroids = np.array([
                Xs[assignments == c].mean(axis=0) if (assignments == c).any() else centroids[c]
                for c in range(k)
            ])
            if np.allclose(new_centroids, centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        out = []
        for c in range(k):
            mask = assignments == c
            if not mask.any():
                continue
            sub = df[mask]
            dominant_diet = sub["Diet_type"].value_counts().idxmax()
            out.append({
                "cluster": int(c),
                "count": int(mask.sum()),
                "avgProtein": float(sub["Protein(g)"].mean()),
                "avgCarbs": float(sub["Carbs(g)"].mean()),
                "avgFat": float(sub["Fat(g)"].mean()),
                "dominantDiet": dominant_diet,
            })

        return _json_response({
            "clusters": out,
            "k": k,
            "source": "azure-blob-storage",
            "executionTimeMs": round((time.time() - start) * 1000, 1),
        })
    except Exception as exc:  # noqa: BLE001
        logging.exception("clusters failed")
        return _json_response({"error": str(exc)}, status_code=500)
