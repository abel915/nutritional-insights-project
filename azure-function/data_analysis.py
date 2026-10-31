#!/usr/bin/env python3
"""
Task 1 - Dataset Analysis and Insights for All_Diets.csv

What this script does
  1. Loads and cleans the data (tidy text, numeric conversion, duplicates, missing values)
  2. Average protein / carbs / fat for each diet type
  3. Top 5 protein-rich recipes for each diet type
  4. The diet type with the highest protein content
  5. The most common cuisines for each diet type
  6. New metrics per recipe: Protein_to_Carbs_ratio and Carbs_to_Fat_ratio
  7. Charts: bar chart, heatmap and scatter plot, saved as PNG files

Usage
  python data_analysis.py                              # reads All_Diets.csv, writes to outputs/
  python data_analysis.py path/to/All_Diets.csv        # custom input path
  python data_analysis.py --out results --show         # custom output folder, open chart windows
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

MACROS = ["Protein(g)", "Carbs(g)", "Fat(g)"]
RATIOS = ["Protein_to_Carbs_ratio", "Carbs_to_Fat_ratio"]
REQUIRED_COLUMNS = ["Diet_type", "Recipe_name", "Cuisine_type", *MACROS]

# Chart colours - a colour-blind-safe categorical order and a one-hue blue ramp
MACRO_COLOURS = {"Protein(g)": "#2a78d6", "Carbs(g)": "#eb6834", "Fat(g)": "#1baf7a"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# --------------------------------------------------------------------------- #
# 1. Load and clean
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a raw All_Diets dataframe. Returns (clean dataframe, cleaning report)."""
    df = raw.copy()
    df.columns = df.columns.str.strip()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}")

    rows_in = len(df)

    # Tidy text so that "Paleo", "paleo" and " Paleo " count as the same diet
    for col in ["Diet_type", "Recipe_name", "Cuisine_type"]:
        df[col] = df[col].astype("string").str.strip()
    df["Diet_type"] = df["Diet_type"].str.lower()
    df["Cuisine_type"] = df["Cuisine_type"].str.lower()

    # Macro columns must be numbers; anything else becomes NaN (treated as missing)
    for col in MACROS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_before = {col: int(n) for col, n in df[MACROS].isna().sum().items()}

    # A recipe without a diet type can't be grouped, so drop it
    has_diet = df["Diet_type"].fillna("") != ""
    dropped_no_diet = int((~has_diet).sum())
    df = df[has_diet].copy()

    # Exact duplicate rows add nothing
    before_dupes = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before_dupes - len(df)

    # Fill missing macros with that diet's average, falling back to the overall average.
    # (The brief suggests df.fillna(df.mean()); on current pandas that fails because of the
    #  text columns, and a per-diet mean is more accurate than one global mean.)
    diet_means = df.groupby("Diet_type")[MACROS].transform("mean")
    df[MACROS] = df[MACROS].fillna(diet_means).fillna(df[MACROS].mean())

    missing_cuisine = int(df["Cuisine_type"].fillna("").eq("").sum())
    df["Cuisine_type"] = df["Cuisine_type"].fillna("unknown").replace("", "unknown")

    report = {
        "rows_read": rows_in,
        "rows_kept": len(df),
        "dropped_no_diet_type": dropped_no_diet,
        "dropped_duplicates": dropped_dupes,
        "missing_macro_values_filled": missing_before,
        "missing_cuisine_set_to_unknown": missing_cuisine,
    }
    return df.reset_index(drop=True), report


def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    return clean_data(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# 2-6. Calculations
# --------------------------------------------------------------------------- #
def add_ratio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add Protein_to_Carbs_ratio and Carbs_to_Fat_ratio. Dividing by 0 gives NaN, not infinity."""
    out = df.copy()
    out["Protein_to_Carbs_ratio"] = (out["Protein(g)"] / out["Carbs(g)"].replace(0, np.nan)).round(3)
    out["Carbs_to_Fat_ratio"] = (out["Carbs(g)"] / out["Fat(g)"].replace(0, np.nan)).round(3)
    return out


def average_macros(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Diet_type")[MACROS].mean().round(2).sort_index()


def top_protein_recipes(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = ["Diet_type", "Recipe_name", "Cuisine_type", *MACROS]
    return (
        df.sort_values("Protein(g)", ascending=False)
        .groupby("Diet_type")
        .head(n)[cols]
        .sort_values(["Diet_type", "Protein(g)"], ascending=[True, False])
        .reset_index(drop=True)
    )


def highest_protein_diet(df: pd.DataFrame) -> dict:
    """Answer 'which diet has the most protein' both ways: per recipe (average) and in total."""
    avg = df.groupby("Diet_type")["Protein(g)"].mean()
    total = df.groupby("Diet_type")["Protein(g)"].sum()
    return {
        "by_average": {"diet": avg.idxmax(), "protein_g": round(float(avg.max()), 2)},
        "by_total": {"diet": total.idxmax(), "protein_g": round(float(total.max()), 2)},
    }


def common_cuisines(df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    known = df[df["Cuisine_type"] != "unknown"]
    counts = known.groupby(["Diet_type", "Cuisine_type"]).size().rename("recipes").reset_index()
    counts["share_%"] = (
        counts["recipes"] / counts.groupby("Diet_type")["recipes"].transform("sum") * 100
    ).round(1)
    counts = counts.sort_values(["Diet_type", "recipes"], ascending=[True, False])
    return counts.groupby("Diet_type").head(top).reset_index(drop=True)


def ratio_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median ratios per diet (median, because a few near-zero denominators create huge outliers)."""
    return df.groupby("Diet_type")[RATIOS].median().round(3).sort_index()


# --------------------------------------------------------------------------- #
# 7. Charts
# --------------------------------------------------------------------------- #
def make_charts(avg: pd.DataFrame, top5: pd.DataFrame, out_dir: str, show: bool = False) -> list[str]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # no screen needed (inside Docker / CI)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    diets = [d.title() for d in avg.index]
    saved = []

    # --- Bar chart: average protein, carbs and fat per diet ------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.26
    x = np.arange(len(avg))
    for i, macro in enumerate(MACROS):
        ax.bar(x + (i - 1) * width, avg[macro], width, label=macro.replace("(g)", ""),
               color=MACRO_COLOURS[macro], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xticks(x, diets)
    ax.set_ylabel("Average per recipe (g)")
    ax.set_title("Average macronutrients by diet type", loc="left", pad=30)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.margins(y=0.05)
    fig.tight_layout()
    saved.append(os.path.join(out_dir, "1_avg_macros_bar.png"))
    fig.savefig(saved[-1], dpi=150)

    # --- Heatmap: macronutrients x diet types --------------------------------
    # Colour = where each diet sits for that nutrient (0 = lowest diet, 1 = highest diet),
    # so carbs' larger numbers don't wash out protein and fat. Cells show the real grams.
    rng = (avg.max() - avg.min()).replace(0, 1)
    relative = (avg - avg.min()) / rng
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(avg) + 2))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    sns.heatmap(relative, annot=avg.values, fmt=".1f", cmap=cmap, vmin=0, vmax=1,
                linewidths=2, linecolor=SURFACE, ax=ax,
                xticklabels=[m.replace("(g)", " (g)") for m in MACROS], yticklabels=diets,
                cbar_kws={"label": "Within each nutrient: lowest → highest diet", "ticks": [0, 1]})
    ax.collections[0].colorbar.set_ticklabels(["lowest", "highest"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.set_title("Average macronutrients (g) by diet type", loc="left")
    fig.tight_layout()
    saved.append(os.path.join(out_dir, "2_macros_heatmap.png"))
    fig.savefig(saved[-1], dpi=150)

    # --- Scatter: top 5 protein recipes per diet, spread across cuisines -----
    # One panel per diet (small multiples) so colour isn't needed to tell diets apart.
    cuisines = sorted(top5["Cuisine_type"].unique())
    ypos = {c: i for i, c in enumerate(cuisines)}
    panel_diets = sorted(top5["Diet_type"].unique())
    fig, axes = plt.subplots(1, len(panel_diets), sharey=True, sharex=True,
                             figsize=(2.6 * len(panel_diets) + 1.5, 0.42 * len(cuisines) + 1.8))
    axes = np.atleast_1d(axes)
    for ax, diet in zip(axes, panel_diets):
        sub = top5[top5["Diet_type"] == diet]
        ax.scatter(sub["Protein(g)"], sub["Cuisine_type"].map(ypos), s=70, color=MACRO_COLOURS["Protein(g)"],
                   edgecolor=SURFACE, linewidth=1.5, alpha=0.9, zorder=3)
        ax.set_title(diet.title(), loc="left", fontsize=11)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        ax.set_xlabel("Protein (g)")
    axes[0].set_yticks(range(len(cuisines)), [c.title() for c in cuisines])
    axes[0].set_ylim(len(cuisines) - 0.4, -0.6)  # A at the top, Z at the bottom
    fig.suptitle("Top 5 protein-rich recipes per diet, by cuisine", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    saved.append(os.path.join(out_dir, "3_top_protein_scatter.png"))
    fig.savefig(saved[-1], dpi=150)

    if show:
        plt.show()
    plt.close("all")
    return saved


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nutritional insights from All_Diets.csv")
    parser.add_argument("csv", nargs="?", default="All_Diets.csv", help="path to All_Diets.csv")
    parser.add_argument("--out", default="outputs", help="folder for result CSVs and charts")
    parser.add_argument("--show", action="store_true", help="also open the charts in windows")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 45)

    print(f"Diet analysis run at {datetime.now():%Y-%m-%d %H:%M:%S}  |  input: {args.csv}")
    if not os.path.exists(args.csv):
        print(f"\nERROR: '{args.csv}' not found.\nDownload All_Diets.csv (Kaggle or course resources) "
              "into this folder, or pass its path:  python data_analysis.py path/to/All_Diets.csv",
              file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    df, report = load_and_clean(args.csv)
    df = add_ratio_metrics(df)

    section("Data cleaning")
    for key, value in report.items():
        print(f"  {key:32} {value}")

    avg = average_macros(df)
    section("Average macronutrient content per diet type (g)")
    print(avg.to_string())

    top5 = top_protein_recipes(df)
    section("Top 5 protein-rich recipes per diet type")
    print(top5.to_string(index=False))

    best = highest_protein_diet(df)
    section("Diet type with the highest protein content")
    print(f"  Highest average per recipe : {best['by_average']['diet']} "
          f"({best['by_average']['protein_g']} g)")
    print(f"  Highest total, all recipes : {best['by_total']['diet']} "
          f"({best['by_total']['protein_g']} g)")

    cuisines = common_cuisines(df)
    section("Most common cuisines per diet type (top 3)")
    print(cuisines.to_string(index=False))

    section("New metrics: median ratios per diet type")
    print(ratio_summary(df).to_string())
    print("\nSample of recipes with the new ratio columns:")
    print(df[["Diet_type", "Recipe_name", *MACROS, *RATIOS]].head(8).to_string(index=False))

    avg.to_csv(os.path.join(args.out, "avg_macros_by_diet.csv"))
    top5.to_csv(os.path.join(args.out, "top5_protein_recipes.csv"), index=False)
    cuisines.to_csv(os.path.join(args.out, "common_cuisines.csv"), index=False)
    df.to_csv(os.path.join(args.out, "recipes_with_ratios.csv"), index=False)

    charts = make_charts(avg, top5, args.out, show=args.show)
    section("Saved files")
    for name in sorted(os.listdir(args.out)):
        print(f"  {os.path.join(args.out, name)}")
    print(f"\nFinished at {datetime.now():%Y-%m-%d %H:%M:%S} - {len(charts)} charts created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
