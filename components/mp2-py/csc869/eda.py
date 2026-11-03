"""Exploratory data analysis figures.

Every figure is exported at 300 dpi into ``reports/figures`` with a stable
filename, and the underlying numbers are written to ``results`` so the report
never has to copy values by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from .data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    class_balance,
    missing_summary,
)
from .paths import ensure_dir


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(train: pd.DataFrame, figures_dir: Path, results_dir: Path) -> Path:
    _style()
    bal = class_balance(train[TARGET_COLUMN])
    df = pd.DataFrame(
        {
            "class": ["<=50K", ">50K"],
            "count": [bal["negative"], bal["positive"]],
            "rate": [bal["negative_rate"], bal["positive_rate"]],
        }
    )
    df.to_csv(results_dir / "eda_class_balance.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(df["class"], df["count"], color=["#4c78a8", "#e45756"])
    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(df["count"]) * 0.015,
            f"{row.count:,}\n({row.rate * 100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Records")
    ax.set_title("Class imbalance in adult.data (n=32,561)")
    ax.set_ylim(0, max(df["count"]) * 1.18)
    path = figures_dir / "eda_class_balance.png"
    _save(fig, path)
    return path


def plot_categorical_income_rates(
    train: pd.DataFrame, figures_dir: Path, results_dir: Path
) -> Path:
    _style()
    cols = [c for c in CATEGORICAL_COLUMNS if c in train.columns]
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.7 * nrows))
    axes = np.asarray(axes).ravel()
    rows = []
    for ax, col in zip(axes, cols):
        s = train[col].fillna("Missing")
        grouped = (
            pd.DataFrame({"category": s, "target": train[TARGET_COLUMN]})
            .groupby("category")["target"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "positive_rate", "count": "n"})
        )
        grouped["column"] = col
        rows.append(grouped)
        # Keep the plot readable for high-cardinality columns.
        plot_df = grouped.sort_values("n", ascending=False).head(15).sort_values("positive_rate")
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["positive_rate"], color="#4c78a8")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["category"], fontsize=7)
        for yi, (rate, n) in enumerate(zip(plot_df["positive_rate"], plot_df["n"])):
            ax.text(min(rate + 0.01, 0.98), yi, f"n={int(n):,}", va="center", fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(>50K)")
        title = col
        if len(grouped) > 15:
            title += f" (top 15 of {len(grouped)} by count)"
        ax.set_title(title)
        ax.axvline(train[TARGET_COLUMN].mean(), color="black", lw=0.7, ls="--", alpha=0.6)
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Income rate by categorical attribute (dashed line = overall 24.08%)", fontsize=11)
    path = figures_dir / "eda_categorical_income_rates.png"
    _save(fig, path)
    pd.concat(rows, ignore_index=True).to_csv(results_dir / "eda_income_rates.csv", index=False)
    return path


def _discretisation_table(
    train: pd.DataFrame, col: str, width: float
) -> pd.DataFrame:
    values = pd.to_numeric(train[col], errors="coerce")
    lo = float(np.floor(values.min()))
    hi = float(np.ceil(values.max()))
    edges = np.arange(lo, hi + width, width)
    if len(edges) < 2:
        edges = np.array([lo, hi + width])
    bins = pd.cut(values, bins=edges, include_lowest=True, right=False)
    grouped = (
        pd.DataFrame({"bin": bins, "target": train[TARGET_COLUMN]})
        .groupby("bin", observed=False)["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "positive_rate", "count": "n"})
    )
    grouped["column"] = col
    grouped["bin_width"] = width
    grouped["bin_label"] = grouped["bin"].astype(str)
    return grouped


def plot_discretisation(train: pd.DataFrame, figures_dir: Path, results_dir: Path) -> list[Path]:
    _style()
    settings = {
        "age": [5, 10, 20],
        "hours-per-week": [5, 10, 20],
        "education-num": [1, 2, 4],
    }
    all_rows = []
    paths = []
    for col, widths in settings.items():
        if col not in train.columns:
            continue
        fig, axes = plt.subplots(1, len(widths), figsize=(4.4 * len(widths), 3.5), sharey=True)
        for ax, width in zip(np.asarray(axes).ravel(), widths):
            tab = _discretisation_table(train, col, width)
            all_rows.append(tab)
            x = np.arange(len(tab))
            ax.bar(x, tab["positive_rate"], color="#4c78a8")
            ax.set_xticks(x)
            ax.set_xticklabels(tab["bin_label"], rotation=45, ha="right", fontsize=6.5)
            for xi, (rate, n) in enumerate(zip(tab["positive_rate"], tab["n"])):
                if n > 0:
                    ax.text(xi, min(rate + 0.01, 0.98), f"{int(n):,}", ha="center", va="bottom", fontsize=5.5, rotation=90)
            ax.set_ylim(0, 1)
            ax.set_title(f"width = {width}")
            ax.set_ylabel("P(>50K)" if width == widths[0] else "")
            ax.set_xlabel(col)
            ax.axhline(train[TARGET_COLUMN].mean(), color="black", lw=0.7, ls="--", alpha=0.6)
        fig.suptitle(f"Discretisation of {col}: income rate per bin (n annotated)", fontsize=11)
        path = figures_dir / f"eda_discretisation_{col.replace('-', '_')}.png"
        _save(fig, path)
        paths.append(path)
    pd.concat(all_rows, ignore_index=True).to_csv(results_dir / "eda_discretisation.csv", index=False)
    return paths


def plot_capital(train: pd.DataFrame, figures_dir: Path, results_dir: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rows = []
    for ax, col in zip(axes, ["capital-gain", "capital-loss"]):
        values = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy()
        zero = int((values == 0).sum())
        positive = values[values > 0]
        ax.hist(values, bins=60, color="#4c78a8", log=True)
        ax.set_title(f"{col}: {zero:,} zeros, {len(positive):,} positive")
        ax.set_xlabel(col)
        ax.set_ylabel("Count (log scale)")
        ax.text(
            0.98,
            0.95,
            f"zero spike = {zero / len(values) * 100:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
        rows.append(
            {
                "column": col,
                "n": int(len(values)),
                "n_zero": zero,
                "n_positive": int(len(positive)),
                "max": float(values.max()),
                "median_positive": float(np.median(positive)) if len(positive) else 0.0,
            }
        )
    fig.suptitle("Capital gain/loss are effectively zero-inflated binary features", fontsize=11)
    path = figures_dir / "eda_capital_gain_loss.png"
    _save(fig, path)
    pd.DataFrame(rows).to_csv(results_dir / "eda_capital_summary.csv", index=False)
    return path


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denom)) if denom else 0.0


def plot_association(train: pd.DataFrame, figures_dir: Path, results_dir: Path) -> Path:
    _style()
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in train.columns]
    num_cols = [c for c in NUMERIC_COLUMNS if c in train.columns]
    cat_frame = train[cat_cols].copy()
    cat_frame["class"] = train[TARGET_COLUMN].map({0: "<=50K", 1: ">50K"})
    cat_labels = cat_cols + ["class"]
    cramers = pd.DataFrame(index=cat_labels, columns=cat_labels, dtype=float)
    for a in cat_labels:
        for b in cat_labels:
            cramers.loc[a, b] = _cramers_v(cat_frame[a], cat_frame[b])
    cramers.to_csv(results_dir / "eda_cramers_v.csv")

    num_frame = train[num_cols + [TARGET_COLUMN]].copy()
    spearman = num_frame.corr(method="spearman")
    spearman.to_csv(results_dir / "eda_spearman.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, matrix, title in [
        (axes[0], cramers, "Cramér's V (categorical)"),
        (axes[1], spearman, "Spearman correlation (numeric)"),
    ]:
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=7)
        ax.grid(False)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                # Black on the bright yellow cells, white elsewhere.
                colour = "black" if values[i, j] >= 0.8 else "white"
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=6,
                )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Association structure: why capital-gain and marital-status matter", fontsize=11)
    path = figures_dir / "eda_association_heatmap.png"
    _save(fig, path)
    return path


def plot_missingness(
    train: pd.DataFrame, test: pd.DataFrame, figures_dir: Path, results_dir: Path
) -> Path:
    _style()
    tr = missing_summary(train)
    te = missing_summary(test)
    merged = tr[["column", "n_missing"]].rename(columns={"n_missing": "train"}).merge(
        te[["column", "n_missing"]].rename(columns={"n_missing": "test"}), on="column"
    )
    merged.to_csv(results_dir / "eda_missingness.csv", index=False)
    merged = merged[(merged["train"] > 0) | (merged["test"] > 0)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(merged))
    width = 0.38
    ax.bar(x - width / 2, merged["train"], width, label="train", color="#4c78a8")
    ax.bar(x + width / 2, merged["test"], width, label="test", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["column"], rotation=25, ha="right")
    ax.set_ylabel("Missing records")
    ax.set_title("Missing values are confined to three categorical columns")
    ax.legend()
    path = figures_dir / "eda_missingness.png"
    _save(fig, path)
    return path


def run_eda(cfg: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, verbose: bool = True) -> list[Path]:
    figures_dir = ensure_dir(cfg["figures_dir"])
    results_dir = ensure_dir(cfg["results_dir"])
    paths = [
        plot_class_balance(train, figures_dir, results_dir),
        plot_categorical_income_rates(train, figures_dir, results_dir),
        *plot_discretisation(train, figures_dir, results_dir),
        plot_capital(train, figures_dir, results_dir),
        plot_association(train, figures_dir, results_dir),
        plot_missingness(train, test, figures_dir, results_dir),
    ]
    if verbose:
        for path in paths:
            print(f"  [eda] wrote {path}", flush=True)
    return paths
