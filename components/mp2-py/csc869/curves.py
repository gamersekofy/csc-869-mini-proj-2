"""ROC, precision-recall, learning-curve, and summary figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import COLORS, DISPLAY_NAMES, MARKERS, MODEL_NAMES
from .paths import ensure_dir


SCENARIO_TITLES = {
    "none": "No resampling",
    "upsampling": "Random upsampling",
    "downsampling": "Random downsampling",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _mean_band(
    df: pd.DataFrame,
    curve: str,
    scenario: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sub = df[(df["curve"] == curve) & (df["scenario"] == scenario) & (df["model"] == model)]
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="x", columns="fold", values="y", aggfunc="mean").sort_index()
    x = pivot.index.to_numpy(dtype=float)
    mean = pivot.mean(axis=1).to_numpy(dtype=float)
    std = pivot.std(axis=1, ddof=0).fillna(0.0).to_numpy(dtype=float)
    return x, mean, std


def make_roc_pr_figures(cfg: dict[str, Any], verbose: bool = True) -> list[Path]:
    """Create ``roc_curves.png`` and ``pr_curves.png`` (one panel per scenario)."""

    _style()
    results_dir = Path(cfg["results_dir"])
    figures_dir = ensure_dir(cfg["figures_dir"])
    curves = pd.read_csv(results_dir / "curve_points.csv")
    summary = pd.read_csv(results_dir / "cv_summary.csv")
    scenarios = list(cfg["scenarios"])
    models = [m for m in MODEL_NAMES if m in set(curves["model"])]

    paths: list[Path] = []
    for curve_name, ylabel, title, metric, filename in [
        ("roc", "True positive rate", "ROC curves", "roc_auc", "roc_curves.png"),
        ("pr", "Precision", "Precision-recall curves", "pr_auc", "pr_curves.png"),
    ]:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 4.2), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]
        for ax, scenario in zip(axes, scenarios):
            for model in models:
                band = _mean_band(curves, curve_name, scenario, model)
                if band is None:
                    continue
                x, mean, std = band
                color = COLORS[model]
                row = summary[
                    (summary["model"] == model)
                    & (summary["scenario"] == scenario)
                    & (summary["metric"] == metric)
                ]
                score = float(row["mean"].iloc[0]) if not row.empty else float("nan")
                label = f"{DISPLAY_NAMES[model]} ({score:.3f})"
                ax.plot(x, mean, color=color, lw=1.4, label=label)
                ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.10)
            if curve_name == "roc":
                ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="Chance")
                ax.set_xlabel("False positive rate")
            else:
                ax.axhline(0.24, color="black", lw=0.8, ls="--", label="Positive rate (0.24)")
                ax.set_xlabel("Recall")
            ax.set_title(SCENARIO_TITLES.get(scenario, scenario))
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower left" if curve_name == "roc" else "upper right", frameon=True)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        if verbose:
            print(f"  [curves] wrote {path}", flush=True)
    return paths


def make_learning_curve_figures(cfg: dict[str, Any], verbose: bool = True) -> list[Path]:
    _style()
    results_dir = Path(cfg["results_dir"])
    figures_dir = ensure_dir(cfg["figures_dir"])
    path_csv = results_dir / "learning_curves.csv"
    if not path_csv.exists():
        return []
    df = pd.read_csv(path_csv)
    scenarios = list(cfg["scenarios"])
    models = [m for m in MODEL_NAMES if m in set(df["model"])]

    paths: list[Path] = []
    for metric, ylabel, filename in [
        ("f1", "F1 score (positive class)", "learning_curves_f1.png"),
        ("roc_auc", "ROC-AUC", "learning_curves_roc_auc.png"),
    ]:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 4.0), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]
        for ax, scenario in zip(axes, scenarios):
            for model in models:
                sub = df[(df["scenario"] == scenario) & (df["model"] == model)]
                if sub.empty:
                    continue
                grouped = sub.groupby("fraction")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(
                    grouped["fraction"],
                    grouped["mean"],
                    color=COLORS[model],
                    marker=MARKERS[model],
                    ms=4,
                    lw=1.2,
                    label=DISPLAY_NAMES[model],
                )
                ax.fill_between(
                    grouped["fraction"],
                    np.clip(grouped["mean"] - grouped["std"].fillna(0), 0, 1),
                    np.clip(grouped["mean"] + grouped["std"].fillna(0), 0, 1),
                    color=COLORS[model],
                    alpha=0.10,
                )
            ax.set_title(SCENARIO_TITLES.get(scenario, scenario))
            ax.set_xlabel("Training-set fraction")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.05, 1.02)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower right", frameon=True)
        fig.suptitle(f"Learning curves: {ylabel}", fontsize=12)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        if verbose:
            print(f"  [curves] wrote {path}", flush=True)
    return paths


def make_summary_heatmap(cfg: dict[str, Any], verbose: bool = True) -> Path | None:
    """Model x scenario heatmap for F1 and PR-AUC."""

    _style()
    results_dir = Path(cfg["results_dir"])
    figures_dir = ensure_dir(cfg["figures_dir"])
    path_csv = results_dir / "cv_summary_wide.csv"
    if not path_csv.exists():
        return None
    df = pd.read_csv(path_csv)
    models = [m for m in MODEL_NAMES if m in set(df["model"])]
    scenarios = list(cfg["scenarios"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric in zip(axes, ["f1", "pr_auc"]):
        mat = np.full((len(models), len(scenarios)), np.nan)
        for i, model in enumerate(models):
            for j, scenario in enumerate(scenarios):
                row = df[(df["model"] == model) & (df["scenario"] == scenario)]
                if not row.empty and metric in row:
                    mat[i, j] = float(row[metric].iloc[0])
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([SCENARIO_TITLES.get(s, s) for s in scenarios], rotation=15, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])
        ax.set_title(metric.replace("_", "-").upper())
        ax.grid(False)
        for i in range(len(models)):
            for j in range(len(scenarios)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean 5-fold CV score by model and sampling scenario", fontsize=12)
    fig.tight_layout()
    path = figures_dir / "cv_metric_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  [curves] wrote {path}", flush=True)
    return path
