"""Error analysis on a fixed 200-record test sample.

The brief asks for a five-model error analysis and a Venn diagram.  A true
5-set Venn does not exist, so the five models are collapsed into three families
(tree-based, linear, instance-based) for a genuine 3-set Venn, and a pairwise
agreement heatmap covers all five models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3

from .data import TARGET_COLUMN
from .models import COLORS, DISPLAY_NAMES, MODEL_NAMES
from .paths import ensure_dir


FAMILIES = {
    "tree-based (DT+RF)": ["decision_tree", "random_forest"],
    "linear (LogReg+SVM)": ["logistic_regression", "svm"],
    "instance-based (KNN)": ["knn"],
}


def _sample_ids(test_df: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    y = test_df[TARGET_COLUMN].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    chosen = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return pd.DataFrame(
        {
            "row_id": test_df["row_id"].to_numpy(dtype=int)[chosen],
            "y_true": y[chosen],
        }
    )


def _family_sets(sample: pd.DataFrame) -> dict[str, set[int]]:
    """A family is wrong when every model in it is wrong."""

    out: dict[str, set[int]] = {}
    for family, models in FAMILIES.items():
        available = [m for m in models if m in sample.columns]
        if not available:
            out[family] = set()
            continue
        wrong = np.ones(len(sample), dtype=bool)
        for model in available:
            wrong &= sample[model].to_numpy(dtype=int) != sample["y_true"].to_numpy(dtype=int)
        out[family] = set(sample.loc[wrong, "row_id"].to_numpy(dtype=int).tolist())
    return out


def _venn_counts(sets: dict[str, set[int]]) -> list[dict[str, Any]]:
    names = list(sets)
    a, b, c = (sets[n] for n in names)
    regions = {
        f"{names[0]} only": a - b - c,
        f"{names[1]} only": b - a - c,
        f"{names[2]} only": c - a - b,
        f"{names[0]} & {names[1]} only": (a & b) - c,
        f"{names[0]} & {names[2]} only": (a & c) - b,
        f"{names[1]} & {names[2]} only": (b & c) - a,
        "all three wrong": a & b & c,
        "at least one family wrong": a | b | c,
    }
    return [{"region": k, "count": len(v)} for k, v in regions.items()]


def _plot_venn(sets_by_scenario: dict[str, dict[str, set[int]]], path: Path) -> None:
    plt.rcParams.update({"font.size": 7, "savefig.dpi": 300})
    scenarios = list(sets_by_scenario)
    n = len(scenarios)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows))
    axes = np.atleast_2d(axes)
    all_wrong: dict[str, int] = {}
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // ncols, idx % ncols]
        sets = sets_by_scenario[scenario]
        names = list(sets)
        venn3(
            [sets[n] for n in names],
            set_labels=names,
            ax=ax,
            alpha=0.55,
        )
        ax.set_title(scenario)
        all_wrong[scenario] = len(set.intersection(*[sets[n] for n in names]))

    # Use the spare cell for a compact summary of the shared hard core.
    for idx in range(n, nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if not all_wrong:
            ax.axis("off")
            continue
        labels = list(all_wrong)
        values = [all_wrong[s] for s in labels]
        y = np.arange(len(labels))
        ax.barh(y, values, height=0.5, color="#7f7f7f")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.25 if max(values) else 1)
        ax.set_xlabel("records (of 200)", fontsize=7)
        ax.set_title("All three families wrong", fontsize=8)
        for yi, value in zip(y, values):
            ax.text(value + 0.4, yi, str(value), va="center", fontsize=7)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Misclassification sets by model family (200-record sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _agreement(preds_wide: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_class_rows = []
    for a in models:
        for b in models:
            agree = preds_wide[a].to_numpy() == preds_wide[b].to_numpy()
            rows.append({"model_a": a, "model_b": b, "agreement": float(agree.mean()), "n": int(len(agree))})
            for cls, label in [(1, "positive"), (0, "negative")]:
                mask = preds_wide["y_true"].to_numpy() == cls
                by_class_rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "class": label,
                        "agreement": float(agree[mask].mean()) if mask.any() else float("nan"),
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(by_class_rows)


def _plot_agreement(matrix: pd.DataFrame, models: list[str], path: Path) -> None:
    plt.rcParams.update({"font.size": 8, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])
    ax.grid(False)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
    ax.set_title("Pairwise prediction agreement on the 200-record sample")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_direction(direction: pd.DataFrame, scenarios: list[str], path: Path) -> None:
    plt.rcParams.update({"font.size": 8, "savefig.dpi": 300})
    models = [m for m in MODEL_NAMES if m in set(direction["model"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 3.8), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    x = np.arange(len(models))
    width = 0.38
    for ax, scenario in zip(axes, scenarios):
        sub = direction[direction["scenario"] == scenario].set_index("model")
        fp = [sub.loc[m, "fp"] if m in sub.index else 0 for m in models]
        fn = [sub.loc[m, "fn"] if m in sub.index else 0 for m in models]
        ax.bar(x - width / 2, fp, width, label="False positives", color="#d62728")
        ax.bar(x + width / 2, fn, width, label="False negatives", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=25, ha="right")
        ax.set_title(scenario)
        ax.set_ylabel("Error count on adult.test")
        ax.legend(frameon=True)
    fig.suptitle("False-positive vs false-negative split", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_error_analysis(
    cfg: dict[str, Any],
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    results_dir = ensure_dir(cfg["results_dir"])
    figures_dir = ensure_dir(cfg["figures_dir"])
    preds_path = results_dir / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found; run the test-evaluation stage first")
    preds = pd.read_csv(preds_path)
    scenarios = list(cfg["scenarios"])
    ea = cfg.get("error_analysis", {})
    n_pos = int(ea.get("n_pos", 100))
    n_neg = int(ea.get("n_neg", 100))
    sample_seed = int(ea.get("sample_seed", 42))
    focus_scenario = ea.get("scenario", "none")

    sample_ids = _sample_ids(test_df, n_pos=n_pos, n_neg=n_neg, seed=sample_seed)
    sample_ids.to_csv(results_dir / "error_sample_ids.csv", index=False)
    sample_set = set(sample_ids["row_id"].tolist())

    venn_by_scenario: dict[str, dict[str, set[int]]] = {}
    venn_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    all_wrong_rows: list[pd.DataFrame] = []

    feature_cols = [
        "row_id",
        TARGET_COLUMN,
        "age",
        "education",
        "education-num",
        "workclass",
        "occupation",
        "hours-per-week",
        "capital-gain",
        "capital-loss",
        "native-country",
    ]
    test_features = test_df[[c for c in feature_cols if c in test_df.columns]].copy()

    for scenario in scenarios:
        sub = preds[(preds["scenario"] == scenario) & (preds["row_id"].isin(sample_set))]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="row_id", columns="model", values="y_pred", aggfunc="first")
        wide["y_true"] = sub.groupby("row_id")["y_true"].first()
        wide = wide.reset_index()
        sets = _family_sets(wide)
        venn_by_scenario[scenario] = sets
        for row in _venn_counts(sets):
            row["scenario"] = scenario
            venn_rows.append(row)

        # All-five-wrong hard cases.
        models_present = [m for m in MODEL_NAMES if m in wide.columns]
        wrong_all = np.ones(len(wide), dtype=bool)
        for model in models_present:
            wrong_all &= wide[model].to_numpy(dtype=int) != wide["y_true"].to_numpy(dtype=int)
        hard = wide.loc[wrong_all, ["row_id", "y_true", *models_present]].merge(
            test_features.drop(columns=[TARGET_COLUMN], errors="ignore"),
            on="row_id",
            how="left",
        )
        hard = hard.rename(columns={"y_true": "actual"})
        hard["scenario"] = scenario
        all_wrong_rows.append(hard)

        # Error direction on the full test set, not just the sample.
        full = preds[preds["scenario"] == scenario]
        for model in models_present:
            m = full[full["model"] == model]
            yt = m["y_true"].to_numpy(dtype=int)
            yp = m["y_pred"].to_numpy(dtype=int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            direction_rows.append(
                {
                    "lang": "python",
                    "model": model,
                    "scenario": scenario,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "fp_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                    "fn_rate": fn / (fn + tp) if (fn + tp) else 0.0,
                }
            )

        # Named error patterns.
        n_all_wrong = int(wrong_all.sum())
        if n_all_wrong:
            h = hard
            high_gain_low_hours = int(((h["capital-gain"] > 0) & (h["hours-per-week"] < 40)).sum())
            mid_education = int(h["education-num"].between(9, 13).sum())
            pattern_rows.append(
                {
                    "scenario": scenario,
                    "pattern": "high_capital_gain_low_hours",
                    "count": high_gain_low_hours,
                    "denominator": n_all_wrong,
                    "rate": high_gain_low_hours / n_all_wrong,
                    "note": "all five models wrong; capital-gain > 0 and hours-per-week < 40",
                }
            )
            pattern_rows.append(
                {
                    "scenario": scenario,
                    "pattern": "mid_education_band_9_13",
                    "count": mid_education,
                    "denominator": n_all_wrong,
                    "rate": mid_education / n_all_wrong,
                    "note": "all five models wrong; education-num between 9 and 13",
                }
            )
        scenario_dir = pd.DataFrame([r for r in direction_rows if r["scenario"] == scenario])
        if not scenario_dir.empty:
            fp_dom = int((scenario_dir["fp"] > scenario_dir["fn"]).sum())
            pattern_rows.append(
                {
                    "scenario": scenario,
                    "pattern": "false_positives_dominate",
                    "count": fp_dom,
                    "denominator": len(scenario_dir),
                    "rate": fp_dom / len(scenario_dir),
                    "note": "models on which FP > FN",
                }
            )

    venn_df = pd.DataFrame(venn_rows)
    venn_df.to_csv(results_dir / "venn_counts.csv", index=False)
    if venn_by_scenario:
        _plot_venn(venn_by_scenario, figures_dir / "venn_misclassification.png")

    direction = pd.DataFrame(direction_rows)
    direction.to_csv(results_dir / "error_direction.csv", index=False)
    if not direction.empty:
        _plot_error_direction(direction, scenarios, figures_dir / "error_direction.png")

    hard_df = pd.concat(all_wrong_rows, ignore_index=True) if all_wrong_rows else pd.DataFrame()
    hard_df.to_csv(results_dir / "hard_cases.csv", index=False)

    patterns = pd.DataFrame(pattern_rows)
    patterns.to_csv(results_dir / "error_patterns.csv", index=False)

    # Pairwise agreement for the focus scenario.
    focus = preds[(preds["scenario"] == focus_scenario) & (preds["row_id"].isin(sample_set))]
    agreement_df = pd.DataFrame()
    agreement_by_class = pd.DataFrame()
    if not focus.empty:
        wide = focus.pivot_table(index="row_id", columns="model", values="y_pred", aggfunc="first")
        wide["y_true"] = focus.groupby("row_id")["y_true"].first()
        wide = wide.reset_index()
        models = [m for m in MODEL_NAMES if m in wide.columns]
        agreement_df, agreement_by_class = _agreement(wide, models)
        agreement_df.to_csv(results_dir / "pairwise_agreement.csv", index=False)
        agreement_by_class.to_csv(results_dir / "pairwise_agreement_by_class.csv", index=False)
        matrix = agreement_df.pivot(index="model_a", columns="model_b", values="agreement").loc[models, models]
        _plot_agreement(matrix, models, figures_dir / "pairwise_agreement.png")

    if verbose:
        print(f"  [error] sample={len(sample_ids)} rows, scenarios={len(venn_by_scenario)}", flush=True)
        if not hard_df.empty:
            print(f"  [error] all-five-wrong records across scenarios: {len(hard_df)}", flush=True)

    return {
        "sample_ids": sample_ids,
        "venn": venn_df,
        "direction": direction,
        "hard_cases": hard_df,
        "patterns": patterns,
        "agreement": agreement_df,
        "agreement_by_class": agreement_by_class,
    }
