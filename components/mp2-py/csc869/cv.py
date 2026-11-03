"""The 5-fold experiment matrix.

For every (scenario, fold, model) combination:

1. take the training portion of the fold,
2. resample it (none / up / down),
3. fit the preprocessor on the resampled training rows only,
4. fit the classifier,
5. score the untouched validation fold.

The tidy CSV produced here is the single source of truth for every table and
figure in the report.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import resolved_paths
from .data import TARGET_COLUMN
from .kfold import stratified_subsample
from .metrics import METRIC_NAMES, compute_metrics, precision_recall_curve
from .models import (
    MODEL_NAMES,
    fit_model,
    predict_scores,
)
from .paths import ensure_dir
from .preprocess import Preprocessor
from .sampling import SCENARIOS, resample_indices


def _curve_points(y_true: np.ndarray, score: np.ndarray, n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, score)
    grid = np.linspace(0.0, 1.0, n_grid)
    tpr_grid = np.interp(grid, fpr, tpr)
    precision, recall = precision_recall_curve(y_true, score)
    prec_grid = np.interp(grid, recall, precision)
    return grid, tpr_grid, prec_grid


def _run_one_fold(
    scenario: str,
    fold_id: int,
    train_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: dict[str, Any],
    model_names: list[str],
    seed: int,
    model_n_jobs: int,
) -> dict[str, Any]:
    scenario_idx = SCENARIOS.index(scenario)
    scenario_seed = seed + fold_id * 17 + scenario_idx * 101
    X_train_raw = train_df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]

    res_idx, counts = resample_indices(y_train, scenario, seed=scenario_seed)
    X_res = X_train_raw.iloc[res_idx].reset_index(drop=True)
    y_res = y_train[res_idx]

    pre = Preprocessor(
        categorical=cfg["missing"]["categorical"],
        numeric=cfg["missing"]["numeric"],
        capital_flags=bool(cfg["missing"].get("capital_flags", False)),
    )
    X_res_p = pre.fit_transform(X_res)
    X_val_raw = train_df.iloc[val_idx].reset_index(drop=True)
    X_val_p = pre.transform(X_val_raw)
    y_val = y[val_idx]
    unseen = dict(pre.unseen_counts_)

    result_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(model_names):
        model_seed = scenario_seed + model_idx * 1009
        model, info = fit_model(
            model_name,
            X_res_p,
            y_res,
            seed=model_seed,
            svm_probability=bool(cfg["cv"].get("svm_probability", False)),
            rf_trees=int(cfg["cv"].get("rf_trees", 300)),
            knn_neighbors=int(cfg["cv"].get("knn_neighbors", 5)),
            n_jobs=model_n_jobs,
            tune_dt=True,
            inner_splits=int(cfg["cv"].get("inner_splits", 3)),
            dt_depths=list(cfg["cv"].get("dt_depths", [3, 5, 7, 10, None])),
        )
        pred = model.predict(X_val_p)
        score = predict_scores(model, X_val_p)
        metrics = compute_metrics(y_val, pred, score)

        for metric in METRIC_NAMES:
            result_rows.append(
                {
                    "lang": "python",
                    "model": model_name,
                    "scenario": scenario,
                    "fold": fold_id,
                    "metric": metric,
                    "value": float(metrics[metric]),
                }
            )

        run_rows.append(
            {
                "lang": "python",
                "model": model_name,
                "scenario": scenario,
                "fold": fold_id,
                "n_train_original": int(len(train_idx)),
                "n_train_resampled": int(len(res_idx)),
                "n_train_positive": int(counts["n_positive"]),
                "n_train_negative": int(counts["n_negative"]),
                "n_val": int(len(val_idx)),
                "n_val_positive": int((y_val == 1).sum()),
                "n_val_negative": int((y_val == 0).sum()),
                "n_features": int(X_res_p.shape[1]),
                "best_depth": info.get("best_depth", ""),
                "fit_seconds": float(info["fit_seconds"]),
            }
        )

        if "dt_tuning" in info:
            for row in info["dt_tuning"]:
                tuning_rows.append(
                    {
                        "lang": "python",
                        "scenario": scenario,
                        "fold": fold_id,
                        "max_depth": row["max_depth"],
                        "mean_f1": row["mean_f1"],
                        "std_f1": row["std_f1"],
                        "selected": row["max_depth"] == info.get("best_depth")
                        or (row["max_depth"] == "None" and info.get("best_depth") is None),
                    }
                )

        grid, tpr_grid, prec_grid = _curve_points(y_val, score)
        for i, x in enumerate(grid):
            curve_rows.append(
                {
                    "lang": "python",
                    "model": model_name,
                    "scenario": scenario,
                    "fold": fold_id,
                    "curve": "roc",
                    "x": float(x),
                    "y": float(tpr_grid[i]),
                }
            )
            curve_rows.append(
                {
                    "lang": "python",
                    "model": model_name,
                    "scenario": scenario,
                    "fold": fold_id,
                    "curve": "pr",
                    "x": float(x),
                    "y": float(prec_grid[i]),
                }
            )

    sampling_row = {
        "lang": "python",
        "scenario": scenario,
        "fold": fold_id,
        "n_before": int(len(train_idx)),
        "pos_before": int((y_train == 1).sum()),
        "neg_before": int((y_train == 0).sum()),
        "positive_rate_before": float((y_train == 1).mean()),
        "n_after": int(counts["n_total"]),
        "pos_after": int(counts["n_positive"]),
        "neg_after": int(counts["n_negative"]),
        "positive_rate_after": float(counts["positive_rate"]),
    }
    unseen_rows = [
        {
            "lang": "python",
            "scenario": scenario,
            "fold": fold_id,
            "column": column,
            "n_unseen": int(count),
        }
        for column, count in unseen.items()
    ]

    return {
        "results": result_rows,
        "runs": run_rows,
        "tuning": tuning_rows,
        "curves": curve_rows,
        "sampling": [sampling_row],
        "unseen": unseen_rows,
    }


def _run_one_learning(
    scenario: str,
    fold_id: int,
    fraction: float,
    train_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: dict[str, Any],
    model_names: list[str],
    seed: int,
    model_n_jobs: int,
) -> list[dict[str, Any]]:
    scenario_idx = SCENARIOS.index(scenario)
    scenario_seed = seed + fold_id * 17 + scenario_idx * 101 + int(round(fraction * 1000))
    X_train_raw = train_df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]

    sub_idx = stratified_subsample(y_train, fraction, seed=scenario_seed)
    X_sub = X_train_raw.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_train[sub_idx]
    res_idx, counts = resample_indices(y_sub, scenario, seed=scenario_seed + 7)
    X_res = X_sub.iloc[res_idx].reset_index(drop=True)
    y_res = y_sub[res_idx]

    pre = Preprocessor(
        categorical=cfg["missing"]["categorical"],
        numeric=cfg["missing"]["numeric"],
        capital_flags=bool(cfg["missing"].get("capital_flags", False)),
    )
    X_res_p = pre.fit_transform(X_res)
    X_val_p = pre.transform(train_df.iloc[val_idx].reset_index(drop=True))
    y_val = y[val_idx]

    rows: list[dict[str, Any]] = []
    for model_idx, model_name in enumerate(model_names):
        model_seed = scenario_seed + model_idx * 1009
        model, info = fit_model(
            model_name,
            X_res_p,
            y_res,
            seed=model_seed,
            svm_probability=False,
            rf_trees=int(cfg["cv"].get("rf_trees", 300)),
            knn_neighbors=int(cfg["cv"].get("knn_neighbors", 5)),
            n_jobs=model_n_jobs,
            tune_dt=False,
            dt_depth=7,
        )
        pred = model.predict(X_val_p)
        score = predict_scores(model, X_val_p)
        metrics = compute_metrics(y_val, pred, score)
        rows.append(
            {
                "lang": "python",
                "model": model_name,
                "scenario": scenario,
                "fold": fold_id,
                "fraction": float(fraction),
                "n_train_original": int(len(sub_idx)),
                "n_train_resampled": int(counts["n_total"]),
                "n_train_positive": int(counts["n_positive"]),
                "n_train_negative": int(counts["n_negative"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "fit_seconds": float(info["fit_seconds"]),
            }
        )
    return rows


def run_cv(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_names: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full CV matrix and write the tidy results."""

    cfg = resolved_paths(cfg)
    results_dir = ensure_dir(cfg["results_dir"])
    model_names = model_names or list(cfg["models"])
    scenarios = scenarios or list(cfg["scenarios"])
    jobs = int(cfg["cv"].get("jobs", 1))
    model_n_jobs = -1 if jobs == 1 else 1

    tasks = [
        (scenario, fold_id, train_idx, val_idx)
        for scenario in scenarios
        for fold_id, (train_idx, val_idx) in enumerate(folds)
    ]

    def _task(args: tuple[Any, ...]) -> dict[str, Any]:
        scenario, fold_id, train_idx, val_idx = args
        return _run_one_fold(
            scenario,
            fold_id,
            train_df,
            y,
            train_idx,
            val_idx,
            cfg,
            model_names,
            int(cfg["seed"]),
            model_n_jobs,
        )

    if jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed

        payloads = Parallel(n_jobs=jobs, verbose=10)(delayed(_task)(t) for t in tasks)
    else:
        payloads = []
        for i, task in enumerate(tasks, start=1):
            if verbose:
                print(f"  [cv] task {i}/{len(tasks)} scenario={task[0]} fold={task[1]}", flush=True)
            payloads.append(_task(task))

    results = pd.DataFrame([row for p in payloads for row in p["results"]])
    runs = pd.DataFrame([row for p in payloads for row in p["runs"]])
    tuning = pd.DataFrame([row for p in payloads for row in p["tuning"]])
    curves = pd.DataFrame([row for p in payloads for row in p["curves"]])
    sampling = pd.DataFrame([row for p in payloads for row in p["sampling"]])
    unseen = pd.DataFrame([row for p in payloads for row in p["unseen"]])

    results = results.sort_values(["scenario", "fold", "model", "metric"]).reset_index(drop=True)
    runs = runs.sort_values(["scenario", "fold", "model"]).reset_index(drop=True)
    sampling = sampling.sort_values(["scenario", "fold"]).reset_index(drop=True)
    curves = curves.sort_values(["scenario", "model", "fold", "curve", "x"]).reset_index(drop=True)

    results.to_csv(results_dir / "cv_tidy.csv", index=False)
    runs.to_csv(results_dir / "cv_runs.csv", index=False)
    sampling.to_csv(results_dir / "sampling_counts.csv", index=False)
    curves.to_csv(results_dir / "curve_points.csv", index=False)
    if not tuning.empty:
        tuning.to_csv(results_dir / "dt_tuning.csv", index=False)
    if not unseen.empty:
        unseen.to_csv(results_dir / "cv_unseen_categories.csv", index=False)
    else:
        pd.DataFrame(
            columns=["lang", "scenario", "fold", "column", "n_unseen"]
        ).to_csv(results_dir / "cv_unseen_categories.csv", index=False)

    summary = (
        results.groupby(["lang", "model", "scenario", "metric"], as_index=False)["value"]
        .agg(mean="mean", std="std", n_folds="count")
        .fillna({"std": 0.0})
    )
    summary.to_csv(results_dir / "cv_summary.csv", index=False)

    wide = summary.pivot_table(
        index=["lang", "model", "scenario"], columns="metric", values="mean"
    ).reset_index()
    wide.columns.name = None
    wide.to_csv(results_dir / "cv_summary_wide.csv", index=False)

    if verbose:
        expected = len(model_names) * len(scenarios) * len(folds) * len(METRIC_NAMES)
        if len(results) != expected:
            print(
                f"  [cv] WARNING: {len(results)} tidy rows, expected {expected} "
                f"({len(model_names)} models x {len(scenarios)} scenarios x "
                f"{len(folds)} folds x {len(METRIC_NAMES)} metrics)",
                flush=True,
            )
        else:
            print(f"  [cv] row-count check OK: {expected} tidy rows", flush=True)
        print(f"  [cv] wrote {len(results)} tidy rows, {len(summary)} summary rows", flush=True)

    return {
        "tidy": results,
        "summary": summary,
        "runs": runs,
        "sampling": sampling,
        "curves": curves,
        "tuning": tuning,
        "unseen": unseen,
    }


def run_learning_curves(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_names: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train-size fraction vs F1 / ROC-AUC on the first fold(s)."""

    cfg = resolved_paths(cfg)
    results_dir = ensure_dir(cfg["results_dir"])
    model_names = model_names or list(cfg["models"])
    scenarios = scenarios or list(cfg["scenarios"])
    fractions = [float(f) for f in cfg["cv"].get("learning_fractions", [0.1, 0.25, 0.5, 0.75, 1.0])]
    n_learning_folds = min(int(cfg["cv"].get("learning_folds", 1)), len(folds))
    jobs = int(cfg["cv"].get("jobs", 1))
    model_n_jobs = -1 if jobs == 1 else 1

    tasks = [
        (scenario, fold_id, fraction, folds[fold_id][0], folds[fold_id][1])
        for scenario in scenarios
        for fold_id in range(n_learning_folds)
        for fraction in fractions
    ]

    def _task(args: tuple[Any, ...]) -> list[dict[str, Any]]:
        scenario, fold_id, fraction, train_idx, val_idx = args
        return _run_one_learning(
            scenario,
            fold_id,
            fraction,
            train_df,
            y,
            train_idx,
            val_idx,
            cfg,
            model_names,
            int(cfg["seed"]),
            model_n_jobs,
        )

    if jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed

        payloads = Parallel(n_jobs=jobs, verbose=10)(delayed(_task)(t) for t in tasks)
    else:
        payloads = []
        for i, task in enumerate(tasks, start=1):
            if verbose:
                print(
                    f"  [learning] task {i}/{len(tasks)} scenario={task[0]} "
                    f"fold={task[1]} fraction={task[2]:.2f}",
                    flush=True,
                )
            payloads.append(_task(task))

    rows = [row for payload in payloads for row in payload]
    df = pd.DataFrame(rows).sort_values(
        ["scenario", "model", "fold", "fraction"]
    ).reset_index(drop=True)
    df.to_csv(results_dir / "learning_curves.csv", index=False)
    if verbose:
        print(f"  [learning] wrote {len(df)} rows", flush=True)
    return df
