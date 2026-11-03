"""Full-training-set evaluation on ``adult.test``.

This is deliberately separate from the CV sweep: each model is trained once on
the entire training split under each sampling scenario, then scored on the
untouched test set.  Predictions are persisted so the error-analysis phase can
reuse them without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import TARGET_COLUMN
from .metrics import METRIC_NAMES, compute_metrics
from .models import MODEL_NAMES, fit_model, predict_scores
from .paths import ensure_dir
from .preprocess import Preprocessor, category_coverage
from .sampling import SCENARIOS, resample_indices


def run_test_eval(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    results_dir = ensure_dir(cfg["results_dir"])
    model_names = model_names or list(cfg["models"])
    scenarios = scenarios or list(cfg["scenarios"])
    seed = int(cfg["seed"])
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int)
    X_test_raw = test_df.drop(columns=[TARGET_COLUMN, "row_id"], errors="ignore")

    metric_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    unseen_rows: list[dict[str, Any]] = []

    for scenario_idx, scenario in enumerate(scenarios):
        scenario_seed = seed + scenario_idx * 101
        res_idx, counts = resample_indices(y_train, scenario, seed=scenario_seed)
        X_res = train_df.iloc[res_idx].reset_index(drop=True)
        y_res = y_train[res_idx]

        pre = Preprocessor(
            categorical=cfg["missing"]["categorical"],
            numeric=cfg["missing"]["numeric"],
            capital_flags=bool(cfg["missing"].get("capital_flags", False)),
        )
        X_res_p = pre.fit_transform(X_res)
        X_test_p = pre.transform(X_test_raw)
        for column, count in pre.unseen_counts_.items():
            unseen_rows.append(
                {
                    "lang": "python",
                    "scenario": scenario,
                    "column": column,
                    "n_unseen_test_rows": int(count),
                }
            )

        if verbose:
            print(
                f"  [test] scenario={scenario} train={len(res_idx):,} "
                f"pos={counts['n_positive']:,} neg={counts['n_negative']:,}",
                flush=True,
            )

        for model_idx, model_name in enumerate(model_names):
            model_seed = scenario_seed + model_idx * 1009
            model, info = fit_model(
                model_name,
                X_res_p,
                y_res,
                seed=model_seed,
                svm_probability=bool(cfg.get("test", {}).get("svm_probability", False)),
                rf_trees=int(cfg["cv"].get("rf_trees", 300)),
                knn_neighbors=int(cfg["cv"].get("knn_neighbors", 5)),
                n_jobs=-1,
                tune_dt=True,
                inner_splits=int(cfg["cv"].get("inner_splits", 3)),
                dt_depths=list(cfg["cv"].get("dt_depths", [3, 5, 7, 10, None])),
            )
            pred = model.predict(X_test_p)
            score = predict_scores(model, X_test_p)
            metrics = compute_metrics(y_test, pred, score)
            for metric in METRIC_NAMES:
                metric_rows.append(
                    {
                        "lang": "python",
                        "model": model_name,
                        "scenario": scenario,
                        "metric": metric,
                        "value": float(metrics[metric]),
                    }
                )
            run_rows.append(
                {
                    "lang": "python",
                    "model": model_name,
                    "scenario": scenario,
                    "n_train": int(len(res_idx)),
                    "n_train_positive": int(counts["n_positive"]),
                    "n_train_negative": int(counts["n_negative"]),
                    "n_test": int(len(test_df)),
                    "n_features": int(X_res_p.shape[1]),
                    "best_depth": info.get("best_depth", ""),
                    "fit_seconds": float(info["fit_seconds"]),
                }
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "lang": "python",
                        "model": model_name,
                        "scenario": scenario,
                        "row_id": test_df["row_id"].to_numpy(dtype=int),
                        "y_true": y_test,
                        "y_pred": np.asarray(pred, dtype=int),
                        "y_score": np.asarray(score, dtype=float),
                    }
                )
            )
            if verbose:
                print(
                    f"    {model_name:<20} acc={metrics['accuracy']:.4f} "
                    f"f1={metrics['f1']:.4f} pr_auc={metrics['pr_auc']:.4f} "
                    f"({info['fit_seconds']:.1f}s)",
                    flush=True,
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["scenario", "model", "metric"]
    ).reset_index(drop=True)
    runs_df = pd.DataFrame(run_rows).sort_values(["scenario", "model"]).reset_index(drop=True)
    preds_df = pd.concat(pred_rows, ignore_index=True)
    unseen_df = pd.DataFrame(
        unseen_rows,
        columns=["lang", "scenario", "column", "n_unseen_test_rows"],
    )

    metrics_df.to_csv(results_dir / "test_metrics.csv", index=False)
    runs_df.to_csv(results_dir / "test_runs.csv", index=False)
    preds_df.to_csv(results_dir / "test_predictions.csv", index=False, float_format="%.6f")
    unseen_df.to_csv(results_dir / "unseen_categories.csv", index=False)

    wide = metrics_df.pivot_table(
        index=["lang", "model", "scenario"], columns="metric", values="value"
    ).reset_index()
    wide.columns.name = None
    wide.to_csv(results_dir / "test_summary_wide.csv", index=False)

    coverage = category_coverage(train_df, test_df)
    coverage.to_csv(results_dir / "category_coverage.csv", index=False)

    # CV vs test side-by-side table.
    cv_path = results_dir / "cv_summary.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        merged = metrics_df.merge(
            cv[["lang", "model", "scenario", "metric", "mean", "std"]],
            on=["lang", "model", "scenario", "metric"],
            how="left",
            suffixes=("_test", "_cv"),
        )
        merged = merged.rename(
            columns={"value": "test", "mean": "cv_mean", "std": "cv_std"}
        )
        merged["delta_test_minus_cv"] = merged["test"] - merged["cv_mean"]
        merged = merged.sort_values(["scenario", "model", "metric"]).reset_index(drop=True)
        merged.to_csv(results_dir / "cv_vs_test.csv", index=False)

    if verbose:
        expected = len(model_names) * len(scenarios) * len(METRIC_NAMES)
        if len(metrics_df) != expected:
            print(
                f"  [test] WARNING: {len(metrics_df)} metric rows, expected {expected}",
                flush=True,
            )
        else:
            print(f"  [test] row-count check OK: {expected} metric rows", flush=True)
        print(f"  [test] wrote {len(metrics_df)} metric rows and {len(preds_df):,} predictions", flush=True)

    return {
        "metrics": metrics_df,
        "runs": runs_df,
        "predictions": preds_df,
        "unseen": unseen_df,
        "coverage": coverage,
    }
