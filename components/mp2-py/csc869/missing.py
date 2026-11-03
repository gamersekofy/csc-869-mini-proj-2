"""Missing-value strategy ablation.

The rubric asks for two strategies per feature type and for both to be
*evaluated*, not just implemented.  This module runs a compact decision-tree CV
over the 2x2 grid:

    categorical: unknown | mode
    numeric:     median  | indicator

Adult has no missing numeric values, so the two numeric variants are identical
in practice; the ablation makes that explicit instead of hiding it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import TARGET_COLUMN
from .metrics import METRIC_NAMES, compute_metrics
from .models import fit_model, predict_scores
from .paths import ensure_dir
from .preprocess import Preprocessor


def run_missing_ablation(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> pd.DataFrame:
    results_dir = ensure_dir(cfg["results_dir"])
    combos = [
        ("unknown", "median"),
        ("unknown", "indicator"),
        ("mode", "median"),
        ("mode", "indicator"),
    ]
    rows: list[dict[str, Any]] = []
    seed = int(cfg["seed"])

    for categorical, numeric in combos:
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            X_train_raw = train_df.iloc[train_idx].reset_index(drop=True)
            y_train = y[train_idx]
            pre = Preprocessor(categorical=categorical, numeric=numeric, capital_flags=False)
            X_train = pre.fit_transform(X_train_raw)
            X_val = pre.transform(train_df.iloc[val_idx].reset_index(drop=True))
            y_val = y[val_idx]

            model, info = fit_model(
                "decision_tree",
                X_train,
                y_train,
                seed=seed + fold_id * 17,
                n_jobs=-1,
                tune_dt=True,
                inner_splits=int(cfg["cv"].get("inner_splits", 3)),
                dt_depths=list(cfg["cv"].get("dt_depths", [3, 5, 7, 10, None])),
            )
            pred = model.predict(X_val)
            score = predict_scores(model, X_val)
            metrics = compute_metrics(y_val, pred, score)
            for metric in METRIC_NAMES:
                rows.append(
                    {
                        "categorical_strategy": categorical,
                        "numeric_strategy": numeric,
                        "fold": fold_id,
                        "metric": metric,
                        "value": float(metrics[metric]),
                        "n_features": int(X_train.shape[1]),
                        "best_depth": info.get("best_depth", ""),
                    }
                )

    df = pd.DataFrame(rows).sort_values(
        ["categorical_strategy", "numeric_strategy", "fold", "metric"]
    ).reset_index(drop=True)
    df.to_csv(results_dir / "missing_ablation.csv", index=False)

    summary = (
        df.groupby(["categorical_strategy", "numeric_strategy", "metric"], as_index=False)["value"]
        .agg(mean="mean", std="std", n_folds="count")
        .fillna({"std": 0.0})
    )
    summary.to_csv(results_dir / "missing_ablation_summary.csv", index=False)

    if verbose:
        print(f"  [missing] wrote {len(df)} ablation rows", flush=True)
    return df
