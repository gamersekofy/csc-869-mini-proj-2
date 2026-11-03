"""Run the carried-forward MP1 Naive Bayes experiment.

Usage::

    python -m csc869.legacy --check
    python -m csc869.legacy --folds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config, resolved_paths
from ..data import TARGET_COLUMN, load_dataset
from ..paths import ensure_dir
from .naive_bayes import run_legacy_cv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MP1 Naive Bayes + k-fold CV sanity run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="run and print the summary")
    args = parser.parse_args(argv)

    cfg = resolved_paths(load_config(args.config))
    train, _ = load_dataset(cfg)
    results_dir = ensure_dir(cfg["results_dir"])

    print(f"legacy NBC: {len(train):,} training rows, {args.folds}-fold stratified CV")
    per_fold = run_legacy_cv(train, n_splits=args.folds, seed=args.seed, verbose=True)
    per_fold.to_csv(results_dir / "legacy_nbc_cv.csv", index=False)

    summary = (
        per_fold.groupby("metric")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_folds"})
    )
    summary.to_csv(results_dir / "legacy_nbc_summary.csv", index=False)
    print("\nlegacy NBC summary:")
    for _, row in summary.iterrows():
        print(f"  {row['metric']:<10} {row['mean']:.4f} +/- {row['std']:.4f} (n={int(row['n_folds'])})")

    acc = float(summary.loc[summary["metric"] == "accuracy", "mean"].iloc[0])
    if args.check:
        if 0.78 <= acc <= 0.88:
            print(f"\ncheck: PASS (accuracy {acc:.4f}; expected MP1 band ~0.81-0.84)")
            return 0
        print(f"\ncheck: WARNING (accuracy {acc:.4f}; expected MP1 band ~0.81-0.84)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
