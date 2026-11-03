"""End-to-end runner for the Python implementation.

Typical use::

    cd components/mp2-py
    python -m csc869.run --config configs/main.yaml --out results/

Stages can be selected individually with ``--stages``, e.g.
``--stages prepare,eda,cv,curves,test,error,legacy,report``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import apply_quick, load_config, resolved_paths
from .data import (
    TARGET_COLUMN,
    class_balance,
    ensure_folds,
    load_dataset,
    missing_summary,
)
from .kfold import stratified_subsample
from .models import MODEL_NAMES
from .paths import ensure_dir, repo_root
from .sampling import SCENARIOS

ALL_STAGES = [
    "prepare",
    "eda",
    "missing",
    "cv",
    "learning",
    "curves",
    "test",
    "error",
    "legacy",
    "report",
    "typst",
]


def _subset_train(train: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or max_rows >= len(train):
        return train
    y = train[TARGET_COLUMN].to_numpy(dtype=int)
    fraction = max(0.01, min(1.0, max_rows / len(train)))
    idx = stratified_subsample(y, fraction=fraction, seed=seed)
    return train.iloc[idx].reset_index(drop=True)


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "csc869": __version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSC 869 MP2 Python experiment runner")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/main.yaml)")
    parser.add_argument("--out", default=None, help="override results directory")
    parser.add_argument("--figures", default=None, help="override figures directory")
    parser.add_argument("--data", default=None, help="directory containing adult.data/adult.test")
    parser.add_argument("--stages", default=None, help=f"comma-separated subset of {ALL_STAGES}")
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--scenarios", default=None, help="comma-separated sampling scenarios")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="stratified training subset for smoke tests")
    parser.add_argument("--force-folds", action="store_true", help="regenerate folds even if the file exists")
    parser.add_argument("--quick", action="store_true", help="use the quick smoke-test profile")
    parser.add_argument("--list-stages", action="store_true")
    args = parser.parse_args(argv)

    if args.list_stages:
        print("\n".join(ALL_STAGES))
        return 0

    config_path = args.config
    if config_path is None:
        default = Path("configs/main.yaml")
        config_path = str(default) if default.exists() else None

    cfg = load_config(config_path)
    if args.quick:
        cfg = apply_quick(cfg, quick=True)

    if args.out:
        cfg["results_dir"] = args.out
    if args.figures:
        cfg["figures_dir"] = args.figures
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.jobs is not None:
        cfg["cv"]["jobs"] = int(args.jobs)
    if args.data:
        cfg["data"]["train"] = str(Path(args.data) / "adult.data")
        cfg["data"]["test"] = str(Path(args.data) / "adult.test")
    if args.max_rows is not None:
        cfg["_max_rows"] = int(args.max_rows)
    if args.force_folds:
        cfg["_force_folds"] = True
    if args.models:
        cfg["models"] = _parse_csv_arg(args.models)
    if args.scenarios:
        cfg["scenarios"] = _parse_csv_arg(args.scenarios)

    bad_models = [m for m in cfg["models"] if m not in MODEL_NAMES]
    if bad_models:
        parser.error(f"unknown models {bad_models}; expected {MODEL_NAMES}")
    bad_scenarios = [s for s in cfg["scenarios"] if s not in SCENARIOS]
    if bad_scenarios:
        parser.error(f"unknown scenarios {bad_scenarios}; expected {SCENARIOS}")

    stages = _parse_csv_arg(args.stages) or list(ALL_STAGES)
    unknown_stages = [s for s in stages if s not in ALL_STAGES]
    if unknown_stages:
        parser.error(f"unknown stages {unknown_stages}; expected {ALL_STAGES}")

    cfg = resolved_paths(cfg)
    results_dir = ensure_dir(cfg["results_dir"])
    figures_dir = ensure_dir(cfg["figures_dir"])
    cfg["figures_dir"] = str(figures_dir)

    print("=" * 78)
    print("CSC 869 MP2 -- Python implementation")
    print("=" * 78)
    print(f"repo root : {cfg['_repo_root']}")
    print(f"results   : {results_dir}")
    print(f"figures   : {figures_dir}")
    print(f"folds     : {cfg['folds']}")
    print(f"models    : {', '.join(cfg['models'])}")
    print(f"scenarios : {', '.join(cfg['scenarios'])}")
    print(f"stages    : {', '.join(stages)}")
    if cfg.get("_quick"):
        print("mode      : QUICK (separate results_quick / figures_quick paths)")
    print("=" * 78, flush=True)

    started = time.perf_counter()
    durations: dict[str, float] = {}

    def timed(stage: str, fn):
        t0 = time.perf_counter()
        print(f"\n--- stage: {stage} ---", flush=True)
        out = fn()
        durations[stage] = float(time.perf_counter() - t0)
        print(f"--- {stage} done in {durations[stage]:.1f}s ---", flush=True)
        return out

    # Data is needed by most stages; load it once.
    train, test = load_dataset(cfg)
    max_rows = cfg.get("_max_rows")
    train = _subset_train(train, max_rows=max_rows, seed=int(cfg["seed"]))
    y = train[TARGET_COLUMN].to_numpy(dtype=int)
    print(f"\nloaded train={len(train):,} test={len(test):,}", flush=True)

    folds = ensure_folds(cfg, y)
    print(f"using {len(folds)} stratified folds from {cfg['folds']}", flush=True)

    if "prepare" in stages:
        def _prepare():
            summary_rows = []
            for name, df in (("train", train), ("test", test)):
                bal = class_balance(df[TARGET_COLUMN])
                summary_rows.append(
                    {
                        "split": name,
                        "n_rows": len(df),
                        "n_positive": bal["positive"],
                        "n_negative": bal["negative"],
                        "positive_rate": bal["positive_rate"],
                        "negative_rate": bal["negative_rate"],
                    }
                )
            pd.DataFrame(summary_rows).to_csv(results_dir / "data_summary.csv", index=False)
            missing_summary(train).to_csv(results_dir / "missing_values_train.csv", index=False)
            missing_summary(test).to_csv(results_dir / "missing_values_test.csv", index=False)

            # Full-training-set sampling counts (the per-fold counts are in
            # sampling_counts.csv from the CV stage).
            from .sampling import scenario_counts

            full_rows = []
            for scenario in cfg["scenarios"]:
                counts = scenario_counts(y, scenario, seed=int(cfg["seed"]))
                full_rows.append(
                    {
                        "lang": "python",
                        "scenario": scenario,
                        "n_before": int(len(y)),
                        "pos_before": int((y == 1).sum()),
                        "neg_before": int((y == 0).sum()),
                        "n_after": int(counts["n_total"]),
                        "pos_after": int(counts["n_positive"]),
                        "neg_after": int(counts["n_negative"]),
                        "positive_rate_after": float(counts["positive_rate"]),
                    }
                )
            pd.DataFrame(full_rows).to_csv(results_dir / "sampling_counts_full.csv", index=False)
            return None

        timed("prepare", _prepare)

    if "eda" in stages:
        from .eda import run_eda

        timed("eda", lambda: run_eda(cfg, train, test, verbose=True))

    if "missing" in stages:
        from .missing import run_missing_ablation

        timed(
            "missing",
            lambda: run_missing_ablation(cfg, train, y, folds, verbose=True),
        )

    if "cv" in stages:
        from .cv import run_cv

        timed(
            "cv",
            lambda: run_cv(
                cfg,
                train,
                y,
                folds,
                model_names=cfg["models"],
                scenarios=cfg["scenarios"],
                verbose=True,
            ),
        )

    if "learning" in stages:
        from .cv import run_learning_curves

        timed(
            "learning",
            lambda: run_learning_curves(
                cfg,
                train,
                y,
                folds,
                model_names=cfg["models"],
                scenarios=cfg["scenarios"],
                verbose=True,
            ),
        )

    if "curves" in stages:
        from .curves import (
            make_learning_curve_figures,
            make_roc_pr_figures,
            make_summary_heatmap,
        )

        def _curves():
            make_roc_pr_figures(cfg, verbose=True)
            make_summary_heatmap(cfg, verbose=True)
            make_learning_curve_figures(cfg, verbose=True)

        timed("curves", _curves)

    if "test" in stages:
        from .test_eval import run_test_eval

        timed(
            "test",
            lambda: run_test_eval(
                cfg,
                train,
                test,
                model_names=cfg["models"],
                scenarios=cfg["scenarios"],
                verbose=True,
            ),
        )

    if "error" in stages:
        from .error_analysis import run_error_analysis

        timed("error", lambda: run_error_analysis(cfg, test, verbose=True))

    if "legacy" in stages:
        from .legacy import run_legacy_cv

        def _legacy():
            per_fold = run_legacy_cv(train, n_splits=10, seed=int(cfg["seed"]), verbose=True)
            per_fold.to_csv(results_dir / "legacy_nbc_cv.csv", index=False)
            summary = (
                per_fold.groupby("metric")["value"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"count": "n_folds"})
            )
            summary.to_csv(results_dir / "legacy_nbc_summary.csv", index=False)
            return summary

        timed("legacy", _legacy)

    if "report" in stages:
        from .report import generate_report

        timed("report", lambda: generate_report(cfg, train, test, verbose=True))

    if "typst" in stages:
        from .typst_report import generate_typst_report

        timed("typst", lambda: generate_typst_report(cfg, train, test, verbose=True))

    elapsed = time.perf_counter() - started
    run_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": elapsed,
        "stages": stages,
        "durations": durations,
    }

    # A partial re-run (e.g. `--stages prepare,missing,report` after fixing a
    # bug) must not erase the record of the expensive stages that produced the
    # existing artifacts.  Keep every invocation in `history` and expose the
    # union of stages/durations at the top level.
    meta_path = results_dir / "run_metadata.json"
    history: list[dict[str, Any]] = []
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        history = list(previous.get("history", []))
        if not history and previous.get("stages"):
            # Metadata written before history existed: treat it as record 1.
            history.append(
                {
                    "timestamp": previous.get("timestamp"),
                    "elapsed_seconds": previous.get("elapsed_seconds", 0.0),
                    "stages": previous.get("stages", []),
                    "durations": previous.get("durations", {}),
                }
            )
    history.append(run_record)

    all_durations: dict[str, float] = {}
    for record in history:
        all_durations.update(record.get("durations", {}))
    all_stages = [s for s in ALL_STAGES if any(s in record.get("stages", []) for record in history)]

    metadata = {
        "version": __version__,
        "timestamp": run_record["timestamp"],
        "elapsed_seconds": float(sum(record.get("elapsed_seconds", 0.0) for record in history)),
        "durations": all_durations,
        "quick": bool(cfg.get("_quick")),
        "stages": all_stages,
        "history": history,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "versions": _versions(),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"done in {elapsed / 60:.1f} min")
    for stage, seconds in durations.items():
        print(f"  {stage:<10} {seconds:7.1f}s")
    print(f"results: {results_dir}")
    print(f"figures: {figures_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
