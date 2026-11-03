"""Adult / Census Income data loading, cleaning, and fold generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, resolved_paths
from .kfold import stratified_kfold_indices, validate_folds
from .paths import ensure_dir, repo_root

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

TARGET_COLUMN = "class"
MISSING_TOKEN = "?"


def _read_adult_rows(path: str | Path, is_test: bool) -> pd.DataFrame:
    """Parse an Adult data file without trusting pandas' header heuristics."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Adult data file not found: {path}")

    records: list[list[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # adult.test starts with the junk line "|1x3 Cross validator".
            if line.startswith("|"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != len(COLUMNS):
                raise ValueError(f"{path}: expected {len(COLUMNS)} fields, got {len(fields)} in {line!r}")
            records.append(fields)

    df = pd.DataFrame(records, columns=COLUMNS)

    # Missing values are encoded as "?" in workclass, occupation, native-country.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].replace({MISSING_TOKEN: np.nan})
        df[col] = df[col].where(~df[col].isin(["", "nan", "NaN", "None"]), np.nan)

    # Numeric columns should parse cleanly; a coercion failure is a data bug.
    for col in NUMERIC_COLUMNS:
        original = df[col]
        parsed = pd.to_numeric(original, errors="coerce")
        bad = parsed.isna() & original.notna()
        if bad.any():
            raise ValueError(f"{path}: non-numeric values in {col}: {original[bad].head().tolist()}")
        df[col] = parsed

    labels = df[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")
    if is_test:
        # adult.test labels end with a period; rstrip above already removed it.
        pass
    mapping = {">50K": 1, "<=50K": 0}
    unknown = sorted(set(labels.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"{path}: unknown class labels {unknown}")
    df[TARGET_COLUMN] = labels.map(mapping).astype(int)

    return df


def load_adult(path: str | Path, is_test: bool = False) -> pd.DataFrame:
    """Load one Adult split with whitespace stripped and labels as 0/1."""

    df = _read_adult_rows(path, is_test=is_test)
    if is_test:
        # Keep an explicit row id for reproducible error analysis.
        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df), dtype=int)
    return df


def load_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_adult(cfg["data"]["train"], is_test=False)
    test = load_adult(cfg["data"]["test"], is_test=True)
    return train, test


def class_balance(y: pd.Series | np.ndarray) -> dict[str, float]:
    arr = np.asarray(y, dtype=int)
    n = len(arr)
    pos = int(arr.sum())
    neg = n - pos
    return {
        "n": n,
        "positive": pos,
        "negative": neg,
        "positive_rate": pos / n if n else 0.0,
        "negative_rate": neg / n if n else 0.0,
    }


def make_folds(
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = stratified_kfold_indices(y, n_splits=n_splits, seed=seed)
    validate_folds(folds, len(y), y)
    return folds


def save_folds(
    path: str | Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series | np.ndarray,
    n_splits: int,
    seed: int,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "seed": int(seed),
        "n_splits": int(n_splits),
        "n_rows": int(len(y)),
        "class_balance": class_balance(y),
        "folds": [
            {
                "fold": int(k),
                "train": [int(i) for i in train_idx],
                "val": [int(i) for i in val_idx],
            }
            for k, (train_idx, val_idx) in enumerate(folds)
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path


def load_folds(
    path: str | Path,
    n_rows: int | None = None,
    y: pd.Series | np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fold file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    folds = [
        (np.asarray(item["train"], dtype=int), np.asarray(item["val"], dtype=int))
        for item in payload["folds"]
    ]
    if n_rows is None:
        n_rows = int(payload["n_rows"])
    validate_folds(folds, n_rows, y)
    return folds


def ensure_folds(
    cfg: dict[str, Any],
    y: pd.Series | np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    path = Path(cfg["folds"])
    if path.exists() and not cfg.get("_force_folds", False):
        folds = load_folds(path, n_rows=len(y), y=y)
        if len(folds) == int(cfg["n_splits"]):
            return folds
        raise ValueError(
            f"{path} contains {len(folds)} folds but config asks for "
            f"{cfg['n_splits']}; delete it, point --folds elsewhere, or pass --force-folds"
        )
    folds = make_folds(y, n_splits=int(cfg["n_splits"]), seed=int(cfg["seed"]))
    save_folds(path, folds, y, n_splits=int(cfg["n_splits"]), seed=int(cfg["seed"]))
    return folds


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COLUMNS:
        if col == TARGET_COLUMN:
            continue
        rows.append(
            {
                "column": col,
                "dtype": "numeric" if col in NUMERIC_COLUMNS else "categorical",
                "n_missing": int(df[col].isna().sum()),
                "pct_missing": float(df[col].isna().mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _run_check(args: argparse.Namespace) -> int:
    cfg = resolved_paths(load_config(args.config))
    if args.data:
        cfg["data"]["train"] = str(Path(args.data) / "adult.data")
        cfg["data"]["test"] = str(Path(args.data) / "adult.test")
    train, test = load_dataset(cfg)

    print("=" * 72)
    print("CSC 869 MP2 -- data check")
    print("=" * 72)
    print(f"train rows: {len(train):,}")
    print(f"test  rows: {len(test):,}")
    for name, df in (("train", train), ("test", test)):
        bal = class_balance(df[TARGET_COLUMN])
        print(
            f"{name} class balance: "
            f">50K {bal['positive']:,} ({bal['positive_rate'] * 100:.2f}%) / "
            f"<=50K {bal['negative']:,} ({bal['negative_rate'] * 100:.2f}%)"
        )

    print("\nmissing values before imputation:")
    for name, df in (("train", train), ("test", test)):
        ms = missing_summary(df)
        nonzero = ms[ms["n_missing"] > 0]
        print(f"  {name}:")
        for _, row in nonzero.iterrows():
            print(f"    {row['column']:<16} {int(row['n_missing']):>5} ({row['pct_missing']:.2f}%)")
        if nonzero.empty:
            print("    none")

    # Fit the default preprocessor on the full training split and confirm that
    # no NaN survives encoding.
    from .preprocess import Preprocessor

    pre = Preprocessor(
        categorical=cfg["missing"]["categorical"],
        numeric=cfg["missing"]["numeric"],
        capital_flags=bool(cfg["missing"].get("capital_flags", False)),
    )
    X_train = pre.fit_transform(train.drop(columns=[TARGET_COLUMN, "row_id"], errors="ignore"))
    X_test = pre.transform(test.drop(columns=[TARGET_COLUMN, "row_id"], errors="ignore"))
    print(
        f"\nafter imputation/encoding: {X_train.shape[1]} features; "
        f"NaNs train={int(np.isnan(X_train).sum())} test={int(np.isnan(X_test).sum())}"
    )
    print(f"categorical strategy: {cfg['missing']['categorical']}")
    print(f"numeric strategy:     {cfg['missing']['numeric']}")

    folds = ensure_folds(cfg, train[TARGET_COLUMN])
    print(f"\nfolds: {len(folds)} x stratified, seed={cfg['seed']} -> {cfg['folds']}")
    for k, (tr, va) in enumerate(folds):
        bal = class_balance(train[TARGET_COLUMN].to_numpy()[va])
        print(
            f"  fold {k}: train={len(tr):,} val={len(va):,} "
            f"val positive={bal['positive_rate'] * 100:.2f}%"
        )

    # MP1 legacy sanity band: a from-scratch mixed Naive Bayes on 10-fold CV
    # should land around 81-84% accuracy on Adult.
    if not args.skip_legacy:
        from .legacy import run_legacy_cv

        print("\nlegacy MP1 Naive Bayes (10-fold CV):")
        legacy = run_legacy_cv(train, n_splits=10, seed=cfg["seed"], verbose=False)
        summary = legacy.groupby("metric")["value"].agg(["mean", "std"]).reset_index()
        for _, row in summary.iterrows():
            print(f"  {row['metric']:<10} {row['mean']:.4f} +/- {row['std']:.4f}")
        acc = float(summary.loc[summary["metric"] == "accuracy", "mean"].iloc[0])
        if 0.78 <= acc <= 0.88:
            print(f"  sanity band: PASS (accuracy {acc:.4f})")
        else:
            print(f"  sanity band: WARNING (accuracy {acc:.4f}, expected ~0.81-0.84)")

    print("\nOK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load, clean, and inspect the Adult data")
    parser.add_argument("--config", default=None, help="optional YAML config")
    parser.add_argument("--data", default=None, help="directory containing adult.data/adult.test")
    parser.add_argument("--check", action="store_true", help="run the full data sanity check")
    parser.add_argument("--skip-legacy", action="store_true", help="skip the MP1 NBC sanity run")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 0
    return _run_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
