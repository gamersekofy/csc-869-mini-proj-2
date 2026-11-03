"""Leak-free preprocessing for the Adult data.

Everything that learns a statistic from data -- category vocabularies, fill
values, medians, means and standard deviations -- is fitted on the training
portion of a fold only.  The validation/test split is transformed with those
frozen statistics.  This is the single most important guard against the
"scaler fit outside the fold" trap.

Two missing-value strategies are supported for each feature type:

categorical
    ``unknown``  treat ``?`` as its own category (usually the best strategy)
    ``mode``     impute the training mode

numeric
    ``median``   median impute
    ``indicator`` median impute plus a missing-indicator column

Adult has no missing numeric values, so the numeric strategies are identical
in practice; the report says so explicitly instead of inventing missingness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN


@dataclass
class Preprocessor:
    categorical: str = "unknown"
    numeric: str = "indicator"
    capital_flags: bool = False
    cat_cols_: list[str] = field(default_factory=list)
    num_cols_: list[str] = field(default_factory=list)
    cat_fill_: dict[str, Any] = field(default_factory=dict)
    cat_mode_: dict[str, Any] = field(default_factory=dict)
    cat_categories_: dict[str, list[Any]] = field(default_factory=dict)
    num_median_: dict[str, float] = field(default_factory=dict)
    num_mean_: dict[str, float] = field(default_factory=dict)
    num_std_: dict[str, float] = field(default_factory=dict)
    indicator_cols_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    unseen_counts_: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.categorical not in {"unknown", "mode"}:
            raise ValueError("categorical strategy must be 'unknown' or 'mode'")
        if self.numeric not in {"median", "indicator"}:
            raise ValueError("numeric strategy must be 'median' or 'indicator'")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> "Preprocessor":
        self.cat_cols_ = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
        self.num_cols_ = [c for c in NUMERIC_COLUMNS if c in X.columns]
        self.cat_fill_ = {}
        self.cat_mode_ = {}
        self.cat_categories_ = {}
        self.num_median_ = {}
        self.num_mean_ = {}
        self.num_std_ = {}
        self.indicator_cols_ = []

        for col in self.cat_cols_:
            series = X[col].astype(object)
            series = series.where(~pd.isna(series), np.nan)
            mode_series = series.dropna().mode()
            mode = mode_series.iloc[0] if len(mode_series) else "Unknown"
            self.cat_mode_[col] = mode
            fill = "Unknown" if self.categorical == "unknown" else mode
            self.cat_fill_[col] = fill
            filled = series.fillna(fill)
            categories = sorted(filled.dropna().unique().tolist(), key=lambda v: str(v))
            self.cat_categories_[col] = categories

        for col in self.num_cols_:
            series = pd.to_numeric(X[col], errors="coerce")
            median = series.median()
            if pd.isna(median):
                median = 0.0
            self.num_median_[col] = float(median)
            filled = series.fillna(median)
            self.num_mean_[col] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.num_std_[col] = std if std > 1e-12 else 1.0
            if self.numeric == "indicator" and series.isna().any():
                self.indicator_cols_.append(col)

        names: list[str] = []
        for col in self.cat_cols_:
            names.extend([f"{col}={cat}" for cat in self.cat_categories_[col]])
        for col in self.num_cols_:
            if col in self.indicator_cols_:
                names.append(f"{col}_missing")
            names.append(col)
        if self.capital_flags:
            names.extend(["capital-gain>0", "capital-loss>0"])
        self.feature_names_ = names
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("Preprocessor must be fitted before transform()")
        self.unseen_counts_ = {}
        parts: list[np.ndarray] = []

        for col in self.cat_cols_:
            if col not in X.columns:
                raise KeyError(f"missing categorical column {col!r}")
            series = X[col].astype(object)
            series = series.where(~pd.isna(series), self.cat_fill_[col])
            categories = self.cat_categories_[col]
            unseen = ~series.isin(categories)
            if bool(unseen.any()):
                self.unseen_counts_[col] = int(unseen.sum())
                series = series.where(~unseen, self.cat_mode_[col])
            for cat in categories:
                parts.append((series == cat).to_numpy(dtype=float))

        for col in self.num_cols_:
            if col not in X.columns:
                raise KeyError(f"missing numeric column {col!r}")
            series = pd.to_numeric(X[col], errors="coerce")
            if col in self.indicator_cols_:
                parts.append(series.isna().to_numpy(dtype=float))
            filled = series.fillna(self.num_median_[col]).to_numpy(dtype=float)
            parts.append((filled - self.num_mean_[col]) / self.num_std_[col])

        if self.capital_flags:
            gain = pd.to_numeric(X["capital-gain"], errors="coerce").fillna(0).to_numpy(dtype=float)
            loss = pd.to_numeric(X["capital-loss"], errors="coerce").fillna(0).to_numpy(dtype=float)
            parts.append((gain > 0).astype(float))
            parts.append((loss > 0).astype(float))

        if not parts:
            return np.empty((len(X), 0), dtype=float)
        return np.column_stack(parts)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_)

    def describe(self) -> dict[str, Any]:
        return {
            "categorical_strategy": self.categorical,
            "numeric_strategy": self.numeric,
            "n_features": len(self.feature_names_),
            "n_categorical_columns": len(self.cat_cols_),
            "n_numeric_columns": len(self.num_cols_),
            "n_missing_indicators": len(self.indicator_cols_),
            "n_unseen_by_column": dict(self.unseen_counts_),
        }


def category_coverage(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Report categories that occur in only one split.

    This is the table behind the "unseen categories at test time" paragraph in
    the report.  The preprocessor maps unseen test categories to the training
    mode bucket, so they never produce an all-zero one-hot row by accident.
    """

    rows: list[dict[str, Any]] = []
    for col in CATEGORICAL_COLUMNS:
        if col not in train.columns or col not in test.columns:
            continue
        train_counts = train[col].dropna().value_counts()
        test_counts = test[col].dropna().value_counts()
        train_set = set(train_counts.index)
        test_set = set(test_counts.index)
        for category in sorted(train_set | test_set, key=str):
            tr = int(train_counts.get(category, 0))
            te = int(test_counts.get(category, 0))
            if tr == 0:
                coverage = "test_only"
            elif te == 0:
                coverage = "train_only"
            else:
                coverage = "both"
            rows.append(
                {
                    "column": col,
                    "category": str(category),
                    "train_count": tr,
                    "test_count": te,
                    "coverage": coverage,
                }
            )
    return pd.DataFrame(rows)
