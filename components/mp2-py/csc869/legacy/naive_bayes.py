"""MP1 modules carried forward: Naive Bayes, k-fold CV, and metrics.

The Naive Bayes implementation below is a mixed categorical/Gaussian model with
Laplace smoothing.  It is deliberately kept independent of scikit-learn so the
MP1 rubric row ("NBC implementation + 0-probability routine") is satisfied by
real code rather than a wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from ..kfold import stratified_kfold_indices
from ..metrics import compute_metrics


class MixedNaiveBayes:
    """Naive Bayes for a mix of categorical and Gaussian numeric features.

    Categorical likelihoods use Laplace (add-*alpha*) smoothing, which is the
    routine that prevents a zero probability from wiping out a class.  Numeric
    features use a per-class Gaussian likelihood.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        categorical: str = "unknown",
        numeric: str = "median",
    ) -> None:
        if categorical not in {"unknown", "mode"}:
            raise ValueError("categorical must be 'unknown' or 'mode'")
        if numeric not in {"median"}:
            raise ValueError("legacy NBC supports numeric='median' only")
        self.alpha = float(alpha)
        self.categorical = categorical
        self.numeric = numeric
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.cat_cols_: list[str] = []
        self.num_cols_: list[str] = []
        self.cat_fill_: dict[str, object] = {}
        self.cat_categories_: dict[str, list[object]] = {}
        self.cat_log_prob_: dict[str, dict[int, dict[object, float]]] = {}
        self.cat_unseen_log_prob_: dict[str, dict[int, float]] = {}
        self.num_median_: dict[str, float] = {}
        self.num_params_: dict[str, dict[int, tuple[float, float]]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MixedNaiveBayes":
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.cat_cols_ = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
        self.num_cols_ = [c for c in NUMERIC_COLUMNS if c in X.columns]

        counts = np.array([(y == cls).sum() for cls in self.classes_], dtype=float)
        self.class_log_prior_ = np.log(counts / counts.sum())

        for col in self.cat_cols_:
            series = X[col].astype(object)
            series = series.where(~pd.isna(series), np.nan)
            mode_series = series.dropna().mode()
            mode = mode_series.iloc[0] if len(mode_series) else "Unknown"
            fill = "Unknown" if self.categorical == "unknown" else mode
            self.cat_fill_[col] = fill
            filled = series.fillna(fill)
            categories = sorted(filled.dropna().unique().tolist(), key=str)
            self.cat_categories_[col] = categories
            k = max(len(categories), 1)
            self.cat_log_prob_[col] = {}
            self.cat_unseen_log_prob_[col] = {}
            for cls, n_cls in zip(self.classes_, counts):
                class_series = filled[y == cls]
                class_counts = class_series.value_counts()
                probs = {}
                for cat in categories:
                    probs[cat] = float(
                        (class_counts.get(cat, 0) + self.alpha) / (n_cls + self.alpha * k)
                    )
                self.cat_log_prob_[col][int(cls)] = {
                    cat: float(np.log(p)) for cat, p in probs.items()
                }
                self.cat_unseen_log_prob_[col][int(cls)] = float(
                    np.log(self.alpha / (n_cls + self.alpha * k))
                )

        for col in self.num_cols_:
            series = pd.to_numeric(X[col], errors="coerce")
            median = float(series.median()) if not pd.isna(series.median()) else 0.0
            self.num_median_[col] = median
            filled = series.fillna(median)
            self.num_params_[col] = {}
            for cls in self.classes_:
                values = filled[y == cls].to_numpy(dtype=float)
                mu = float(values.mean())
                var = float(values.var(ddof=0))
                var = max(var, 1e-9)
                self.num_params_[col][int(cls)] = (mu, var)
        return self

    def _joint_log_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None or self.class_log_prior_ is None:
            raise RuntimeError("MixedNaiveBayes must be fitted first")
        n = len(X)
        jll = np.tile(self.class_log_prior_, (n, 1))

        for col in self.cat_cols_:
            series = X[col].astype(object)
            series = series.where(~pd.isna(series), self.cat_fill_[col])
            categories = set(self.cat_categories_[col])
            filled = series.to_numpy(dtype=object)
            for j, cls in enumerate(self.classes_):
                log_probs = self.cat_log_prob_[col][int(cls)]
                unseen = self.cat_unseen_log_prob_[col][int(cls)]
                contribution = np.full(n, unseen, dtype=float)
                for cat, logp in log_probs.items():
                    contribution[filled == cat] = logp
                jll[:, j] += contribution

        for col in self.num_cols_:
            values = pd.to_numeric(X[col], errors="coerce").fillna(self.num_median_[col]).to_numpy(dtype=float)
            for j, cls in enumerate(self.classes_):
                mu, var = self.num_params_[col][int(cls)]
                jll[:, j] += -0.5 * np.log(2.0 * np.pi * var) - ((values - mu) ** 2) / (2.0 * var)
        return jll

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        max_jll = jll.max(axis=1, keepdims=True)
        log_norm = max_jll + np.log(np.exp(jll - max_jll).sum(axis=1, keepdims=True))
        return jll - log_norm

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))


def run_legacy_cv(
    train_df: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 42,
    alpha: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the MP1 NBC through a from-scratch stratified k-fold CV."""

    y = train_df[TARGET_COLUMN].to_numpy(dtype=int)
    folds = stratified_kfold_indices(y, n_splits=n_splits, seed=seed)
    feature_cols = [c for c in train_df.columns if c not in {TARGET_COLUMN, "row_id"}]
    rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        model = MixedNaiveBayes(alpha=alpha, categorical="unknown", numeric="median")
        model.fit(train_df.iloc[train_idx][feature_cols], y[train_idx])
        pred = model.predict(train_df.iloc[val_idx][feature_cols])
        score = model.predict_proba(train_df.iloc[val_idx][feature_cols])[:, 1]
        metrics = compute_metrics(y[val_idx], pred, score)
        for metric, value in metrics.items():
            rows.append({"fold": fold, "metric": metric, "value": float(value)})
        if verbose:
            print(
                f"  [legacy] fold {fold}: acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}",
                flush=True,
            )

    return pd.DataFrame(rows)
