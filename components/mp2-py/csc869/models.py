"""The five classifiers used in the experiment matrix.

Python uses off-the-shelf scikit-learn estimators, as permitted by the brief.
The Rust component re-implements the tractable learners; the Python numbers are
the reference the Rust numbers are checked against.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .kfold import stratified_kfold_indices
from .metrics import f1_score

MODEL_NAMES = [
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "svm",
    "knn",
]

DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (RBF)",
    "knn": "KNN (k=5)",
}

COLORS = {
    "decision_tree": "#1f77b4",
    "random_forest": "#2ca02c",
    "logistic_regression": "#d62728",
    "svm": "#9467bd",
    "knn": "#ff7f0e",
}

MARKERS = {
    "decision_tree": "o",
    "random_forest": "s",
    "logistic_regression": "^",
    "svm": "D",
    "knn": "v",
}

# Static notes used by the generated decision table / report.
INTERPRETABILITY = {
    "decision_tree": "high",
    "random_forest": "medium",
    "logistic_regression": "high",
    "svm": "low",
    "knn": "medium",
}

MP1_EVIDENCE = {
    "decision_tree": "MP1 baseline tree; depth tuning carried forward",
    "random_forest": "new in MP2; bagged trees",
    "logistic_regression": "new in MP2; linear reference",
    "svm": "new in MP2; RBF kernel, calibrated for probabilities",
    "knn": "new in MP2; scaled Euclidean / Hamming distance",
}


def build_model(
    name: str,
    seed: int = 42,
    *,
    svm_probability: bool = False,
    rf_trees: int = 300,
    knn_neighbors: int = 5,
    n_jobs: int = -1,
) -> Any:
    """Build an unfitted estimator for *name*."""

    if name == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            random_state=seed,
            class_weight=None,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(rf_trees),
            random_state=seed,
            n_jobs=n_jobs,
            class_weight=None,
        )
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        class_weight=None,
                    ),
                ),
            ]
        )
    if name == "svm":
        base = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            cache_size=2000,
            random_state=seed,
            class_weight=None,
        )
        if svm_probability:
            # Platt/sigmoid calibration.  ``ensemble=False`` uses the
            # cross-validated predictions to fit one calibrator, which is the
            # cheapest faithful Platt implementation in scikit-learn 1.9+.
            return CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv=3,
                ensemble=False,
            )
        return base
    if name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=int(knn_neighbors),
                        algorithm="ball_tree",
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return a continuous score for the positive class.

    Preference order: ``predict_proba`` > ``decision_function`` > hard labels.
    ROC/PR curves only need a monotone score, so SVM margins are acceptable
    when probability calibration is disabled for a CV sweep.
    """

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()


def tune_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    depths: list[int | None] | None = None,
    inner_splits: int = 3,
) -> tuple[int | None, list[dict[str, Any]]]:
    """Choose ``max_depth`` with an inner stratified 3-fold CV on F1."""

    if depths is None:
        depths = [3, 5, 7, 10, None]
    rows: list[dict[str, Any]] = []
    best_depth: int | None = None
    best_score = -np.inf

    inner_folds = stratified_kfold_indices(y, n_splits=inner_splits, seed=seed)
    for depth in depths:
        scores = []
        for train_idx, val_idx in inner_folds:
            clf = DecisionTreeClassifier(
                criterion="gini",
                max_depth=depth,
                random_state=seed,
                class_weight=None,
            )
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], pred))
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        rows.append(
            {
                "max_depth": depth if depth is not None else "None",
                "mean_f1": mean_score,
                "std_f1": std_score,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth
    return best_depth, rows


def fit_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    *,
    svm_probability: bool = False,
    rf_trees: int = 300,
    knn_neighbors: int = 5,
    n_jobs: int = -1,
    tune_dt: bool = True,
    dt_depth: int | None = 7,
    inner_splits: int = 3,
    dt_depths: list[int | None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit one model and return it with timing / tuning metadata."""

    start = time.perf_counter()
    info: dict[str, Any] = {}
    if name == "decision_tree":
        if tune_dt:
            best_depth, tuning_rows = tune_decision_tree(
                X, y, seed=seed, depths=dt_depths, inner_splits=inner_splits
            )
            info["best_depth"] = best_depth
            info["dt_tuning"] = tuning_rows
        else:
            best_depth = dt_depth
            info["best_depth"] = best_depth
        model = build_model(name, seed=seed, n_jobs=n_jobs)
        model.set_params(max_depth=best_depth)
    else:
        model = build_model(
            name,
            seed=seed,
            svm_probability=svm_probability,
            rf_trees=rf_trees,
            knn_neighbors=knn_neighbors,
            n_jobs=n_jobs,
        )
    model.fit(X, y)
    info["fit_seconds"] = float(time.perf_counter() - start)
    return model, info
