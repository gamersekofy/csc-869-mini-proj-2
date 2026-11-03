"""Classification metrics implemented locally.

The four classic metrics (accuracy, precision, recall, F1) and the two ranking
metrics (ROC-AUC, PR-AUC / average precision) are implemented here rather than
imported from scikit-learn so the MP1 evaluation module remains a genuine
from-scratch implementation.  The tests compare the results against
``sklearn.metrics`` on random data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def _as_arrays(y_true: Iterable[int], y_pred: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return yt, yp


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, int]:
    yt, yp = _as_arrays(y_true, y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(yt == yp)) if len(yt) else 0.0


def precision_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    c = confusion_counts(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fp"])


def recall_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    c = confusion_counts(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fn"])


def f1_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return _safe_div(2.0 * p * r, p + r)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks with ties receiving the average rank."""

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # i and j are 0-based, ranks are 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Mann-Whitney U ROC-AUC with tie-aware average ranks."""

    yt = np.asarray(y_true, dtype=int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape:
        raise ValueError("y_true and y_score must have the same shape")
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(ys)
    auc = (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(precision, recall)`` at every distinct score threshold.

    The first point is ``(1.0, 0.0)`` and the last point is the full-threshold
    point, mirroring ``sklearn.metrics.precision_recall_curve`` closely enough
    for curve plotting and average precision.
    """

    yt = np.asarray(y_true, dtype=int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if len(yt) == 0:
        return np.array([1.0]), np.array([0.0])

    order = np.argsort(-ys, kind="mergesort")
    y_sorted = yt[order]
    s_sorted = ys[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = tp[-1] if len(tp) else 0
    if n_pos == 0:
        return np.array([1.0]), np.array([0.0])

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # Keep the last index of each run of equal scores so the threshold is the
    # same for all tied predictions.
    change = np.flatnonzero(np.diff(s_sorted) != 0)
    last_of_group = np.r_[change, len(s_sorted) - 1]
    precision = precision[last_of_group]
    recall = recall[last_of_group]

    return np.r_[1.0, precision], np.r_[0.0, recall]


def average_precision_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Area under the precision-recall curve (step interpolation)."""

    precision, recall = precision_recall_curve(y_true, y_score)
    if len(recall) < 2:
        return 0.0
    # AP = sum (R_i - R_{i-1}) * P_i.  ``recall`` is non-decreasing.
    return float(np.sum(np.diff(recall) * precision[1:]))


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> dict[str, float]:
    """Return all six metrics as a dict."""

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_score is None:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


def summarize(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))
