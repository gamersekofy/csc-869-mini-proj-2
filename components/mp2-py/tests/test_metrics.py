from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from csc869.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def test_classic_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    assert accuracy_score(y_true, y_pred) == skm.accuracy_score(y_true, y_pred)
    assert abs(precision_score(y_true, y_pred) - skm.precision_score(y_true, y_pred, zero_division=0)) < 1e-12
    assert abs(recall_score(y_true, y_pred) - skm.recall_score(y_true, y_pred, zero_division=0)) < 1e-12
    assert abs(f1_score(y_true, y_pred) - skm.f1_score(y_true, y_pred, zero_division=0)) < 1e-12


def test_aucs_match_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=400)
    score = np.round(rng.normal(size=400), 1)  # force ties
    assert abs(roc_auc_score(y_true, score) - skm.roc_auc_score(y_true, score)) < 1e-12
    assert abs(average_precision_score(y_true, score) - skm.average_precision_score(y_true, score)) < 1e-9
