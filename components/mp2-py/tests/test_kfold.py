from __future__ import annotations

import numpy as np

from csc869.kfold import stratified_kfold_indices, validate_folds


def test_folds_partition_and_stratify():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    folds = stratified_kfold_indices(y, n_splits=5, seed=42)
    validate_folds(folds, len(y), y)
    seen = np.zeros(len(y), dtype=int)
    for _, val in folds:
        seen[val] += 1
    assert np.all(seen == 1)
    for _, val in folds:
        assert 0.15 < y[val].mean() < 0.85


def test_reproducible():
    y = np.array([0] * 50 + [1] * 20)
    a = stratified_kfold_indices(y, n_splits=5, seed=7)
    b = stratified_kfold_indices(y, n_splits=5, seed=7)
    for (ta, va), (tb, vb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(va, vb)
