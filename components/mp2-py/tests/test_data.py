from __future__ import annotations

import numpy as np

from csc869.config import load_config, resolved_paths
from csc869.data import CATEGORICAL_COLUMNS, TARGET_COLUMN, class_balance, load_dataset, make_folds
from csc869.preprocess import Preprocessor


def _data():
    cfg = resolved_paths(load_config(None))
    return load_dataset(cfg)


def test_row_counts_and_labels():
    train, test = _data()
    assert len(train) == 32561
    assert len(test) == 16281
    assert set(train[TARGET_COLUMN].unique()) == {0, 1}
    assert set(test[TARGET_COLUMN].unique()) == {0, 1}
    bal = class_balance(train[TARGET_COLUMN])
    assert abs(bal["positive_rate"] - 0.2408) < 0.001


def test_missing_values_are_null_not_strings():
    train, test = _data()
    for col in CATEGORICAL_COLUMNS:
        assert train[col].isna().sum() >= 0
    assert train["workclass"].isna().sum() == 1836
    assert train["occupation"].isna().sum() == 1843
    assert train["native-country"].isna().sum() == 583


def test_preprocessor_produces_no_nan_and_handles_unseen():
    train, test = _data()
    pre = Preprocessor(categorical="unknown", numeric="indicator")
    X_train = pre.fit_transform(train.drop(columns=[TARGET_COLUMN]))
    X_test = pre.transform(test.drop(columns=[TARGET_COLUMN, "row_id"], errors="ignore"))
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] > 80


def test_folds_are_a_partition():
    train, _ = _data()
    y = train[TARGET_COLUMN].to_numpy()
    folds = make_folds(y, n_splits=5, seed=42)
    assert len(folds) == 5
    seen = np.zeros(len(y), dtype=int)
    for _, val in folds:
        seen[val] += 1
    assert np.all(seen == 1)
