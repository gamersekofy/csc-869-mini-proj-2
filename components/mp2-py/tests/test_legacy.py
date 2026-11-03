from __future__ import annotations

import numpy as np

from csc869.config import load_config, resolved_paths
from csc869.data import TARGET_COLUMN, load_dataset
from csc869.legacy import MixedNaiveBayes, run_legacy_cv


def test_nbc_fits_and_beats_chance():
    cfg = resolved_paths(load_config(None))
    train, _ = load_dataset(cfg)
    # A 4,000-row stratified-ish slice keeps the test fast.
    train = train.iloc[:4000].reset_index(drop=True)
    y = train[TARGET_COLUMN].to_numpy()
    model = MixedNaiveBayes(alpha=1.0)
    model.fit(train.drop(columns=[TARGET_COLUMN]), y)
    pred = model.predict(train.drop(columns=[TARGET_COLUMN]))
    assert 0.70 < float(np.mean(pred == y)) < 1.0


def test_legacy_cv_runs():
    cfg = resolved_paths(load_config(None))
    train, _ = load_dataset(cfg)
    train = train.iloc[:3000].reset_index(drop=True)
    df = run_legacy_cv(train, n_splits=3, seed=42, verbose=False)
    assert set(df["metric"]) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert len(df) == 3 * 6
