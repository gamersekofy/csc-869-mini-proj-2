from __future__ import annotations

import numpy as np

from csc869.sampling import resample_indices


def test_none_keeps_everything():
    y = np.array([0] * 80 + [1] * 20)
    idx, counts = resample_indices(y, "none", seed=1)
    assert len(idx) == len(y)
    assert counts["n_positive"] == 20
    assert counts["n_negative"] == 80


def test_upsampling_reaches_parity():
    y = np.array([0] * 80 + [1] * 20)
    idx, counts = resample_indices(y, "upsampling", seed=1)
    assert counts["n_positive"] == counts["n_negative"]
    assert counts["n_total"] == 160


def test_downsampling_reaches_parity():
    y = np.array([0] * 80 + [1] * 20)
    idx, counts = resample_indices(y, "downsampling", seed=1)
    assert counts["n_positive"] == counts["n_negative"] == 20
    assert counts["n_total"] == 40
