"""Imbalance handling: none, random upsampling, random downsampling.

Resampling is always applied to the *training portion of a fold*, never to the
validation fold and never to the test set.  The functions here return positional
indices into the training array, which keeps the caller in control of the
train/validation boundary.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

SCENARIOS = ["none", "upsampling", "downsampling"]


def resample_indices(
    y: Iterable[int],
    scenario: str,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return ``(indices, counts)`` for one sampling scenario.

    ``indices`` are positions into ``y``.  For ``none`` they are simply
    ``0..n-1``.  ``upsampling`` duplicates minority rows with replacement until
    both classes are equal.  ``downsampling`` removes majority rows without
    replacement until both classes are equal.
    """

    y_arr = np.asarray(y, dtype=int)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("resampling requires both classes in the training fold")

    if scenario == "none":
        idx = np.arange(len(y_arr), dtype=int)
    elif scenario == "upsampling":
        target = max(len(pos), len(neg))
        if len(pos) < target:
            pos = np.concatenate([pos, rng.choice(pos, size=target - len(pos), replace=True)])
        if len(neg) < target:
            neg = np.concatenate([neg, rng.choice(neg, size=target - len(neg), replace=True)])
        idx = np.concatenate([pos, neg])
    else:  # downsampling
        target = min(len(pos), len(neg))
        pos = rng.choice(pos, size=target, replace=False)
        neg = rng.choice(neg, size=target, replace=False)
        idx = np.concatenate([pos, neg])

    rng.shuffle(idx)
    y_res = y_arr[idx]
    counts = {
        "n_total": int(len(idx)),
        "n_positive": int((y_res == 1).sum()),
        "n_negative": int((y_res == 0).sum()),
        "positive_rate": float((y_res == 1).mean()) if len(idx) else 0.0,
    }
    return idx, counts


def scenario_counts(y: Iterable[int], scenario: str, seed: int = 42) -> dict[str, int]:
    _, counts = resample_indices(y, scenario, seed=seed)
    return counts
