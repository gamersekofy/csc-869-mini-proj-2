"""Stratified k-fold cross-validation implemented from scratch.

MP1 used a hand-written k-fold routine; this module carries it forward and
generalises it to any number of folds.  The same fold assignments are written
to ``data/folds.json`` so the Rust implementation can consume them verbatim.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def stratified_kfold_indices(
    y: Iterable[int],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, validation_idx), ...]`` with stratified folds.

    Each class is shuffled independently and dealt round-robin into folds.  The
    implementation is intentionally simple and deterministic so that the same
    seed reproduces the same folds in Python and in the Rust code that reads
    ``folds.json``.
    """

    y_arr = np.asarray(y, dtype=int)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > len(y_arr):
        raise ValueError("n_splits cannot exceed the number of samples")

    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y_arr), dtype=int)

    for cls in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == cls)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_splits

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        val_idx = np.flatnonzero(fold_of == fold)
        train_idx = np.flatnonzero(fold_of != fold)
        if len(val_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"empty fold {fold}")
        folds.append((train_idx, val_idx))
    return folds


def kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plain (unstratified) k-fold, kept for completeness."""

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    chunks = np.array_split(idx, n_splits)
    folds = []
    for fold in range(n_splits):
        val_idx = np.sort(chunks[fold])
        train_idx = np.sort(np.concatenate([chunks[i] for i in range(n_splits) if i != fold]))
        folds.append((train_idx, val_idx))
    return folds


def validate_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
    y: Iterable[int] | None = None,
) -> None:
    """Raise ``ValueError`` if folds are not a proper partition."""

    seen = np.zeros(n_samples, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(folds):
        if len(np.intersect1d(train_idx, val_idx)):
            raise ValueError(f"fold {fold}: train and validation overlap")
        seen[val_idx] += 1
        if y is not None:
            y_arr = np.asarray(y)
            for name, idx in (("train", train_idx), ("validation", val_idx)):
                classes = np.unique(y_arr[idx])
                if len(classes) < 2:
                    raise ValueError(f"fold {fold}: {name} split has a single class")
    if not np.all(seen == 1):
        raise ValueError("validation folds do not cover every row exactly once")


def stratified_subsample(
    y: Iterable[int],
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """Return stratified positions covering approximately *fraction* of rows."""

    y_arr = np.asarray(y, dtype=int)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for cls in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == cls)
        n_take = max(1, int(round(len(idx) * fraction)))
        n_take = min(n_take, len(idx))
        chosen.append(rng.choice(idx, size=n_take, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out
