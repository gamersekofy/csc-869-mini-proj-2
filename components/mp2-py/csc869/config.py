"""Configuration loading for the Python experiment harness."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import repo_root, resolve


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "n_splits": 5,
    "data": {
        "train": "data/adult/adult.data",
        "test": "data/adult/adult.test",
    },
    "folds": "data/folds.json",
    "results_dir": "results",
    "figures_dir": "reports/figures",
    "missing": {
        "categorical": "unknown",  # unknown | mode
        "numeric": "indicator",  # median | indicator
        "capital_flags": False,
    },
    "scenarios": ["none", "upsampling", "downsampling"],
    "models": [
        "decision_tree",
        "random_forest",
        "logistic_regression",
        "svm",
        "knn",
    ],
    "cv": {
        "inner_splits": 3,
        "dt_depths": [3, 5, 7, 10, None],
        "learning_fractions": [0.1, 0.25, 0.5, 0.75, 1.0],
        "learning_folds": 1,
        "svm_probability": False,
        "rf_trees": 300,
        "knn_neighbors": 5,
        "jobs": 1,
    },
    "test": {
        "svm_probability": True,
    },
    "error_analysis": {
        "scenario": "none",
        "n_pos": 100,
        "n_neg": 100,
        "sample_seed": 42,
    },
    "quick": {
        "folds": 2,
        "rf_trees": 40,
        "dt_depths": [5],
        "learning_fractions": [0.25, 1.0],
        "max_rows": 8000,
    },
}


def _deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML config on top of :data:`DEFAULTS`.

    Relative paths in the returned dict are left as strings; call
    :func:`resolved_paths` to turn them into absolute paths.
    """

    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config {path!r} must contain a mapping at the top level")
        _deep_update(cfg, user)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def resolved_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *cfg* with data/fold/result/figure paths absolute."""

    out = copy.deepcopy(cfg)
    root = repo_root()
    out["_repo_root"] = str(root)
    out["data"]["train"] = str(resolve(cfg["data"]["train"], root))
    out["data"]["test"] = str(resolve(cfg["data"]["test"], root))
    out["folds"] = str(resolve(cfg["folds"], root))
    out["results_dir"] = str(resolve(cfg["results_dir"], root))
    out["figures_dir"] = str(resolve(cfg["figures_dir"], root))
    return out


def apply_quick(cfg: dict[str, Any], quick: bool) -> dict[str, Any]:
    """Apply the ``quick`` profile for smoke tests."""

    if not quick:
        return cfg
    q = cfg.get("quick", {})
    out = copy.deepcopy(cfg)
    out["n_splits"] = int(q.get("folds", 1))
    out["cv"]["rf_trees"] = int(q.get("rf_trees", 40))
    out["cv"]["dt_depths"] = list(q.get("dt_depths", [5]))
    out["cv"]["learning_fractions"] = list(q.get("learning_fractions", [0.25, 1.0]))
    out["cv"]["learning_folds"] = 1
    out["cv"]["svm_probability"] = False
    out["test"]["svm_probability"] = False
    out["_quick"] = True
    out["_max_rows"] = q.get("max_rows")
    # Keep smoke-test artifacts away from the real results and, critically,
    # away from the frozen data/folds.json consumed by the Rust component.
    out["folds"] = "data/folds_quick.json"
    out["results_dir"] = "results_quick"
    out["figures_dir"] = "reports/figures_quick"
    return out
