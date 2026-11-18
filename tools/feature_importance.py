#!/usr/bin/env python3
"""Feature importance for the two tree models, on the full training set.

Writes results/feature_importance.csv (per encoded column) and
results/feature_importance_grouped.csv (one row per source attribute), plus a
horizontal bar chart at reports/figures/feature_importance.png and a dark copy
for the deck.

Impurity-based importance (mean decrease in Gini) from the fitted models.  No
resampling: this answers "what does the model lean on", not "what does the
imbalance do".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "components/mp2-py"))

from csc869.config import load_config, resolved_paths          # noqa: E402
from csc869.data import TARGET_COLUMN, load_dataset           # noqa: E402
from csc869.preprocess import Preprocessor                    # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from sklearn.ensemble import RandomForestClassifier           # noqa: E402
from sklearn.tree import DecisionTreeClassifier               # noqa: E402


def group_of(name: str) -> str:
    """Map an encoded column back to its source attribute."""
    if "=" in name:
        return name.split("=", 1)[0]
    if name.endswith("__isna"):
        return name[: -len("__isna")]
    return name


def main() -> int:
    cfg = resolved_paths(load_config(None))
    train, _ = load_dataset(cfg)
    y = train[TARGET_COLUMN].to_numpy(dtype=int)

    pre = Preprocessor(
        categorical=cfg["missing"]["categorical"],
        numeric=cfg["missing"]["numeric"],
        capital_flags=bool(cfg["missing"].get("capital_flags", False)),
    )
    X = pre.fit_transform(train)
    names = pre.get_feature_names_out()
    print(f"encoded features: {len(names)}  rows: {X.shape[0]}")

    rows = []
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight=None
        ),
        "decision_tree": DecisionTreeClassifier(
            criterion="gini", random_state=42, class_weight=None
        ),
    }
    for label, model in models.items():
        model.fit(X, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1]
        for i in order:
            rows.append(
                {
                    "model": label,
                    "feature": names[i],
                    "group": group_of(names[i]),
                    "importance": float(imp[i]),
                }
            )
        print(f"  {label}: top5 " + ", ".join(f"{names[i]}={imp[i]:.3f}" for i in order[:5]))

    per_col = pd.DataFrame(rows)
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    per_col.to_csv(out / "feature_importance.csv", index=False)

    grouped = (
        per_col.groupby(["model", "group"], as_index=False)["importance"].sum()
        .sort_values(["model", "importance"], ascending=[True, False])
    )
    grouped.to_csv(out / "feature_importance_grouped.csv", index=False)

    # --- figure: one panel per model, top 12 source attributes ---
    plt.rcParams.update({"font.size": 8, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, label in zip(axes, ["random_forest", "decision_tree"]):
        sub = grouped[grouped["model"] == label].head(12).iloc[::-1]
        ax.barh(sub["group"], sub["importance"], color="#4c72b0")
        ax.set_title(f"{label.replace('_', ' ').title()} — Gini importance")
        ax.set_xlabel("mean decrease in impurity")
    fig.suptitle("Where the tree models put their weight (full training set, no resampling)")
    fig.tight_layout()
    figdir = Path(cfg["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'feature_importance.csv'}")
    print(f"wrote {figdir / 'feature_importance.png'}")
    print(grouped[grouped.model == 'random_forest'].head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
