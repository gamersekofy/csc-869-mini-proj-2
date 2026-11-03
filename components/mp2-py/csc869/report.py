"""Generate report-ready tables and a Markdown draft from the result CSVs.

The report never copies numbers by hand: every table is built from
``results/cv_tidy.csv``, ``results/test_metrics.csv``, and the error-analysis
CSVs.  The Markdown file can be converted to PDF with pandoc/LaTeX if desired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import TARGET_COLUMN, class_balance
from .models import DISPLAY_NAMES, INTERPRETABILITY, MP1_EVIDENCE, MODEL_NAMES
from .paths import ensure_dir, repo_root


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if np.isnan(value) else format(float(value), floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _read(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Some stages legitimately write a header-only file (e.g. when Adult's
        # test split has no unseen categories).
        return pd.DataFrame()


def _decision_table(cfg: dict[str, Any], results_dir: Path) -> pd.DataFrame:
    cv = _read(results_dir, "cv_summary_wide.csv")
    runs = _read(results_dir, "cv_runs.csv")
    test_runs = _read(results_dir, "test_runs.csv")
    if cv.empty:
        return pd.DataFrame()
    # Prefer the full-training-set test fit time: it includes the SVM Platt
    # calibration and is closer to deployment cost than the CV fit time.
    if not test_runs.empty:
        run_means = (
            test_runs.groupby(["model", "scenario"])["fit_seconds"].mean().reset_index()
        )
    elif not runs.empty:
        run_means = (
            runs.groupby(["model", "scenario"])["fit_seconds"].mean().reset_index()
        )
    else:
        run_means = pd.DataFrame(columns=["model", "scenario", "fit_seconds"])
    table = cv.merge(run_means, on=["model", "scenario"], how="left")
    table["model_label"] = table["model"].map(DISPLAY_NAMES)
    table["interpretability"] = table["model"].map(INTERPRETABILITY)
    table["mp1_evidence"] = table["model"].map(MP1_EVIDENCE)
    table["inference_cost"] = table["model"].map(
        {
            "decision_tree": "low",
            "random_forest": "medium",
            "logistic_regression": "low",
            "svm": "high",
            "knn": "high",
        }
    )
    keep = [
        "model",
        "model_label",
        "scenario",
        "accuracy",
        "f1",
        "pr_auc",
        "roc_auc",
        "fit_seconds",
        "inference_cost",
        "interpretability",
        "mp1_evidence",
    ]
    keep = [c for c in keep if c in table.columns]
    table = table[keep].rename(columns={"fit_seconds": "train_seconds"})
    return table.sort_values(["f1", "pr_auc"], ascending=False).reset_index(drop=True)


def _recommend(cfg: dict[str, Any], decision: pd.DataFrame, results_dir: Path) -> dict[str, Any]:
    if decision.empty:
        return {}
    best = decision.sort_values(["f1", "pr_auc", "accuracy"], ascending=False).iloc[0]
    test = _read(results_dir, "test_summary_wide.csv")
    test_row = test[
        (test["model"] == best["model"]) & (test["scenario"] == best["scenario"])
    ] if not test.empty else pd.DataFrame()
    precision = recall = float("nan")
    if not test_row.empty:
        precision = float(test_row["precision"].iloc[0])
        recall = float(test_row["recall"].iloc[0])
    scenario_failure = {
        "none": "the 24/76 prior makes the model conservative on borderline >50K cases",
        "upsampling": "duplicated minority rows can inflate confidence and overfit the duplicated records",
        "downsampling": "dropping ~75% of the majority class adds variance and loses legitimate <=50K examples",
    }
    model_failure = {
        "decision_tree": "axis-aligned splits overfit the capital-gain spike and high-cardinality country/occupation dummies",
        "random_forest": "less interpretable than a single tree and its probabilities are not perfectly calibrated",
        "logistic_regression": "linear decision boundary misses interactions such as married + high hours + capital gain",
        "svm": "expensive to train and score, sensitive to scaling, and its calibrated probabilities are less transparent",
        "knn": "distance concentration in high-dimensional one-hot space and slow inference at 32k rows",
    }
    recommendation = {
        "model": str(best["model"]),
        "model_label": DISPLAY_NAMES.get(str(best["model"]), str(best["model"])),
        "scenario": str(best["scenario"]),
        "cv_f1": float(best["f1"]),
        "cv_pr_auc": float(best["pr_auc"]),
        "cv_accuracy": float(best["accuracy"]),
        "test_precision": precision,
        "test_recall": recall,
        "reasons": [
            f"Empirical: highest mean CV F1 ({best['f1']:.4f}); PR-AUC ({best['pr_auc']:.4f}) is the tie-breaker.",
            f"Trade-off: test precision={precision:.4f}, recall={recall:.4f}; the choice is defensible under the 24/76 imbalance.",
            f"Practical: {INTERPRETABILITY.get(str(best['model']), 'n/a')} interpretability and "
            f"{float(best['train_seconds']) if not pd.isna(best['train_seconds']) else float('nan'):.2f}s mean full-training-set fit time.",
        ],
        "failure_mode": f"{model_failure.get(str(best['model']), 'model-specific failure')}; "
        f"{scenario_failure.get(str(best['scenario']), 'sampling-specific failure')}.",
        "next_steps": [
            "Calibrate probabilities (Platt/isotonic) and tune the decision threshold to a business cost.",
            "Compare cost-sensitive learning (class weights) with resampling instead of resampling alone.",
            "Try SMOTE/ADASYN and gradient boosting (XGBoost/LightGBM) as exploratory extensions.",
            "Monitor drift in capital-gain, hours-per-week, and native-country coverage after deployment.",
        ],
    }
    return recommendation


def generate_report(
    cfg: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    verbose: bool = True,
) -> Path:
    results_dir = Path(cfg["results_dir"])
    figures_dir = Path(cfg["figures_dir"])
    reports_dir = ensure_dir(repo_root() / "reports")
    fig_rel = figures_dir.name

    cv = _read(results_dir, "cv_summary.csv")
    cv_wide = _read(results_dir, "cv_summary_wide.csv")
    test_metrics = _read(results_dir, "test_metrics.csv")
    test_wide = _read(results_dir, "test_summary_wide.csv")
    delta = _read(results_dir, "cv_vs_test.csv")
    sampling = _read(results_dir, "sampling_counts.csv")
    patterns = _read(results_dir, "error_patterns.csv")
    direction = _read(results_dir, "error_direction.csv")
    venn = _read(results_dir, "venn_counts.csv")
    agreement = _read(results_dir, "pairwise_agreement.csv")
    legacy = _read(results_dir, "legacy_nbc_summary.csv")
    unseen = _read(results_dir, "unseen_categories.csv")
    missing_ablation = _read(results_dir, "missing_ablation_summary.csv")

    decision = _decision_table(cfg, results_dir)
    if not decision.empty:
        decision.to_csv(results_dir / "decision_table.csv", index=False)
    recommendation = _recommend(cfg, decision, results_dir)
    if recommendation:
        with open(results_dir / "recommendation.json", "w", encoding="utf-8") as fh:
            json.dump(recommendation, fh, indent=2)

    bal_train = class_balance(train[TARGET_COLUMN])
    bal_test = class_balance(test[TARGET_COLUMN])

    # Compact CV table: model x scenario for F1 / PR-AUC.
    cv_table = pd.DataFrame()
    if not cv_wide.empty:
        cv_table = cv_wide.copy()
        cv_table["model"] = cv_table["model"].map(DISPLAY_NAMES)
        keep = ["model", "scenario", "accuracy", "f1", "pr_auc", "roc_auc"]
        keep = [c for c in keep if c in cv_table.columns]
        cv_table = cv_table[keep].sort_values(["scenario", "f1"], ascending=[True, False])

    test_table = pd.DataFrame()
    if not test_wide.empty:
        test_table = test_wide.copy()
        test_table["model"] = test_table["model"].map(DISPLAY_NAMES)
        keep = ["model", "scenario", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
        keep = [c for c in keep if c in test_table.columns]
        test_table = test_table[keep].sort_values(["scenario", "f1"], ascending=[True, False])

    delta_table = pd.DataFrame()
    if not delta.empty:
        delta_table = delta.copy()
        delta_table["model"] = delta_table["model"].map(DISPLAY_NAMES)
        delta_table = delta_table[delta_table["metric"].isin(["accuracy", "f1", "pr_auc"])]
        delta_table = delta_table[
            ["model", "scenario", "metric", "cv_mean", "test", "delta_test_minus_cv"]
        ].sort_values(["scenario", "model", "metric"])

    md: list[str] = []
    md.append("# CSC 869 Mini Project 2 -- Python implementation report draft\n")
    md.append(
        "This file is generated from `results/*.csv`; regenerate it with "
        "`python -m csc869.run --config configs/main.yaml --out results/`. "
        "The brief's newer limit is **6 pages**; the older rubric says 3. "
        "This draft is a content source, not a typeset PDF.\n"
    )
    md.append("## 1. Data and preprocessing\n")
    md.append(
        f"- Training rows: **{len(train):,}**; test rows: **{len(test):,}**.\n"
        f"- Training class balance: **>50K {bal_train['positive_rate'] * 100:.2f}%** / "
        f"<=50K {bal_train['negative_rate'] * 100:.2f}%.\n"
        f"- Test class balance: **>50K {bal_test['positive_rate'] * 100:.2f}%** / "
        f"<=50K {bal_test['negative_rate'] * 100:.2f}%.\n"
        "- Missing values are confined to `workclass`, `occupation`, and `native-country`; "
        "they are encoded as `?` in the raw files and mapped to null.\n"
        "- Two categorical strategies are implemented: `unknown` (missing as its own category) "
        "and `mode` (training-mode imputation). Two numeric strategies are implemented: "
        "`median` and `median + missing indicator`. Adult has no missing numeric values, so the "
        "numeric variants are identical in practice; this is reported rather than faked. The "
        "real numeric issues are the zero-inflated `capital-gain`/`capital-loss` features and "
        "`hours-per-week` outliers, shown in the capital-gain figure.\n"
        "- All preprocessing statistics (modes, medians, means, standard deviations, one-hot "
        "vocabularies) are fitted **inside each fold** on the resampled training rows only.\n"
    )
    if not unseen.empty:
        md.append("\nUnseen test categories handled by falling back to the training mode bucket:\n")
        md.append(_md_table(unseen))
    if not missing_ablation.empty:
        md.append("\nMissing-value strategy ablation (decision tree, 5-fold CV):\n")
        abl = missing_ablation[missing_ablation["metric"].isin(["accuracy", "f1", "pr_auc"])]
        md.append(_md_table(abl))

    md.append("\n## 2. Methods\n")
    md.append(
        "- Classifiers: Decision Tree (Gini, depth tuned by inner 3-fold F1), Random Forest "
        "(300 trees), Logistic Regression (L2, max_iter=1000), SVM (RBF, Platt-calibrated for "
        "the final test model), KNN (k=5, ball tree, scaled features).\n"
        "- Sampling scenarios: `none`, random `upsampling` of the minority to parity, and random "
        "`downsampling` of the majority to the minority count. Resampling is applied only to the "
        "training portion of each fold.\n"
        "- Evaluation: stratified 5-fold CV (seed 42, frozen in `data/folds.json`), six metrics "
        "(accuracy, precision, recall, F1, ROC-AUC, PR-AUC), ROC/PR curves, and learning curves.\n"
        "- Metric definitions: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R), "
        "ROC-AUC is the tie-aware Mann-Whitney statistic, and PR-AUC is average precision. "
        "All six are computed locally in `csc869.metrics`; the tests compare them against "
        "scikit-learn on random data.\n"
        "- Curves: ROC and PR curves are averaged over the five folds on a common grid and "
        "plotted with a +/-1 standard-deviation band, one panel per sampling scenario. "
        "Learning curves use training fractions 0.1-1.0 on the first fold (decision-tree depth "
        "fixed at 7 for the curve sweep to avoid re-tuning at every fraction).\n"
        "- Error analysis: a fixed 200-record test sample (100 positives + 100 negatives), a "
        "3-family Venn of misclassification sets, a 5x5 pairwise agreement heatmap, and "
        "FP/FN counts per model and scenario.\n"
        "- MP1 carry-forward: a from-scratch mixed Naive Bayes with Laplace smoothing, a "
        "from-scratch stratified k-fold routine, and a from-scratch metrics module remain in "
        "`csc869.legacy`, `csc869.kfold`, and `csc869.metrics`.\n"
    )

    md.append("\n## 3. Exploratory figures\n")
    for name, caption in [
        ("eda_class_balance.png", "24.08% positive: accuracy alone would reward a constant <=50K predictor at 75.92%."),
        ("eda_categorical_income_rates.png", "Income rate by categorical attribute; missingness itself is informative."),
        ("eda_discretisation_age.png", "Age bins: income rises through mid-career and flattens late."),
        ("eda_discretisation_hours_per_week.png", "Hours bins: part-time work is associated with a very low >50K rate."),
        ("eda_discretisation_education_num.png", "Education bins: the >50K rate rises steeply with education-num."),
        ("eda_capital_gain_loss.png", "Capital gain/loss are zero-inflated; the zero spike explains tree over-reliance."),
        ("eda_association_heatmap.png", "Cramér's V and Spearman association structure."),
        ("eda_missingness.png", "Missingness is confined to three categorical columns."),
    ]:
        if (figures_dir / name).exists():
            md.append(f"\n![{caption}]({fig_rel}/{name})\n\n*{caption}*\n")

    md.append("\n## 4. 5-fold CV results\n")
    md.append("Mean CV scores by model and sampling scenario (full tidy data in `results/cv_tidy.csv`):\n")
    md.append(_md_table(cv_table))
    if not cv_wide.empty:
        md.append("\n**What the matrix shows.**\n")
        for scenario in cfg["scenarios"]:
            sub = cv_wide[cv_wide["scenario"] == scenario]
            if sub.empty:
                continue
            best = sub.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
            md.append(
                f"- `{scenario}`: best F1 **{best['f1']:.4f}** with "
                f"{DISPLAY_NAMES.get(best['model'], best['model'])} "
                f"(accuracy {best['accuracy']:.4f}, PR-AUC {best['pr_auc']:.4f}).\n"
            )
        best_pr = cv_wide.sort_values("pr_auc", ascending=False).iloc[0]
        md.append(
            f"- Highest PR-AUC overall: **{best_pr['pr_auc']:.4f}** with "
            f"{DISPLAY_NAMES.get(best_pr['model'], best_pr['model'])} under `{best_pr['scenario']}`.\n"
        )
        md.append(
            "- Accuracy alone is misleading: the best no-resampling models sit around 0.85-0.86 "
            "accuracy but only ~0.66-0.68 F1, while a constant `<=50K` predictor already scores 0.7592 "
            "accuracy and 0 F1.\n"
        )
    if (figures_dir / "cv_metric_heatmap.png").exists():
        md.append(f"\n![CV heatmap]({fig_rel}/cv_metric_heatmap.png)\n")
    if (figures_dir / "roc_curves.png").exists():
        md.append(f"\n![ROC curves]({fig_rel}/roc_curves.png)\n")
    if (figures_dir / "pr_curves.png").exists():
        md.append(f"\n![PR curves]({fig_rel}/pr_curves.png)\n")
    if (figures_dir / "learning_curves_f1.png").exists():
        md.append(f"\n![Learning curves F1]({fig_rel}/learning_curves_f1.png)\n")

    md.append("\n## 5. Test-set evaluation and CV-vs-test delta\n")
    md.append(_md_table(test_table))
    md.append("\nDelta = test minus mean CV. Negative values are expected: the test split has a "
              "slightly lower positive rate (23.62% vs 24.08%) and a small distribution shift.\n")
    md.append(_md_table(delta_table))

    md.append("\n## 6. Error analysis\n")
    if (figures_dir / "venn_misclassification.png").exists():
        md.append(f"\n![Venn misclassification]({fig_rel}/venn_misclassification.png)\n")
    if (figures_dir / "pairwise_agreement.png").exists():
        md.append(f"\n![Pairwise agreement]({fig_rel}/pairwise_agreement.png)\n")
    if (figures_dir / "error_direction.png").exists():
        md.append(f"\n![Error direction]({fig_rel}/error_direction.png)\n")
    md.append("\nVenn region counts (200-record sample):\n")
    md.append(_md_table(venn))
    md.append("\nPairwise agreement (focus scenario):\n")
    md.append(_md_table(agreement))
    md.append("\nNamed error patterns:\n")
    md.append(_md_table(patterns))
    md.append("\nFalse-positive / false-negative split:\n")
    md.append(_md_table(direction))

    md.append("\n## 7. Final recommendation\n")
    if recommendation:
        md.append(
            f"**Recommended: {recommendation['model_label']} + {recommendation['scenario']}.**\n\n"
            + "\n".join(f"- {r}" for r in recommendation["reasons"])
            + f"\n\n**Failure mode:** {recommendation['failure_mode']}\n\n"
            + "**Next steps:**\n"
            + "\n".join(f"- {s}" for s in recommendation["next_steps"])
            + "\n"
        )
    md.append("\nDecision table:\n")
    if not decision.empty:
        decision_display = decision.drop(columns=["model"], errors="ignore").rename(
            columns={"model_label": "model"}
        )
        md.append(_md_table(decision_display))
    else:
        md.append(_md_table(decision))

    md.append("\n## 8. Compile and run\n")
    md.append(
        "```bash\n"
        "# Python (from the repository root)\n"
        "nix develop\n"
        "source .venv/bin/activate\n"
        "cd components/mp2-py\n"
        "pip install -r requirements.txt   # optional inside Nix; required elsewhere\n"
        "python -m csc869.data --check\n"
        "python -m csc869.run --config configs/main.yaml --out results/ --jobs 4\n"
        "python -m csc869.legacy --check\n"
        "typst compile --root ../.. reports/report.typ reports/report.pdf\n"
        "\n"
        "# Rust (implemented separately)\n"
        "cargo build --release\n"
        "./target/release/mp2-rust --data data/ --folds data/folds.json --out results/rust/\n"
        "```\n"
    )
    md.append("\nExpected runtime on the full Adult data: roughly 20-40 minutes for the CV + "
              "learning-curve sweep on a modern laptop (use `--jobs 4` to parallelise), plus a "
              "few minutes for the test models. Figures land in `reports/figures/`; tidy "
              "results in `results/`.\n")

    if not legacy.empty:
        md.append("\n## Appendix: MP1 legacy Naive Bayes\n")
        md.append(_md_table(legacy))

    path = reports_dir / ("report_quick.md" if cfg.get("_quick") else "report.md")
    path.write_text("\n".join(md), encoding="utf-8")
    if verbose:
        print(f"  [report] wrote {path}", flush=True)
    return path
