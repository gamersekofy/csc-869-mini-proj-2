# CSC 869 Mini Project 2 — Python and native Rust

Imbalanced-data study on the UCI **Census Income / Adult** dataset. This
repository contains the full Python experiment matrix and a native Rust workspace
component. The Rust CV/test CLI and differential harness are documented in
[`components/mp2-rust/README.md`](components/mp2-rust/README.md).

## What is implemented

- **Data cleaning** — 15-column Adult loader that strips whitespace, removes the
  `adult.test` junk header line and trailing label periods, and maps `?` to
  null. Verified row counts: **32,561 train / 16,281 test**.
- **Missing-value strategies** — two per feature type:
  - categorical: `unknown` (missing as its own category) and `mode` imputation;
  - numeric: `median` and `median + missing indicator`.
  Adult has no missing numeric values, so the numeric variants are identical in
  practice; the report says this explicitly instead of faking missingness.
- **Leak-free preprocessing** — modes, medians, means, standard deviations and
  one-hot vocabularies are fitted **inside each fold** on the resampled training
  rows only. Unseen test categories fall back to the training mode bucket.
- **Imbalance handling** — `none`, random `upsampling` to parity, and random
  `downsampling` to the minority count. Resampling is applied only to the
  training portion of a fold; the validation fold and test set are untouched.
- **Five classifiers** — Decision Tree (Gini, depth tuned by inner 3-fold F1),
  Random Forest (300 trees), Logistic Regression (L2, max_iter=1000), SVM
  (RBF; Platt/sigmoid calibration for the final test models), KNN (k=5,
  ball tree, scaled features).
- **Evaluation** — stratified 5-fold CV, six metrics (accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC), ROC/PR curves with mean ± std bands, learning
  curves, full-test-set evaluation, CV-vs-test delta table.
- **Error analysis** — fixed 200-record test sample (100 positives + 100
  negatives), 3-family Venn diagram (tree-based / linear / instance-based),
  pairwise agreement heatmap, hard-case table, FP/FN split, named error
  patterns.
- **MP1 carry-forward** — from-scratch mixed Naive Bayes with Laplace
  smoothing (`csc869.legacy`), from-scratch stratified k-fold
  (`csc869.kfold`), and from-scratch metrics (`csc869.metrics`). The NBC
  scores ~83.3% on 10-fold CV, inside the MP1 sanity band.
- **Report source** — `reports/report.md` is generated from the result CSVs;
  every table is built from `results/cv_tidy.csv`, `results/test_metrics.csv`,
  and the error-analysis CSVs. No numbers are copied by hand.
- **Typeset PDF** — `reports/report.typ` is generated from the same CSVs and
  compiled with Typst into `reports/report.pdf`: a title page, a table of
  contents, then the 6 numbered body pages (8 physical pages in total). The body
  has a stat strip, booktabs tables, default quote blocks, a figure grid, and
  highlighted `python` / `yaml` / `bash` snippets. It uses Typst's stock fonts
  and colours, not a custom theme. The Nix shells include `typst`.

## Repository layout

```
components/mp2-py/
  csc869/
    data.py            # Adult loader, folds.json generation, --check
    preprocess.py      # leak-free preprocessing
    sampling.py        # none / upsampling / downsampling
    models.py          # five classifiers + DT depth tuning
    metrics.py         # accuracy, precision, recall, F1, ROC-AUC, PR-AUC
    kfold.py           # from-scratch stratified k-fold
    cv.py              # 5-fold experiment matrix + learning curves
    curves.py          # ROC / PR / learning-curve figures
    test_eval.py       # full-training-set evaluation on adult.test
    error_analysis.py  # 200-record sample, Venn, agreement, hard cases
    eda.py             # exploratory figures
    legacy/            # MP1 Naive Bayes + k-fold + metrics carry-forward
    report.py          # Markdown report + decision table + recommendation
    typst_report.py    # typeset Typst report + PDF
    run.py             # end-to-end runner
  configs/main.yaml
  requirements.txt
  pyproject.toml
  tests/
data/
  adult/               # UCI Adult files
  folds.json           # frozen 5-fold split, shared with Rust
results/               # generated CSVs
reports/figures/       # generated 300-dpi figures
reports/figures/dark/  # Mocha-recoloured copies used by the deck (tools/dark_figures.py)
docs/                  # execution plan + presentation guide (talk script, code map, FAQ)
tools/dark_figures.py  # rebuild reports/figures/dark/ from the light figures
```

## Setup

### With Nix (recommended)

From the repository root:

```bash
nix develop          # or: direnv allow
source .venv/bin/activate
cd components/mp2-py
pip install -r requirements.txt   # optional inside Nix; required elsewhere
```

`flake.nix` now provides the Python packages and `zlib` (needed by the
manylinux NumPy wheel). If you change the Python version, delete the relevant
`.venv` and reload the shell.

### Without Nix

```bash
cd components/mp2-py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From `components/mp2-py` (the package directory):

```bash
# 1. Data sanity check: row counts, class balance, missingness, encoding,
#    fold generation, and the MP1 NBC sanity band.
python -m csc869.data --check

# 2. MP1 legacy Naive Bayes on 10-fold CV (also run by step 1).
python -m csc869.legacy --check

# 3. Full experiment matrix (5 models x 3 scenarios x 5 folds).
#    Writes results/, reports/figures/, reports/report.md and the Typst PDF.
python -m csc869.run --config configs/main.yaml --out results/

# 4. Regenerate only the reports (Markdown + Typst/PDF) from existing results.
python -m csc869.run --config configs/main.yaml --out results/ --stages report,typst

# 5. Fast smoke test on an 8k-row stratified subset with separate
#    results_quick/ and reports/figures_quick/ paths.
python -m csc869.run --config configs/main.yaml --quick

# 6. Tests.
python -m pytest -q
```

The Typst stage writes `reports/report.typ` and compiles it to
`reports/report.pdf` with `typst compile --root ../..`. If `typst` is not on
PATH, the `.typ` source is still written and the stage prints a hint.

Useful runner options:

```bash
python -m csc869.run --help
python -m csc869.run --stages prepare,eda,missing,cv,curves,test,error,legacy,report,typst
python -m csc869.run --models decision_tree,random_forest --scenarios none,upsampling
python -m csc869.run --jobs 4        # parallelise (scenario, fold) tasks
python -m csc869.run --force-folds   # regenerate data/folds.json (avoid unless intentional)
```

### Expected runtime

On a modern laptop, the full run is roughly **20–40 minutes**:

- CV sweep (75 fits, including DT inner tuning): ~10–25 min depending on SVM;
- learning curves (75 fits on one fold): ~5–10 min;
- test models (15 fits, SVM calibrated): ~5–10 min;
- EDA, curves, error analysis, legacy NBC: ~1–2 min.

The quick profile runs in ~1–2 minutes.

## Outputs

| Path | Contents |
| --- | --- |
| `data/folds.json` | Frozen stratified 5-fold split (seed 42); the Rust component reads the same file. |
| `results/cv_tidy.csv` | One row per `lang, model, scenario, fold, metric, value` — **450 rows** for Python (900 when Rust is added). |
| `results/cv_summary.csv` | Mean/std across folds for every metric. |
| `results/cv_summary_wide.csv` | One row per model/scenario with all metric means. |
| `results/cv_runs.csv` | Per model/scenario fit time for each CV task. |
| `results/sampling_counts.csv` | Positive/negative counts before and after resampling, per fold. |
| `results/sampling_counts_full.csv` | Full-training-set counts for none / up / down. |
| `results/missing_ablation.csv` | Decision-tree CV over the 2x2 categorical/numeric missing-value strategy grid. |
| `results/missing_ablation_summary.csv` | Mean/std of the missing-value ablation. |
| `results/curve_points.csv` | Interpolated ROC/PR points (200 per fold/curve). |
| `results/learning_curves.csv` | F1 / ROC-AUC / PR-AUC vs training fraction. |
| `results/test_metrics.csv` | Six metrics per model/scenario on `adult.test`. |
| `results/test_runs.csv` | Full-training-set fit time per model/scenario. |
| `results/test_predictions.csv` | Row-level test predictions and scores (reused by error analysis). |
| `results/category_coverage.csv` | Category overlap between train and test, including test-only values. |
| `results/cv_vs_test.csv` | CV mean vs test score with delta. |
| `results/error_sample_ids.csv` | The fixed 200-record sample ids. |
| `results/venn_counts.csv` | 3-family Venn region counts. |
| `results/pairwise_agreement.csv` | 5x5 model agreement matrix. |
| `results/error_direction.csv` | TP/TN/FP/FN per model/scenario. |
| `results/hard_cases.csv` | All-five-models-wrong records with features. |
| `results/error_patterns.csv` | Named error patterns with counts and rates. |
| `results/decision_table.csv` | Model x scenario decision table. |
| `results/recommendation.json` | Data-driven recommendation, reasons, failure mode, next steps. |
| `results/unseen_categories.csv` | Test-split unseen-category counts (header-only: the Adult test split has none). |
| `results/run_metadata.json` | Versions, config, per-stage durations, and a history of invocations. |
| `reports/figures/*.png` | 300-dpi EDA, ROC/PR, learning-curve, Venn, agreement, error figures. |
| `reports/report.md` | Generated Markdown report with every table and figure. |
| `reports/report.typ` | Generated Typst source (same numbers, print layout). |
| `reports/report.pdf` | Typeset report compiled by Typst: title page, contents, 6 numbered body pages. |
| `reports/slides.pdf` | 13-slide 16:9 deck; `slides.typ` is hand-written, `slides_notes.md` is the 5-minute cue card. |
| `docs/presentation-guide.md` | Talk script, "where is X" code tour with file:line refs, and FAQ. |

## Design notes / traps avoided

- **Scaler fitted inside the fold** — the preprocessor is fitted on the
  resampled training rows only, then applied to the validation/test rows.
- **Resampling before splitting** — never done. `sampling.py` returns indices
  into the training portion; the validation fold is scored untouched.
- **`adult.test` header and trailing periods** — the loader skips the
  `|1x3 Cross validator` line and strips the trailing `.` from labels.
- **Unseen categories** — the preprocessor maps test-only categories to the
  training mode bucket. The full Adult train/test split happens to contain no
  test-only categories, so `results/unseen_categories.csv` is header-only;
  individual CV validation folds can still contain categories absent from their
  training portion, and the same fallback applies.
- **Accuracy as the headline metric** — the report leads with F1 and PR-AUC;
  a constant `<=50K` predictor would already score 75.92% accuracy.
- **Different folds in Rust and Python** — one frozen `data/folds.json`,
  generated by the from-scratch stratified k-fold in `csc869.kfold`.
- **A 5-set Venn diagram** — not attempted. The five models are collapsed into
  three families for a true 3-set Venn, plus a 5x5 agreement heatmap.
- **Hand-copied numbers** — every table is generated from the CSVs.

## Reproducibility

- Seed: 42 everywhere.
- Folds: `data/folds.json`, generated by `csc869.kfold.stratified_kfold_indices`.
  Schema:

  ```json
  {
    "seed": 42,
    "n_splits": 5,
    "n_rows": 32561,
    "class_balance": {"n": 32561, "positive": 7841, "negative": 24720,
                      "positive_rate": 0.2408, "negative_rate": 0.7592},
    "folds": [
      {"fold": 0, "train": [/* 26,048 indices */], "val": [/* 6,513 indices */]},
      ...
    ]
  }
  ```

  The Rust component should read the same file and use the same index order.
- The preprocessing vocabulary is rebuilt inside every fold, so changing the
  fold file changes the feature count slightly when a category is absent from a
  training fold; this is expected and is the correct leak-free behaviour.
- Model seeds are derived deterministically from the fold, scenario, and model
  index.

## Can Rust call Python / scikit-learn?

Yes, but it is usually the wrong tool for this assignment. The practical
options are:

1. **Embed a Python interpreter** with `pyo3` + `pythonize`/`numpy` bindings,
   or `inline-python`. Rust can import `sklearn`, pass NumPy arrays, and call
   `fit`/`predict`. This works, but it drags a Python runtime and the full
   scientific stack into the Rust binary and makes the "two implementations"
   comparison less meaningful.
2. **Run Python as a subprocess** (CLI or JSON/Arrow over stdin/stdout). This is
   simple, robust, and keeps the Rust and Python experiments independent.
3. **Call a Python model server** (FastAPI/Flask) over HTTP. Useful for
   deployment, overkill for a course project.
4. **Re-implement the learner in Rust** — the approach the brief actually
   wants for the tractable models (CART, random forest, logistic regression,
   KNN), with `linfa-svm` for the RBF SVM.

For MP2, the cleanest architecture is: Python is the reference implementation;
Rust reads the same `data/folds.json` and writes the same tidy CSV schema. A
small `pyo3` bridge is a nice extension if you want to compare a Rust
preprocessing pipeline against sklearn, but it should not replace the Rust
implementation of the five classifiers.

## Native Rust build and run

From the repository root:

```bash
nix develop .#mp2-rust --command cargo build --release
./target/release/csc869 --data data/ --folds data/folds.json --out results/rust/
# Quick frozen-fold parity profile; does not alter data or reports:
./target/release/csc869 --data data/ --folds data/folds.json --out results_quick/rust/ --quick
```

The full command uses all five existing outer folds, 300 forest trees and
three-fold Platt calibration for final-test SVMs. The quick command uses
40 trees/depth 5 and projects the first two original frozen folds onto the
first 8,000 source rows; it deliberately does not regenerate folds or claim
the stock Python quick profile uses that same partition. See the component
README for the unchanged-Python reference harness, per-cell comparison,
metric definitions and numerical-parity limitations. The Rust binary never
invokes Python.

## Data

UCI Machine Learning Repository, Census Income / Adult, dataset id 2:
<https://archive.ics.uci.edu/dataset/2/adult>

- `data/adult/adult.data`: 32,561 rows, no header.
- `data/adult/adult.test`: 16,281 rows, one junk header line, labels end in `.`.

## Notes for the report

- The MP2 brief says **no more than 6 pages**; the older rubric says 3. This
  implementation targets 6 and flags the discrepancy in the report draft.
- The MP1 NBC, 0-probability (Laplace) routine, k-fold CV, and metrics module
  are carried forward in `csc869.legacy`, `csc869.kfold`, and `csc869.metrics`
  to cover the carried-over rubric rows.
- `reports/report.pdf` is the typeset deliverable: a title page, a table of
  contents, and 6 numbered body pages (8 physical pages). `reports/report.md`
  is the Markdown content source with the full set of tables and figures. Both
  are generated from the CSVs, so regenerating the results updates both.
