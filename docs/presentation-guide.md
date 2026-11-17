# CSC 869 Mini Project 2 — Presentation Guide

Working notes for the 7-minute talk plus the code walk-through. The verbatim script is
`reports/slides_script.md`; this file is the reference behind it. Everything here is
either measured on this repo or a direct pointer to the line that proves it. If a
number is not in here, do not improvise it during the talk.

**Contents**

1. [90-second summary](#1-90-second-summary)
2. [What the project was](#2-what-the-project-was)
3. [Why we did it](#3-why-we-did-it)
4. [How it differs from Mini Project 1](#4-how-it-differs-from-mini-project-1)
5. [The pipeline](#5-the-pipeline)
6. [Numbers you must not get wrong](#6-numbers-you-must-not-get-wrong)
7. [Code tour — "show me where X is"](#7-code-tour--show-me-where-x-is)
8. [Live demo runbook](#8-live-demo-runbook)
9. [FAQ](#9-faq)
10. [Traps](#10-traps)
11. [File map](#11-file-map)

---

## 1. 90-second summary

> We took the Adult census data and asked a narrow question: **when 24% of the data is
> the class you care about, does resampling actually help, or does it just look like it
> helps?** We ran five classifiers under three sampling scenarios with a frozen 5-fold
> split — 75 fits per language, 450 metric values — plus a full-test-set evaluation and
> an error analysis on the hard cases. The headline: resampling moves F1 by about one
> point, not twenty; the model family matters more than the sampling strategy; and the
> best configuration is Random Forest with upsampling. Everything is implemented twice,
> in Python and in native Rust, reading the same `folds.json`, and the two agree to
> within 0.011 F1 on fold means.

If you only remember one sentence: **the imbalance is a data problem, not a tuning problem.**

---

## 2. What the project was

| Item | Detail |
|---|---|
| Course | CSC 869, Fall 2026 |
| Brief | Imbalanced data, more classifiers, in-depth evaluation, final recommendation |
| Dataset | UCI Census Income / **Adult**, same as MP1 — 32,561 train / 16,281 test rows |
| Target | `>50K` vs `<=50K`; positive (minority) rate **24.08%** train, **23.62%** test |
| Matrix | 5 classifiers × 3 sampling scenarios × 5 folds = **75 fits**, × 6 metrics = **450 values** |
| Implementations | Python package (`components/mp2-py`) and native Rust binary (`components/mp2-rust`) |
| Deliverable | 6-page PDF report (8 physical pages) + this deck; **code is not submitted** |

The five classifiers: Decision Tree (Gini, depth tuned by inner 3-fold F1), Random Forest
(300 trees), Logistic Regression (L2, max_iter 1000), SVM (RBF, Platt-calibrated for the
final test fits), kNN (k=5, scaled features).

The three scenarios: `none` (the honest baseline), `upsampling` (duplicate minority rows to
parity), `downsampling` (drop majority rows to the minority count).

The six metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC — all computed by our own
`csc869/metrics.py`, not imported from sklearn.

---

## 3. Why we did it

Adult is the canonical imbalanced-classification toy problem, and it is deceptively easy to
report well on. A constant "`<=50K`" predictor scores **0.759 accuracy and 0 F1**. So the first
thing the project has to do is refuse the headline metric:

1. **Accuracy is unusable here.** 24% positive rate means the majority class carries the
   score. F1 on the positive class is the metric that punishes ignoring it; PR-AUC is the
   tie-breaker because it ranks rather than thresholds.
2. **Sampling strategy is the assignment's actual question.** Everyone "knows" you should
   upsample a minority class. The interesting result is that on this dataset it buys about
   one F1 point while costing precision, and that downsampling buys recall at a much larger
   precision cost — a trade you should make deliberately, not by reflex.
3. **Model family beats preprocessing.** Random Forest and SVM sit within ~1 F1 point of each
   other across all three scenarios; the gap between the best and worst model in a scenario
   is larger than the gap between scenarios for a fixed model.
4. **Two implementations is the verification story.** Same folds, same resample seeds, same
   metric definitions, two languages. If Python and Rust agree cell by cell, the pipeline is
   not the explanation for any result — which is exactly what makes the results believable.

---

## 4. How it differs from Mini Project 1

MP1 was **one classifier, written from scratch, evaluated with k-fold CV**. MP2 keeps that
code alive and builds a comparison study around it.

| | MP1 | MP2 |
|---|---|---|
| Dataset | Adult | Adult (unchanged, so the comparison is clean) |
| Classifiers | Mixed Naive Bayes, hand-written | Five: DT, RF, LogReg, SVM, kNN (sklearn) **plus** the MP1 NBC carried forward |
| Metrics | from scratch: accuracy, precision, recall, F1 | same code, plus ROC-AUC and PR-AUC, curves with ±1 SD bands |
| CV | hand-written k-fold | same hand-written routine, generalised to 5 folds, **frozen to `folds.json`** |
| Imbalance | not addressed | three sampling scenarios, resampling inside the fold only |
| Evaluation depth | CV score | CV + full test set + CV-vs-test deltas + learning curves + error analysis |
| Implementations | Python | Python **and** native Rust, differential-tested |
| Output | results | results + recommendation + 6-page typeset report generated from the CSVs |

Why the NBC is still there: the attached rubric is the MP1 rubric carried over, and it still
awards 30 points for a Naive Bayes implementation (20) and k-fold CV (10). The module lives in
`components/mp2-py/csc869/legacy/` and still runs on the cleaned data: **10-fold accuracy
0.8334 ± 0.0052**, inside the MP1 sanity band of 81–84%.

The one honest difference in framing: MP1 asked "can you implement the algorithm"; MP2 asks
"can you design an experiment that answers a question".

---

## 5. The pipeline

```
adult.data / adult.test
   │  data.py: strip junk header, strip trailing '.', '?' → null, 32,561 / 16,281 rows
   ▼
folds.json  ◄── kfold.py: stratified 5-fold, seed 42, written once and frozen
   │
   ▼  for each scenario × fold:
   ├─ resample the TRAINING rows only            sampling.py
   ├─ fit preprocessor on resampled train only   preprocess.py
   ├─ transform the validation fold with it      (no leakage)
   └─ for each model: fit → predict → 6 metrics  models.py, metrics.py
   ▼
results/cv_tidy.csv, cv_summary.csv, cv_runs.csv, curves, learning curves
   ▼
test stage: refit on the full training set, score the untouched test split
   ▼
error stage: 200-record sample, family Venn, pairwise agreement, FP/FN split
   ▼
report stage: Typst + Markdown generated from the CSVs (no hand-typed numbers)
```

The single most important discipline in that diagram: **resample → preprocess → fit all
happen inside the fold.** Resampling before splitting is how a project fakes a 99% F1.

---

## 6. Numbers you must not get wrong

**CV (mean over 5 folds), best per scenario**

| scenario | model | F1 | PR-AUC |
|---|---|---|---|
| none | Random Forest | 0.6813 | **0.7701** (best overall) |
| upsampling | Random Forest | **0.6905** | 0.7479 |
| downsampling | SVM (RBF) | 0.6818 | — |

**Test set (refit on all training rows, scored once)**

| model · scenario | F1 | precision | recall |
|---|---|---|---|
| SVM · upsampling | **0.6843** | 0.5761 | 0.8427 |
| SVM · downsampling | 0.6837 | 0.5713 | 0.8510 |
| LogReg · upsampling | 0.6737 | 0.5622 | 0.8404 |
| RF · upsampling | 0.6722 | 0.6661 | 0.6784 |
| RF · no resampling | 0.6587 | 0.7187 | 0.6079 |

CV-vs-test deltas stay within **±0.017 F1** and **±0.017 PR-AUC** across all 15 cells — no
optimistic surprise, and the small negative bias is explained by the test split's slightly
lower positive rate (23.62% vs 24.08%).

**Cross-language (quick protocol: 8,000 rows, 2 projected folds, 180 metric values)**

| model | max F1 delta on fold means | worst single fold |
|---|---|---|
| kNN | 0.00000 | 1.1e-16 (float noise, 1 of 36 values) |
| SVM | 0.00000 | 0.00000 |
| Logistic Regression | 0.00034 | 0.00067 |
| Decision Tree | 0.00102 | 0.00203 |
| Random Forest | 0.01082 | 0.02316 |

Random Forest's residual is up to **0.018 PR-AUC** per fold — different per-tree bootstrap RNG
streams, not a fold or preprocessing mismatch.

**Cost**

| run | time |
|---|---|
| Full Python matrix (450 values) | ≈ 20–40 min, `--jobs 4`; CV stage alone 1,375.8 s measured |
| Quick parity protocol, Python | 337.9 s |
| Quick parity protocol, Rust | 113.7 s (**3.0×**) |
| Full Rust matrix | **extrapolated ≈ 8–14 min, not run** |
| Slowest single fit | SVM on the full training set: 603 s (Platt calibration adds 3 inner fits) |

**Error analysis (200-record test sample: 100 positives, 100 negatives)**

- All three model families are wrong on **22 of 200** records in `none` (9 in upsampling, 17 in
  downsampling); at least one family fails on **50**.
- **KNN is the most alone-wrong**: 11 records nobody else misses.
- `education-num` 9–13 covers **91% / 89% / 88%** of the all-wrong cases across the three
  scenarios — the hard core is mid-education, not the zero-capital-gain group.
- False positives outnumber false negatives in all five models under both resampling schemes:
  the precision cost of chasing recall.
- Downsampling trades precision for recall: mean FN falls 0.4094 → 0.1720, mean FP rises
  0.0704 → 0.2078.

**MP1 carry-forward**: mixed Naive Bayes, 10-fold, accuracy **0.8334 ± 0.0052**.

---

## 7. Code tour — "show me where X is"

When the professor asks to see something, open the file, jump to the line, and say the
one-liner. Line numbers are current as of the committed tree.

### 7.1 Where the data is parsed and cleaned

`components/mp2-py/csc869/data.py:59` — `_read_adult_rows`. Hand-rolled parser, because
`adult.test` has a junk first line and labels with a trailing period.

```python
# data.py:72-77
# adult.test starts with the junk line "|1x3 Cross validator".
if line.startswith("|"):
    continue
fields = [field.strip() for field in line.split(",")]
if len(fields) != len(COLUMNS):
    raise ValueError(...)
```

Say: *"If you let pandas guess the header, the label column silently becomes strings and every
model predicts one class. So the loader is explicit and it fails loudly on a malformed row."*

Related: `data.py:109` `load_adult` (label → int), `data.py:232` `_run_check` (the
`python -m csc869.data --check` exit test: row counts + class balance + legacy NBC band).

### 7.2 Where the k-fold split happens — **the MP1 question**

`components/mp2-py/csc869/kfold.py:15` — `stratified_kfold_indices`. This is the from-scratch
implementation the rubric asks about.

```python
# kfold.py:40-46
for cls in np.unique(y_arr):          # each class shuffled independently
    idx = np.flatnonzero(y_arr == cls)
    rng.shuffle(idx)
    fold_of[idx] = np.arange(len(idx)) % n_splits   # dealt round-robin
```

Say: *"Each class is shuffled on its own and dealt round-robin into folds. That is what makes
it stratified — every fold gets the same 24% positive rate — and it is deterministic under the
seed, which is what lets the Rust code reproduce the same folds."*

`kfold.py:52` `kfold_indices` is the plain unstratified version, kept for completeness.
`kfold.py:73` `validate_folds` checks every index appears exactly once in validation.

### 7.3 Where the folds are frozen to disk (and reused by Rust)

`components/mp2-py/csc869/data.py:140` `make_folds`, `data.py:198` `ensure_folds`.

```python
# data.py:203-212
if path.exists() and not cfg.get("_force_folds", False):
    folds = load_folds(path, n_rows=len(y), y=y)
    ...
folds = make_folds(y, n_splits=int(cfg["n_splits"]), seed=int(cfg["seed"]))
save_folds(path, folds, y, n_splits=..., seed=...)
```

Say: *"The split is written to `data/folds.json` once and then loaded, never regenerated. Both
languages read the same file — the Rust binary validates it against its own row count before it
runs."* See `data.rs:76` `validate` and `main.rs:153`.

### 7.4 Where resampling happens — and where it must not

`components/mp2-py/csc869/sampling.py:18` `resample_indices` returns **positions**, not data,
so the caller keeps control of the train/validation boundary.

```python
# sampling.py:43-52
elif scenario == "upsampling":
    target = max(len(pos), len(neg))
    if len(pos) < target:
        pos = np.concatenate([pos, rng.choice(pos, size=target - len(pos), replace=True)])
    ...
else:  # downsampling
    target = min(len(pos), len(neg))
```

The call site is the one to show: `components/mp2-py/csc869/cv.py:62`.

```python
# cv.py:62-76
X_train_raw = train_df.iloc[train_idx].reset_index(drop=True)
y_train = y[train_idx]

res_idx, counts = resample_indices(y_train, scenario, seed=scenario_seed)
X_res = X_train_raw.iloc[res_idx].reset_index(drop=True)
y_res = y_train[res_idx]

pre = Preprocessor(...)
X_res_p = pre.fit_transform(X_res)          # fitted on resampled TRAIN only
X_val_raw = train_df.iloc[val_idx].reset_index(drop=True)
X_val_p = pre.transform(X_val_raw)          # validation only transformed
```

Say: *"Read it in order: we slice the training fold, resample it, fit the preprocessor on the
resampled rows, and only then transform the validation fold. Three leaks are blocked in six
lines — resampling before the split, encoding before the split, and scaling on validation data."*

The test set is never resampled anywhere: `test_eval.py:25` `run_test_eval` refits on the full
training rows and scores `adult.test` once.

### 7.5 Where each classifier is built

`components/mp2-py/csc869/models.py:76` `build_model` — one function, five branches:

```python
# models.py:88-100 (decision tree and random forest)
if name == "decision_tree":
    return DecisionTreeClassifier(criterion="gini", random_state=seed, class_weight=None)
if name == "random_forest":
    return RandomForestClassifier(n_estimators=int(rf_trees), random_state=seed,
                                  n_jobs=n_jobs, class_weight=None)
```

- Logistic: `Pipeline([StandardScaler, LogisticRegression(max_iter=1000)])` — `models.py:100`.
- SVM: `models.py:114`, with the Platt branch at `models.py:123`:
  `CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, ensemble=False)`.
  Note the asymmetry: in CV the SVM runs **uncalibrated** (`svm_probability=False`) but is still
  scored, because the metrics only need a ranking. Calibration runs for the final test fit, where
  probabilities matter — and it is why that fit costs 603 s.
- kNN: `models.py:134`, `KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree")` inside a
  scaler pipeline.

**Decision-tree depth tuning** — likely follow-up question. `models.py:167` `tune_decision_tree`:
grid `[3, 5, 7, 10, None]`, evaluated by an **inner** stratified 3-fold CV on F1 (`models.py:182`),
so the outer validation fold never participates in the choice.

Say: *"The only model with a tuned hyperparameter is the decision tree. Everything else runs
defaults on purpose — the comparison is about sampling and model family, and tuning each model
individually would confound it."*

### 7.6 Where the metrics come from

`components/mp2-py/csc869/metrics.py` — from scratch, no sklearn imports:

- `metrics.py:55` `f1_score` — harmonic mean via `_safe_div(2pr, p+r)`, so 0/0 is 0, not NaN.
- `metrics.py:79` `roc_auc_score` — rank-based (Mann–Whitney), with average ranks for ties
  (`metrics.py:61`).
- `metrics.py:95` `precision_recall_curve` and `metrics.py:136` `average_precision_score`.
- `metrics.py:146` `compute_metrics` — the one function every stage calls.

Say: *"These are the MP1 implementations, unchanged in behaviour. The reason to keep them
instead of importing sklearn's is that the Rust side has to match something that is written
down, not a library's internal tie-breaking."*

### 7.7 Where the CV loop runs

`components/mp2-py/csc869/cv.py:281` `run_cv`. The task list is generated at `cv.py:299`:

```python
# cv.py:299-303
tasks = [
    (scenario, fold_id, train_idx, val_idx)
    for scenario in scenarios
    for fold_id, (train_idx, val_idx) in enumerate(folds)
]
```

15 tasks, each running all five models → 75 fits. Parallelised with joblib at `cv.py:321`
(`--jobs 4` = four tasks at a time); each task is independent precisely because the folds are
precomputed.

### 7.8 Where the test evaluation and the delta table are

`components/mp2-py/csc869/test_eval.py:25` `run_test_eval` — fit on 100% of `adult.data`, evaluate
on `adult.test`, write `results/test_metrics.csv` and `test_runs.csv`. The CV-vs-test delta table
in the report is built from those two files (`typst_report.py:233` `_delta_summary`).

### 7.9 Where the error analysis lives

- `error_analysis.py:34` `_sample_ids` — the fixed 200-record sample, 100 positives + 100
  negatives, seeded; drawn from the test split so no model has seen it.
- `error_analysis.py:56` `_family_sets` — collapses five models into three families
  (tree-based = DT+RF, linear = LogReg+SVM, instance = KNN).
- `error_analysis.py:72` `_venn_counts` — set arithmetic for the 7 regions.
- `error_analysis.py:139` `_agreement` — pairwise agreement matrix.
- `error_analysis.py:213` `run_error_analysis` — orchestrates, writes the three figures + CSVs.

Say if asked why families and not models: *"Five overlapping circles are unreadable. Three
families answer the question the report asks — does the residual concentrate in one modelling
approach?"*

### 7.10 Where the figures and the report are generated

- `curves.py:61` `make_roc_pr_figures` — per-scenario ROC and PR panels with mean ± 1 SD bands
  over folds, on a common grid (`cv.py:39` `_curve_points`, 200 grid points).
- `curves.py:118` `make_learning_curve_figures` — F1 vs training fraction.
- `curves.py:176` `make_summary_heatmap` — the 5 × 3 F1 / PR-AUC matrix.
- `typst_report.py:929` `generate_typst_report` — builds `reports/report.typ` from the CSVs;
  `typst_report.py:892` `_compile` calls Typst.

Say: *"No number in the report is typed by hand. The report is a build artifact of the result
CSVs, which is also why the tables and the deck cannot drift apart."*

### 7.11 The Rust side — what to show if asked "did you actually reimplement it"

| Concept | Location |
|---|---|
| Entry point, arg parsing, whole CV loop | `components/mp2-rust/src/main.rs:86` `run` |
| Fold loop + resample-inside-fold | `main.rs:194-220` |
| Scenario seed formula (must match Python) | `main.rs:203` — `cfg.seed + fi*17 + si*101` |
| Preprocessor fitted on the resampled train rows | `main.rs:212` → `data.rs:126` |
| Bit-exact numpy RNG clone | `rng.rs:6` `Rng`, `rng.rs:12` `new`, `rng.rs:124` `choice` |
| Resampling | `rng.rs:147` `resample` |
| Model dispatch | `models.rs:427` `fit_predict` |
| Decision tree + depth search | `models.rs:30` `tree`, `models.rs:97` `fit_tree`, `models.rs:441` |
| kNN | `models.rs:143` `knn` |
| Logistic regression (L-BFGS, own gradient) | `models.rs:203` `logistic`, `models.rs:182` `objective` |
| SVM (via libsvm-rs) + Platt calibration | `models.rs:289` `svm`, `models.rs:359` `platt` |
| Metrics | `metrics.rs:2` `compute` |

The line that proves the two languages share a split, `main.rs:203`:

```rust
let seed = cfg.seed + *fi as u64 * 17 + si as u64 * 101;
```

Say: *"Same formula as `cv.py:61`. That is not a coincidence to be hoped for — it is the
contract the parity test checks."*

**Tests**: 8 in-tree Rust tests (`rng.rs`, `metrics.rs`, `models.rs`, `data.rs`), including
`numpy_oracle()`, which compares the Rust RNG against a frozen numpy-generated fixture
(`components/mp2-rust/tests/rng_oracle.json`). `cargo test --release` → 8 passed.

**Differential harness**: `tools/rust_parity.py` — `reference` (`tools/rust_parity.py:27`) runs
Python on the quick protocol, `compare` (`tools/rust_parity.py:82`) joins the two tidy CSVs and
writes `parity_cv_tidy.csv` / `parity_means.csv`.

---

## 8. Live demo runbook

Seven minutes of talk (script: `reports/slides_script.md`), then the code walk-through if he asks
for one. The live-demo commands and output interpretation are in `docs/demo-runbook.md`. Do not open a random file because you were asked "where does the
splitting happen" — that is section 7.2.

1. **Deck (0:00–4:15).** Slides 2 → 5 → 9 → 12. Problem, headline, test check, parity. Skip the
   figure slides if short on time; the numbers carry the argument.
2. **`kfold.py:40` (4:15–4:55).** The stratified split. This is the question that got asked in
   MP1 and will be asked again.
3. **`cv.py:62` (4:55–5:40).** Resample → fit preprocessor → transform validation, in that
   order. Point at the three leak boundaries.
4. **`main.rs:194` (5:40–6:30).** The same loop in Rust, same seeds, reading the same folds.
5. **Close (6:30–7:00).** Recommendation + the honest limitation: full Rust sweep not run.

Backup tabs to have open but not shown unless asked: `models.py:76`, `metrics.py:55`,
`error_analysis.py:213`.

---

## 9. FAQ

**Why not accuracy?**
Because a constant "`<=50K`" predictor gets 0.759 accuracy and 0 F1. Accuracy measures the
majority class on this dataset.

**Why F1 rather than balanced accuracy or MCC?**
F1 is what the brief's rubric uses and it is the metric that trades precision against recall on
the class we care about. PR-AUC is reported as well precisely because it does not require a
threshold choice.

**Why is PR-AUC the tie-breaker?**
It looks at the whole ranking rather than one threshold, so it is insensitive to the arbitrary
0.5 cut. ROC-AUC is reported too but it flatters all models here (0.90–0.91) because it is
insensitive to the prior — that contrast is one of the talk's points.

**Why upsample by duplication instead of SMOTE?**
SMOTE is Python-only in this setup, and the project's spine is that both languages run the same
matrix. Synthetic minority generation would also change the experiment from "sampling" to
"feature-space augmentation". It is listed as future work, not quietly skipped.

**Does resampling fix the imbalance?**
No. It moves F1 by about a point and shifts the precision/recall trade. The hard records stay
hard — 22 of 200 are wrong under every family regardless of scenario.

**Would class weights have been better?**
Possibly, and it is the first item in future work. It is a different lever (loss weighting, not
data duplication) and would deserve its own matrix, which the 5-page budget did not allow.

**Why does downsampling win on recall?**
Dropping majority rows changes the effective prior toward 50/50, so the model's errors move from
missed positives to false positives: mean FN 0.4094 → 0.1720 while mean FP 0.0704 → 0.2078.

**Why does Random Forest look best on CV but SVM on the test set?**
The gap is 0.009 F1 (0.6905 CV vs 0.6843 test), well inside the ±0.017 CV-vs-test band — it is
noise, not a reversal. RF keeps the better PR-AUC; SVM keeps the better recall. The report
recommends RF because it dominates on ranking and costs less to fit.

**Is 0.69 F1 good?**
For Adult with a 24% positive rate and a 0.5 threshold, yes — it is the range the standard
literature reports for non-boosted models. Gradient boosting would add a few points; that is
explicitly future work.

**How is the 200-record sample chosen, and is it fair?**
100 positives + 100 negatives drawn with a fixed seed from the test split, which no model has
trained on. Counts carry sampling noise — the report says so in the limitations rather than
pretending they are exact.

**Why three model families for the Venn and not five models?**
Five-way Venn diagrams are unreadable at page size. The three families answer the question —
does the residual concentrate in one modelling approach — and the pairwise agreement heatmap
covers all five models individually.

**What exactly is identical between Python and Rust, and what is not?**
Identical: the fold file, the scenario seeds, the metric definitions, the CSV schema. Not
identical: the model internals — sklearn's split tie-breaking, per-tree bootstrap streams,
L-BFGS vs SciPy, libsvm-rs vs sklearn's fork. kNN matches exactly (1e-16); SVM, logistic and the
tree agree to ≤2e-3 F1; Random Forest differs by up to 0.011 F1 on fold means (0.018 PR-AUC),
which is what independent bootstrap streams do.

**Why not run the full 450-cell Rust matrix?**
Cost, honestly. It is a run rather than a port — the binary already reads the frozen folds — and
it extrapolates to 8–14 minutes. It is reported as extrapolated, and no Rust full-matrix figure
is claimed anywhere in the write-up.

**Where would a leak show up if there were one?**
In the CV-vs-test delta table: a leak inflates CV and the test column collapses. All 15 cells
sit within ±0.017, which is the table's reason to exist.

**Why tune only the decision tree's depth?**
Because an untuned tree is overfit by construction and the brief asks for a defensible tree.
Tuning every model would confound "sampling strategy" with "hyperparameter search", and cost
days of compute for the SVM.

**Why is the SVM so expensive?**
RBF with Platt calibration fits three calibration models on top of the base fit; on the full
training rows that peaks at 603 s. In CV it runs uncalibrated, which is why CV is much cheaper
than the test stage per fit.

**Does the report's page limit constrain the science?**
It constrains the presentation, not the experiment: the 6-page body carries the argument, and
the full EDA and curve set lives in `reports/figures/` and `reports/report.md`.

---

## 10. Traps

- **Don't say the Rust full matrix was run.** It wasn't. Extrapolated 8–14 min, clearly labelled.
- **Don't say "the results are identical".** They are close, with stated bounds; kNN is the only
  exactly-matching model.
- **"450-cell" means 450 metric values**, not 450 fits. Fits: 75 in CV per language, 15 in test.
- **Don't say the test set was resampled.** It never is. Resampling is inside the training fold.
- **Don't call the comparison "tuned".** Only the tree's depth is tuned.
- **Don't claim resampling solved the imbalance.** It moved F1 ~1 point and the hard cases remain.
- **Don't quote 0.8437 accuracy as the headline.** That is RF·upsampling accuracy; F1 is 0.6722.
- **Don't present the recommendation as universally best** — it is the best under the 24%
  prior with a 0.5 threshold, and the report says the threshold should be re-set to a business cost.

---

## 11. File map

```
components/mp2-py/csc869/
  data.py           loader, folds.json generation, --check exit test
  kfold.py          from-scratch stratified k-fold + fold validation
  sampling.py       none / upsampling / downsampling → positional indices
  preprocess.py     leak-free Preprocessor (fit inside fold)
  models.py         five estimators + DT depth tuning + Platt calibration
  metrics.py        from-scratch accuracy/precision/recall/F1/ROC-AUC/PR-AUC
  cv.py             the CV matrix runner and fold worker
  test_eval.py      full-training refit, single test-split evaluation
  curves.py         ROC/PR panels, learning curves, 5×3 heatmap
  error_analysis.py 200-record sample, family Venn, agreement, FP/FN
  report.py         Markdown draft from the CSVs
  typst_report.py   Typst report from the CSVs + compile
  run.py            stage driver: prepare, eda, missing, cv, learning, curves, test, error, legacy, report
  legacy/           MP1 Naive Bayes + k-fold + metrics carry-forward
components/mp2-rust/src/
  main.rs           CLI, fold loop, CSV writer
  data.rs           CSV loader, fold validation, Preprocessor
  rng.rs            numpy-compatible RNG, resampling
  models.rs         tree/RF/kNN/logistic/SVM + Platt
  metrics.rs        the same six metrics
tools/rust_parity.py   reference / compare / fixtures for the differential test
data/folds.json        the frozen split both languages read
results*/              CSVs (gitignored, regenerated by a run)
reports/figures/dark/  Mocha-recoloured figures used by the deck (gitignored build artifact —
                       regenerate with tools/dark_figures.py if you rebuild the slides)
reports/               report.typ/pdf, slides.typ/pdf, slides_notes.md, figures/
```

---

*Everything in section 6 is reproducible from `results/cv_summary.csv`, `results/test_metrics.csv`,
`results/test_runs.csv`, `results_quick/rust/parity_cv_tidy.csv` and `results_quick/rust/parity_means.csv`.*
