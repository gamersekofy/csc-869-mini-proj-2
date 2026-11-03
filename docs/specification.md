# CSC 869 — Mini Project 2: Execution Plan

**Assignment:** Imbalanced data, more classifiers, in-depth evaluation, final recommendation **Dataset:** Census Income / Adult (UCI), same as Mini Project 1 **Released:** Wed 2026-10-28  •  **Canvas due:** Wed 2026-11-18 **Working deadline (submission ready):** Sun 2026-11-15  — 3-day buffer, matches his usual early-date habit **Two implementations:** Rust and Python, same experiment matrix in both **Suggested repo:** \`\~/Projects/SFSU/fall2026/csc869-mini-project-2/\`

---

## 0\. What is actually being graded

The attached rubric is the **Mini Project 1 rubric**, carried over — it still asks for the Naive Bayes implementation (20 pts) and k-fold CV (10 pts), which are *not* in the Mini Project 2 brief. Two things follow from that:

1. **Do not drop the NBC code.** Carry MP1's Naive Bayes \+ k-fold CV \+ evaluation module forward into the new repo as a working module and mention it in the "main methods adopted" section. It costs \~1 hour and covers 30 rubric points if the old rubric is what gets used.  
2. **Flag the page limit.** Rubric row says "within 3 pages?"; the MP2 statement says "no more than 6 pages". Target **6 pages** (the newer, project-specific instruction) but put a note in the doc you send him so he can confirm with Prof. Yang in class if he cares.

| Rubric row | Pts | Where it is earned |
| :---- | :---- | :---- |
| How to compile and run | 5 | README \+ report §1 run block (both languages) |
| Main methods discussion | 5 | Report §2 |
| Evaluation strategies | 5 | Report §3 (CV protocol, metrics, curve definitions) |
| Results analysis \+ next steps | 10 | Report §4–5 |
| Page limit | 5 | Compile check at the end |
| Successful demonstration | 10 | 5-min live demo, rehearsed |
| Explorative data visualisation | 10 | Phase 2 figures |
| Handling missing values | 10 | Phase 1 (two strategies per type) |
| NBC implementation \+ 0-prob routine | 20 | MP1 module carried forward, still runs |
| k-fold CV implementation | 10 | MP1 module (generalised to 5-fold) |
| Evaluation measurements | 5 | Phase 5 metrics module |
| Decision Tree on the same data | 5 | Phase 4 (also RF/SVM/LogReg/KNN) |

**Bonus vs. this rubric:** the MP2 brief itself requires 5 classifiers × 3 sampling scenarios × 5-fold CV, ROC/PR/learning curves, test-set evaluation, 200-record error analysis with a Venn diagram, and a final recommendation. That is where the real marks live. Treat the rubric as the floor.

---

## 1\. Deliverables

- [ ] Python implementation (full experiment matrix) \+ requirements  
- [ ] Rust implementation (same matrix)  
- [ ] PDF report, ≤ 6 pages  
- [ ] Source code, commented, both languages  
- [ ] README: exact compile \+ run commands for both  
- [ ] 5-min live demo, rehearsed twice  
- [ ] 2–3 min summary \+ reflection  
- [ ] Canvas upload (PDF \+ code zip) — verify the upload landed, do not trust the spinner

---

## 2\. Phase 1 — Data download and cleaning (\~3 h, Oct 29–31)

1. **Download.** UCI Adult, id 2: \`\`\`bash mkdir \-p data && cd data curl \-LO [https\://archive.ics.uci.edu/static/public/2/adult.zip](https://archive.ics.uci.edu/static/public/2/adult.zip) unzip adult.zip   \# adult.data, adult.test, adult.names, old versions \`\`\`  
2. **Verify sizes.** \`adult.data\` \= 32,561 rows; \`adult.test\` \= 16,281 rows. No header in \`adult.data\`. \`adult.test\` starts with one junk line (\`\|1x3 Cross validator\`) and its labels end with a trailing period (\`\>50K.\` / \`\<=50K.\`) — strip both, or the label column silently becomes strings and every model predicts one class.  
3. **Column names** (15): age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class.  
4. **Whitespace \+ missing.** Strip surrounding whitespace on every field; map \`?\` to null. \`?\` only appears in workclass, occupation, native-country (\~2,399 / 2,809 / 857 rows).  
5. **Missing-value strategies — two per type, both evaluated** (this is a 10-point rubric row):  
   - Categorical: (a) impute **mode**; (b) treat \`?\` as its own **"Unknown"** category. The second usually wins because the missingness is not random — it correlates with low income.  
   - Numeric: (a) **median** impute; (b) median \+ **missing-indicator** column. \`fnlwgt\` has no missing values, so the numeric exercise mainly covers capital-gain/loss zeros and hours-per-week outliers — say so in the report rather than faking missingness.  
   - Do **not** drop rows: you lose 7% of the data and change the class balance.  
6. **Encoding.** One-hot for nominal categoricals, standardise numeric (fit scaler **inside** each fold — a scaler fit on the full set leaks test information and inflates CV accuracy by a couple of points).  
7. **Freeze the split protocol.** Write one \`folds.json\` (stratified 5-fold, seed 42\) and have *both* Rust and Python read it. Same folds in both languages is the whole point of a two-implementation comparison.  
8. **Carry forward MP1** now, while the cleaning code is fresh: bring the Naive Bayes classifier, its 0-probability routine (Laplace/alpha smoothing), the k-fold CV module, and the accuracy/precision/recall/F1 module into \`legacy/\` and make them run on the new cleaned data. One command each.

**Exit test:** \`python \-m csc869.data \--check\` prints 32,561 / 16,281 rows, class balance 24.08% / 75.92%, zero nulls after imputation, and the legacy NBC scores ≈ 81–84% accuracy on 10-fold CV (sanity band; it was in that range in MP1).

---

## 3\. Phase 2 — Exploratory visualisation (\~4 h, Oct 31–Nov 1\)

Produce figures that *argue* something, not decoration. Deliverables for the 10-point row:

1. **Class balance** bar \+ the income rate by each categorical attribute (horizontal bars sorted by rate, with counts annotated). This is the figure that justifies the whole report.  
2. **Discretisation experiment** on 2–3 continuous attributes (age, hours-per-week, education-num) at **three bin widths** each, showing income rate per bin with n annotated. The brief asks for the exploration; keep one small figure per attribute.  
3. **capital-gain / capital-loss** as a near-binary feature — a log-scale histogram showing the spike at 0\. Explains why tree models over-rely on it.  
4. **Correlation / association** heatmap (Cramér's V for categoricals, Spearman for numeric) to justify feature choices in the error analysis later.  
5. Export every figure at 300 dpi into \`reports/figures/\` with stable filenames; the report pulls from there, so regenerating never breaks the doc.

**Exit test:** five figures exist, each with a one-sentence caption that states an insight ("never-married \+ low hours drives \>50K rate below 3%").

---

## 4\. Phase 3 — Imbalance handling (\~4 h, Nov 1–2)

Implement exactly the **three scenarios the brief compares**, in both languages:

| Scenario | Rule | Notes |
| :---- | :---- | :---- |
| none | original data | the baseline everything is measured against |
| upsampling | random oversample minority to parity | duplicate rows with replacement |
| downsampling | random undersample majority to the minority count | drops \~75% of rows — expect variance |

- The brief says *at least two* methods; the comparison section requires all three, so three is the minimum that satisfies both.  
- **SMOTE is optional and risky.** It needs neighbour search on encoded numeric space, is fiddly to port to Rust, and would break the "same experiment in both languages" symmetry. If you want it for the analysis, do it only in Python and label it exploratory. Decide this on Nov 2 and do not revisit.  
- **Resample inside the fold only.** Resample the training portion of each CV fold, fit, then score on the untouched validation fold. Resampling before splitting is the classic mistake here and it produces a beautiful, fake F1.  
- **Never resample the test set.** It must stay at 24% positive.  
- Log the resulting positive/negative counts per fold in a CSV — you will need them for the report table.

**Exit test:** \`results/sampling\_counts.csv\` shows, per fold: none 24/76, up 50/50, down 50/50 with row counts 32.5k / 48k / 7.8k.

---

## 5\. Phase 4 — Classifiers (\~12 h, Nov 3–7)

**Python (off-the-shelf, permitted):** DecisionTree (Gini, tuned depth via inner 3-fold), SVM (RBF, \`StandardScaler\` pipeline, probability via Platt/\`CalibratedClassifierCV\`), LogisticRegression (max\_iter 1000, class\_weight=None — sampling handles balance), KNN (k=5, scaled, \`algorithm='ball\_tree'\`), RandomForest (300 trees, \`n\_jobs=-1\`).

**Rust:** implement by hand what is tractable, take off-the-shelf where porting would eat the project:

- **Decision tree** — CART with Gini, depth limit, min-samples-leaf. This is the core; do it properly.  
- **Random forest** — bag your own trees (bootstrap \+ feature subsampling). Reuses the DT, \~100 lines.  
- **Logistic regression** — batch gradient descent with L2, standardised features, 1,000 epochs.  
- **KNN** — brute force with \`rayon\` over rows, or a kd-tree if the brute force is slow. Distance on standardised numerics \+ one-hot Hamming.  
- **SVM** — use \`linfa-svm\` (RBF). The statement explicitly permits off-the-shelf implementations; writing SMO from scratch is a week, not a night. Document the choice in the methods section — that is a strength, not a corner cut.

**Order matters:** Python first (Nov 3), because it gives you the reference numbers the Rust code must reproduce. Rust DT \+ KNN by Nov 5, Rust LogReg \+ RF \+ SVM by Nov 7\.

**Exit test:** both languages produce a per-fold metrics CSV for one classifier with the same schema, and the two agree within \~1 point on accuracy. Divergence \>3 points means a bug (usually: different fold assignment, or Rust not standardising).

---

## 6\. Phase 5 — 5-fold CV, metrics, and curves (\~8 h, Nov 8–9)

1. **Metrics per fold:** accuracy, precision, recall, F1 (positive class \= \`\>50K\`), plus **ROC-AUC** and **PR-AUC / average precision**. Report mean ± std across folds — never a single number.  
2. **ROC curves** — mean curve over folds with the ±1 std band, one panel per sampling scenario, 5 models overlaid.  
3. **Precision–Recall curves** — same layout. This is the figure that shows why accuracy is the wrong headline metric under imbalance.  
4. **Learning curves** — train-size fraction 0.1→1.0 vs F1 (and ROC-AUC) for each model under each scenario. This is where you can see whether upsampling helps by adding information or just by adding duplicates.  
5. **Result matrix:** 5 classifiers × 3 scenarios × 5 folds × 2 languages. Store one tidy CSV (\`lang, model, scenario, fold, metric, value\`) and generate every table and figure from it. Do not hand-copy numbers into the report.  
6. Runtime sanity: 75 fits per language is fine in Python; in Rust keep the DT depth limited or the CV sweep will crawl.

**Exit test:** \`results/cv\_tidy.csv\` has 5 × 3 × 5 × 6 × 2 \= 900 rows; the summary table renders from it with no manual editing.

---

## 7\. Phase 6 — Test-set evaluation (\~3 h, Nov 10\)

1. Train each model on the **full training set** under each sampling scenario (no CV here), predict \`adult.test\`, report the same six metrics. The brief asks for the four classic ones \+ curves; include AUCs because they cost nothing.  
2. **Compare CV vs test side by side** in one table, with the delta column. Discuss: test accuracy usually sits 1–3 points below CV because the test set has a slightly different positive rate (23.9% vs 24.1%) and unseen categories.  
3. Watch for **category coverage**: some native-country / occupation values appear only in test, or only once in train. Note how you handle unseen categories (fallback to the mode bucket) — this is a real bug source and a good report paragraph.

**Exit test:** \`results/test\_metrics.csv\` \+ a CV-vs-test delta table with commentary.

---

## 8\. Phase 7 — Error analysis \+ Venn (\~4 h, Nov 11\)

1. Draw the **200-record sample**: 100 positives \+ 100 negatives from the test set, \`random\_state\` fixed, sample ids recorded in a CSV so the figure is reproducible.  
2. **A 5-set Venn diagram does not exist** — do not fake one. Use two panels:  
   - collapse the five models into three families — **tree-based** (DT \+ RF), **linear** (LogReg \+ SVM), **instance-based** (KNN) — and draw a true 3-set Venn of their **misclassification** sets. The intersection labelled "all three wrong" is the interesting cell.  
   - a **pairwise agreement heatmap** for all five models (accuracy-on-agreement matrix), which is what the brief actually wants to see: who agrees with whom, and on which class.  
3. **Inspect the hard cases.** For the "all wrong" cell, tabulate age / education / hours / capital-gain / actual vs predicted. Expect: very high capital-gain with low hours, and borderline education-num bands. Quantify, don't just show one row.  
4. **Direction of error.** False positives (predicted \>50K, actually ≤50K) usually dominate. Report the FP/FN split per model and per sampling scenario — downsampling buys recall at the cost of precision, and this is where you show it.

**Exit test:** one figure \+ one table \+ three named error patterns, each with counts.

---

## 9\. Phase 8 — Final recommendation (\~3 h, Nov 12–13)

1. Build a **decision table**: model × scenario → accuracy, F1, PR-AUC, training time, inference cost, interpretability. Add a column for "MP1 evidence" so the recommendation genuinely spans both projects.  
2. Recommend **one classifier \+ one sampling strategy**, with three explicit reasons: empirical (metric), trade-off (precision vs recall under imbalance), and practical (training time, interpretability, deployment).  
3. State the **failure mode** of your choice — a recommendation without a stated failure mode reads as marketing.  
4. Next steps (10-point rubric row): calibrated probabilities, cost-sensitive learning instead of resampling, SMOTE/ADASYN, gradient boosting, threshold tuning to a business cost, drift monitoring.

---

## 10\. Phase 9 — Report, ≤ 6 pages (\~8 h, Nov 13–14)

Page budget — hold to it and the page-limit mark is free:

| Page | Content |
| :---- | :---- |
| 1 | Problem, dataset, preprocessing, missing-value strategies, imbalance methods |
| 2 | Methods: five classifiers, sampling protocol, 5-fold CV, metrics, curves; compile/run block |
| 3 | EDA figures (2–3 panels) with insights |
| 4 | CV results table \+ ROC/PR figures |
| 5 | Test-set comparison \+ learning curves |
| 6 | Error analysis (Venn \+ table), recommendation, limitations, next steps |

Rules that keep it at 6 pages: figures at 2 per row, no full-page screenshots, one results table (the rest in an appendix), 10.5pt body, references as footnote-style URLs.

**Compile/run section (5 pts) — paste real, tested commands:**

\`\`\`bash

# Python

python3 \-m venv .venv && source .venv/bin/activate pip install \-r requirements.txt python \-m csc869.run \--config configs/main.yaml \--out results/

# Rust

cargo build \--release ./target/release/csc869 \--data data/ \--folds data/folds.json \--out results/rust/ \`\`\`

Include expected runtime and the exact figures path. If a command has not been run from a clean shell on the day of submission, it is not tested.

---

## 11\. Phase 10 — Demonstration (\~3 h, Nov 14–15)

5-minute script, rehearsed twice with a timer:

| Time | Beat |
| :---- | :---- |
| 0:00 | Dataset \+ imbalance in one slide: 24% positive, why accuracy lies |
| 0:30 | Pipeline diagram: clean → fold → resample(train only) → fit → score |
| 1:15 | Live: run one fold in Python, show the metrics print |
| 2:15 | Same fold in Rust, same numbers — this is the demo's punchline |
| 3:15 | Results dashboard: CV table \+ ROC/PR |
| 4:00 | Venn \+ the one error pattern you can name from memory |
| 4:30 | Recommendation \+ failure mode, 30 s |

Reflection (2–3 min): what surprised you, what you would change, what the imbalance taught you about metric choice. Write three bullets, do not improvise.

**Demo-day kit:** repo open at the run command, \`results/\` pre-populated so a failed live run costs nothing, figures open in a second window, PDF of the report on the desktop.

---

## 12\. Day-by-day schedule

| Date | Work | Hours |
| :---- | :---- | :---- |
| Wed 10-29 | Download, verify sizes, cleaning \+ missing-value strategies | 3 |
| Thu 10-30 | Repo scaffold, folds.json, carry MP1 NBC \+ k-fold \+ metrics forward | 3 |
| Sat 10-31 | EDA figures 1–3 | 3 |
| Sun 11-01 | EDA figures 4–5, discretisation experiment, sampling module | 4 |
| Mon 11-02 | Sampling inside folds, counts CSV, Python harness skeleton | 2 |
| Tue 11-03 | Python: 5 classifiers \+ metrics module (after class) | 3 |
| Thu 11-05 | Rust: DT \+ KNN | 4 |
| Sat 11-07 | Rust: LogReg \+ RF \+ linfa SVM; cross-language agreement check | 5 |
| Sun 11-08 | Full CV sweep both languages | 5 |
| Mon 11-09 | ROC / PR / learning curves | 4 |
| Tue 11-10 | Test-set evaluation \+ CV-vs-test comparison (after class) | 3 |
| Wed 11-11 | Error analysis, Venn, agreement heatmap (holiday, clear day) | 4 |
| Fri 11-13 | Results tables \+ figures, draft methods/eval sections | 4 |
| Sat 11-14 | Analysis \+ recommendation \+ compile/run appendix; page check | 5 |
| Sun 11-15 | Demo rehearsal ×2, reflection, submit PDF \+ code | 3 |

Total ≈ 55 h across 18 days. The three heavy days are Nov 7–8 (Rust finishing \+ CV sweep) — protect the weekend.

---

## 13\. Traps that cost marks

1. **Scaler fit outside the fold** — inflated CV, and the CV-vs-test gap gets blamed on the model instead of the bug.  
2. **Resampling before splitting** — the single most common way this project gets a fake 99% F1.  
3. **\`adult.test\` header and trailing periods** — silently turns labels into strings.  
4. **Unseen categories at test time** — handle explicitly and say how.  
5. **Accuracy as the headline metric** — under 24/76 imbalance, a constant "≤50K" predictor scores 76%. Lead with F1 and PR-AUC.  
6. **Different folds in Rust and Python** — destroys the two-implementation comparison. One \`folds.json\`.  
7. **A 5-set Venn diagram** — collapse to three families instead.  
8. **Copying numbers into the report by hand** — generate every table from \`cv\_tidy.csv\`.  
9. **Untested compile/run commands** — 5 free points, lost in 30 seconds.  
10. **Submitting at 23:50 on Nov 18** — the plan lands it on Nov 15\.