# CSC 869 MP2 — presentation script

Deck: `reports/slides.pdf` — 13 slides, 16:9. **Allocation: 7:00.**

Blockquotes are what you say. [Square brackets] are stage directions — never spoken. Bold marks the
one sentence per slide that has to land; if you lose your place, find the bold line and work
backward from it.

Each slide opens with **On screen:** — that is your cue for what to point at, and the wording that
follows refers to it. Don't read numbers off a slide that aren't in this script; every figure traces
to `results/cv_summary.csv`, `results/test_metrics.csv` or the error-analysis CSVs.

Rehearse once out loud with a timer. The per-slide clock is computed at 145 wpm from the word count
and lands at 6:53, so there are 7 seconds of slack before the hard stop. If you're behind at slide 6,
use the cut list at the bottom.

---

## 1 · Title — 0:00–0:17 (17 s)

**On screen:** title, subtitle, your name.

> Same dataset as MP1, different question. Last time I had to build the algorithm. This time: when
> only a quarter of the data is the class you care about, does resampling actually help — or does it
> just move the numbers around?

---

## 2 · Accuracy lies — 0:17–0:53 (36 s)

**On screen:** four stat lines and the class-balance bar chart on the left, the 0.759 callout on the right.

> Adult, same census data as MP1: 32,561 training rows, 16,281 test, and a quarter of them earning
> over 50K — the short bar there.
>
> Here's the trap that shaped everything. Predict "under 50K" for everybody and you score 0.759
> accuracy — seventy-six percent — with an F1 of exactly zero.
>
> **So accuracy is out.** I score F1 on the positive class and break ties with PR-AUC, because
> PR-AUC judges the ranking instead of one threshold. ROC-AUC is in the report too; I'll show you why
> it flatters everything.

---

## 3 · Pipeline — 0:53–1:34 (41 s)

**On screen:** five boxes, left to right.

> Five stages: clean, fold, resample, fit, score.
>
> Cleaning is MP1's problem again — same two files, same junk header in `adult.test`, same trailing
> period on test labels. That code transferred straight across.
>
> What's new is the middle: five stratified folds, seed 42, written to a file once, and both
> languages read that same file. That's what makes the two-language comparison legitimate.
>
> [Point at the resample box.] And resampling sits *inside* the fold — train part, resample, fit the
> encoder on those rows, then only transform the validation fold. **Resample before splitting and you
> get a 99% F1 that means nothing.**

---

## 4 · Methods — 1:34–2:13 (39 s)

**On screen:** models on the left, scenarios on the right, the 75-fits line underneath.

> Five models: decision tree, random forest, logistic regression, an RBF SVM, k-nearest neighbours.
> Three scenarios: leave it alone, upsample the minority until the classes match, or downsample the
> majority until they do.
>
> Five models, three scenarios, five folds — seventy-five fits per language, six metrics each: the
> 450 numbers in the report.
>
> Defaults on purpose, except the tree's depth, tuned on an inner three-fold split of training rows
> only. And MP1's Naive Bayes still runs: 0.833 on ten-fold, exactly the band it hit in MP1.
> **That's how I know the cleaning didn't move underneath me.**

---

## 5 · Headline result — 2:13–2:47 (34 s)

**On screen:** the three-row table on the left, "what actually matters" on the right.

> Here's the headline, in the left-hand table. Best F1 per scenario: 0.6813 with no resampling,
> 0.6905 with upsampling, 0.6818 with downsampling.
>
> Three numbers inside a point of each other. **So resampling — the thing everyone reaches for first
> — buys about one F1 point, and it doesn't rescue the minority class.**
>
> What it changes is the trade: downsampling takes false negatives from 0.41 to 0.17 while false
> positives rise from 0.07 to 0.21. Upsampling holds precision and pays in duplicated rows.

---

## 6 · The matrix — 2:47–3:02 (15 s)

**On screen:** the 5 × 3 heatmap.

> The same fifteen cells, as a picture. Don't read every number — read the shape. KNN is the pale
> row in all three scenarios; everything else sits in a narrow band.
>
> **The narrow band is the result.**

---

## 7 · ROC and PR — 3:02–3:31 (29 s)

**On screen:** ROC on top, PR underneath, one caption.

> Two views of ranking, stacked so you can compare them.
>
> Top panel, ROC: everything sits near 0.90, including models I know behave differently. At a 24%
> positive rate ROC is dominated by true negatives, so it can't separate them.
>
> Bottom panel, precision-recall: now they separate. PR-AUC runs from 0.47 — decision tree,
> upsampling — to 0.77 for random forest with no resampling.
>
> **ROC flatters everything here. PR tells the truth.**

---

## 8 · Learning curves — 3:31–3:47 (16 s)

**On screen:** three panels of F1 against training fraction.

> F1 against training-set size. The curves climb fast, then flatten — doubling the data buys almost
> nothing.
>
> **The ceiling is signal in the features, not the number of rows**, so I won't claim more data would
> fix this.

---

## 9 · Test set — 3:47–4:20 (33 s)

**On screen:** five-row table on the left, delta note on the right.

> The check that matters: refit on the full training set, score the test split once.
>
> Test minus CV stays inside ±0.017 F1 across all fifteen cells. **That's the number I'd defend in an
> interview**, because leakage shows up exactly here — inflated CV, collapsed test. No collapse.
>
> Best test F1 is SVM with downsampling, 0.6837 — but read its row: recall 0.85 against precision
> 0.57. Random forest with upsampling gets 0.6722 and is the only balanced row: 0.67 and 0.68.

---

## 10 · Where the errors live — 4:20–4:58 (38 s)

**On screen:** Venn on the left, pairwise agreement matrix on the right.

> Two hundred test records, a hundred per class, fixed seed — and I asked where the models disagree.
>
> In the no-resampling scenario all three families are wrong on twenty-two of them, and at least one
> family misses fifty. KNN is the most alone-wrong: eleven records nobody else misses. **Not a
> threshold problem — a model with no near neighbour to lean on.**
>
> And the hard cases have a shape: education-num 9 to 13 covers 91, 89 and 88 per cent of the
> all-wrong records — mid-education, not the capital-gain group I expected.

---

## 11 · Recommendation — 4:58–5:37 (39 s)

**On screen:** "Why" on the left, "Failure mode" on the right.

> Random forest with upsampling: best mean CV F1 at 0.6905, PR-AUC 0.7479 as tie-breaker, sixteen
> seconds to fit on the full resampled training set, and the only balanced row on the test set.
>
> The failure mode, plainly: probabilities aren't calibrated, so 0.5 is arbitrary; duplicated rows
> inflate confidence on the cases they duplicate; and a forest is harder to explain than a tree.
>
> Next: calibrate and cost-tune the threshold, class weights against resampling, then SMOTE and
> gradient boosting. SMOTE stayed out because it's Python-only here, and the two-language comparison
> is the spine of this project.

[If asked "why not logistic regression?" — best PR-AUC of the resampled options at 0.7651, but lower
F1 and lower precision at 0.5. Pick it only for a ranking system.]

---

## 12 · Two languages, one answer — 5:37–6:24 (47 s)

**On screen:** the delta table on the left, "why the residual exists" on the right.

> All of this also runs as native Rust: same fold file, same resample seeds, same metric code. MP1
> shipped the same pair — one classifier, both languages — so this is that discipline at five times
> the scale: **when the two disagree, find out why before trusting a number.**
>
> On the quick protocol — 8,000 rows, two folds — kNN agrees to one part in ten to the sixteenth and
> SVM matches on F1. Logistic is out by 0.0003, the tree by a thousandth, random forest by 0.011 —
> per-tree bootstrap randomness, not pipeline.
>
> **If two implementations agree cell by cell, the pipeline isn't an explanation for any result I
> just showed you.**

---

## 13 · Cost and close — 6:24–6:53 (29 s)

**On screen:** Python column, Rust column, closing line.

> Cost, for the record: twenty to forty minutes for the full Python matrix, ninety seconds on the
> quick protocol. Rust does that in 114 seconds — threefold — and a full Rust sweep extrapolates to
> eight to fourteen minutes. I didn't run that one, so it's labelled extrapolated.
>
> **Same data as MP1, one fold file, two languages, one answer: resampling is a one-point lever, not
> a twenty-point one.** Thank you.

---

## Timing checkpoints

Glance at the slide number, not a clock.

| Clock | Should be finishing | If you're past this |
|---|---|---|
| 0:53 | slide 2 | drop the ROC-AUC sentence |
| 1:34 | slide 3 | drop the `adult.test` cleaning detail |
| 2:13 | slide 4 | drop the Naive Bayes line |
| 2:47 | slide 5 | cut the FN/FP numbers, keep "one point" |
| 3:31 | slide 7 | slide 6 is skippable in a pinch |
| 4:20 | slide 9 | cut the precision/recall reading |
| 4:58 | slide 10 | cut the education-num detail |
| 5:37 | slide 11 | shorten next steps to "calibration, then class weights" |
| 6:53 | slide 13 (script ends; 7 s of slack to 7:00) | — |

**Running long — cut in this order:** slide 8 entirely → the education-num sentence on 10 → the
FN/FP trade on 5 → the failure-mode paragraph on 11. About 40 seconds, and the argument survives.

**Running short — add:** on slide 3, the missing-value handling (categoricals become an explicit
Unknown level, numerics get an indicator column); on slide 9, that the recommendation is
threshold-sensitive — re-set the cut to a business cost and the ranking changes.

## If he asks for code mid-talk

Three jumps cover what he asked for in MP1:

1. `components/mp2-py/csc869/kfold.py:40` — the stratified split (MP1's routine, generalised to 5 folds).
2. `components/mp2-py/csc869/cv.py:62` — resample, fit preprocessor, transform validation, in that order.
3. `components/mp2-rust/src/main.rs:194` — the same loop in Rust, same seeds, same folds.

Full map with the lines to point at: `docs/presentation-guide.md` §7. Live-demo commands:
`docs/demo-runbook.md`.
