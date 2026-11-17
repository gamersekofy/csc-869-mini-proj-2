# CSC 869 MP2 — presentation script

Deck: `reports/slides.pdf` — 13 slides, 16:9. **Allocation: 7:00.**

About 1,000 spoken words: roughly 6:40 of speech at a normal pace, leaving ~20 s for pointing at
figures and pausing. Rehearse it once out loud with a timer; if you land under 6:30 you're rushing.

How to use this: blockquoted text is what you say, close to verbatim. Anything in [square brackets]
is a stage direction and is never spoken. Bold is the sentence that has to land — if you lose your
place, find the bold line and work back from it.

Do not read numbers off the slide that aren't in this script. Every figure here traces to
`results/cv_summary.csv`, `results/test_metrics.csv` or the error-analysis CSVs. If he asks for code,
the map is `docs/presentation-guide.md` §7.

---

## 1 · Title — 0:00–0:15 (15 s)

> This is Mini Project 2. The question is narrow: when about a quarter of the data is the class you
> care about, does resampling actually help — or does it just move the numbers around?

[Don't introduce yourself. The question is the whole opening.]

---

## 2 · Accuracy lies — 0:15–0:55 (40 s)

> This is the Adult census dataset. 32,561 training rows, 16,281 test, 24% of them earning over 50K.
>
> Here's the problem. Predict "under 50K" for every single person and you score 0.759 accuracy —
> seventy-six percent — with an F1 of zero.
>
> **So accuracy isn't a metric here, it's a trap.** I score F1 on the positive class, and PR-AUC
> breaks ties because it judges the ranking rather than one threshold. ROC-AUC is in the report too,
> and I'll show you why it's the weaker of the two.

---

## 3 · Pipeline — 0:55–1:30 (35 s)

> Five stages: clean, fold, resample, fit, score.
>
> Two things matter. The folds are stratified and frozen — generated once, seed 42, written to a
> file, and both implementations read that same file. And resampling and preprocessing happen
> **inside** the fold, on training rows only.
>
> [Point at the resample and fit boxes.] I slice the training part, resample it, fit the encoder and
> scaler on the resampled rows, then only *transform* the validation fold with them.
>
> **That ordering is the difference between a real 0.69 F1 and a fake 0.99.**

---

## 4 · Methods — 1:30–2:10 (40 s)

> Five models: decision tree, random forest, logistic regression, an RBF support vector machine,
> k-nearest neighbours. Three scenarios: no resampling, upsampling the minority to parity,
> downsampling the majority to the minority count.
>
> Five models, three scenarios, five folds — seventy-five fits per language, six metrics each, which
> is where the report's 450 numbers come from.
>
> Hyperparameters are defaults on purpose. The one exception is the tree's depth, chosen by an inner
> three-fold search on training rows only. **Tuning everything would have turned a sampling
> comparison into a hyperparameter competition.**

---

## 5 · Headline result — 2:10–2:55 (45 s)

> [Point at the three rows as you say them.] Best F1 per scenario: 0.6813 with no resampling, 0.6905
> with upsampling, 0.6818 with downsampling.
>
> **That's the result: resampling moves F1 by about one point. It doesn't rescue the minority class.**
>
> What it changes is the trade. Downsampling pushes toward recall — false negatives fall from 0.41 to
> 0.17 — while false positives rise from 0.07 to 0.21. Upsampling keeps precision and pays with
> duplicated rows.
>
> Model family matters more than the sampling choice: the spread between the best and worst model
> inside one scenario is wider than the spread between scenarios for a fixed model.

---

## 6 · The matrix — 2:55–3:15 (20 s)

> Same numbers, all fifteen cells. Don't read every cell, read the shape. KNN is the pale row
> everywhere — a distance metric on mostly-categorical data. Everything else sits in a narrow band.
>
> **The narrow band is the finding.**

---

## 7 · ROC and PR — 3:15–3:50 (35 s)

> Two views of ranking. On the left, ROC: everything sits near 0.90, including models I know behave
> differently. With a 24% positive rate, ROC is dominated by true negatives, so it can't separate them.
>
> On the right, precision-recall, and now the differences show: PR-AUC runs from 0.47 to 0.77. Best
> is random forest with no resampling; worst is the decision tree with upsampling.
>
> **ROC tells you nothing on this dataset. PR tells you everything.**

---

## 8 · Learning curves — 3:50–4:10 (20 s)

> F1 against training-set size: the curves climb fast and then flatten. Doubling the data buys almost
> nothing.
>
> **The ceiling is the signal in the features, not the sample size** — which is why I'm not going to
> claim that more data would fix this.

---

## 9 · Test set — 4:10–4:50 (40 s)

> The honest check: refit on the full training set, score the test split once.
>
> Test minus CV stays within plus or minus 0.017 F1 across all fifteen cells. **That's the number I
> care about most**, because leakage shows up exactly here — inflated CV, collapsed test. There's no
> collapse.
>
> Best test F1 is the SVM with downsampling, 0.6837 — but that's recall 0.85 against precision 0.57.
> Random forest with upsampling scores 0.6722 and is the only balanced one: 0.67 precision, 0.68
> recall.

---

## 10 · Where the errors live — 4:50–5:35 (45 s)

> I took two hundred test records, a hundred per class, and looked at where the models disagree.
>
> In the no-resampling scenario the three model families are all wrong on twenty-two of them; at
> least one family is wrong on fifty. K-nearest-neighbours is the most alone-wrong — eleven records
> nobody else misses. **That's not a threshold problem, that's an instance-based model with no near
> neighbour to lean on.**
>
> And the hard core has a shape: education-num between 9 and 13 covers 91, 89 and 88 per cent of the
> all-wrong cases. The ambiguous records are mid-education — not the zero-capital-gain group I
> expected going in.

---

## 11 · Recommendation — 5:35–6:10 (35 s)

> Recommendation: random forest with upsampling — best mean CV F1 at 0.6905, PR-AUC 0.7479, and the
> only balanced operating point on the test set.
>
> Failure mode, plainly: the probabilities aren't calibrated, so 0.5 is an arbitrary threshold;
> duplicated rows inflate confidence on exactly the cases they duplicate; and a forest is less
> interpretable than a tree.
>
> Next, in order: calibrate and cost-tune the threshold, compare class weights against resampling,
> then SMOTE and gradient boosting. SMOTE stayed out because it's Python-only here, and the
> two-language comparison is the spine of this project.

[If he asks "why not logistic regression?" — it ranks better, 0.765 PR-AUC, but its F1 and its
precision at 0.5 are lower. If you were shipping a ranking system, that's the pick instead.]

---

## 12 · Two languages, one answer — 6:10–6:45 (35 s)

> All of this also runs as a native Rust binary: same fold file, same seeds, same metric definitions.
>
> On the quick protocol — 8,000 rows, two folds — kNN agrees to one part in ten to the sixteenth:
> exact, not close. SVM matches on F1 as well. Logistic regression is out by 0.0003 and the tree by
> about a thousandth. Random forest is the loosest at 0.011 F1 on fold means, and that's per-tree
> bootstrap randomness, not a pipeline difference.
>
> **If two implementations agree cell by cell, the pipeline isn't an explanation for any result I
> just showed you.**

---

## 13 · Cost and close — 6:45–7:00 (15 s)

> Cost: the full Python matrix is twenty to forty minutes; the quick protocol, ninety seconds. Rust
> does that in 114 seconds — threefold — and a full Rust sweep extrapolates to eight to fourteen
> minutes. **I didn't run that one, so it's labelled extrapolated.** Thank you.

---

## Timing checkpoints

Glance at the slide number, not a clock.

| Clock | Should be finishing | If you're past this |
|---|---|---|
| 0:55 | slide 2 | drop the ROC-AUC sentence on slide 2 |
| 1:30 | slide 3 | cut "only *transform* the validation fold" |
| 2:10 | slide 4 | cut the hyperparameter-competition line |
| 2:55 | slide 5 | cut the FN/FP numbers, keep "one point" |
| 3:50 | slide 7 | slide 6 is skippable in a pinch |
| 4:50 | slide 9 | cut the balanced-precision sentence |
| 5:35 | slide 10 | cut the education-num detail |
| 6:10 | slide 11 | shorten next steps to "calibration, then class weights" |
| 7:00 | slide 13 | — |

**Running long — cut in this order:** slide 8 entirely → the education-num sentence on 10 → the
FN/FP trade on 5 → the failure-mode paragraph on 11. About 40 seconds, and the argument survives.

**Running short — add:** on slide 3, the missing-value handling (categoricals become an explicit
Unknown level, numerics get an indicator column); on slide 9, that the recommendation is
threshold-sensitive — re-set the cut to a business cost and the ranking changes.

## If he asks for code mid-talk

Three jumps cover what he asked for in MP1:

1. `components/mp2-py/csc869/kfold.py:40` — the stratified split.
2. `components/mp2-py/csc869/cv.py:62` — resample, fit preprocessor, transform validation, in order.
3. `components/mp2-rust/src/main.rs:194` — the same loop in Rust, same seeds, same folds.

Full map with the lines to point at: `docs/presentation-guide.md` §7.
