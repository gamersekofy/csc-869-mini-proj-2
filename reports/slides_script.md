# CSC 869 MP2 — presentation script

Deck: `reports/slides.pdf` — 13 slides. Allocation: 7:00.

Written to be said, not read. The quotes are close to word-for-word but meant to come out loosely —
if a sentence lands differently and the number is right, that's fine. The one rule is not to invent a
number that isn't on this page.

[Square brackets] are things to do or point at, never spoken. Timings assume about 145 words a minute:
the talk proper ends at 7:00 on slide 12, and slide 13 is optional cost detail.

---

## 1 · Title — 0:00–0:22 (22 s)

[Say this before you touch anything.]

> So, mini project two. Same Adult data as MP1, but a different question. Last time I had to build the
> model; this time I wanted to know whether resampling actually helps when the class I care about is
> only a quarter of the data — or whether it just shuffles the numbers.

---

## 2 · Accuracy lies — 0:22–1:01 (39 s)

[Row counts top left, bar chart bottom left, the 0.759 callout on the right.]

> Quick context. Thirty-two and a half thousand rows to train on, sixteen thousand to test, and only
> about a quarter of them earn over fifty K — that's the short bar in the chart.
>
> Here's the problem. If I write a model that just says "under fifty K" to everybody, it scores 0.759
> accuracy, with an F1 of zero.
>
> So accuracy is useless here. I score on F1 for the positive class and use PR-AUC to break ties.
> ROC-AUC is in the report as well; I'll come back to why it isn't much help.

---

## 3 · Pipeline — 1:01–1:41 (40 s)

[Five boxes. Point at them as you name them.]

> Five steps: cleaning, folding, resampling, fitting, scoring. Cleaning is the same chore as MP1 —
> same files, the junk header in adult.test, the trailing periods on the test labels.
>
> What's new is the middle. I make five stratified folds once, with seed 42, write them to a file, and
> both implementations read that same file. That's the only reason a two-language comparison means
> anything.
>
> And resampling sits inside the fold: resample the training part, fit the encoder on the resampled
> rows, then transform the validation fold. Resample first and you get a beautiful F1 that's fiction.

---

## 4 · Methods — 1:41–2:22 (41 s)

[Models left, scenarios right.]

> Five models — decision tree, random forest, logistic regression, an RBF SVM, k-nearest neighbours.
> And three ways of handling the imbalance: leave it alone, duplicate the minority until the classes
> are even, or throw majority rows away until they are.
>
> Fifteen cells, five folds each, so seventy-five fits per language, six metrics per fit.
>
> Hyperparameters are defaults on purpose, except the tree's depth, tuned on an inner three-fold split
> of the training rows. MP1's naive Bayes is still in here too, at 0.833 on ten-fold, right where it
> was, so I know the cleaning hasn't shifted underneath me.

---

## 5 · Headline result — 2:22–3:05 (43 s)

[Table on the left — read the three numbers off it.]

> This is the result I'd lead with. Best F1 in each scenario: 0.6813 with no resampling, 0.6905 with
> upsampling, 0.6818 with downsampling.
>
> Which is basically the same number, three times. So upsampling — the thing everyone reaches for
> first — is worth about one F1 point, and it doesn't rescue the minority class.
>
> What it changes is where the errors land. Downsample and false negatives drop from 0.41 to 0.17,
> while false positives climb from 0.07 to 0.21. Upsample and you keep precision, but you pay in
> duplicated rows.
>
> What surprised me was that changing model moves F1 more than changing sampling strategy.

---

## 6 · The matrix — 3:05–3:22 (17 s)

[The heatmap. Gesture at it, don't read cells.]

> Same table as a picture. You don't need every cell — just the shape. KNN is the pale row in all
> three scenarios, everything else sits in a fairly narrow band, and that band is sort of the whole
> finding.

---

## 7 · ROC and PR — 3:22–3:53 (31 s)

[Top panel first, then the bottom.]

> Two ways of looking at ranking. Up top is ROC, and everything's around 0.90, including models I know
> aren't equally good. With only 24% positives, ROC is dominated by true negatives, so it can't tell
> them apart.
>
> The bottom panel is precision-recall, and there the differences show: 0.47 for the decision tree
> with upsampling, up to 0.77 for random forest with no resampling.
>
> So ROC flatters everything here, and PR is the honest one.

---

## 8 · Learning curves — 3:53–4:10 (17 s)

[Three panels.]

> Supporting evidence more than a result. F1 against how much training data the model got: the curves
> rise quickly and flatten. Doubling the data adds almost nothing, so the limit is the signal in the
> features, not the number of rows.

---

## 9 · Test set — 4:10–4:48 (38 s)

[Table left, delta note right.]

> This is the check I trust most. Refit on the whole training set, score the test set once.
>
> Test minus CV stays inside about 0.017 F1 across all fifteen cells, and that's exactly where a leak
> would show up — CV looks great, then the test number falls over. It doesn't.
>
> The best test F1 is actually the SVM with downsampling, 0.6837, but look how it gets there: recall
> 0.85, precision 0.57. Random forest with upsampling is a little lower at 0.6722 and it's the only
> balanced one, 0.67 and 0.68.

---

## 10 · Where the errors live — 4:48–5:30 (42 s)

[Venn left, agreement matrix right.]

> Then I took two hundred test rows, a hundred of each class, and looked at where the models hurt.
>
> With no resampling, all three model families are wrong on twenty-two of them, and at least one of
> the three is wrong on fifty. KNN is the most alone-wrong — eleven rows nothing else misses. That's
> not a threshold problem; that's a nearest-neighbour model with no neighbour nearby.
>
> The hard rows have a pattern too. Education-num between nine and thirteen covers about ninety
> percent of the all-wrong cases in every scenario, so it's mid-education rather than the
> zero-capital-gain people I assumed going in.

---

## 11 · Recommendation — 5:30–6:11 (41 s)

[Why left, failure mode right.]

> So I'd ship random forest with upsampling. Best mean CV F1 at 0.6905, PR-AUC 0.7479, sixteen seconds
> to fit on the full resampled set, and on test it's the balanced one.
>
> It isn't free. The probabilities aren't calibrated, so 0.5 isn't a meaningful threshold; duplicating
> rows makes it over-confident on exactly the rows it duplicated; and a forest is harder to explain
> than a single tree.
>
> Next: calibrate and set the threshold from a real cost, compare class weights against resampling,
> then SMOTE or gradient boosting. SMOTE is Python-only here, and I didn't want the two
> implementations to drift apart.

---

## 12 · Two languages, one answer — 6:11–7:00 (49 s)

[Delta table left, explanation right.]

> It all runs as Rust too, and MP1 did the same thing — one classifier, both languages — so this is
> that idea at a bigger scale. Same fold file, same seeds, same metric code, and when the two disagree
> I find out why before I trust the number.
>
> On the quick protocol, 8,000 rows and two folds, kNN agrees to about 1e-16, so exactly, and SVM
> matches on F1. Logistic is out by 0.0003, the tree by around a thousandth, random forest by 0.011,
> which is per-tree randomness rather than the pipeline.
>
> That's the bit I'd leave you with: if two implementations line up cell by cell, the pipeline isn't
> the explanation for any of these results. Thanks.

---

## 13 · Cost and close — 7:00–7:27 (27 s)

[Optional buffer slide — the talk has already ended on the previous line. Use it only if you're ahead.]

> Cost, quickly: twenty to forty minutes for the full Python matrix, ninety seconds for the quick one.
> Rust does the quick one in 114 seconds, so three times faster, and at full scale that'd be around
> eight to fourteen minutes, though I didn't run that one.
>
> So: same data as MP1, one fold file, two implementations, and resampling is worth about one point,
> not twenty.

---

## Timing checkpoints

| Clock | Should be finishing | If you're past it |
|---|---|---|
| 1:05 | slide 2 | drop the ROC-AUC sentence |
| 1:48 | slide 3 | drop the adult.test detail |
| 2:30 | slide 4 | drop the naive Bayes line |
| 3:12 | slide 5 | cut the FN/FP numbers, keep "one point" |
| 3:59 | slide 7 | slide 6 is skippable entirely |
| 4:58 | slide 9 | cut the precision/recall reading |
| 5:40 | slide 10 | cut the education-num detail |
| 6:22 | slide 11 | shorten next steps to "calibration, then class weights" |
| 7:00 | slide 12 — script ends there; slide 13 is optional | — |

Running long? Cut in this order: slide 8, then the education-num sentence on 10, then the error-trade
numbers on 5, then the failure-mode paragraph on 11. About 40 seconds, and the argument still stands.

Running early? Add the missing-value handling on slide 3 (categoricals get an explicit Unknown level,
numerics get an indicator column), or on slide 9 mention that the recommendation is threshold-sensitive.

## If he asks to see code

1. `components/mp2-py/csc869/kfold.py:40` — the stratified split, MP1's routine extended to 5 folds.
2. `components/mp2-py/csc869/cv.py:62` — resample, fit preprocessor, transform validation, in that order.
3. `components/mp2-rust/src/main.rs:194` — same loop in Rust, same seeds, same folds.

Full map: `docs/presentation-guide.md` §7. Live-demo commands: `docs/demo-runbook.md`.
