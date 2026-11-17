# CSC 869 MP2 — speaker notes
Deck: `reports/slides.pdf` (13 slides, 16:9). Target: 5:00 hard stop, 5:30 with questions.

Rule for the whole talk: never read a number off the slide you cannot explain. Every figure in the deck comes from `results/cv_tidy.csv`.

---

## 1 · Title (0:00–0:10)
"Predicting >50K income on the Adult census data — five classifiers, three sampling strategies, two implementations."

Do not introduce yourself at length. The professor knows who you are.

## 2 · Accuracy lies (0:10–0:45)
- 32,561 train rows, 16,281 test, 24% positive.
- Say the punchline first: **a constant "≤50K" predictor scores 0.759 accuracy and 0 F1.**
- Therefore: F1 on the positive class is the headline, PR-AUC breaks ties.
- If asked why not balanced accuracy: mention it as a secondary check, not the headline.

## 3 · Pipeline (0:45–1:15)
- Walk the five boxes left to right in one breath: clean → fold → resample → fit → score.
- The one line that matters: **resampling happens inside the fold, on train rows only.** Preprocessing is fit on the resampled train rows and merely applied to validation.
- This is spec trap #1 and #2; say it before anyone asks.
- Missing values: categorical → explicit Unknown level (unseen test categories map there); numeric → indicator column.

## 4 · Methods (1:15–1:45)
- Five models, three scenarios, 15 cells, 5 folds = 75 fits per language.
- Hyperparameters: RF 300 trees, KNN k=5, DT depth tuned over {3,5,7,10,∞}, SVM RBF and Platt-calibrated for the final test fit.
- One folds file, `data/folds.json`, shared by Python and Rust — that's what makes the comparison legal.

## 5 · Headline result (1:45–2:15)
- Best F1 per scenario: none 0.6813 (RF), upsampling 0.6905 (RF), downsampling 0.6818 (SVM).
- Highest PR-AUC overall: 0.7701, RF with no resampling.
- The honest read: **resampling moves F1 by about one point.** It does not rescue the minority class. Downsampling pays accuracy for F1; upsampling pays with duplicated rows.

## 6 · Heatmap (2:15–2:35)
- Point at the diagonal pattern, not every cell: KNN is the weak row everywhere; tree/linear models cluster within ~1 point.
- Say: "the colour band is narrow — that's the result."

## 7 · ROC and PR (2:35–2:55)
- ROC sits at 0.90 for almost everything, which is why it hides the imbalance.
- PR is where the differences show; PR-AUC ranges 0.47–0.77.
- If asked about the ±1 SD band: averaged over folds on a common grid.

## 8 · Learning curves (2:55–3:10)
- More training rows give a small, flattening gain — the ceiling is signal, not sample size.
- Keep this to one sentence; it's supporting evidence.

## 9 · Test set (3:10–3:35)
- Deltas test-minus-CV stay within ±0.017 on F1 and PR-AUC. No leakage, no optimistic surprise.
- The test split has a slightly lower positive rate (23.62% vs 24.08%) which explains the small negative bias.
- Best test F1: SVM downsampling 0.6837; RF upsampling 0.6722 with balanced precision/recall.

## 10 · Error analysis (3:35–4:00)
- 200-record sample: 100 positives, 100 negatives, fixed seed.
- In the none scenario: all three families wrong on 22, at least one family wrong on 50.
- KNN is the most alone-wrong — 11 records nobody else misses. That's an instance-based failure, not a threshold problem.
- This is the slide to sound like you understand the data, not the code.

## 11 · Recommendation (4:00–4:25)
- **Random Forest + upsampling.** Highest mean CV F1 0.6905, PR-AUC 0.7479, 1.69 s full-train fit.
- Failure mode, said plainly: probabilities are not calibrated; duplicated minority rows can inflate confidence; less interpretable than one tree.
- Next steps in one breath: calibration + cost-based threshold, class weights vs resampling, SMOTE/ADASYN, gradient boosting.

## 12 · Two languages, one answer (4:25–4:45)
- Same folds, same resample seeds, same metric code. kNN agrees to 1e-16 — that is exact, not "close".
- Logistic and SVM within 5e-4 F1; decision tree ~1e-3; random forest up to 0.011 F1 / 0.018 PR-AUC.
- Say the reason out loud: different per-tree RNG streams and different solvers, not different data. If the professor pushes, the honest line is "the pipeline is verified; the models are different implementations of the same specification".
- Speed: 337.9 s Python vs 113.7 s Rust on the quick protocol, repeat run byte-identical.

## 13 · Reproduction cost (4:45–5:00)
- No command walk-through — the professor does not need it. Give the cost instead: 20–40 min Python for the full 450-cell matrix, 1.5 min for the quick protocol.
- Rust: 113.7 s measured on the quick protocol (3.0×), so a full Rust sweep extrapolates to roughly 8–14 min. Say "extrapolated" out loud — that sweep was not run.
- Close on the point of the project: one folds file, two languages, same metrics — the comparison is real, not rhetorical.

---

### Q&A prep (one line each)
- **Why not SMOTE?** Python-only in this repo; kept exploratory so the two implementations stay comparable.
- **Why is SVM so slow?** 308 s in the upsampling scenario because the calibrated RBF fit is quadratic; the downsampling fit is 18 s.
- **Why not just use class weights?** That's the natural next experiment — resampling was the assignment's lever.
- **Are Rust and Python numbers identical?** Identical folds, preprocessing and metric definitions; model-level results agree to within library differences (see parity slide if asked).
- **Which metric would you ship on?** PR-AUC for ranking, then a cost-tuned threshold — not F1 at 0.5.
