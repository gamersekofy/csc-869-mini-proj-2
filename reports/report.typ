#set page(
  paper: "a4",
  margin: (x: 1.6cm, y: 1.5cm),
  header: context {
    set text(size: 8pt)
    if counter(page).get().first() > 1 [
      Predicting >50K Income under Class Imbalance #h(1fr) Uzair Hamed Mohammed
    ]
  },
  footer: context {
    set text(size: 8.5pt)
    counter(page).display("1 / 1", both: true)
  },
)
#set text(size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.")
#show figure.caption: it => text(size: 8pt, it)
#show figure: set block(breakable: false)

#page(numbering: none, header: none, footer: none)[
  #set align(center)
  #v(3.2cm)
  #text(size: 11pt)[CSC 869 · Mini Project 2]
  #v(2.2cm)
  #text(size: 24pt, weight: "bold")[Predicting >50K Income under Class Imbalance]
  #v(0.6cm)
  #text(size: 11.5pt)[A comparison of sampling strategies across five classifiers]
  #v(2.6cm)
  #text(size: 14pt)[Uzair Hamed Mohammed]
  #v(0.5cm)
  #text(size: 11pt)[San Francisco State University]
  #v(2.2cm)
  #text(size: 11pt)[Python and native Rust implementations]
  #v(0.4cm)
  #text(size: 11pt)[2026-11-01]
  #v(1.2cm)
  #line(length: 55%, stroke: 0.6pt)
]
#page(numbering: none, header: none, footer: none)[
  #outline(title: [Contents], indent: auto, depth: 2)
]
#counter(page).update(1)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  block(width: 100%)[
  #set align(center)
  #text(size: 14pt, weight: "bold")[32,561]
  #linebreak()
  #text(size: 8pt)[training rows]
],
  block(width: 100%)[
  #set align(center)
  #text(size: 14pt, weight: "bold")[16,281]
  #linebreak()
  #text(size: 8pt)[test rows]
],
  block(width: 100%)[
  #set align(center)
  #text(size: 14pt, weight: "bold")[24.08%]
  #linebreak()
  #text(size: 8pt)[positive class (>50K)]
],
  block(width: 100%)[
  #set align(center)
  #text(size: 14pt, weight: "bold")[75]
  #linebreak()
  #text(size: 8pt)[CV fits (5 × 3 × 5)]
],
)

= Problem and data

The task is binary income prediction on the UCI Adult data: *24.08%* of the 32,561 training rows earn more than 50K, so a constant `<=50K` predictor already scores *75.92%* accuracy while finding zero positive cases. Every result below therefore leads with F1 and PR-AUC, reports mean ± standard deviation over five frozen folds, and compares three sampling scenarios (none, upsampling, downsampling) for five classifiers. The Python component is the reference implementation; the native Rust component reads the same `data/folds.json`, resamples with the same seeds, and emits the same tidy schema.

#quote(block: true)[
  *Why accuracy is the wrong headline* A constant `<=50K` classifier scores 75.92% accuracy and F1 = 0. The best no-resampling models score about 0.85 accuracy but only 0.66–0.68 F1, so the ranking that matters is F1 / PR-AUC.
]

= Preprocessing and missing values

- *Cleaning.* The loader strips whitespace, skips the `adult.test` junk header, removes the trailing period from test labels, and maps `?` to null. Row counts are 32,561 train / 16,281 test, matching UCI exactly.
- *Two strategies per feature type.* Categorical: `unknown` (missing as its own level) and training-mode imputation. Numeric: median and median + missing indicator. Adult has no missing numeric values, so the numeric variants are identical in practice; the real numeric issue is the zero-inflated `capital-gain`/`capital-loss` pair, which is shown in the EDA section instead of faking missingness.
- *Leak-free by construction.* Modes, medians, means, standard deviations and one-hot vocabularies are fitted inside each fold on the resampled training rows only; resampling happens after the split and never touches validation or test rows. Test-only categories fall back to the training mode bucket (the full Adult test split has none; individual CV folds can).
- *No rows dropped.* Dropping the 7% with missing categorical values would change the class balance and shrink the already-small positive class.

== Missing-value ablation (decision tree, 5-fold CV)

#text(size: 7.8pt, table(
  columns: (1.1fr, 1.1fr, 1fr, 1fr, 1fr),
  align: (left, left, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Categorical*],
    [*Numeric*],
    [*Accuracy*],
    [*F1*],
    [*PR-AUC*],
  ),
  table.hline(stroke: 0.5pt),
  [*mode*],
  [*indicator*],
  [*0.8531*],
  [*0.6624*],
  [*0.7559*],
  [mode],
  [median],
  [0.8531],
  [0.6624],
  [0.7559],
  [unknown],
  [indicator],
  [0.8525],
  [0.6615],
  [0.7545],
  [unknown],
  [median],
  [0.8525],
  [0.6615],
  [0.7545],
  table.hline(stroke: 0.8pt),
))

Mode imputation edges out the explicit `unknown` level by about 0.001 F1; the numeric variants are identical, as expected. The ablation is small on purpose: it answers the rubric question without pretending Adult has numeric missingness.

= Methods

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    *Classifiers.* Decision Tree (Gini, depth tuned by inner 3-fold F1 over
    3/5/7/10/unlimited), Random Forest (300 trees), Logistic Regression
    (L2, `max_iter=1000`), RBF SVM (`C=1`, `gamma="scale"`, Platt-calibrated
    for the final test fit), and KNN (`k=5`, ball tree, scaled features).
    `class_weight` is left at `None` so the sampling comparison is not
    confounded by cost-sensitive learning.

    *Metrics.* Accuracy, precision, recall, F1, ROC-AUC and PR-AUC are computed
    locally in `csc869.metrics`; the unit tests compare them with scikit-learn
    on random data.
  ],
  [
    *Protocol.* Stratified 5-fold CV (seed 42) frozen in `data/folds.json` and
    shared with Rust. Each fold resamples only its training portion, fits the
    preprocessor and model, then scores the untouched validation fold.

    *Curves.* ROC and PR curves are averaged over folds on a common grid with a
    ±1 standard-deviation band, one panel per scenario. Learning curves sweep
    10–100% of the first fold's training rows.

    *Error analysis.* A fixed 200-record test sample (100 positives, 100
    negatives), a 3-family Venn of misclassification sets, a 5×5 agreement
    matrix, and per-model FP/FN counts.
  ],
)

#show raw: set text(size: 6.6pt)
#grid(columns: (1fr, 1fr), gutter: 5pt,
  [
    ```python
# csc869/cv.py -- one (scenario, fold) slice
idx, _ = resample_indices(y_train, scenario, seed=s)
pre = Preprocessor(categorical=cat, numeric=num)
X_res = pre.fit_transform(X_train.iloc[idx])  # train only
X_val = pre.transform(X_val_raw)  # never resampled
model, _ = fit_model(name, X_res, y_res, seed=s)
m = compute_metrics(y_val, model.predict(X_val),
                    predict_scores(model, X_val))
    ```
  ],
  [
    ```rust
// components/mp2-rust/src/main.rs -- same slice
let rel = rng::resample(&y, scenario, seed);
let idx: Vec<_> = rel.iter().map(|&i| tr[i]).collect();
let yr = yy(&train, &idx);
let pre = Preprocessor::fit(&train, &idx, /* cfg */);
let x = pre.transform(&train, &idx); // train only
let z = pre.transform(&train, va);   // never resampled
let m = metrics::compute(&yv, &pred, &score);
    ```
  ],
)

*Cross-language check.* The Rust binary reads the same #raw("data/folds.json"), resamples train rows with the same seeds, and emits the same tidy schema. On the quick protocol (first 8,000 rows, two projected folds) kNN agrees to #raw("1e-16"), logistic regression and SVM to at most #raw("5e-4") F1, the decision tree to about #raw("1e-3"), and the random forest to at most #raw("0.018") PR-AUC -- the residual is per-tree RNG streams, not a fold or preprocessing mismatch.

*MP1 carry-forward.* The from-scratch mixed Naive Bayes, stratified k-fold and metrics modules still run on the cleaned data: 10-fold accuracy *0.8334 ± 0.0052*.

= Exploratory analysis

The figures argue four things: the class prior makes accuracy misleading; missingness and part-time work are informative; capital gain is effectively a spike at zero; and education-num is the strongest monotone numeric signal.

#grid(
  columns: (1fr, 1fr),
  gutter: (10pt, 5pt),
  figure(
  image("figures/eda_class_balance.png", width: 100%),
  caption: [24.08% positive: a constant \`\<=50K\` predictor already scores 75.92%.],
),
  figure(
  image("figures/eda_categorical_income_rates.png", width: 100%),
  caption: [Income rate by categorical level; the \`?\` level is not missing at random.],
),
  figure(
  image("figures/eda_capital_gain_loss.png", width: 100%),
  caption: [Capital gain/loss are zero-inflated, so trees over-rely on them.],
),
  figure(
  image("figures/eda_discretisation_age.png", width: 100%),
  caption: [Age bins: income rises through mid-career and flattens late.],
),
  figure(
  image("figures/eda_discretisation_hours_per_week.png", width: 100%),
  caption: [Hours bins: part-time work has a very low \>50K rate.],
),
  figure(
  image("figures/eda_discretisation_education_num.png", width: 100%),
  caption: [Education-num bins: a steep, near-monotone rise.],
),
  figure(
  image("figures/eda_association_heatmap.png", width: 100%),
  caption: [Cramér's V and Spearman association structure.],
),
  figure(
  image("figures/eda_missingness.png", width: 100%),
  caption: [Missingness is confined to three categorical columns.],
),
)

#quote(block: true)[
  *What the EDA changes* Missingness is informative, not noise; part-time work and low education are the clearest low-income markers; and the capital-gain spike means a handful of rows can dominate a tree split. These observations motivate the error patterns in §8.
]

#pagebreak()

= Cross-validation results

Mean scores over the five frozen folds; the best F1 in each scenario is bold. Full tidy data: `results/cv_tidy.csv` (450 Python rows; 900 with Rust).

#text(size: 7.9pt, table(
  columns: (1.7fr, 1.2fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, left, right, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Model*],
    [*Scenario*],
    [*Accuracy*],
    [*F1*],
    [*PR-AUC*],
    [*ROC-AUC*],
  ),
  table.hline(stroke: 0.5pt),
  [*Random Forest*],
  [*None*],
  [*0.8569*],
  [*0.6813*],
  [*0.7701*],
  [*0.9054*],
  [Logistic Regression],
  [None],
  [0.8517],
  [0.6615],
  [0.7671],
  [0.9066],
  [Decision Tree],
  [None],
  [0.8525],
  [0.6615],
  [0.7545],
  [0.8957],
  [SVM (RBF)],
  [None],
  [0.8555],
  [0.6612],
  [0.7694],
  [0.9020],
  [KNN (k=5)],
  [None],
  [0.8224],
  [0.6062],
  [0.6094],
  [0.8397],
  [*Random Forest*],
  [*Upsampling*],
  [*0.8494*],
  [*0.6905*],
  [*0.7479*],
  [*0.9020*],
  [SVM (RBF)],
  [Upsampling],
  [0.8101],
  [0.6856],
  [0.7460],
  [0.9046],
  [Logistic Regression],
  [Upsampling],
  [0.8109],
  [0.6820],
  [0.7651],
  [0.9064],
  [Decision Tree],
  [Upsampling],
  [0.8134],
  [0.6123],
  [0.4684],
  [0.7446],
  [KNN (k=5)],
  [Upsampling],
  [0.7647],
  [0.6081],
  [0.5455],
  [0.8225],
  [*SVM (RBF)*],
  [*Downsampling*],
  [*0.8049*],
  [*0.6818*],
  [*0.7552*],
  [*0.9062*],
  [Logistic Regression],
  [Downsampling],
  [0.8101],
  [0.6815],
  [0.7621],
  [0.9055],
  [Random Forest],
  [Downsampling],
  [0.8119],
  [0.6810],
  [0.7474],
  [0.9043],
  [Decision Tree],
  [Downsampling],
  [0.7972],
  [0.6715],
  [0.7139],
  [0.8944],
  [KNN (k=5)],
  [Downsampling],
  [0.7699],
  [0.6284],
  [0.5817],
  [0.8491],
  table.hline(stroke: 0.8pt),
))

- `none`: *Random Forest* at F1 0.6813, accuracy 0.8569, PR-AUC 0.7701.
- `upsampling`: *Random Forest* at F1 0.6905, accuracy 0.8494, PR-AUC 0.7479.
- `downsampling`: *SVM (RBF)* at F1 0.6818, accuracy 0.8049, PR-AUC 0.7552.
- Highest PR-AUC overall: *0.7701* with Random Forest under `none`.
- The decision tree's PR-AUC collapses to *0.4684* under upsampling: duplicating positives distorts its leaf probabilities even though F1 looks respectable.

#figure(
  image("figures/cv_metric_heatmap.png", width: 92%),
  caption: [F1 and PR-AUC across the full 5 × 3 matrix.],
)

Top five configurations with fold-level spread (mean ± standard deviation):

#text(size: 7.8pt, table(
  columns: (1.55fr, 1.1fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, left, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Model*],
    [*Scenario*],
    [*F1*],
    [*PR-AUC*],
    [*Accuracy*],
  ),
  table.hline(stroke: 0.5pt),
  [Random Forest],
  [Upsampling],
  [0.6905 ± 0.0080],
  [0.7479 ± 0.0109],
  [0.8494 ± 0.0037],
  [SVM (RBF)],
  [Upsampling],
  [0.6856 ± 0.0060],
  [0.7460 ± 0.0086],
  [0.8101 ± 0.0062],
  [Logistic Regression],
  [Upsampling],
  [0.6820 ± 0.0044],
  [0.7651 ± 0.0064],
  [0.8109 ± 0.0051],
  [SVM (RBF)],
  [Downsampling],
  [0.6818 ± 0.0080],
  [0.7552 ± 0.0104],
  [0.8049 ± 0.0087],
  [Logistic Regression],
  [Downsampling],
  [0.6815 ± 0.0042],
  [0.7621 ± 0.0060],
  [0.8101 ± 0.0056],
  table.hline(stroke: 0.8pt),
))

#quote(block: true)[
  *How stable is the ranking?* Fold-to-fold F1 standard deviations run from 0.0042 to 0.0116 (median 0.0079). The top two configurations — Random Forest under `upsampling` and SVM (RBF) under `upsampling` — are separated by only 0.0049 F1, less than one fold standard deviation. Treat the recommendation as a defensible trade-off among near-ties, not a decisive win; the test ranking agrees at the top.
]

#pagebreak()

= Curves and learning behaviour

#figure(
  image("figures/roc_curves.png", width: 100%),
  caption: [ROC curves: mean over five folds with a ±1 std band, one panel per scenario.],
)

#figure(
  image("figures/pr_curves.png", width: 100%),
  caption: [Precision–recall curves: the honest view under 24/76 imbalance.],
)

#figure(
  image("figures/learning_curves_f1.png", width: 100%),
  caption: [Learning curves: F1 vs training fraction on the first fold.],
)

The learning curves show whether resampling adds information or only duplicates: Random Forest F1 rises from 0.6642 at 10% of the training rows to 0.6865 at 100%; Logistic Regression F1 rises from 0.6733 at 10% of the training rows to 0.6733 at 100%; SVM (RBF) F1 rises from 0.6637 at 10% of the training rows to 0.6730 at 100%. The no-resampling curves plateau early, which is the expected signature of a fixed 24% prior rather than a data-starved model. The ROC-AUC learning curves are in `reports/figures/learning_curves_roc_auc.png`.

#quote(block: true)[
  *Reading the curves* ROC-AUC is similar across models (about 0.90–0.91) because it is insensitive to the prior, while PR-AUC separates them (about 0.61–0.77). The PR panel is where the sampling choice shows: no resampling keeps precision high at low recall, while downsampling moves mass toward recall. Learning curves flatten after roughly 25–50% of the training rows, so more rows alone will not close the gap; better features and threshold tuning will.
]

#pagebreak()

= Test-set evaluation

Each model is refit on the full training set under each scenario and scored once on `adult.test`; the test split is never resampled.

#text(size: 7.7pt, table(
  columns: (1.6fr, 1.1fr, 0.95fr, 0.95fr, 0.9fr, 0.9fr, 0.95fr),
  align: (left, left, right, right, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Model*],
    [*Scenario*],
    [*Accuracy*],
    [*Precision*],
    [*Recall*],
    [*F1*],
    [*PR-AUC*],
  ),
  table.hline(stroke: 0.5pt),
  [*Decision Tree*],
  [*None*],
  [*0.8605*],
  [*0.7613*],
  [*0.5962*],
  [*0.6687*],
  [*0.7602*],
  [SVM (RBF)],
  [None],
  [0.8591],
  [0.7639],
  [0.5840],
  [0.6620],
  [0.7688],
  [Random Forest],
  [None],
  [0.8512],
  [0.7187],
  [0.6079],
  [0.6587],
  [0.7619],
  [Logistic Regression],
  [None],
  [0.8527],
  [0.7292],
  [0.5985],
  [0.6574],
  [0.7622],
  [KNN (k=5)],
  [None],
  [0.8242],
  [0.6457],
  [0.5663],
  [0.6034],
  [0.6002],
  [*SVM (RBF)*],
  [*Upsampling*],
  [*0.8164*],
  [*0.5761*],
  [*0.8427*],
  [*0.6843*],
  [*0.7349*],
  [Logistic Regression],
  [Upsampling],
  [0.8077],
  [0.5622],
  [0.8404],
  [0.6737],
  [0.7599],
  [Random Forest],
  [Upsampling],
  [0.8437],
  [0.6661],
  [0.6784],
  [0.6722],
  [0.7437],
  [Decision Tree],
  [Upsampling],
  [0.8156],
  [0.6088],
  [0.6134],
  [0.6111],
  [0.4647],
  [KNN (k=5)],
  [Upsampling],
  [0.7636],
  [0.4997],
  [0.7535],
  [0.6009],
  [0.5307],
  [*SVM (RBF)*],
  [*Downsampling*],
  [*0.8140*],
  [*0.5713*],
  [*0.8510*],
  [*0.6837*],
  [*0.7479*],
  [Random Forest],
  [Downsampling],
  [0.8109],
  [0.5684],
  [0.8292],
  [0.6745],
  [0.7392],
  [Logistic Regression],
  [Downsampling],
  [0.8072],
  [0.5613],
  [0.8414],
  [0.6734],
  [0.7579],
  [Decision Tree],
  [Downsampling],
  [0.8047],
  [0.5586],
  [0.8248],
  [0.6661],
  [0.7217],
  [KNN (k=5)],
  [Downsampling],
  [0.7664],
  [0.5035],
  [0.7938],
  [0.6161],
  [0.5655],
  table.hline(stroke: 0.8pt),
))

Across the 15 model/scenario pairs the test-minus-CV F1 delta ranges from -0.0226 to 0.0073 (mean -0.0059); the largest drop is Random Forest under None. PR-AUC deltas stay between -0.0162 and 0.0078.

The test positive rate is *23.62%* versus 24.08% in training, so small negative deltas are distribution shift, not a bug.

= Error analysis

#grid(
  columns: (1fr, 1fr),
  gutter: (10pt, 4pt),
  figure(
  image("figures/venn_misclassification.png", width: 100%),
  caption: [3-family Venn of misclassification sets with the all-three-wrong summary (200-record sample).],
),
  figure(
  image("figures/pairwise_agreement.png", width: 100%),
  caption: [Pairwise agreement across the five models.],
),
)

All three families miss the same records in every scenario (see the summary panel), so the hard core is a stable set of difficult records rather than independent model noise.

#text(size: 7.7pt, table(
  columns: (2.1fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Named error pattern*],
    [*None*],
    [*Upsampling*],
    [*Downsampling*],
  ),
  table.hline(stroke: 0.5pt),
  [High capital gain, low hours],
  [0/22 (0%)],
  [0/9 (0%)],
  [0/17 (0%)],
  [Education-num 9–13],
  [20/22 (91%)],
  [8/9 (89%)],
  [15/17 (88%)],
  [FP \> FN models],
  [0/5 (0%)],
  [5/5 (100%)],
  [5/5 (100%)],
  table.hline(stroke: 0.8pt),
))

Downsampling trades precision for recall: the mean false-negative rate falls from 0.4094 to 0.1720 while the mean false-positive rate rises from 0.0704 to 0.2078.

#quote(block: true)[
  *What the errors say* Education-num 9–13 accounts for 91% / 89% / 88% of the all-wrong cases across the three scenarios, so the shared hard core is mid-education rather than the zero-inflated capital feature. False positives outnumber false negatives in all five models under both resampling scenarios, which is the precision cost of chasing recall.
]

#pagebreak()

= Final recommendation

#quote(block: true)[
  *Recommended: Random Forest + Upsampling* 
  - Empirical: highest mean CV F1 (0.6905); PR-AUC (0.7479) is the tie-breaker.
  - Trade-off: test precision=0.6661, recall=0.6784; the choice is defensible under the 24/76 imbalance.
  - Practical: medium interpretability and 1.69s mean full-training-set fit time.

  *Failure mode.* less interpretable than a single tree and its probabilities are not perfectly calibrated; duplicated minority rows can inflate confidence and overfit the duplicated records.
]

#text(size: 7.4pt, table(
  columns: (1.5fr, 1.05fr, 0.8fr, 0.85fr, 0.8fr, 0.85fr, 0.75fr),
  align: (left, left, right, right, right, right, right),
  inset: (x: 4.5pt, y: 3pt),
  stroke: none,
  table.header(repeat: true,
    table.hline(stroke: 0.8pt),
    [*Model*],
    [*Scenario*],
    [*F1*],
    [*PR-AUC*],
    [*Train s*],
    [*Inference*],
    [*Interp.*],
  ),
  table.hline(stroke: 0.5pt),
  [*Random Forest*],
  [*Upsampling*],
  [*0.6905*],
  [*0.7479*],
  [*1.69*],
  [*medium*],
  [*medium*],
  [SVM (RBF)],
  [Upsampling],
  [0.6856],
  [0.7460],
  [308.01],
  [high],
  [low],
  [Logistic Regression],
  [Upsampling],
  [0.6820],
  [0.7651],
  [0.22],
  [low],
  [high],
  [SVM (RBF)],
  [Downsampling],
  [0.6818],
  [0.7552],
  [18.01],
  [high],
  [low],
  [Logistic Regression],
  [Downsampling],
  [0.6815],
  [0.7621],
  [0.05],
  [low],
  [high],
  [Random Forest],
  [None],
  [0.6813],
  [0.7701],
  [1.32],
  [medium],
  [medium],
  [Random Forest],
  [Downsampling],
  [0.6810],
  [0.7474],
  [0.83],
  [medium],
  [medium],
  [Decision Tree],
  [Downsampling],
  [0.6715],
  [0.7139],
  [0.54],
  [low],
  [high],
  [Logistic Regression],
  [None],
  [0.6615],
  [0.7671],
  [0.20],
  [low],
  [high],
  [Decision Tree],
  [None],
  [0.6615],
  [0.7545],
  [1.34],
  [low],
  [high],
  [SVM (RBF)],
  [None],
  [0.6612],
  [0.7694],
  [83.10],
  [high],
  [low],
  [KNN (k=5)],
  [Downsampling],
  [0.6284],
  [0.5817],
  [0.09],
  [high],
  [medium],
  [Decision Tree],
  [Upsampling],
  [0.6123],
  [0.4684],
  [2.06],
  [low],
  [high],
  [KNN (k=5)],
  [Upsampling],
  [0.6081],
  [0.5455],
  [0.33],
  [high],
  [medium],
  [KNN (k=5)],
  [None],
  [0.6062],
  [0.6094],
  [0.21],
  [high],
  [medium],
  table.hline(stroke: 0.8pt),
))

- *Calibrate and threshold.* Fit Platt/isotonic probabilities and tune the decision threshold to an explicit business cost instead of 0.5.
- *Try cost-sensitive learning.* Compare class weights with resampling rather than assuming resampling is the only lever.
- *Extend the model space.* SMOTE/ADASYN and gradient boosting (XGBoost/LightGBM) are the natural next experiments; SMOTE is Python-only and should stay exploratory.
- *Monitor drift.* Watch capital-gain, hours-per-week and native-country coverage after deployment; these are the features that drive the hard cases.

= Reproduce

The experiment matrix is declared in the config:

```yaml
# configs/main.yaml (excerpt)
seed: 42
n_splits: 5
scenarios: [none, upsampling, downsampling]
models: [decision_tree, random_forest, logistic_regression, svm, knn]
missing: {categorical: unknown, numeric: indicator}
```

Both implementations consume the same `folds.json` and emit the same tidy CSV schema, so the two tracks are comparable cell by cell. On the hardware used here ($8$ cores, four parallel workers) the full $450$-cell Python matrix takes roughly 20–40 minutes, of which one CV stage alone accounts for 23 minutes; the quick parity protocol finishes in about 1.5 minutes. The Rust binary runs that protocol in 113.7 s, a factor of (3.0), so a full $450$-cell Rust sweep extrapolates to roughly 8–14 minutes — a run rather than a port. That sweep was not executed, and no Rust figure is claimed beyond the quick protocol.

#quote(block: true)[
  *Limitations* The error analysis uses a single 200-record sample, so its counts carry sampling uncertainty; Random Forest and SVM keep mostly default hyperparameters; SMOTE/ADASYN and gradient boosting were left as next steps; and the cross-language check runs on the quick parity protocol rather than a full 450-cell Rust sweep. The full EDA and curve set lives in `reports/figures/` and `reports/report.md`, which this 6-page summary deliberately compresses.
]

#context [#metadata(counter(page).final().first()) <pagecount>]
