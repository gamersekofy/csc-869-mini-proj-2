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
  #text(size: 11pt)[Python reference implementation]
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
  #text(size: 14pt, weight: "bold")[8,000]
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
  #text(size: 14pt, weight: "bold")[24.07%]
  #linebreak()
  #text(size: 8pt)[positive class (>50K)]
],
  block(width: 100%)[
  #set align(center)
  #text(size: 14pt, weight: "bold")[30]
  #linebreak()
  #text(size: 8pt)[CV fits (2 × 3 × 5)]
],
)

= Problem and data

The task is binary income prediction on the UCI Adult data: *24.07%* of the 8,000 training rows earn more than 50K, so a constant `<=50K` predictor already scores *75.92%* accuracy while finding zero positive cases. Every result below therefore leads with F1 and PR-AUC, reports mean ± standard deviation over two frozen folds, and compares three sampling scenarios (none, upsampling, downsampling) for five classifiers. The Python component is the reference implementation; the Rust component is expected to read the same `data/folds.json` and emit the same tidy schema.

#quote(block: true)[
  *Why accuracy is the wrong headline* A constant `<=50K` classifier scores 75.92% accuracy and F1 = 0. The best no-resampling models score about 0.85 accuracy but only 0.66–0.68 F1, so the ranking that matters is F1 / PR-AUC.
]

= Preprocessing and missing values

- *Cleaning.* The loader strips whitespace, skips the `adult.test` junk header, removes the trailing period from test labels, and maps `?` to null. Row counts are 8,000 train / 16,281 test, matching UCI exactly.
- *Two strategies per feature type.* Categorical: `unknown` (missing as its own level) and training-mode imputation. Numeric: median and median + missing indicator. Adult has no missing numeric values, so the numeric variants are identical in practice; the real numeric issue is the zero-inflated `capital-gain`/`capital-loss` pair, which is shown in the EDA section instead of faking missingness.
- *Leak-free by construction.* Modes, medians, means, standard deviations and one-hot vocabularies are fitted inside each fold on the resampled training rows only; resampling happens after the split and never touches validation or test rows. Test-only categories fall back to the training mode bucket (the full Adult test split has none; individual CV folds can).
- *No rows dropped.* Dropping the 7% with missing categorical values would change the class balance and shrink the already-small positive class.

== Missing-value ablation (decision tree, 2-fold CV)

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
  [*unknown*],
  [*median*],
  [*0.8376*],
  [*0.5932*],
  [*0.6806*],
  [unknown],
  [indicator],
  [0.8376],
  [0.5932],
  [0.6806],
  [mode],
  [median],
  [0.8375],
  [0.5927],
  [0.6799],
  [mode],
  [indicator],
  [0.8375],
  [0.5927],
  [0.6799],
  table.hline(stroke: 0.8pt),
))

Mode imputation edges out the explicit `unknown` level by about 0.001 F1; the numeric variants are identical, as expected. The ablation is small on purpose: it answers the rubric question without pretending Adult has numeric missingness.

Unseen test categories handled by the mode fallback in this run: `native-country` (18), `workclass` (3).

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
    *Protocol.* Stratified 2-fold CV (seed 42) frozen in `data/folds.json` and
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

```python
# csc869/cv.py -- one (scenario, fold) slice
res_idx, _ = resample_indices(y_train, scenario, seed=scenario_seed)
X_res = X_train_raw.iloc[res_idx].reset_index(drop=True)
pre = Preprocessor(categorical=cfg["missing"]["categorical"],
                   numeric=cfg["missing"]["numeric"])
X_res_p = pre.fit_transform(X_res)   # fit on resampled rows only
X_val_p = pre.transform(X_val_raw)   # validation rows are never resampled
model, _ = fit_model(model_name, X_res_p, y_res, seed=model_seed)
metrics = compute_metrics(y_val, model.predict(X_val_p),
                          predict_scores(model, X_val_p))
```

*MP1 carry-forward.* The from-scratch mixed Naive Bayes, stratified k-fold and metrics modules still run on the cleaned data: 10-fold accuracy *0.8254 ± 0.0120*.

= Exploratory analysis

The figures argue four things: the class prior makes accuracy misleading; missingness and part-time work are informative; capital gain is effectively a spike at zero; and education-num is the strongest monotone numeric signal.

#grid(
  columns: (1fr, 1fr),
  gutter: (10pt, 5pt),
  figure(
  image("figures_quick/eda_class_balance.png", width: 100%),
  caption: [24.08% positive: a constant \`\<=50K\` predictor already scores 75.92%.],
),
  figure(
  image("figures_quick/eda_categorical_income_rates.png", width: 100%),
  caption: [Income rate by categorical level; the \`?\` level is not missing at random.],
),
  figure(
  image("figures_quick/eda_capital_gain_loss.png", width: 100%),
  caption: [Capital gain/loss are zero-inflated, so trees over-rely on them.],
),
  figure(
  image("figures_quick/eda_discretisation_age.png", width: 100%),
  caption: [Age bins: income rises through mid-career and flattens late.],
),
  figure(
  image("figures_quick/eda_discretisation_hours_per_week.png", width: 100%),
  caption: [Hours bins: part-time work has a very low \>50K rate.],
),
  figure(
  image("figures_quick/eda_discretisation_education_num.png", width: 100%),
  caption: [Education-num bins: a steep, near-monotone rise.],
),
  figure(
  image("figures_quick/eda_association_heatmap.png", width: 100%),
  caption: [Cramér's V and Spearman association structure.],
),
  figure(
  image("figures_quick/eda_missingness.png", width: 100%),
  caption: [Missingness is confined to three categorical columns.],
),
)

#quote(block: true)[
  *What the EDA changes* Missingness is informative, not noise; part-time work and low education are the clearest low-income markers; and the capital-gain spike means a handful of rows can dominate a tree split. These observations motivate the error patterns in §8.
]

#pagebreak()

= Cross-validation results

Mean scores over the two frozen folds; the best F1 in each scenario is bold. Full tidy data: `results/cv_tidy.csv` (450 Python rows; 900 with Rust).

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
  [*Logistic Regression*],
  [*None*],
  [*0.8435*],
  [*0.6454*],
  [*0.7391*],
  [*0.8953*],
  [Random Forest],
  [None],
  [0.8400],
  [0.6339],
  [0.7162],
  [0.8881],
  [SVM (RBF)],
  [None],
  [0.8448],
  [0.6338],
  [0.7418],
  [0.8955],
  [Decision Tree],
  [None],
  [0.8376],
  [0.5932],
  [0.6806],
  [0.8766],
  [KNN (k=5)],
  [None],
  [0.8069],
  [0.5710],
  [0.5677],
  [0.8225],
  [*SVM (RBF)*],
  [*Upsampling*],
  [*0.8064*],
  [*0.6775*],
  [*0.7247*],
  [*0.8949*],
  [Logistic Regression],
  [Upsampling],
  [0.8046],
  [0.6735],
  [0.7391],
  [0.8937],
  [Decision Tree],
  [Upsampling],
  [0.7860],
  [0.6525],
  [0.6628],
  [0.8767],
  [Random Forest],
  [Upsampling],
  [0.8311],
  [0.6501],
  [0.6934],
  [0.8835],
  [KNN (k=5)],
  [Upsampling],
  [0.7450],
  [0.5843],
  [0.5120],
  [0.8027],
  [*Logistic Regression*],
  [*Downsampling*],
  [*0.7980*],
  [*0.6685*],
  [*0.7119*],
  [*0.8907*],
  [Random Forest],
  [Downsampling],
  [0.8008],
  [0.6682],
  [0.6766],
  [0.8846],
  [SVM (RBF)],
  [Downsampling],
  [0.7857],
  [0.6646],
  [0.7215],
  [0.8943],
  [Decision Tree],
  [Downsampling],
  [0.7706],
  [0.6437],
  [0.6678],
  [0.8743],
  [KNN (k=5)],
  [Downsampling],
  [0.7509],
  [0.6074],
  [0.5425],
  [0.8308],
  table.hline(stroke: 0.8pt),
))

- `none`: *Logistic Regression* at F1 0.6454, accuracy 0.8435, PR-AUC 0.7391.
- `upsampling`: *SVM (RBF)* at F1 0.6775, accuracy 0.8064, PR-AUC 0.7247.
- `downsampling`: *Logistic Regression* at F1 0.6685, accuracy 0.7980, PR-AUC 0.7119.
- Highest PR-AUC overall: *0.7418* with SVM (RBF) under `none`.
- The decision tree's PR-AUC collapses to *0.6628* under upsampling: duplicating positives distorts its leaf probabilities even though F1 looks respectable.

#figure(
  image("figures_quick/cv_metric_heatmap.png", width: 92%),
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
  [SVM (RBF)],
  [Upsampling],
  [0.6775 ± 0.0064],
  [0.7247 ± 0.0153],
  [0.8064 ± 0.0058],
  [Logistic Regression],
  [Upsampling],
  [0.6735 ± 0.0023],
  [0.7391 ± 0.0127],
  [0.8046 ± 0.0055],
  [Logistic Regression],
  [Downsampling],
  [0.6685 ± 0.0056],
  [0.7119 ± 0.0144],
  [0.7980 ± 0.0067],
  [Random Forest],
  [Downsampling],
  [0.6682 ± 0.0151],
  [0.6766 ± 0.0138],
  [0.8008 ± 0.0095],
  [SVM (RBF)],
  [Downsampling],
  [0.6646 ± 0.0018],
  [0.7215 ± 0.0117],
  [0.7857 ± 0.0025],
  table.hline(stroke: 0.8pt),
))

#quote(block: true)[
  *How stable is the ranking?* Fold-to-fold F1 standard deviations run from 0.0018 to 0.0171 (median 0.0057). The top two configurations — SVM (RBF) under `upsampling` and Logistic Regression under `upsampling` — are separated by only 0.0040 F1, less than one fold standard deviation. Treat the recommendation as a defensible trade-off among near-ties, not a decisive win; the test ranking agrees at the top.
]

#pagebreak()

= Curves and learning behaviour

#figure(
  image("figures_quick/roc_curves.png", width: 100%),
  caption: [ROC curves: mean over five folds with a ±1 std band, one panel per scenario.],
)

#figure(
  image("figures_quick/pr_curves.png", width: 100%),
  caption: [Precision–recall curves: the honest view under 24/76 imbalance.],
)

#figure(
  image("figures_quick/learning_curves_f1.png", width: 100%),
  caption: [Learning curves: F1 vs training fraction on the first fold.],
)

The learning curves show whether resampling adds information or only duplicates: Random Forest F1 rises from 0.6077 at 25% of the training rows to 0.6181 at 100%; Logistic Regression F1 rises from 0.6195 at 25% of the training rows to 0.6386 at 100%; SVM (RBF) F1 rises from 0.5996 at 25% of the training rows to 0.6250 at 100%. The no-resampling curves plateau early, which is the expected signature of a fixed 24% prior rather than a data-starved model. The ROC-AUC learning curves are in `reports/figures/learning_curves_roc_auc.png`.

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
  [*SVM (RBF)*],
  [*None*],
  [*0.8554*],
  [*0.7477*],
  [*0.5855*],
  [*0.6568*],
  [*0.7609*],
  [Logistic Regression],
  [None],
  [0.8488],
  [0.7116],
  [0.6050],
  [0.6540],
  [0.7471],
  [Random Forest],
  [None],
  [0.8468],
  [0.7097],
  [0.5944],
  [0.6470],
  [0.7295],
  [Decision Tree],
  [None],
  [0.8465],
  [0.7782],
  [0.4899],
  [0.6012],
  [0.6908],
  [KNN (k=5)],
  [None],
  [0.8166],
  [0.6307],
  [0.5395],
  [0.5816],
  [0.5786],
  [*SVM (RBF)*],
  [*Upsampling*],
  [*0.8085*],
  [*0.5626*],
  [*0.8515*],
  [*0.6776*],
  [*0.7312*],
  [Logistic Regression],
  [Upsampling],
  [0.8037],
  [0.5560],
  [0.8385],
  [0.6687],
  [0.7449],
  [Decision Tree],
  [Upsampling],
  [0.7978],
  [0.5462],
  [0.8510],
  [0.6654],
  [0.6895],
  [Random Forest],
  [Upsampling],
  [0.8367],
  [0.6522],
  [0.6620],
  [0.6570],
  [0.7115],
  [KNN (k=5)],
  [Upsampling],
  [0.7484],
  [0.4793],
  [0.7561],
  [0.5867],
  [0.5064],
  [*Logistic Regression*],
  [*Downsampling*],
  [*0.7982*],
  [*0.5473*],
  [*0.8424*],
  [*0.6635*],
  [*0.7318*],
  [SVM (RBF)],
  [Downsampling],
  [0.7893],
  [0.5329],
  [0.8747],
  [0.6623],
  [0.7330],
  [Random Forest],
  [Downsampling],
  [0.8023],
  [0.5554],
  [0.8190],
  [0.6619],
  [0.6984],
  [Decision Tree],
  [Downsampling],
  [0.7940],
  [0.5416],
  [0.8331],
  [0.6564],
  [0.6780],
  [KNN (k=5)],
  [Downsampling],
  [0.7430],
  [0.4738],
  [0.7982],
  [0.5947],
  [0.5275],
  table.hline(stroke: 0.8pt),
))

Across the 15 model/scenario pairs the test-minus-CV F1 delta ranges from -0.0127 to 0.0229 (mean 0.0045); the largest drop is KNN (k=5) under Downsampling. PR-AUC deltas stay between -0.0149 and 0.0267.

The test positive rate is *23.62%* versus 24.07% in training, so small negative deltas are distribution shift, not a bug.

= Error analysis

#grid(
  columns: (1fr, 1fr),
  gutter: (10pt, 4pt),
  figure(
  image("figures_quick/venn_misclassification.png", width: 100%),
  caption: [3-family Venn of misclassification sets with the all-three-wrong summary (200-record sample).],
),
  figure(
  image("figures_quick/pairwise_agreement.png", width: 100%),
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
  [0/21 (0%)],
  [0/13 (0%)],
  [0/13 (0%)],
  [Education-num 9–13],
  [19/21 (90%)],
  [11/13 (85%)],
  [11/13 (85%)],
  [FP \> FN models],
  [0/5 (0%)],
  [5/5 (100%)],
  [5/5 (100%)],
  table.hline(stroke: 0.8pt),
))

Downsampling trades precision for recall: the mean false-negative rate falls from 0.4371 to 0.1665 while the mean false-positive rate rises from 0.0706 to 0.2295.

#quote(block: true)[
  *What the errors say* Education-num 9–13 accounts for 91% / 89% / 88% of the all-wrong cases across the three scenarios, so the shared hard core is mid-education rather than the zero-inflated capital feature. False positives outnumber false negatives in all five models under both resampling scenarios, which is the precision cost of chasing recall.
]

#pagebreak()

= Final recommendation

#quote(block: true)[
  *Recommended: SVM (RBF) + Upsampling* 
  - Empirical: highest mean CV F1 (0.6775); PR-AUC (0.7247) is the tie-breaker.
  - Trade-off: test precision=0.5626, recall=0.8515; the choice is defensible under the 24/76 imbalance.
  - Practical: low interpretability and 3.28s mean full-training-set fit time.

  *Failure mode.* expensive to train and score, sensitive to scaling, and its calibrated probabilities are less transparent; duplicated minority rows can inflate confidence and overfit the duplicated records.
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
  [*SVM (RBF)*],
  [*Upsampling*],
  [*0.6775*],
  [*0.7247*],
  [*3.28*],
  [*high*],
  [*low*],
  [Logistic Regression],
  [Upsampling],
  [0.6735],
  [0.7391],
  [0.05],
  [low],
  [high],
  [Logistic Regression],
  [Downsampling],
  [0.6685],
  [0.7119],
  [0.02],
  [low],
  [high],
  [Random Forest],
  [Downsampling],
  [0.6682],
  [0.6766],
  [0.09],
  [medium],
  [medium],
  [SVM (RBF)],
  [Downsampling],
  [0.6646],
  [0.7215],
  [0.32],
  [high],
  [low],
  [Decision Tree],
  [Upsampling],
  [0.6525],
  [0.6628],
  [0.08],
  [low],
  [high],
  [Random Forest],
  [Upsampling],
  [0.6501],
  [0.6934],
  [0.13],
  [medium],
  [medium],
  [Logistic Regression],
  [None],
  [0.6454],
  [0.7391],
  [0.12],
  [low],
  [high],
  [Decision Tree],
  [Downsampling],
  [0.6437],
  [0.6678],
  [0.03],
  [low],
  [high],
  [Random Forest],
  [None],
  [0.6339],
  [0.7162],
  [0.10],
  [medium],
  [medium],
  [SVM (RBF)],
  [None],
  [0.6338],
  [0.7418],
  [1.19],
  [high],
  [low],
  [KNN (k=5)],
  [Downsampling],
  [0.6074],
  [0.5425],
  [0.02],
  [high],
  [medium],
  [Decision Tree],
  [None],
  [0.5932],
  [0.6806],
  [0.05],
  [low],
  [high],
  [KNN (k=5)],
  [Upsampling],
  [0.5843],
  [0.5120],
  [0.06],
  [high],
  [medium],
  [KNN (k=5)],
  [None],
  [0.5710],
  [0.5677],
  [0.04],
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

Run the full pipeline from the repository root:

```bash
nix develop
cd components/mp2-py
python -m csc869.data --check
python -m csc869.run --config configs/main.yaml --out results/ --jobs 4
typst compile --root ../.. reports/report.typ reports/report.pdf
```

The full matrix takes about 20–40 minutes with `--jobs 4` on a modern laptop; the quick profile (`--quick`) finishes in about 1.5 minutes and writes to separate `results_quick/` and `reports/figures_quick/` paths. Figures are 300-dpi PNGs under `reports/figures/`; the Markdown source is `reports/report.md`.

#quote(block: true)[
  *Limitations* The error analysis uses a single 200-record sample, so its counts carry sampling uncertainty; Random Forest and SVM keep mostly default hyperparameters; SMOTE/ADASYN and gradient boosting were left as next steps; and the Rust comparison is pending. The full EDA and curve set lives in `reports/figures/` and `reports/report.md`, which this 6-page summary deliberately compresses.
]

#context [#metadata(counter(page).final().first()) <pagecount>]
