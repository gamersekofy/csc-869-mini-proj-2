// CSC 869 Mini Project 2 — presentation deck (16:9, Catppuccin Mocha)
#set page(
  paper: "presentation-16-9",
  margin: (x: 1.4cm, y: 1.0cm),
  fill: rgb("#1e1e2e"),
  header: context {
    set text(size: 8.5pt, fill: rgb("#6c7086"))
    if counter(page).get().first() > 1 [
      CSC 869 · Mini Project 2 #h(1fr) Predicting >50K income
    ]
  },
  footer: context {
    set text(size: 8.5pt, fill: rgb("#6c7086"))
    h(1fr)
    counter(page).display("1 / 1", both: true)
  },
)

#set text(size: 13pt, fill: rgb("#cdd6f4"), font: "DejaVu Sans")
#set par(leading: 0.58em)
#set list(marker: text(fill: rgb("#cba6f7"))[•], indent: 0.45em, body-indent: 0.35em)
#set table(stroke: none, inset: (x: 5pt, y: 3pt))

#let accent = rgb("#cba6f7")
#let blue = rgb("#89b4fa")
#let green = rgb("#a6e3a1")
#let yellow = rgb("#f9e2af")
#let red = rgb("#f38ba8")
#let peach = rgb("#fab387")
#let sub = rgb("#a6adc8")
#let panel = rgb("#313244")

#let slide(title: none, body) = {
  pagebreak(weak: false)
  set align(left + top)
  if title != none [
    #v(5pt)
    #text(size: 20pt, weight: "bold", fill: blue)[#title]
    #v(-9pt)
    #line(length: 100%, stroke: 1.4pt + accent)
    #v(7pt)
  ]
  v(1fr)
  body
  v(1fr)
}

#let kv(k, v) = [#text(fill: sub, size: 12pt)[#k] #h(5pt) #v]

// ---- 1. title ----
#page(fill: rgb("#181825"), header: none, footer: none)[
  #set align(center + horizon)
  #text(size: 11pt, fill: accent)[CSC 869 · Mini Project 2]
  #v(12pt)
  #text(size: 30pt, weight: "bold", fill: rgb("#cdd6f4"))[Predicting >50K Income under Class Imbalance]
  #v(8pt)
  #text(size: 15pt, fill: sub)[A comparison of sampling strategies across five classifiers]
  #v(30pt)
  #text(size: 14pt)[Uzair Hamed Mohammed]
  #v(3pt)
  #text(size: 12pt, fill: sub)[San Francisco State University · 2026-11-17]
]

// ---- 2. the problem ----
#slide(title: [Accuracy lies on this dataset])[
  #grid(columns: (1fr, 1fr), gutter: 18pt,
    [
      #kv[Training rows][32,561]
      #v(4pt)
      #kv[Test rows][16,281]
      #v(4pt)
      #kv[Positive (>50K)][24.08% train · 23.62% test]
      #v(4pt)
      #kv[Folds][5 frozen stratified folds, shared by both languages]
    ],
    [
      #text(size: 13.5pt, weight: "bold", fill: yellow)[A constant "≤50K" predictor
scores #text(fill: red)[0.759] accuracy and #text(fill: red)[0 F1].]
      #v(6pt)
      #text(size: 12.5pt)[So the headline metric is F1 on the positive class, with PR-AUC as the ranking tie-breaker.]
    ],
  )
  #v(6pt)
  #image("figures/dark/eda_class_balance.png", width: 34%)
]

// ---- 3. pipeline ----
#slide(title: [One pipeline, two implementations])[
  #grid(columns: (1fr, 1fr, 1fr, 1fr, 1fr), gutter: 6pt,
    ..(
      ("clean", "27 raw columns →\ntyped frame"),
      ("fold", "5 stratified folds,\nfrozen in JSON"),
      ("resample", "train rows only —\nnever the validation fold"),
      ("fit", "preprocess inside\nthe fold, then train"),
      ("score", "F1 · PR-AUC ·\nROC-AUC per fold"),
    ).map(((h, d)) => block(fill: panel, inset: 8pt, radius: 4pt, width: 100%)[
      #set align(center)
      #text(size: 12.5pt, weight: "bold", fill: green)[#h]
      #v(3pt)
      #text(size: 10pt, fill: sub)[#d]
    ])
  )
  #v(10pt)
  #text(size: 13pt)[The order is the whole point: #text(fill: yellow)[resample → preprocess → fit] happens *inside* the fold. Resampling before splitting is how this project fakes a 99% F1.]
  #v(6pt)
  #text(size: 12.5pt)[Missing values: categoricals become an explicit `Unknown` level (test-time unseen categories map there too); numeric missingness gets an indicator column.]
]

// ---- 4. methods ----
#slide(title: [Five classifiers, three sampling scenarios])[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    [
      #text(size: 13pt, weight: "bold", fill: accent)[Models]
      #v(2pt)
      - Decision Tree — depth tuned over {3, 5, 7, 10, ∞}
      - Random Forest — 300 trees
      - Logistic Regression — scaled linear reference
      - SVM — RBF, Platt-calibrated for the final test fit
      - KNN — k = 5, scaled Euclidean / Hamming
    ],
    [
      #text(size: 13pt, weight: "bold", fill: accent)[Scenarios]
      #v(2pt)
      - `none` — the honest baseline
      - `upsampling` — minority duplicated to balance
      - `downsampling` — majority cut to balance
      #v(6pt)
      #text(size: 11.5pt, fill: sub)[15 model × scenario cells × 5 folds = 75 fits per language, all on the same folds.]
    ],
  )
  #v(8pt)
  #text(size: 12.5pt)[Metrics: accuracy, precision, recall, F1, PR-AUC, ROC-AUC — per fold, then averaged. Every number here is generated from `results/cv_tidy.csv`, never typed by hand.]
]

// ---- 5. headline result ----
#slide(title: [Resampling moves F1 by ~1 point, not 20])[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    [
      #text(size: 12.5pt, weight: "bold", fill: green)[Best F1 per scenario]
      #v(4pt)
      #table(columns: 3,
        table.header([#text(fill: sub, size: 10.5pt)[scenario]], [#text(fill: sub, size: 10.5pt)[best F1]], [#text(fill: sub, size: 10.5pt)[model]]),
        [none], [#text(weight: "bold")[0.6813]], [Random Forest],
        [upsampling], [#text(weight: "bold", fill: green)[0.6905]], [Random Forest],
        [downsampling], [#text(weight: "bold")[0.6818]], [SVM (RBF)],
      )
      #v(6pt)
      #text(size: 11.5pt)[Highest PR-AUC overall: #text(weight: "bold")[0.7701] — Random Forest, no resampling.]
    ],
    [
      #text(size: 12.5pt, weight: "bold", fill: yellow)[What actually matters]
      #v(4pt)
      #text(size: 12pt)[No scenario rescues the minority class. Downsampling buys F1 by spending accuracy; upsampling buys it with duplicated rows. Best and worst tree/linear models sit within ~1 F1 point — the imbalance is a data problem, not a tuning problem.]
    ],
  )
]

// ---- 6. heatmap ----
#slide(title: [The full 5 × 3 matrix])[
  #image("figures/dark/cv_metric_heatmap.png", width: 74%)
]

// ---- 7. curves ----
#slide(title: [Ranking quality: ROC and PR])[
  #align(center, image("figures/dark/roc_curves.png", width: 62%))
  #v(3pt)
  #align(center, image("figures/dark/pr_curves.png", width: 62%))
  #v(3pt)
  #text(size: 11.5pt, fill: sub)[Averaged over folds on a common grid, ±1 SD band, one panel per scenario. ROC hides the imbalance; PR does not.]
]

// ---- 8. learning curves ----
#slide(title: [More data helps less than more signal])[
  #image("figures/dark/learning_curves_f1.png", width: 100%)
]

// ---- 9. test set ----
#slide(title: [CV held up on the test split])[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    [
      #table(columns: 4,
        table.header([#text(fill: sub, size: 10.5pt)[model · scenario]], [#text(fill: sub, size: 10.5pt)[F1]], [#text(fill: sub, size: 10.5pt)[prec]], [#text(fill: sub, size: 10.5pt)[rec]]),
        [SVM · down], [0.6837], [0.5713], [0.8510],
        [RF · up], [0.6722], [0.6661], [0.6784],
        [LogReg · up], [0.6737], [0.5622], [0.8404],
        [DT · none], [0.6687], [0.7613], [0.5962],
        [RF · none], [0.6587], [0.7187], [0.6079],
      )
    ],
    [
      #text(size: 12.5pt)[Test-minus-CV deltas stay within ±0.017 on F1 and PR-AUC — no fold-level leak, no optimistic surprise.]
      #v(6pt)
      #text(size: 11.5pt, fill: sub)[The test split carries a slightly lower positive rate (23.62% vs 24.08%), which explains most of the small negative bias.]
    ],
  )
]

// ---- 10. error analysis ----
#slide(title: [Where the remaining errors live])[
  #grid(columns: (1fr, 1fr), gutter: 10pt,
    image("figures/dark/venn_misclassification.png", width: 100%),
    image("figures/dark/pairwise_agreement.png", width: 100%),
  )
  #v(4pt)
  #text(size: 12pt)[In the `none` scenario all three families are wrong on 22 of 200 records, and at least one family fails on 50. KNN is the most alone-wrong: 11 records nobody else misses.]
]

// ---- 11. recommendation ----
#slide(title: [Recommendation: Random Forest + upsampling])[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    [
      #text(size: 12.5pt, weight: "bold", fill: green)[Why]
      #v(2pt)
      - Highest mean CV F1: #text(weight: "bold")[0.6905]
      - PR-AUC #text(weight: "bold")[0.7479] as tie-breaker
      - 1.69 s full-train fit — cheap to retrain
      - Test: precision 0.6661 / recall 0.6784
    ],
    [
      #text(size: 12.5pt, weight: "bold", fill: red)[Failure mode]
      #v(2pt)
      - Probabilities are not calibrated
      - Duplicated minority rows can inflate confidence
      - Less interpretable than a single tree
      #v(6pt)
      #text(size: 11pt, fill: sub)[Next: Platt/isotonic calibration + cost-based threshold, class weights vs resampling, SMOTE/ADASYN, gradient boosting.]
    ],
  )
]

// ---- 12. parity ----
#slide(title: [Two languages, one answer])[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    [
      #text(size: 12.5pt, weight: "bold", fill: green)[F1 delta, Rust − Python]
      #v(4pt)
      #table(columns: 3,
        table.header([#text(fill: sub, size: 10.5pt)[model]], [#text(fill: sub, size: 10.5pt)[max |Δ|·means]], [#text(fill: sub, size: 10.5pt)[worst fold]]),
        [kNN], [#text(fill: green)[0.00000]], [#text(fill: green)[0.00000]],
        [SVM], [#text(fill: green)[0.00000]], [#text(fill: green)[0.00000]],
        [Logistic], [#text(fill: yellow)[0.00034]], [#text(fill: yellow)[0.00067]],
        [Decision Tree], [#text(fill: yellow)[0.00102]], [#text(fill: yellow)[0.00203]],
        [Random Forest], [#text(fill: peach)[0.01082]], [#text(fill: peach)[0.02316]],
      )
    ],
    [
      #text(size: 12.5pt, weight: "bold", fill: accent)[Why the residual exists]
      #v(4pt)
      #text(size: 12pt)[Same folds, same resample seeds, same metric definitions. What differs is the model internals: sklearn's split-tie traversal, per-tree bootstrap streams, L-BFGS vs SciPy, libsvm-rs vs sklearn's fork.]
      #v(6pt)
      #text(size: 11.5pt, fill: sub)[Quick protocol: 8,000 rows, 2 projected folds. Python 337.9 s, Rust 113.7 s — 3.0× faster, and the repeat run is byte-identical.]
    ],
  )
]

// ---- 13. reproduce ----
#slide(title: [Reproduction cost])[
  #grid(columns: (1fr, 1fr), gutter: 20pt,
    [
      #text(size: 12pt, weight: "bold", fill: blue)[Python]
      #v(4pt)
      #kv[Full 450-cell matrix][≈ 20–40 min · \#jobs 4]
      #v(5pt)
      #kv[Quick parity protocol][≈ 1.5 min]
      #v(5pt)
      #kv[Single CV stage][23 min measured]
    ],
    [
      #text(size: 12pt, weight: "bold", fill: peach)[Rust]
      #v(4pt)
      #kv[Full 450-cell matrix][≈ 8–14 min · extrapolated]
      #v(5pt)
      #kv[Quick parity protocol][113.7 s measured]
      #v(5pt)
      #kv[Speed-up, same protocol][3.0×]
    ],
  )
  #v(10pt)
  #text(size: 11.5pt)[Both implementations read the same `folds.json` and emit the same tidy CSV schema, so a full Rust sweep is a run, not a port. The extrapolation above assumes the quick-protocol ratio holds at full scale — it was not run, and no Rust figure is claimed for it.]
]
