# Measured frozen-fold quick parity

Generated from results_quick/rust/parity_means.csv, not manually transcribed. Values below are Python / Rust / (Rust − Python), averaged after per-fold evaluation.

Protocol: first 8000 source training rows; first two ORIGINAL data/folds.json folds projected onto those rows; 40 forest trees, DT depth 5; quick SVM calibration off. Full original test split: 16281 rows. This is not the stock Python quick repartition.
Reference frozen-fold SHA-256: 2ce31e4856814942a3b29fedc56a68a697b2c3327b16c3e5f5bd2a4b1250d947.
Measured reference CV+test: 337.919353 s. Measured Rust end-to-end: 113.718859 s; repeat: 112.808399 s. Nix startup/build time excluded. The repeat produced byte-identical cv_tidy.csv and test_metrics.csv. Format, strict Clippy, eight unit tests, release build and missing-fold/no-write guards passed.
Input checks: {"folds":true,"resampling":true,"features":true,"preprocessing_head":true}. Numeric transformed heads agree at atol/rtol 1e-10 (observed maximum approximately 1.04e-13). Outer rows, sampled rows and feature names match exactly.
Joined CV cells: 180; joined test cells: 90. CSV schema/order and counts match.
Maximum individual cell absolute delta: CV 0.029490616621983823; test 0.018896068329937288.

| Model/scenario | Mean F1 P/R/Δ | Mean ROC-AUC P/R/Δ | Mean AP P/R/Δ |
|---|---|---|---|
| decision_tree/downsampling | 0.64374 / 0.64467 / +0.00093 | 0.87891 / 0.87952 / +0.00060 | 0.65913 / 0.66095 / +0.00183 |
| decision_tree/none | 0.61292 / 0.61393 / +0.00102 | 0.87732 / 0.87860 / +0.00128 | 0.68694 / 0.68820 / +0.00126 |
| decision_tree/upsampling | 0.64357 / 0.64322 / -0.00035 | 0.88125 / 0.88029 / -0.00096 | 0.67366 / 0.67465 / +0.00099 |
| knn/downsampling | 0.58861 / 0.58861 / +0.00000 | 0.82542 / 0.82542 / +0.00000 | 0.53806 / 0.53806 / -0.00000 |
| knn/none | 0.55589 / 0.55589 / +0.00000 | 0.81496 / 0.81496 / +0.00000 | 0.56879 / 0.56879 / +0.00000 |
| knn/upsampling | 0.56889 / 0.56889 / +0.00000 | 0.79826 / 0.79826 / +0.00000 | 0.49387 / 0.49387 / +0.00000 |
| logistic_regression/downsampling | 0.66041 / 0.66074 / +0.00033 | 0.89270 / 0.89268 / -0.00003 | 0.72676 / 0.72625 / -0.00051 |
| logistic_regression/none | 0.64191 / 0.64191 / +0.00000 | 0.89348 / 0.89350 / +0.00002 | 0.73972 / 0.73973 / +0.00000 |
| logistic_regression/upsampling | 0.66723 / 0.66757 / +0.00034 | 0.89245 / 0.89244 / -0.00001 | 0.73972 / 0.73971 / -0.00001 |
| random_forest/downsampling | 0.65032 / 0.66115 / +0.01082 | 0.88781 / 0.89309 / +0.00528 | 0.70421 / 0.71442 / +0.01021 |
| random_forest/none | 0.63974 / 0.64259 / +0.00285 | 0.89384 / 0.89341 / -0.00044 | 0.74057 / 0.74823 / +0.00765 |
| random_forest/upsampling | 0.65280 / 0.65956 / +0.00676 | 0.88974 / 0.88997 / +0.00023 | 0.70807 / 0.72569 / +0.01762 |
| svm/downsampling | 0.66166 / 0.66166 / +0.00000 | 0.89484 / 0.89484 / +0.00001 | 0.73525 / 0.73524 / -0.00001 |
| svm/none | 0.64759 / 0.64759 / +0.00000 | 0.89609 / 0.89608 / -0.00001 | 0.75390 / 0.75389 / -0.00001 |
| svm/upsampling | 0.66444 / 0.66444 / +0.00000 | 0.89669 / 0.89669 / +0.00000 | 0.73446 / 0.73446 / +0.00001 |

kNN quick metrics agree to floating-point precision; equal-distance traversal can diverge on other data. DT uses the same Gini/midpoint/depth-grid contract but different private feature tie traversal. RF uses different per-tree bootstrap/feature streams. Logistic L-BFGS and native LIBSVM solver paths can differ numerically. All discrepancies are retained in parity_cv_tidy.csv and parity_test_metrics.csv; none are overwritten with reference values.

Full five-fold/300-tree mode was not run. Three-fold final-test Platt calibration is implemented and unit-tested on a synthetic example, not verified on the full Adult sweep. Learning curves, plots, reports and error analysis remain the Python/report lane, not this native CV/test component.
