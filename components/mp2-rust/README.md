# Native Rust component

The workspace builds the pure-Rust binary `csc869`. It does not invoke Python, load Python predictions, or read reference result CSVs. Python is used only by the external differential-test harness.

## Build and run (repository root)

~~~sh
nix develop .#mp2-rust
cargo build --release
./target/release/csc869 --data data/ --folds data/folds.json --out results/rust/
~~~

The last command is the FULL profile (five frozen folds, 300 forest trees, inner-CV depths 3/5/7/10/unlimited, three-fold out-of-fold Platt calibration for final test SVMs). It is documented, not run as part of the quick verification.

~~~sh
# Same executable, smaller matching parity workload.
./target/release/csc869 --data data/ --folds data/folds.json --out results_quick/rust/ --quick --jobs 8
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
~~~

Use `nix develop .#mp2-rust --command ...` for a non-interactive invocation. Dependencies are centralized and pinned in the root workspace manifest and Cargo.lock. No global package installation or flake changes are needed. `--config` accepts the Python YAML configuration (default components/mp2-py/configs/main.yaml).

## Quick profile: why the stock runner is not used verbatim

The stock Python `--quick` changes the sample, fold count and fold path to data/folds_quick.json, and its default stages write reports. That conflicts with immutable data/reports and the requirement to read ONLY data/folds.json.

The comparison harness calls unchanged Python CV and test functions directly, using the quick model settings (40 trees, depth 5, no SVM calibration), the first 8,000 original training rows, and the first two frozen outer folds filtered to that row prefix. It never re-partitions those rows. This is a **frozen-fold quick profile**, not a claim to reproduce the stock runner's independently repartitioned quick numbers. The entire original Adult test split is evaluated. Full mode uses every original row and frozen fold.

~~~sh
PYTHONDONTWRITEBYTECODE=1 nix develop --command python -B tools/rust_parity.py reference --quick --out results_quick/python --jobs 2
nix develop .#mp2-rust --command cargo build --release
./target/release/csc869 --data data/ --folds data/folds.json --out results_quick/rust/ --quick --jobs 8
PYTHONDONTWRITEBYTECODE=1 nix develop --command python -B tools/rust_parity.py compare --python results_quick/python --rust results_quick/rust
~~~

The harness checks identical CSV column names/order and complete, unique metric cells, then writes every joined cell/delta and per-model/scenario means. It also checks sampling counts and compares actual training/validation/sampled row IDs, feature names, and transformed matrix heads. Reference metrics are computed by the original csc869.metrics module.

## Input and evaluation contract

- Frozen outer train/validation arrays are read and validated for range, duplicates, overlap, complete partition and seed agreement. Outer folds are never generated or rewritten. The separate inner DT tuning folds are generated using the reference's NumPy-compatible stratified algorithm, not substituted for the outer folds.
- Train-only sampling uses seed + 17*fold + 101*scenario_index; learner seeds add 1009*model_index. Test sampling uses seed + 101*scenario_index. SeedSequence/PCG64, bounded draws, Floyd/tail choices and final shuffles are implemented in Rust and tested against independent NumPy fixtures.
- The preprocessor is fitted AFTER sampling, as in Python. Categorical order, missing fill, alphabetical mode tie-break, unseen-to-mode fallback, one-hot column order, population numeric scaling, training-only numeric indicators and optional capital flags mirror the reference. Logistic/kNN perform the additional all-feature StandardScaler stage used in the Python pipelines; SVM uses the preprocessor output without an extra scaler, exactly as models.py does.
- Metrics use positive label >50K, zero-safe F1, tie-aware Mann–Whitney ROC-AUC and distinct-threshold step-integrated average precision (not trapezoidal PR area). Metrics are calculated per fold before means. Summary SD uses ddof=1, matching the exported pandas summary.
- Output schemas: cv_tidy.csv is lang,model,scenario,fold,metric,value (six rows per model/scenario/fold); test_metrics.csv omits fold. Also exported: cv_summary.csv, cv_runs.csv, dt_tuning.csv, sampling_counts.csv, test_runs.csv, test_predictions.csv, input_audit.json and run_metadata.json.

## Learners and known numerical differences

Like the reference, inner tree-depth tuning uses the already resampled/preprocessed outer-training matrix. Duplicate source rows can therefore cross inner tuning folds. The protected outer validation/test rows remain separate; changing the inner duplicate-grouping methodology would require a shared reference change, not a silent Rust-only correction.

Measured per-model/scenario differences and runtimes are archived in [PARITY.md](PARITY.md).

CART uses float32 feature comparisons, Gini-equivalent split gain, midpoint thresholds and inner F1 depth selection. Split ties use a reproducible Rust feature permutation rather than sklearn's private MT19937/Cython splitter traversal; exact tree identity is not promised. The forest averages native CART probabilities with sqrt(p) per-split feature sampling; its per-tree/bootstrap streams differ from sklearn. Logistic regression solves the same L2 C=1 objective with native L-BFGS, max 1000 iterations and gradient tolerance 1e-4; solver paths can differ from SciPy. kNN uses the same twice-scaled Euclidean representation and k=5, with deterministic source-order resolution of equal-distance neighbor-boundary ties (sklearn ball-tree traversal can resolve those differently). SVM uses the FFI-free libsvm-rs 0.9.2 RBF C-SVC with C=1, gamma=scale and tolerance 0.001; it need not be bit-identical to sklearn's LIBSVM fork. Final-test calibration uses three stratified, unshuffled out-of-fold fits, sigmoid targets with Bayesian smoothing, then a full-data refit; quick mode disables it to match Python's quick setting.

No learning curves, plots, report regeneration or error-analysis work is performed by the native command; its requested scope is frozen-fold CV and test-table parity. Do not treat a successful quick run as measured evidence of full-mode runtime or full-sweep model parity.
