# Live demo runbook — showing it working

The professor wants to see the code run. Two tracks, one of which you should rehearse once on the
laptop you'll present from.

**Recommendation: demo the Rust binary, then show the Python side doing cheap visible work.** The
Rust run is one command, prints progress line by line, and finishes in single-digit minutes. The
Python CV matrix is 23 minutes and must never be run live.

---

## 0 · Before the demo starts

```bash
cd ~/Documents/SFSU/masters/fall-2026/csc-869/mini-project-2
git status --short          # should be clean or show only ignored artifacts
ls data/ data/folds.json    # adult.data, adult.test and the frozen split must exist
```

Terminal font at least 14 pt, and clear the scrollback so the first thing on screen is your prompt.
The run writes into `results/rust/`; that directory is gitignored, so nothing you do here dirties the repo.

---

## 1 · Rust — the main event

```bash
cargo run --release
```

That is the whole command. From the repo root, the binary defaults to
`--data data/ --folds data/folds.json --out results/rust/` and runs the **full** matrix: five
models × three scenarios × five folds, then the test stage on all 16,281 test rows.

Flags, if he asks:

| flag | effect |
|---|---|
| `--quick` | first 8,000 rows, first two frozen folds. Use it if you have under 3 minutes. |
| `--jobs N` | worker threads; defaults to min(cores, 8) |
| `--audit-only` | writes `input_audit.json` (per-scenario, per-fold resample counts) and exits — a 10-second proof that the machinery and the fold file line up |
| `--data`, `--folds`, `--out` | override the paths above |

### What you should have seen, and what it means

```
Compiling mp2-rust v0.1.0 (...)
    Finished `release` profile [optimized] target(s) in 6.42s
     Running `target/release/csc869`
[rust cv] none/0/decision_tree: F1=0.66737, 1.56s
[rust cv] none/0/random_forest: F1=0.68947, 3.78s
[rust cv] none/0/logistic_regression: F1=0.67331, 0.19s
[rust cv] none/0/svm: F1=0.67300, 15.08s
[rust cv] none/0/knn: F1=0.61972, 1.67s
```

- `none/0/decision_tree` is **scenario / fold / model**. `none` = no resampling, `0` = first of the
  five frozen folds.
- `F1=0.66737` is that model's F1 on that fold's held-out rows — one cell of the 75.
- `1.56s` is fit + predict for that cell, which is why the order matters: the SVM line is 10× the
  rest and it is the one you point at when asked about cost.
- The same fold is reused across the three scenarios, so identical fold numbers with different
  scenarios are expected and correct.

After the last CV line the binary refits each model on the full training set and prints
`[rust test] <scenario>/<model>: F1=..., ...` lines, then a single final line:

```
RUST_SECONDS=472.913000
```

**Write that number down.** It turns the deck's "≈ 8–14 min extrapolated" into a measured figure,
which removes a limitation from the talk for free.

### Files it leaves behind

`results/rust/`: `input_audit.json`, `sampling_counts.csv`, `cv_tidy.csv`, `cv_runs.csv`,
`dt_tuning.csv`, `cv_summary.csv`, `test_metrics.csv`, `test_runs.csv`, `test_predictions.csv`,
`run_metadata.json`. Same schema as the Python outputs, which is what makes them joinable.

### Sanity check while it runs

At the pace above, a full CV pass is about 5½ minutes (15 scenario×fold blocks × ~22 s) and the
test stage adds roughly 2 minutes. If it is much slower than that, stop pointing at the wall clock
and use `--quick` instead.

---

## 2 · Python — show it runs without waiting for it

Do not run the full Python matrix live. Three commands, about 15 seconds total, prove the pipeline
is real:

```bash
cd ~/Documents/SFSU/masters/fall-2026/csc-869/mini-project-2
nix develop                       # or: direnv allow, then skip this line
cd components/mp2-py

python -m csc869.run --list-stages   # prepare, eda, missing, cv, learning, curves, test, error, legacy, report, typst
python -m csc869.run --stages curves --jobs 4   # ~8 s: rebuilds the ROC/PR, learning-curve and heatmap figures
python -m csc869.run --stages error --jobs 4    # ~3 s: the Venn, agreement matrix and FP/FN figures
```

If he wants to see the data checks:

```bash
python -m csc869.data --check
```

which prints 32,561 / 16,281 rows, the 24.08 / 75.92 class balance, the fold file summary and the
MP1 Naive Bayes sanity band (0.8334 ± 0.0052). About ten seconds — the legacy 10-fold NBC is the
3-second part.

Costs, so you can answer "why aren't you running that one?" honestly:

| command | time |
|---|---|
| `--stages curves` | ~8 s |
| `--stages error` | ~3 s |
| `--stages cv` (full) | ~23 min |
| `--stages test` (full) | ~20 min |
| `--quick` end to end | a few minutes, SVM-dominated |

---

## 3 · What to say while it runs

Keep it to three sentences and let the terminal scroll:

> "This is the native Rust implementation: same frozen folds as Python, same resample seeds, same
> metric definitions. Each line is one model on one fold, and the timings are per fit — the SVM is
> the expensive one. What it writes is the same tidy CSV schema Python writes, so I can join them
> cell by cell."

If he asks whether the numbers match, quote the parity result rather than waiting for this run to
finish: kNN exact to 1e-16, SVM exact on F1, logistic out by 0.0003, tree by about a thousandth,
random forest by 0.011 on fold means.

### Expected values, so you can sanity-check live

Python CV F1 means for the `none` scenario — the block your first Rust lines correspond to:

| model | Python CV mean F1 |
|---|---|
| Random Forest | 0.6813 |
| Decision Tree | 0.6615 |
| Logistic Regression | 0.6615 |
| SVM (RBF) | 0.6612 |
| KNN (k=5) | 0.6062 |

If a live Rust fold mean lands within ~0.01 F1 of the Python mean for the same model, that is
parity behaving as documented; say so out loud, because it is the claim the whole project rests on.

---

## 4 · Fallbacks

- **Rust build is cold and slow to compile** — pre-build before the session: `cargo build --release`,
  then run `./target/release/csc869` directly so the demo never waits on rustc.
- **Run is too slow for the slot** — Ctrl-C is safe: outputs are written per stage, not per fit.
  Restart with `--quick`.
- **Something fails** — fall back to the precomputed evidence, which is committed:
  `results/cv_summary.csv`, `results/test_metrics.csv`, `results_quick/rust/parity_means.csv`.
  Open the CSV and read the row you need; the demo becomes "here is the output it produced".
- **Nix shell won't start on the presentation machine** — the committed figures and CSVs still
  support the talk; `python -m csc869.run --list-stages` needs the venv, so skip the Python half
  and demo only the Rust binary.
