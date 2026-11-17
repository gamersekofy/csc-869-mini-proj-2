mod data;
mod metrics;
mod models;
mod rng;
use data::{Audit, Frozen, Preprocessor, Row};
use serde::Deserialize;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
struct Config {
    seed: u64,
    missing: Missing,
    cv: Cv,
    models: Vec<String>,
    scenarios: Vec<String>,
    test: Test,
    quick: Quick,
}
#[derive(Deserialize)]
struct Missing {
    categorical: String,
    numeric: String,
    #[serde(default)]
    capital_flags: bool,
}
#[derive(Deserialize)]
struct Cv {
    inner_splits: usize,
    dt_depths: Vec<Option<usize>>,
    rf_trees: usize,
    knn_neighbors: usize,
    svm_probability: bool,
}
#[derive(Deserialize)]
struct Test {
    svm_probability: bool,
}
#[derive(Deserialize)]
struct Quick {
    folds: usize,
    rf_trees: usize,
    dt_depths: Vec<Option<usize>>,
    max_rows: usize,
}
fn save(
    out: &Path,
    name: &str,
    header: &[&str],
    mut rows: Vec<Vec<String>>,
    keys: &[usize],
) -> Result<(), Box<dyn Error>> {
    rows.sort_by(|a, b| {
        for &k in keys {
            let o = if header[k] == "fold" {
                a[k].parse::<usize>().unwrap().cmp(&b[k].parse().unwrap())
            } else {
                a[k].cmp(&b[k])
            };
            if !o.is_eq() {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    let mut w = csv::Writer::from_path(out.join(name))?;
    w.write_record(header)?;
    for r in rows {
        w.write_record(r)?;
    }
    w.flush()?;
    Ok(())
}
fn yy(rows: &[Row], idx: &[usize]) -> Vec<u8> {
    idx.iter().map(|&i| rows[i].y).collect()
}
fn count(y: &[u8]) -> usize {
    y.iter().map(|&v| v as usize).sum()
}
fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}
fn run() -> Result<(), Box<dyn Error>> {
    let (mut data_dir, mut fold_path, mut out, mut config) = (
        PathBuf::from("data"),
        PathBuf::from("data/folds.json"),
        PathBuf::from("results/rust"),
        PathBuf::from("components/mp2-py/configs/main.yaml"),
    );
    let mut quick = false;
    let mut audit_only = false;
    let mut jobs = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data_dir = args.next().ok_or("--data needs a directory")?.into(),
            "--folds" => fold_path = args.next().ok_or("--folds needs a path")?.into(),
            "--out" => out = args.next().ok_or("--out needs a directory")?.into(),
            "--config" => config = args.next().ok_or("--config needs a path")?.into(),
            "--quick" => quick = true,
            "--audit-only" => audit_only = true,
            "--jobs" => jobs = args.next().ok_or("--jobs needs a number")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "csc869 --data data/ --folds data/folds.json --out results/rust/ [--quick] [--jobs N] [--config PATH] [--audit-only]\nQuick: first 8000 rows, first two frozen outer folds; no fold regeneration."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if jobs == 0 {
        return Err("--jobs must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()?;
    libsvm_rs::set_quiet(true);
    let cfg: Config = serde_yaml::from_str(&fs::read_to_string(&config)?)?;
    if !["unknown", "mode"].contains(&cfg.missing.categorical.as_str())
        || !["median", "indicator"].contains(&cfg.missing.numeric.as_str())
    {
        return Err("invalid missing-value strategy".into());
    }
    if cfg.cv.inner_splits < 2 || cfg.cv.rf_trees == 0 {
        return Err("invalid model configuration".into());
    }
    for m in &cfg.models {
        if ![
            "decision_tree",
            "random_forest",
            "logistic_regression",
            "svm",
            "knn",
        ]
        .contains(&m.as_str())
        {
            return Err(format!("unknown model {m}").into());
        }
    }
    let started = Instant::now();
    let adult = if data_dir.join("adult.data").exists() {
        data_dir.clone()
    } else {
        data_dir.join("adult")
    };
    let mut train = data::load(&adult.join("adult.data"))?;
    let test = data::load(&adult.join("adult.test"))?;
    let frozen: Frozen = serde_json::from_str(&fs::read_to_string(&fold_path)?)?;
    data::validate(&frozen, train.len()).map_err(|e| format!("invalid frozen folds: {e}"))?;
    if frozen.seed != cfg.seed {
        return Err("config seed differs from frozen fold seed".into());
    }
    if quick {
        train.truncate(cfg.quick.max_rows);
    }
    fs::create_dir_all(&out)?;
    let n = train.len();
    let folds: Vec<_> = frozen
        .folds
        .iter()
        .take(if quick {
            cfg.quick.folds
        } else {
            frozen.folds.len()
        })
        .map(|f| {
            (
                f.fold,
                f.train
                    .iter()
                    .copied()
                    .filter(|&i| i < n)
                    .collect::<Vec<_>>(),
                f.val.iter().copied().filter(|&i| i < n).collect::<Vec<_>>(),
            )
        })
        .collect();
    let depths = if quick {
        &cfg.quick.dt_depths
    } else {
        &cfg.cv.dt_depths
    };
    let trees = if quick {
        cfg.quick.rf_trees
    } else {
        cfg.cv.rf_trees
    };
    let (mut tidy, mut runs, mut sampling, mut tuning, mut audit) =
        (vec![], vec![], vec![], vec![], vec![]);
    for scenario in &cfg.scenarios {
        let si = ["none", "upsampling", "downsampling"]
            .iter()
            .position(|s| s == scenario)
            .ok_or("invalid scenario")?;
        for (fi, tr, va) in &folds {
            if tr.is_empty() || va.is_empty() {
                return Err("quick projection yielded an empty frozen split".into());
            }
            let seed = cfg.seed + *fi as u64 * 17 + si as u64 * 101;
            let y = yy(&train, tr);
            if count(&y) == 0 || count(&y) == y.len() {
                return Err("single-class frozen training split".into());
            }
            let relative = rng::resample(&y, scenario, seed);
            let sampled: Vec<_> = relative.iter().map(|&i| tr[i]).collect();
            let yr = yy(&train, &sampled);
            let yv = yy(&train, va);
            let pre = Preprocessor::fit(
                &train,
                &sampled,
                &cfg.missing.categorical,
                &cfg.missing.numeric,
                cfg.missing.capital_flags,
            );
            let x = pre.transform(&train, &sampled);
            let z = pre.transform(&train, va);
            audit.push(Audit {
                scenario: scenario.clone(),
                fold: *fi,
                seed,
                train: tr.clone(),
                val: va.clone(),
                sampled: sampled.clone(),
                feature_names: pre.names.clone(),
                train_head: x.iter().take(3).cloned().collect(),
                val_head: z.iter().take(3).cloned().collect(),
            });
            let p0 = count(&y);
            let p1 = count(&yr);
            sampling.push(vec![
                "rust".into(),
                scenario.clone(),
                fi.to_string(),
                tr.len().to_string(),
                p0.to_string(),
                (y.len() - p0).to_string(),
                (p0 as f64 / y.len() as f64).to_string(),
                yr.len().to_string(),
                p1.to_string(),
                (yr.len() - p1).to_string(),
                (p1 as f64 / yr.len() as f64).to_string(),
            ]);
            if audit_only {
                continue;
            }
            for (mi, name) in cfg.models.iter().enumerate() {
                let t = Instant::now();
                let (pred, score, info) = models::fit_predict(
                    name,
                    &x,
                    &yr,
                    &z,
                    seed + mi as u64 * 1009,
                    depths,
                    cfg.cv.inner_splits,
                    trees,
                    cfg.cv.knn_neighbors,
                    !quick && cfg.cv.svm_probability,
                );
                let seconds = t.elapsed().as_secs_f64();
                let m = metrics::compute(&yv, &pred, &score);
                for (j, metric) in metrics::NAMES.iter().enumerate() {
                    tidy.push(vec![
                        "rust".into(),
                        name.clone(),
                        scenario.clone(),
                        fi.to_string(),
                        metric.to_string(),
                        m[j].to_string(),
                    ]);
                }
                runs.push(vec![
                    "rust".into(),
                    name.clone(),
                    scenario.clone(),
                    fi.to_string(),
                    tr.len().to_string(),
                    yr.len().to_string(),
                    p1.to_string(),
                    (yr.len() - p1).to_string(),
                    va.len().to_string(),
                    count(&yv).to_string(),
                    (yv.len() - count(&yv)).to_string(),
                    pre.names.len().to_string(),
                    info.depth.map_or(String::new(), |d| d.to_string()),
                    seconds.to_string(),
                ]);
                for (d, mean, sd) in info.tuning {
                    tuning.push(vec![
                        "rust".into(),
                        scenario.clone(),
                        fi.to_string(),
                        d.map_or("None".into(), |d| d.to_string()),
                        mean.to_string(),
                        sd.to_string(),
                        (d == info.depth).to_string(),
                    ]);
                }
                eprintln!(
                    "[rust cv] {scenario}/{fi}/{name}: F1={:.5}, {:.2}s",
                    m[3], seconds
                );
            }
        }
    }
    fs::write(out.join("input_audit.json"), serde_json::to_vec(&audit)?)?;
    save(
        &out,
        "sampling_counts.csv",
        &[
            "lang",
            "scenario",
            "fold",
            "n_before",
            "pos_before",
            "neg_before",
            "positive_rate_before",
            "n_after",
            "pos_after",
            "neg_after",
            "positive_rate_after",
        ],
        sampling,
        &[1, 2],
    )?;
    if audit_only {
        return Ok(());
    }
    save(
        &out,
        "cv_tidy.csv",
        &["lang", "model", "scenario", "fold", "metric", "value"],
        tidy.clone(),
        &[2, 3, 1, 4],
    )?;
    save(
        &out,
        "cv_runs.csv",
        &[
            "lang",
            "model",
            "scenario",
            "fold",
            "n_train_original",
            "n_train_resampled",
            "n_train_positive",
            "n_train_negative",
            "n_val",
            "n_val_positive",
            "n_val_negative",
            "n_features",
            "best_depth",
            "fit_seconds",
        ],
        runs,
        &[2, 3, 1],
    )?;
    save(
        &out,
        "dt_tuning.csv",
        &[
            "lang",
            "scenario",
            "fold",
            "max_depth",
            "mean_f1",
            "std_f1",
            "selected",
        ],
        tuning,
        &[1, 2],
    )?;
    let mut groups = std::collections::BTreeMap::<(String, String, String), Vec<f64>>::new();
    for r in tidy {
        groups
            .entry((r[1].clone(), r[2].clone(), r[4].clone()))
            .or_default()
            .push(r[5].parse()?);
    }
    let mut summary = vec![];
    for ((model, scenario, metric), v) in groups {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = if v.len() > 1 {
            (v.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        } else {
            0.
        };
        summary.push(vec![
            "rust".into(),
            model,
            scenario,
            metric,
            mean.to_string(),
            sd.to_string(),
            v.len().to_string(),
        ]);
    }
    save(
        &out,
        "cv_summary.csv",
        &[
            "lang", "model", "scenario", "metric", "mean", "std", "n_folds",
        ],
        summary,
        &[1, 2, 3],
    )?;
    let (mut test_metrics, mut test_runs, mut test_predictions) = (vec![], vec![], vec![]);
    let train_ids: Vec<_> = (0..train.len()).collect();
    let test_ids: Vec<_> = (0..test.len()).collect();
    let y = yy(&train, &train_ids);
    let yt = yy(&test, &test_ids);
    for (si, scenario) in cfg.scenarios.iter().enumerate() {
        let seed = cfg.seed + si as u64 * 101;
        let sampled = rng::resample(&y, scenario, seed);
        let yr = yy(&train, &sampled);
        let pre = Preprocessor::fit(
            &train,
            &sampled,
            &cfg.missing.categorical,
            &cfg.missing.numeric,
            cfg.missing.capital_flags,
        );
        let x = pre.transform(&train, &sampled);
        let z = pre.transform(&test, &test_ids);
        for (mi, name) in cfg.models.iter().enumerate() {
            let t = Instant::now();
            let (pred, score, info) = models::fit_predict(
                name,
                &x,
                &yr,
                &z,
                seed + mi as u64 * 1009,
                depths,
                cfg.cv.inner_splits,
                trees,
                cfg.cv.knn_neighbors,
                !quick && cfg.test.svm_probability,
            );
            let seconds = t.elapsed().as_secs_f64();
            let m = metrics::compute(&yt, &pred, &score);
            for (j, metric) in metrics::NAMES.iter().enumerate() {
                let mut r = strings(&["rust", name, scenario, metric]);
                r.push(m[j].to_string());
                test_metrics.push(r);
            }
            let pos = count(&yr);
            test_runs.push(vec![
                "rust".into(),
                name.clone(),
                scenario.clone(),
                yr.len().to_string(),
                pos.to_string(),
                (yr.len() - pos).to_string(),
                yt.len().to_string(),
                pre.names.len().to_string(),
                info.depth.map_or(String::new(), |d| d.to_string()),
                seconds.to_string(),
            ]);
            for i in 0..test.len() {
                test_predictions.push(vec![
                    "rust".into(),
                    name.clone(),
                    scenario.clone(),
                    i.to_string(),
                    yt[i].to_string(),
                    pred[i].to_string(),
                    format!("{:.6}", score[i]),
                ]);
            }
            eprintln!(
                "[rust test] {scenario}/{name}: F1={:.5}, {:.2}s",
                m[3], seconds
            );
        }
    }
    save(
        &out,
        "test_metrics.csv",
        &["lang", "model", "scenario", "metric", "value"],
        test_metrics,
        &[2, 1, 3],
    )?;
    save(
        &out,
        "test_runs.csv",
        &[
            "lang",
            "model",
            "scenario",
            "n_train",
            "n_train_positive",
            "n_train_negative",
            "n_test",
            "n_features",
            "best_depth",
            "fit_seconds",
        ],
        test_runs,
        &[2, 1],
    )?;
    save(
        &out,
        "test_predictions.csv",
        &[
            "lang", "model", "scenario", "row_id", "y_true", "y_pred", "y_score",
        ],
        test_predictions,
        &[2, 1],
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    fs::write(
        out.join("run_metadata.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"language":"rust","quick":quick,"protocol":if quick{"first-8000/first-two-frozen-folds"}else{"full-frozen-folds"},"folds_source":fold_path,"train_rows":train.len(),"test_rows":test.len(),"rf_trees":trees,"dt_depths":depths,"test_svm_platt":!quick&&cfg.test.svm_probability,"elapsed_seconds":elapsed,"outer_folds_regenerated":false,"implementation":"native Rust CART, forest, L-BFGS logistic, kNN, libsvm-rs SVM; NumPy-compatible sampling"}),
        )?,
    )?;
    println!("RUST_SECONDS={elapsed:.6}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("csc869: {e}");
        std::process::exit(1);
    }
}
