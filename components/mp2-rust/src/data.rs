use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    path::Path,
};
pub const CATS: [&str; 8] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
];
pub const NUMS: [&str; 6] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
];
#[derive(Clone, Debug)]
pub struct Row {
    pub cats: Vec<Option<String>>,
    pub nums: [f64; 6],
    pub y: u8,
}
pub type Matrix = Vec<Vec<f64>>;
pub fn load(path: &Path) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .comment(Some(b'|'))
        .from_path(path)?;
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        if r.len() != 15 {
            return Err(format!("{}: expected 15 columns, got {}", path.display(), r.len()).into());
        }
        let cats = [1, 3, 5, 6, 7, 8, 9, 13]
            .iter()
            .map(|&i| match &r[i] {
                "?" | "" | "nan" | "NaN" | "None" => None,
                s => Some(s.to_string()),
            })
            .collect();
        let mut nums = [0.; 6];
        for (j, i) in [0, 2, 4, 10, 11, 12].iter().enumerate() {
            nums[j] = r[*i].parse()?;
        }
        let y = match r[14].trim_end_matches('.') {
            ">50K" => 1,
            "<=50K" => 0,
            s => return Err(format!("unknown target {s}").into()),
        };
        rows.push(Row { cats, nums, y });
    }
    Ok(rows)
}
#[derive(Deserialize)]
pub struct Frozen {
    pub seed: u64,
    pub n_rows: usize,
    pub folds: Vec<Fold>,
}
#[derive(Deserialize)]
pub struct Fold {
    pub fold: usize,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
}
pub fn validate(f: &Frozen, n: usize) -> Result<(), String> {
    if f.n_rows != n {
        return Err(format!(
            "fold n_rows={} differs from loaded rows={n}",
            f.n_rows
        ));
    }
    if f.folds.len() < 2 {
        return Err("need at least two frozen folds".into());
    }
    let mut seen = vec![0; n];
    for (k, fold) in f.folds.iter().enumerate() {
        if fold.fold != k {
            return Err("nonconsecutive frozen fold IDs".into());
        }
        let mut flag = vec![0; n];
        for &i in &fold.train {
            if i >= n || flag[i] != 0 {
                return Err("invalid/duplicate frozen train index".into());
            }
            flag[i] = 1;
        }
        for &i in &fold.val {
            if i >= n || flag[i] != 0 {
                return Err("invalid/overlapping frozen validation index".into());
            }
            flag[i] = 2;
            seen[i] += 1;
        }
        if flag.contains(&0) {
            return Err("frozen fold is not a full partition".into());
        }
    }
    if seen.iter().any(|&x| x != 1) {
        return Err("frozen validation folds do not partition source rows".into());
    }
    Ok(())
}
#[derive(Clone, Debug)]
pub struct Preprocessor {
    modes: Vec<String>,
    fills: Vec<String>,
    categories: Vec<Vec<String>>,
    medians: [f64; 6],
    means: [f64; 6],
    stds: [f64; 6],
    indicators: [bool; 6],
    flags: bool,
    pub names: Vec<String>,
}
impl Preprocessor {
    pub fn fit(rows: &[Row], idx: &[usize], categorical: &str, numeric: &str, flags: bool) -> Self {
        let mut p = Self {
            modes: vec![],
            fills: vec![],
            categories: vec![],
            medians: [0.; 6],
            means: [0.; 6],
            stds: [1.; 6],
            indicators: [false; 6],
            flags,
            names: vec![],
        };
        for (c, name) in CATS.iter().enumerate() {
            let mut counts = BTreeMap::<String, usize>::new();
            for &i in idx {
                if let Some(s) = &rows[i].cats[c] {
                    *counts.entry(s.clone()).or_default() += 1;
                }
            }
            let mut mode = "Unknown".to_string();
            let mut best = 0;
            for (s, n) in counts {
                if n > best {
                    mode = s;
                    best = n;
                }
            }
            let fill = if categorical == "unknown" {
                "Unknown".to_string()
            } else {
                mode.clone()
            };
            let cats: BTreeSet<_> = idx
                .iter()
                .map(|&i| rows[i].cats[c].clone().unwrap_or_else(|| fill.clone()))
                .collect();
            let cats: Vec<_> = cats.into_iter().collect();
            for s in &cats {
                p.names.push(format!("{name}={s}"));
            }
            p.categories.push(cats);
            p.modes.push(mode);
            p.fills.push(fill);
        }
        for (c, name) in NUMS.iter().enumerate() {
            let mut vals: Vec<_> = idx
                .iter()
                .map(|&i| rows[i].nums[c])
                .filter(|x| x.is_finite())
                .collect();
            p.indicators[c] = numeric == "indicator" && vals.len() != idx.len();
            vals.sort_by(f64::total_cmp);
            let med = if vals.is_empty() {
                0.
            } else if vals.len() % 2 == 1 {
                vals[vals.len() / 2]
            } else {
                (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.
            };
            p.medians[c] = med;
            let filled: Vec<_> = idx
                .iter()
                .map(|&i| {
                    let v = rows[i].nums[c];
                    if v.is_finite() { v } else { med }
                })
                .collect();
            let mean = filled.iter().sum::<f64>() / filled.len() as f64;
            let std = (filled.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / filled.len() as f64)
                .sqrt();
            p.means[c] = mean;
            p.stds[c] = if std > 1e-12 { std } else { 1. };
            if p.indicators[c] {
                p.names.push(format!("{name}_missing"));
            }
            p.names.push(name.to_string());
        }
        if flags {
            p.names
                .extend(["capital-gain>0".into(), "capital-loss>0".into()]);
        }
        p
    }
    pub fn transform(&self, rows: &[Row], idx: &[usize]) -> Matrix {
        idx.iter()
            .map(|&i| {
                let row = &rows[i];
                let mut x = Vec::with_capacity(self.names.len());
                for c in 0..8 {
                    let s = row.cats[c].as_ref().unwrap_or(&self.fills[c]);
                    let s = if self.categories[c].binary_search(s).is_ok() {
                        s
                    } else {
                        &self.modes[c]
                    };
                    x.extend(
                        self.categories[c]
                            .iter()
                            .map(|cat| if cat == s { 1. } else { 0. }),
                    );
                }
                for c in 0..6 {
                    let v = row.nums[c];
                    if self.indicators[c] {
                        x.push(if v.is_nan() { 1. } else { 0. });
                    }
                    x.push(
                        (if v.is_finite() { v } else { self.medians[c] } - self.means[c])
                            / self.stds[c],
                    );
                }
                if self.flags {
                    x.push(if row.nums[3] > 0. { 1. } else { 0. });
                    x.push(if row.nums[4] > 0. { 1. } else { 0. });
                }
                x
            })
            .collect()
    }
}
#[derive(Serialize)]
pub struct Audit {
    pub scenario: String,
    pub fold: usize,
    pub seed: u64,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    pub sampled: Vec<usize>,
    pub feature_names: Vec<String>,
    pub train_head: Matrix,
    pub val_head: Matrix,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_and_indicators() {
        let rows = vec![
            Row {
                cats: vec![Some("a".into()); 8],
                nums: [1.; 6],
                y: 0,
            },
            Row {
                cats: vec![None; 8],
                nums: [f64::NAN; 6],
                y: 1,
            },
            Row {
                cats: vec![Some("unseen".into()); 8],
                nums: [3.; 6],
                y: 0,
            },
        ];
        let p = Preprocessor::fit(&rows, &[0, 1], "unknown", "indicator", true);
        let x = p.transform(&rows, &[2]);
        assert_eq!(p.names.len(), 30);
        assert_eq!(x[0][0], 0.);
        assert_eq!(x[0][1], 1.);
        assert_eq!(x[0][16], 0.);
    }
}
