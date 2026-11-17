use crate::{
    data::Matrix,
    metrics,
    rng::{Rng, stratified},
};
use libsvm_rs::{
    KernelType, SvmNode, SvmParameter, SvmProblem, check_parameter, predict::predict_values,
    train::svm_train,
};
use rayon::prelude::*;
#[derive(Clone)]
struct Node {
    prob: f64,
    split: Option<(usize, f64, Box<Node>, Box<Node>)>,
}
impl Node {
    fn score(&self, x: &[f64]) -> f64 {
        match &self.split {
            None => self.prob,
            Some((j, t, l, r)) => {
                if (x[*j] as f32 as f64) <= *t {
                    l.score(x)
                } else {
                    r.score(x)
                }
            }
        }
    }
}
fn tree(
    x: &Matrix,
    y: &[u8],
    idx: &[usize],
    depth: usize,
    max_depth: usize,
    mtry: usize,
    rng: &mut Rng,
) -> Node {
    let n = idx.len();
    let total = idx.iter().map(|&i| y[i] as usize).sum::<usize>();
    let mut node = Node {
        prob: total as f64 / n as f64,
        split: None,
    };
    if total == 0 || total == n || depth >= max_depth || n < 2 {
        return node;
    }
    let p = x[0].len();
    let features = rng.choice(p, p, false);
    let mut visited = 0;
    let (mut best, mut feature, mut threshold) = (-f64::INFINITY, 0, 0.);
    for f in features {
        let mut vals: Vec<_> = idx.iter().map(|&i| (x[i][f] as f32, y[i])).collect();
        vals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        if vals[n - 1].0 as f64 <= vals[0].0 as f64 + 1e-7 {
            continue;
        }
        visited += 1;
        let mut left = 0usize;
        for i in 0..n - 1 {
            left += vals[i].1 as usize;
            if vals[i + 1].0 as f64 <= vals[i].0 as f64 + 1e-7 {
                continue;
            }
            let right = total - left;
            let proxy =
                (left * left) as f64 / (i + 1) as f64 + (right * right) as f64 / (n - i - 1) as f64;
            if proxy > best + 1e-12 {
                best = proxy;
                feature = f;
                threshold = (vals[i].0 as f64 + vals[i + 1].0 as f64) / 2.;
            }
        }
        if visited >= mtry {
            break;
        }
    }
    let parent = (total * total) as f64 / n as f64;
    if !best.is_finite() || best + 1e-12 < parent {
        return node;
    }
    let (l, r): (Vec<_>, Vec<_>) = idx
        .iter()
        .copied()
        .partition(|&i| (x[i][feature] as f32 as f64) <= threshold);
    if l.is_empty() || r.is_empty() {
        return node;
    }
    node.split = Some((
        feature,
        threshold,
        Box::new(tree(x, y, &l, depth + 1, max_depth, mtry, rng)),
        Box::new(tree(x, y, &r, depth + 1, max_depth, mtry, rng)),
    ));
    node
}
fn fit_tree(
    x: &Matrix,
    y: &[u8],
    idx: &[usize],
    depth: Option<usize>,
    mtry: usize,
    seed: u64,
) -> Node {
    tree(
        x,
        y,
        idx,
        0,
        depth.unwrap_or(usize::MAX),
        mtry,
        &mut Rng::new(seed),
    )
}
fn select(x: &Matrix, ids: &[usize]) -> Matrix {
    ids.iter().map(|&i| x[i].clone()).collect()
}
fn labels(y: &[u8], ids: &[usize]) -> Vec<u8> {
    ids.iter().map(|&i| y[i]).collect()
}
fn hard(score: &[f64]) -> Vec<u8> {
    score.iter().map(|&s| u8::from(s > 0.5)).collect()
}
fn standardize(x: &Matrix, z: &Matrix) -> (Matrix, Matrix) {
    let n = x.len() as f64;
    let p = x[0].len();
    let means: Vec<_> = (0..p)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n)
        .collect();
    let scales: Vec<_> = (0..p)
        .map(|j| {
            let sd = (x.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / n).sqrt();
            if sd > 1e-12 { sd } else { 1. }
        })
        .collect();
    let trans = |a: &Matrix| {
        a.iter()
            .map(|r| (0..p).map(|j| (r[j] - means[j]) / scales[j]).collect())
            .collect()
    };
    (trans(x), trans(z))
}
fn knn(x: &Matrix, y: &[u8], z: &Matrix, k: usize) -> Vec<f64> {
    z.par_iter()
        .map(|q| {
            let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
            let mut worst = f64::INFINITY;
            for (i, r) in x.iter().enumerate() {
                let mut d = 0.;
                for (a, b) in r.iter().zip(q) {
                    d += (a - b) * (a - b);
                    if d > worst {
                        break;
                    }
                }
                if best.len() < k || d < worst {
                    let at = best.partition_point(|&(v, ix)| v < d || (v == d && ix < i));
                    best.insert(at, (d, i));
                    if best.len() > k {
                        best.pop();
                    }
                    if best.len() == k {
                        worst = best[k - 1].0;
                    }
                }
            }
            best.iter().map(|&(_, i)| y[i] as f64).sum::<f64>() / k as f64
        })
        .collect()
}
fn sigmoid(z: f64) -> f64 {
    if z >= 0. {
        1. / (1. + (-z).exp())
    } else {
        let e = z.exp();
        e / (1. + e)
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn objective(x: &Matrix, y: &[u8], w: &[f64]) -> (f64, Vec<f64>) {
    let p = x[0].len();
    let n = x.len() as f64;
    let mut g = vec![0.; p + 1];
    let mut loss = 0.;
    for (r, &yi) in x.iter().zip(y) {
        let z = dot(r, &w[..p]) + w[p];
        loss += z.max(0.) + (-z.abs()).exp().ln_1p() - yi as f64 * z;
        let d = sigmoid(z) - yi as f64;
        for j in 0..p {
            g[j] += d * r[j];
        }
        g[p] += d;
    }
    for j in 0..p {
        loss += 0.5 * w[j] * w[j];
        g[j] = (g[j] + w[j]) / n;
    }
    g[p] /= n;
    (loss / n, g)
}
fn logistic(x: &Matrix, y: &[u8], z: &Matrix) -> Vec<f64> {
    let p = x[0].len();
    let mut w = vec![0.; p + 1];
    let (mut f, mut g) = objective(x, y, &w);
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    for _ in 0..1000 {
        if g.iter().map(|v| v.abs()).fold(0., f64::max) < 1e-4 {
            break;
        }
        let mut q = g.clone();
        let mut alphas = vec![];
        for (s, v, rho) in history.iter().rev() {
            let a = rho * dot(s, &q);
            for (j, qj) in q.iter_mut().enumerate() {
                *qj -= a * v[j];
            }
            alphas.push(a);
        }
        if let Some((s, v, _)) = history.last() {
            let scale = dot(s, v) / dot(v, v);
            for qj in &mut q {
                *qj *= scale;
            }
        }
        for ((s, v, rho), a) in history.iter().zip(alphas.into_iter().rev()) {
            let b = rho * dot(v, &q);
            for j in 0..q.len() {
                q[j] += s[j] * (a - b);
            }
        }
        let mut d: Vec<_> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&g, &d);
        if slope >= 0. {
            d = g.iter().map(|v| -v).collect();
            slope = -dot(&g, &g);
        }
        let mut step = 1.;
        let mut accepted = None;
        for _ in 0..30 {
            let nw: Vec<_> = w.iter().zip(&d).map(|(a, b)| a + step * b).collect();
            let (nf, ng) = objective(x, y, &nw);
            if nf <= f + 1e-4 * step * slope {
                accepted = Some((nw, nf, ng));
                break;
            }
            step *= 0.5;
        }
        let Some((nw, nf, ng)) = accepted else {
            break;
        };
        let s: Vec<_> = nw.iter().zip(&w).map(|(a, b)| a - b).collect();
        let v: Vec<_> = ng.iter().zip(&g).map(|(a, b)| a - b).collect();
        let ys = dot(&s, &v);
        if ys > 1e-12 {
            if history.len() == 10 {
                history.remove(0);
            }
            history.push((s, v, 1. / ys));
        }
        w = nw;
        g = ng;
        f = nf;
    }
    z.par_iter()
        .map(|r| sigmoid(dot(r, &w[..p]) + w[p]))
        .collect()
}
fn sparse(x: &Matrix) -> Vec<Vec<SvmNode>> {
    x.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .filter_map(|(i, &v)| {
                    if v == 0. {
                        None
                    } else {
                        Some(SvmNode {
                            index: (i + 1) as i32,
                            value: v,
                        })
                    }
                })
                .collect()
        })
        .collect()
}
fn svm(x: &Matrix, y: &[u8], z: &Matrix) -> (Vec<u8>, Vec<f64>) {
    let n = (x.len() * x[0].len()) as f64;
    let mean = x.iter().flatten().sum::<f64>() / n;
    let var = x.iter().flatten().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let gamma = if var == 0. {
        1.
    } else {
        1. / (x[0].len() as f64 * var)
    };
    let param = SvmParameter {
        kernel_type: KernelType::Rbf,
        c: 1.,
        gamma,
        cache_size: 2000.,
        eps: 0.001,
        probability: false,
        ..Default::default()
    };
    let problem = SvmProblem {
        labels: y.iter().map(|&v| v as f64).collect(),
        instances: sparse(x),
    };
    check_parameter(&problem, &param).expect("valid SVM parameters");
    let model = svm_train(&problem, &param);
    let nodes = sparse(z);
    let out: Vec<_> = nodes
        .par_iter()
        .map(|r| {
            let mut values = [0.];
            let label = predict_values(&model, r, &mut values);
            (
                u8::from(label == 1.),
                if model.label[0] == 1 {
                    values[0]
                } else {
                    -values[0]
                },
            )
        })
        .collect();
    out.into_iter().unzip()
}
fn calibration_folds(y: &[u8]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let first = y[0];
    let mut assigned = vec![0; y.len()];
    let mut offset = 0;
    for cls in [first, 1 - first] {
        let ids: Vec<_> = (0..y.len()).filter(|&i| y[i] == cls).collect();
        let mut counts = [0; 3];
        for j in 0..ids.len() {
            counts[(offset + j) % 3] += 1;
        }
        let mut pos = 0;
        for (f, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                assigned[ids[pos]] = f;
                pos += 1;
            }
        }
        offset += ids.len();
    }
    (0..3)
        .map(|f| {
            (
                (0..y.len()).filter(|&i| assigned[i] != f).collect(),
                (0..y.len()).filter(|&i| assigned[i] == f).collect(),
            )
        })
        .collect()
}
fn platt(scores: &[f64], y: &[u8]) -> (f64, f64) {
    let pos = y.iter().map(|&v| v as usize).sum::<usize>() as f64;
    let neg = y.len() as f64 - pos;
    let targets: Vec<_> = y
        .iter()
        .map(|&v| {
            if v == 1 {
                (pos + 1.) / (pos + 2.)
            } else {
                1. / (neg + 2.)
            }
        })
        .collect();
    let (mut a, mut b) = (0., ((pos + 1.) / (neg + 1.)).ln());
    let loss = |a: f64, b: f64| {
        scores
            .iter()
            .zip(&targets)
            .map(|(&s, &t)| {
                let z = a * s + b;
                z.max(0.) + (-z.abs()).exp().ln_1p() - t * z
            })
            .sum::<f64>()
    };
    let mut value = loss(a, b);
    for _ in 0..100 {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0., 0., 1e-12, 0., 1e-12);
        for (&s, &t) in scores.iter().zip(&targets) {
            let p = sigmoid(a * s + b);
            let v = p * (1. - p);
            ga += (p - t) * s;
            gb += p - t;
            haa += v * s * s;
            hab += v * s;
            hbb += v;
        }
        if ga.abs().max(gb.abs()) < 1e-6 {
            break;
        }
        let det = haa * hbb - hab * hab;
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        let mut step = 1.;
        let mut accepted = false;
        for _ in 0..30 {
            let nv = loss(a - step * da, b - step * db);
            if nv < value {
                a -= step * da;
                b -= step * db;
                value = nv;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    (a, b)
}
#[derive(Default)]
pub struct Info {
    pub depth: Option<usize>,
    pub tuning: Vec<(Option<usize>, f64, f64)>,
}
// Keep the reference runner's learner/config arguments explicit at this boundary.
#[allow(clippy::too_many_arguments)]
pub fn fit_predict(
    name: &str,
    x: &Matrix,
    y: &[u8],
    z: &Matrix,
    seed: u64,
    depths: &[Option<usize>],
    inner: usize,
    rf_trees: usize,
    k: usize,
    calibrate: bool,
) -> (Vec<u8>, Vec<f64>, Info) {
    let mut info = Info::default();
    match name {
        "decision_tree" => {
            let folds = stratified(y, inner, seed);
            let mut best = -1.;
            let mut depth = None;
            for &d in depths {
                let mut scores = vec![];
                for (tr, va) in &folds {
                    let model = fit_tree(x, y, tr, d, x[0].len(), seed);
                    let s: Vec<_> = va.iter().map(|&i| model.score(&x[i])).collect();
                    scores.push(metrics::compute(&labels(y, va), &hard(&s), &s)[3]);
                }
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let sd = (scores.iter().map(|v| (v - avg).powi(2)).sum::<f64>()
                    / scores.len() as f64)
                    .sqrt();
                info.tuning.push((d, avg, sd));
                if avg > best {
                    best = avg;
                    depth = d;
                }
            }
            info.depth = depth;
            let model = fit_tree(
                x,
                y,
                &(0..x.len()).collect::<Vec<_>>(),
                depth,
                x[0].len(),
                seed,
            );
            let score: Vec<_> = z.par_iter().map(|r| model.score(r)).collect();
            (hard(&score), score, info)
        }
        "random_forest" => {
            let trees: Vec<_> = (0..rf_trees)
                .into_par_iter()
                .map(|b| {
                    let seed = seed.wrapping_add((b as u64 + 1) * 104729);
                    let mut r = Rng::new(seed);
                    let ids = r.choice(x.len(), x.len(), true);
                    fit_tree(
                        x,
                        y,
                        &ids,
                        None,
                        (x[0].len() as f64).sqrt() as usize,
                        seed + 17,
                    )
                })
                .collect();
            let score: Vec<_> = z
                .par_iter()
                .map(|row| trees.iter().map(|t| t.score(row)).sum::<f64>() / trees.len() as f64)
                .collect();
            (hard(&score), score, info)
        }
        "knn" => {
            assert!(k > 0 && k <= x.len());
            let (a, b) = standardize(x, z);
            let score = knn(&a, y, &b, k);
            (hard(&score), score, info)
        }
        "logistic_regression" => {
            let (a, b) = standardize(x, z);
            let score = logistic(&a, y, &b);
            (hard(&score), score, info)
        }
        "svm" => {
            if calibrate {
                let mut oof = vec![0.; y.len()];
                for (tr, va) in calibration_folds(y) {
                    let (_, s) = svm(&select(x, &tr), &labels(y, &tr), &select(x, &va));
                    for (&i, v) in va.iter().zip(s) {
                        oof[i] = v;
                    }
                }
                let (a, b) = platt(&oof, y);
                let (_, s) = svm(x, y, z);
                let score: Vec<_> = s.iter().map(|&v| sigmoid(a * v + b)).collect();
                (hard(&score), score, info)
            } else {
                let (p, s) = svm(x, y, z);
                (p, s, info)
            }
        }
        _ => panic!("unknown model"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_gain_xor_split() {
        let x = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        let y = vec![0, 1, 1, 0];
        let t = fit_tree(&x, &y, &[0, 1, 2, 3], Some(2), 2, 42);
        assert_eq!(hard(&x.iter().map(|r| t.score(r)).collect::<Vec<_>>()), y);
    }
    #[test]
    fn calibrated_svm_scores() {
        libsvm_rs::set_quiet(true);
        let x: Matrix = (-6..=6)
            .filter(|&v| v != 0)
            .map(|v| vec![v as f64])
            .collect();
        let y: Vec<u8> = x.iter().map(|r| u8::from(r[0] > 0.)).collect();
        let (_, score, _) = fit_predict("svm", &x, &y, &x, 42, &[Some(2)], 3, 8, 1, true);
        assert!(score.iter().all(|v| v.is_finite() && *v > 0. && *v < 1.));
        assert!(score[0] < score[score.len() - 1]);
    }
    #[test]
    fn models_separate() {
        libsvm_rs::set_quiet(true);
        let x: Matrix = vec![
            vec![-3.],
            vec![-2.],
            vec![-1.],
            vec![1.],
            vec![2.],
            vec![3.],
        ];
        let y = vec![0, 0, 0, 1, 1, 1];
        for name in [
            "decision_tree",
            "random_forest",
            "logistic_regression",
            "svm",
            "knn",
        ] {
            let (p, s, _) = fit_predict(name, &x, &y, &x, 42, &[Some(2)], 3, 8, 1, false);
            assert_eq!(p, y, "{name}");
            assert!(s.iter().all(|v| v.is_finite()));
        }
    }
}
