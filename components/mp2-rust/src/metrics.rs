pub const NAMES: [&str; 6] = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"];
pub fn compute(y: &[u8], pred: &[u8], score: &[f64]) -> [f64; 6] {
    assert_eq!(y.len(), pred.len());
    assert_eq!(y.len(), score.len());
    let (mut tp, mut tn, mut fp, mut fn_) = (0., 0., 0., 0.);
    for (&a, &b) in y.iter().zip(pred) {
        match (a, b) {
            (1, 1) => tp += 1.,
            (0, 0) => tn += 1.,
            (0, 1) => fp += 1.,
            _ => fn_ += 1.,
        }
    }
    let div = |a: f64, b: f64| if b == 0. { 0. } else { a / b };
    let precision = div(tp, tp + fp);
    let recall = div(tp, tp + fn_);
    let f1 = div(2. * precision * recall, precision + recall);
    let pos = y.iter().map(|&v| v as usize).sum::<usize>();
    let neg = y.len() - pos;
    let mut order: Vec<_> = (0..y.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    let (mut i, mut seen_pos, mut seen_neg) = (0, 0usize, 0usize);
    let (mut ap, mut u) = (0., 0.);
    while i < order.len() {
        let mut j = i;
        let (mut gp, mut gn) = (0, 0);
        while j < order.len() && score[order[j]] == score[order[i]] {
            if y[order[j]] == 1 {
                gp += 1
            } else {
                gn += 1
            }
            j += 1;
        }
        seen_pos += gp;
        seen_neg += gn;
        if pos > 0 {
            ap += gp as f64 / pos as f64 * seen_pos as f64 / (seen_pos + seen_neg) as f64;
        }
        u += gp as f64 * ((neg - seen_neg) as f64 + 0.5 * gn as f64);
        i = j;
    }
    [
        div(tp + tn, y.len() as f64),
        precision,
        recall,
        f1,
        if pos == 0 || neg == 0 {
            f64::NAN
        } else {
            u / (pos * neg) as f64
        },
        ap,
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties() {
        let m = compute(&[0, 1, 0, 1], &[0, 0, 0, 0], &[0.5; 4]);
        assert_eq!(m[0], 0.5);
        assert_eq!(m[3], 0.);
        assert_eq!(m[4], 0.5);
        assert_eq!(m[5], 0.5);
    }
    #[test]
    fn perfect() {
        let m = compute(&[0, 1, 0, 1], &[0, 1, 0, 1], &[0., 1., 0., 1.]);
        assert_eq!(m, [1.; 6]);
    }
    #[test]
    fn no_positives() {
        let m = compute(&[0, 0], &[0, 0], &[0., 1.]);
        assert!(m[4].is_nan());
        assert_eq!(m[5], 0.);
    }
}
