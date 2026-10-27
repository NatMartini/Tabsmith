//! Binary classification metrics (definitions match scikit-learn's).

use rustpy_contracts::evaluation::{BinaryMetrics, ConfusionMatrix};
use rustpy_contracts::METRICS_VERSION;

/// Probability clipping used by log loss: machine epsilon of f64, as in
/// scikit-learn >= 1.5 (`eps="auto"` for float64 inputs).
pub const LOG_LOSS_EPS: f64 = f64::EPSILON;

/// Computes metrics for probabilities `p` (of the positive class) and labels `y`.
pub fn binary_metrics(p: &[f64], y: &[bool], threshold: f64) -> BinaryMetrics {
    assert_eq!(p.len(), y.len());
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    let mut log_loss = 0.0;
    let mut brier = 0.0;
    for (&pi, &yi) in p.iter().zip(y) {
        let pred = pi >= threshold;
        match (pred, yi) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
        let pc = pi.clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS);
        log_loss -= if yi { pc.ln() } else { (1.0 - pc).ln() };
        let t = if yi { 1.0 } else { 0.0 };
        brier += (pi - t) * (pi - t);
    }
    let n = p.len() as u64;
    let nf = n.max(1) as f64;
    let ratio = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    let recall = ratio(tp, tp + fn_);
    let specificity = ratio(tn, tn + fp);
    let precision = ratio(tp, tp + fp);
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    BinaryMetrics {
        metrics_version: METRICS_VERSION.to_string(),
        n,
        n_positive: tp + fn_,
        n_negative: tn + fp,
        threshold,
        confusion: ConfusionMatrix { tp, fp, tn, fn_ },
        accuracy: ratio(tp + tn, n),
        balanced_accuracy: (recall + specificity) / 2.0,
        precision,
        recall,
        specificity,
        f1,
        log_loss: log_loss / nf,
        brier: brier / nf,
        roc_auc: roc_auc(p, y),
    }
}

/// ROC AUC via the Mann–Whitney statistic with average ranks for ties.
pub fn roc_auc(p: &[f64], y: &[bool]) -> Option<f64> {
    let n_pos = y.iter().filter(|&&v| v).count();
    let n_neg = y.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && p[idx[j + 1]] == p[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            if y[k] {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let np = n_pos as f64;
    Some((rank_sum_pos - np * (np + 1.0) / 2.0) / (np * n_neg as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_checked_values() {
        let p = [0.9, 0.8, 0.3, 0.6, 0.2, 0.1];
        let y = [true, true, true, false, false, false];
        let m = binary_metrics(&p, &y, 0.5);
        assert_eq!((m.confusion.tp, m.confusion.fp, m.confusion.tn, m.confusion.fn_), (2, 1, 2, 1));
        assert!((m.precision - 2.0 / 3.0).abs() < 1e-12);
        assert!((m.recall - 2.0 / 3.0).abs() < 1e-12);
        assert!((m.balanced_accuracy - 2.0 / 3.0).abs() < 1e-12);
        // AUC: pairs (pos > neg): 0.9>all(3), 0.8>all(3), 0.3>{0.2,0.1}(2) => 8/9
        assert!((m.roc_auc.unwrap() - 8.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn ties_and_degenerate() {
        assert_eq!(roc_auc(&[0.5, 0.5], &[true, false]), Some(0.5));
        assert_eq!(roc_auc(&[0.5, 0.5], &[true, true]), None);
        let m = binary_metrics(&[0.1, 0.2], &[false, false], 0.5);
        assert_eq!(m.precision, 0.0);
        assert_eq!(m.f1, 0.0);
    }
}
