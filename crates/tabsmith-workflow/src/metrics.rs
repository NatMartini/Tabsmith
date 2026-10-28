//! Metrics of the three tasks (definitions match scikit-learn's: binary metrics;
//! macro-averaged multiclass recall, F1 and one-vs-rest ROC AUC; RMSE, MAE, R²).

use tabsmith_contracts::METRICS_VERSION;
use tabsmith_contracts::evaluation::{BinaryMetrics, ConfusionMatrix, Metrics, MulticlassMetrics, RegressionMetrics};
use tabsmith_contracts::schema::OutputSchema;
use tabsmith_runtime::Matrix;
use tabsmith_runtime::model::{sigmoid, softmax};

use crate::data::Targets;

/// Metrics of model outputs (`rows x outputs`, as the runtime computes them)
/// against targets, read with the output schema: probabilities for
/// classification, values in the target's units for regression.
pub fn task_metrics(outputs: &Matrix, targets: &Targets, schema: &OutputSchema) -> Metrics {
    match targets {
        Targets::Binary(y) => {
            let p: Vec<f64> = outputs.data.iter().map(|&z| sigmoid(f64::from(z))).collect();
            Metrics::Binary(binary_metrics(&p, y, schema.threshold))
        }
        Targets::Classes { y, k } => {
            let p: Vec<Vec<f64>> = (0..outputs.rows).map(|r| softmax(outputs.row(r))).collect();
            Metrics::Multiclass(multiclass_metrics(&p, y, *k))
        }
        Targets::Values(y) => {
            let s = schema.target_scale.expect("regression schema");
            let v: Vec<f64> = outputs.data.iter().map(|&z| s.mean + s.std * f64::from(z)).collect();
            Metrics::Regression(regression_metrics(&v, y))
        }
    }
}

/// Multiclass metrics of per-row class probabilities `p` and class indices `y`.
/// Means over classes skip classes absent from `y` (scikit-learn warns and
/// counts them as 0; here they would only measure the sample, not the model).
pub fn multiclass_metrics(p: &[Vec<f64>], y: &[u32], k: usize) -> MulticlassMetrics {
    assert_eq!(p.len(), y.len());
    let mut confusion = vec![vec![0u64; k]; k];
    let mut log_loss = 0.0;
    for (pi, &yi) in p.iter().zip(y) {
        let mut pred = 0;
        for (c, &v) in pi.iter().enumerate() {
            if v > pi[pred] {
                pred = c;
            }
        }
        confusion[yi as usize][pred] += 1;
        log_loss -= pi[yi as usize].clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS).ln();
    }
    let n = y.len() as u64;
    let class_counts: Vec<u64> = confusion.iter().map(|row| row.iter().sum()).collect();
    let present: Vec<usize> = (0..k).filter(|&c| class_counts[c] > 0).collect();
    let mean = |f: &dyn Fn(usize) -> f64| present.iter().map(|&c| f(c)).sum::<f64>() / present.len().max(1) as f64;
    let ratio = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    let recall = |c: usize| ratio(confusion[c][c], class_counts[c]);
    let predicted = |c: usize| (0..k).map(|a| confusion[a][c]).sum::<u64>();
    let f1 = |c: usize| {
        let (pr, re) = (ratio(confusion[c][c], predicted(c)), recall(c));
        if pr + re == 0.0 { 0.0 } else { 2.0 * pr * re / (pr + re) }
    };
    let correct: u64 = (0..k).map(|c| confusion[c][c]).sum();
    let roc_auc = (present.len() >= 2).then(|| {
        mean(&|c| {
            let score: Vec<f64> = p.iter().map(|pi| pi[c]).collect();
            let is_c: Vec<bool> = y.iter().map(|&v| v as usize == c).collect();
            roc_auc(&score, &is_c).unwrap_or(0.5)
        })
    });
    let (accuracy, balanced_accuracy, f1) = (ratio(correct, n), mean(&recall), mean(&f1));
    MulticlassMetrics {
        metrics_version: METRICS_VERSION.to_string(),
        n,
        class_counts: class_counts.clone(),
        accuracy,
        balanced_accuracy,
        f1,
        log_loss: log_loss / n.max(1) as f64,
        roc_auc,
        confusion: confusion.clone(),
    }
}

/// Regression metrics of predicted values `v` against targets `y`, in their units.
pub fn regression_metrics(v: &[f64], y: &[f64]) -> RegressionMetrics {
    assert_eq!(v.len(), y.len());
    let n = y.len();
    let nf = n.max(1) as f64;
    let mean = y.iter().sum::<f64>() / nf;
    let (mut sse, mut sae, mut sst) = (0.0, 0.0, 0.0);
    for (&vi, &yi) in v.iter().zip(y) {
        sse += (vi - yi) * (vi - yi);
        sae += (vi - yi).abs();
        sst += (yi - mean) * (yi - mean);
    }
    RegressionMetrics {
        metrics_version: METRICS_VERSION.to_string(),
        n: n as u64,
        rmse: (sse / nf).sqrt(),
        mae: sae / nf,
        r2: if sst > 0.0 { 1.0 - sse / sst } else { 0.0 },
        mean,
    }
}

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
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
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

    #[test]
    fn multiclass_hand_checked() {
        // Class 0: 2 rows (1 right), class 1: 1 row (right), class 2: 1 row (predicted 1).
        let p = vec![vec![0.7, 0.2, 0.1], vec![0.2, 0.5, 0.3], vec![0.1, 0.8, 0.1], vec![0.3, 0.4, 0.3]];
        let y = [0, 0, 1, 2];
        let m = multiclass_metrics(&p, &y, 3);
        assert_eq!(m.confusion, vec![vec![1, 1, 0], vec![0, 1, 0], vec![0, 1, 0]]);
        assert!((m.accuracy - 0.5).abs() < 1e-12);
        // recalls 1/2, 1, 0 => 1/2; F1: class 0 (p 1, r 1/2) 2/3, class 1 (p 1/3, r 1) 1/2, class 2: 0.
        assert!((m.balanced_accuracy - 0.5).abs() < 1e-12);
        assert!((m.f1 - (2.0 / 3.0 + 0.5) / 3.0).abs() < 1e-12);
        let ll = -(0.7f64.ln() + 0.2f64.ln() + 0.8f64.ln() + 0.3f64.ln()) / 4.0;
        assert!((m.log_loss - ll).abs() < 1e-12);
    }

    #[test]
    fn regression_hand_checked() {
        let m = regression_metrics(&[1.0, 2.0, 4.0], &[1.0, 3.0, 5.0]);
        assert!((m.rmse - (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((m.mae - 2.0 / 3.0).abs() < 1e-12);
        // mean 3, SST = 4 + 0 + 4 = 8, SSE = 2 => R² = 0.75
        assert!((m.r2 - 0.75).abs() < 1e-12);
    }
}
