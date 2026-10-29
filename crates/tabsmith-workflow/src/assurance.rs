//! What a bundle says about its own uncertainty and its inputs, computed on the
//! validation split when the bundle is built: split conformal prediction
//! (`conformal.json`) and permutation importance (`importance.json`).
//!
//! Both use the validation rows, which also selected the model (early stopping and
//! the choice between candidates), so the conformal guarantee is approximate: the
//! exact one needs rows the model never saw. The test split is not touched: it is
//! used once, for the evaluation, and nothing else.

use tabsmith_contracts::config::SelectionMetric;
use tabsmith_contracts::data::SplitName;
use tabsmith_contracts::manifest::{Conformal, FieldImportance, Importance};
use tabsmith_contracts::run::RunRecord;
use tabsmith_contracts::schema::OutputSchema;
use tabsmith_contracts::transform::FittedTransform;
use tabsmith_contracts::{Phase, Result};
use tabsmith_runtime::model::{Model, sigmoid, softmax};
use tabsmith_runtime::{Column, Matrix, Table, apply_transform_rows};

use crate::data::{Targets, targets};
use crate::metrics::task_metrics;
use crate::pipeline::Workspace;
use crate::rng::SplitMix64;
use crate::train::par_outputs;

/// Validation rows used at most (a prefix, in split order): enough for a stable
/// quantile and importance, bounded so that a bundle of a large project builds fast.
const MAX_ROWS: usize = 10_000;
/// Shuffles per field for the importance.
const REPEATS: usize = 3;
/// Fewer calibration rows than this give no conformal file.
const MIN_CALIBRATION_ROWS: usize = 20;

/// The default miscoverage of the conformal intervals and sets.
pub const DEFAULT_ALPHA: f64 = 0.1;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Assurance {
    pub conformal: Option<Conformal>,
    pub importance: Option<Importance>,
}

impl Workspace {
    /// Conformal calibration and permutation importance of `model` (the weights of
    /// `run`) on the run's validation split.
    pub fn assurance(&self, run: &RunRecord, model: &Model, alpha: f64) -> Result<Assurance> {
        let p = self.prepare_for_run(run)?;
        let mut rows = p.split.rows(SplitName::Valid);
        rows.truncate(MAX_ROWS);
        let sub = p.table.take(&rows);
        let x = apply_transform_rows(&p.fitted.transform, &sub, Phase::Bundle, &rows)?;
        let y = targets(&sub, &p.features.target)?;
        let mut schema = p.fitted.output_schema.clone();
        schema.threshold = run.threshold;
        let outputs = par_outputs(&x, |x| model.outputs(x));
        let importance = importance(
            &p.fitted.transform,
            &sub,
            &rows,
            &y,
            &schema,
            model,
            p.resolved.config.metric(),
            p.resolved.config.data.seed,
            &outputs,
        );
        Ok(Assurance {
            conformal: conformal(&outputs, &y, &schema, alpha),
            importance,
        })
    }
}

/// Split conformal calibration: the nonconformity score of every calibration row
/// (|y − value| for a regression, 1 − p(true label) for a classification) and its
/// finite-sample `1 − alpha` quantile, the ⌈(n + 1)(1 − α)⌉-th smallest score.
pub fn conformal(outputs: &Matrix, y: &Targets, schema: &OutputSchema, alpha: f64) -> Option<Conformal> {
    let n = outputs.rows;
    if n < MIN_CALIBRATION_ROWS || !(alpha > 0.0 && alpha < 1.0) {
        return None;
    }
    let (scores, method): (Vec<f64>, &str) = match y {
        Targets::Values(y) => {
            let s = schema.target_scale?;
            let residuals = outputs
                .data
                .iter()
                .zip(y)
                .map(|(&z, &t)| (t - (s.mean + s.std * f64::from(z))).abs())
                .collect();
            (residuals, "split-conformal/absolute-residual")
        }
        Targets::Binary(y) => {
            let scores = outputs
                .data
                .iter()
                .zip(y)
                .map(|(&z, &t)| {
                    let p = sigmoid(f64::from(z));
                    1.0 - if t { p } else { 1.0 - p }
                })
                .collect();
            (scores, "split-conformal/lac")
        }
        Targets::Classes { y, k } => {
            let scores = (0..n)
                .map(|r| 1.0 - softmax(&outputs.data[r * k..(r + 1) * k])[y[r] as usize])
                .collect();
            (scores, "split-conformal/lac")
        }
    };
    let mut sorted = scores.clone();
    sorted.sort_by(f64::total_cmp);
    let rank = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
    if rank == 0 || rank > n {
        // Too few rows for this alpha: no finite quantile covers it.
        return None;
    }
    let quantile = sorted[rank - 1];
    if !quantile.is_finite() {
        return None;
    }
    let covered = scores.iter().filter(|&&s| s <= quantile).count();
    Some(Conformal {
        alpha,
        calibration_split: "valid".into(),
        n_calibration: n,
        quantile,
        calibration_coverage: covered as f64 / n as f64,
        method: method.into(),
        note: "calibrated on the validation split, which also selected the model: coverage on new rows is \
               approximate; rows must be distributed like the validation rows"
            .into(),
    })
}

/// Permutation importance: the drop of the selection metric on `sub` when one input
/// field's values are shuffled across its rows (the field's own values, in another
/// order, so its distribution is unchanged), averaged over `REPEATS` shuffles. Lower-is-
/// better metrics are flipped, so a positive importance always means shuffling hurt.
#[allow(clippy::too_many_arguments)]
fn importance(
    transform: &FittedTransform,
    sub: &Table,
    rows: &[usize],
    y: &Targets,
    schema: &OutputSchema,
    model: &Model,
    metric: SelectionMetric,
    seed: u64,
    baseline_outputs: &Matrix,
) -> Option<Importance> {
    let name = metric.as_str();
    let score = |outputs: &Matrix| task_metrics(outputs, y, schema).get(name);
    let baseline = score(baseline_outputs)?;
    let sign = if metric.higher_is_better() { 1.0 } else { -1.0 };
    let n = sub.n_rows();
    let mut fields = Vec::new();
    for f in &transform.features {
        let col = sub.column(f.name())?;
        let mut drops = Vec::with_capacity(REPEATS);
        for r in 0..REPEATS {
            let mut perm: Vec<usize> = (0..n).collect();
            SplitMix64::derive(seed, &format!("importance/{}/{r}", f.name())).shuffle(&mut perm);
            let shuffled = col.take(&perm);
            let columns: Vec<Column> = sub
                .columns()
                .iter()
                .map(|c| {
                    if c.name == f.name() {
                        Column {
                            name: c.name.clone(),
                            data: shuffled.clone(),
                        }
                    } else {
                        c.clone()
                    }
                })
                .collect();
            let table = Table::new(columns).ok()?;
            let x = apply_transform_rows(transform, &table, Phase::Bundle, rows).ok()?;
            let outputs = par_outputs(&x, |x| model.outputs(x));
            drops.push(sign * (baseline - score(&outputs)?));
        }
        let mean = drops.iter().sum::<f64>() / REPEATS as f64;
        let var = drops.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / REPEATS as f64;
        fields.push(FieldImportance {
            name: f.name().to_string(),
            importance: mean,
            std: var.sqrt(),
        });
    }
    fields.sort_by(|a, b| b.importance.total_cmp(&a.importance).then(a.name.cmp(&b.name)));
    Some(Importance {
        metric: name.into(),
        split: "valid".into(),
        n_rows: n,
        repeats: REPEATS,
        baseline,
        fields,
        note: "the drop of the metric when the field's values are shuffled across the validation rows; a field \
               the model never reads drops nothing, a field two others can stand in for drops little"
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabsmith_contracts::config::Task;
    use tabsmith_contracts::schema::TargetScale;

    fn regression_schema() -> OutputSchema {
        OutputSchema {
            target: "y".into(),
            task: Task::Regression,
            negative_label: String::new(),
            positive_label: String::new(),
            probability: String::new(),
            threshold: 0.0,
            decision: "value".into(),
            classes: vec![],
            target_scale: Some(TargetScale { mean: 10.0, std: 2.0 }),
        }
    }

    #[test]
    fn the_quantile_covers_at_least_one_minus_alpha_of_the_calibration_rows() {
        // 100 rows whose predictions miss by 0, 1, 2, ... units (in the target's units).
        let n = 100;
        let outputs = Matrix {
            rows: n,
            cols: 1,
            data: vec![0.0; n],
        };
        let y = Targets::Values((0..n).map(|i| 10.0 + i as f64).collect());
        let c = conformal(&outputs, &y, &regression_schema(), 0.1).unwrap();
        // ⌈101 × 0.9⌉ = 91st smallest residual: 90.
        assert_eq!(c.quantile, 90.0);
        assert!(c.calibration_coverage >= 0.9, "{}", c.calibration_coverage);
        assert_eq!(c.method, "split-conformal/absolute-residual");
        // Too few rows for the alpha: no quantile.
        let few = Matrix {
            rows: 20,
            cols: 1,
            data: vec![0.0; 20],
        };
        assert!(conformal(&few, &Targets::Values(vec![10.0; 20]), &regression_schema(), 0.01).is_none());
    }

    #[test]
    fn classification_sets_use_one_minus_the_true_probability() {
        let n = 50;
        // Logits that put 0.88 on the true label for every row: the score is 0.12.
        let z = (0.88f64 / 0.12).ln() as f32;
        let outputs = Matrix {
            rows: n,
            cols: 1,
            data: (0..n).map(|i| if i % 2 == 0 { z } else { -z }).collect(),
        };
        let y = Targets::Binary((0..n).map(|i| i % 2 == 0).collect());
        let schema = OutputSchema {
            target: "y".into(),
            task: Task::Binary,
            negative_label: "0".into(),
            positive_label: "1".into(),
            probability: "P(1)".into(),
            threshold: 0.5,
            decision: "threshold".into(),
            classes: vec![],
            target_scale: None,
        };
        let c = conformal(&outputs, &y, &schema, 0.1).unwrap();
        // The logit is an f32, so the probability is 0.88 to about 1e-8.
        assert!((c.quantile - 0.12).abs() < 1e-6, "{}", c.quantile);
        assert_eq!(c.calibration_coverage, 1.0);
        assert_eq!(c.method, "split-conformal/lac");
    }
}
