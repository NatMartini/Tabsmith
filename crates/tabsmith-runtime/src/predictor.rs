//! Application-facing prediction API.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::Serialize;
use tabsmith_contracts::config::{ExtraColumnPolicy, Task};
use tabsmith_contracts::manifest::BundleManifest;
use tabsmith_contracts::schema::{InputSchema, OutputSchema};
use tabsmith_contracts::{ErrorCode, Phase, Result, TABSMITH_VERSION, TabsmithError};

use crate::bundle::{Bundle, LoadOptions};
use crate::model::{sigmoid, softmax};
use crate::transform::{
    Matrix, UnseenCategories, apply_transform, check_table_columns, extra_column_error, missing_column_error, unseen_categories,
    unseen_fields_by_row,
};
use crate::value::{Record, Table};

/// What the model predicts for a record, by task.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Outcome {
    /// Two classes.
    Binary {
        /// P(target == positive label): the score the threshold applies to.
        probability: f64,
        /// `probability >= threshold`: the label is the positive label.
        is_positive: bool,
    },
    /// Three or more classes.
    Multiclass {
        /// P(target == label).
        probability: f64,
        /// P(class) for every class of the output schema, in its order.
        probabilities: Vec<f64>,
    },
    /// A number (also in the prediction's label, as text).
    Regression {
        /// The predicted value, in the target's units.
        value: f64,
    },
}

/// One prediction. Serialized flat: `label`, the fields of the [`Outcome`], and
/// `unseen_categories` when there are any.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prediction {
    /// The predicted label: a class, or the predicted value as text (regression).
    pub label: String,
    /// Probabilities or value, by task.
    #[serde(flatten)]
    pub outcome: Outcome,
    /// Categorical input fields whose value was not seen in training. With
    /// `unknown_category = "ignore"` such a value contributes nothing (all-zero
    /// encoding), so the prediction relies on the other fields only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unseen_categories: Vec<String>,
}

impl Prediction {
    /// Binary: P(positive label). Multiclass: P(label). Regression: `None`.
    pub fn probability(&self) -> Option<f64> {
        match &self.outcome {
            Outcome::Binary { probability, .. } | Outcome::Multiclass { probability, .. } => Some(*probability),
            Outcome::Regression { .. } => None,
        }
    }

    /// Binary: whether the positive label was predicted. Other tasks: false.
    pub fn is_positive(&self) -> bool {
        matches!(self.outcome, Outcome::Binary { is_positive: true, .. })
    }

    /// Multiclass: P(class) for every class, in the output schema's order.
    pub fn probabilities(&self) -> Option<&[f64]> {
        match &self.outcome {
            Outcome::Multiclass { probabilities, .. } => Some(probabilities),
            _ => None,
        }
    }

    /// Regression: the predicted value.
    pub fn value(&self) -> Option<f64> {
        match self.outcome {
            Outcome::Regression { value } => Some(value),
            _ => None,
        }
    }
}

/// Reads one row of model outputs with the output schema: the sigmoid and the
/// threshold (binary), the softmax and its most probable class, the first in class
/// order on a tie (multiclass), or the standardised value scaled back (regression).
pub fn decode(schema: &OutputSchema, outputs: &[f32]) -> (String, Outcome) {
    match schema.task {
        Task::Binary => {
            let probability = sigmoid(f64::from(outputs[0]));
            let is_positive = probability >= schema.threshold;
            let label = if is_positive {
                &schema.positive_label
            } else {
                &schema.negative_label
            };
            (label.clone(), Outcome::Binary { probability, is_positive })
        }
        Task::Multiclass => {
            let probabilities = softmax(outputs);
            let best = tabsmith_embedded::kernels::argmax(&probabilities);
            (
                schema.classes[best].clone(),
                Outcome::Multiclass {
                    probability: probabilities[best],
                    probabilities,
                },
            )
        }
        Task::Regression => {
            let scale = schema.target_scale.expect("checked when the bundle was loaded");
            let value = tabsmith_embedded::kernels::unscale(outputs[0], scale.mean, scale.std);
            (value.to_string(), Outcome::Regression { value })
        }
    }
}

/// Result of [`Predictor::self_check`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CheckReport {
    /// The checked bundle's ID.
    pub bundle_id: String,
    /// The checked bundle's directory.
    pub bundle_dir: String,
    /// Files whose SHA-256 matched the manifest.
    pub files_verified: usize,
    /// Example records predicted.
    pub examples_checked: usize,
    /// Largest difference from an expected probability (regression: from an expected
    /// value, in standard deviations of the training targets).
    pub max_abs_probability_diff: f64,
    /// Largest difference that still passes.
    pub tolerance: f64,
    /// Examples whose predicted label differs from the expected one.
    pub label_mismatches: usize,
    /// Version of this runtime.
    pub runtime_version: String,
    /// Device the examples were predicted on.
    pub device: String,
    /// Operating system of this process.
    pub os: String,
    /// CPU architecture of this process.
    pub arch: String,
    /// Time taken to load and verify the bundle, in milliseconds.
    pub load_ms: f64,
    /// Whether every check passed.
    pub passed: bool,
}

/// A loaded, verified model ready for prediction. Cheap to share behind an `Arc`.
#[derive(Debug, Clone)]
pub struct Predictor {
    bundle: Bundle,
    load_ms: f64,
}

/// `Instant::now` panics on `wasm32-unknown-unknown`, which has no clock in `std`;
/// load times read 0 there.
fn clock() -> Option<Instant> {
    if cfg!(target_arch = "wasm32") { None } else { Some(Instant::now()) }
}

fn ms_since(start: Option<Instant>) -> f64 {
    start.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1e3)
}

impl Predictor {
    /// Loads and verifies a bundle directory. Nothing is loaded if any check fails.
    pub fn load(dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let start = clock();
        let bundle = Bundle::open(dir, &opts)?;
        Self::loaded(bundle, start, &opts)
    }

    /// Like [`Predictor::load`], for a bundle whose files are already in memory;
    /// see [`Bundle::from_memory`].
    pub fn from_memory(label: impl Into<PathBuf>, files: BTreeMap<String, Vec<u8>>, opts: LoadOptions) -> Result<Self> {
        let start = clock();
        let bundle = Bundle::from_memory(label, files, &opts)?;
        Self::loaded(bundle, start, &opts)
    }

    fn loaded(bundle: Bundle, start: Option<Instant>, opts: &LoadOptions) -> Result<Self> {
        let p = Self {
            bundle,
            load_ms: ms_since(start),
        };
        if opts.self_check {
            let report = p.self_check()?;
            if !report.passed {
                return Err(TabsmithError::new(
                    ErrorCode::BundleCorrupt,
                    Phase::Load,
                    "bundle self-check failed: example predictions differ from the expected outputs",
                )
                .field(p.bundle.dir.display().to_string())
                .actual(format!(
                    "max |Δp| = {:.3e} (tolerance {:.1e}), {} label mismatches",
                    report.max_abs_probability_diff, report.tolerance, report.label_mismatches
                ))
                .next("rebuild the bundle; if it persists, report the runtime/platform combination"));
            }
        }
        Ok(p)
    }

    /// The loaded bundle's ID (`b-…`, derived from its content).
    pub fn bundle_id(&self) -> &str {
        self.bundle.id()
    }

    /// The loaded bundle's manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.bundle.manifest
    }

    /// The fields a record must provide.
    pub fn input_schema(&self) -> &InputSchema {
        &self.bundle.input_schema
    }

    /// Target, labels and decision threshold.
    pub fn output_schema(&self) -> &OutputSchema {
        &self.bundle.output_schema
    }

    /// The whole verified bundle.
    pub fn bundle(&self) -> &Bundle {
        &self.bundle
    }

    /// Predicts a batch of records.
    pub fn predict(&self, records: &[Record]) -> Result<Vec<Prediction>> {
        let table = self.records_table(records, &[])?;
        self.predict_table(&table)
    }

    /// Predicts a single record.
    pub fn predict_one(&self, record: &Record) -> Result<Prediction> {
        Ok(self
            .predict(std::slice::from_ref(record))?
            .pop()
            .expect("one record in, one prediction out"))
    }

    /// Predicts from a JSON object (one record) or array of objects.
    pub fn predict_json(&self, value: &serde_json::Value) -> Result<Vec<Prediction>> {
        let records = match value {
            serde_json::Value::Array(items) => items.iter().map(Record::from_json).collect::<Result<Vec<_>>>()?,
            other => vec![Record::from_json(other)?],
        };
        self.predict(&records)
    }

    /// Predicts from any `Serialize` rows whose fields match the input schema
    /// (e.g. your own `#[derive(Serialize)] struct Reading { .. }`).
    pub fn predict_serialize<T: Serialize>(&self, rows: &[T]) -> Result<Vec<Prediction>> {
        let records = rows
            .iter()
            .map(|r| {
                let v = serde_json::to_value(r)
                    .map_err(|e| TabsmithError::new(ErrorCode::SchemaTypeMismatch, Phase::Predict, format!("cannot serialize row: {e}")))?;
                Record::from_json(&v)
            })
            .collect::<Result<Vec<_>>>()?;
        self.predict(&records)
    }

    /// Predicts a columnar table (used by offline batch prediction and evaluation).
    pub fn predict_table(&self, table: &Table) -> Result<Vec<Prediction>> {
        let outputs = self.outputs(table)?;
        let unseen = unseen_fields_by_row(&self.bundle.transform, table);
        let schema = &self.bundle.output_schema;
        Ok(unseen
            .into_iter()
            .enumerate()
            .map(|(r, unseen_categories)| {
                let (label, outcome) = decode(schema, outputs.row(r));
                Prediction {
                    label,
                    outcome,
                    unseen_categories,
                }
            })
            .collect())
    }

    /// Per-field counts of categorical values in `table` that were not seen in
    /// training (useful to detect drift such as a new month or product code).
    pub fn unseen_categories(&self, table: &Table) -> Vec<UnseenCategories> {
        unseen_categories(&self.bundle.transform, table)
    }

    /// The model's outputs for a table, `rows x outputs`: logits (classification)
    /// or standardised values (regression); [`decode`] reads them.
    pub fn outputs(&self, table: &Table) -> Result<Matrix> {
        check_table_columns(&self.bundle.input_schema, table, Phase::Predict)?;
        let x = apply_transform(&self.bundle.transform, table, Phase::Predict)?;
        Ok(self.outputs_matrix(&x))
    }

    /// The model's outputs for an already-transformed matrix.
    pub fn outputs_matrix(&self, x: &Matrix) -> Matrix {
        self.bundle.model.outputs(x)
    }

    /// Checks records against the input contract and turns them into a table with
    /// the schema's fields, plus the `keep` columns (identifiers to carry along,
    /// allowed even when extra columns are refused) where records have them.
    pub fn records_table(&self, records: &[Record], keep: &[String]) -> Result<Table> {
        let schema = &self.bundle.input_schema;
        for (i, r) in records.iter().enumerate() {
            for field in schema.fields.iter().filter(|f| f.required) {
                if !r.contains(&field.name) {
                    let present: Vec<&str> = r.names().collect();
                    return Err(missing_column_error(
                        &field.name,
                        field.kind,
                        field.nullable,
                        &present,
                        Some(i),
                        Phase::Predict,
                    ));
                }
            }
            if schema.extra_columns == ExtraColumnPolicy::Error {
                for name in r.names() {
                    let known =
                        schema.field(name).is_some() || schema.passthrough_id.as_deref() == Some(name) || keep.iter().any(|k| k == name);
                    if !known {
                        return Err(extra_column_error(name, Phase::Predict));
                    }
                }
            }
        }
        let mut names: Vec<&str> = schema
            .fields
            .iter()
            .filter(|f| f.required || records.iter().any(|r| r.contains(&f.name)))
            .map(|f| f.name.as_str())
            .collect();
        for k in keep {
            if schema.field(k).is_none() && !names.contains(&k.as_str()) && records.iter().any(|r| r.contains(k)) {
                names.push(k.as_str());
            }
        }
        Ok(Table::from_records(records, &names))
    }

    /// Runs the bundled synthetic examples and compares them with the outputs
    /// recorded at build time.
    pub fn self_check(&self) -> Result<CheckReport> {
        let b = &self.bundle;
        let preds = self.predict_json(&serde_json::Value::Array(b.examples.records.clone()))?;
        let mut max_diff = 0f64;
        let mut mismatches = 0usize;
        let std = b.output_schema.target_scale.map_or(1.0, |s| s.std);
        for (p, e) in preds.iter().zip(&b.expected.predictions) {
            let diff = match &p.outcome {
                Outcome::Binary { probability, .. } => (probability - e.probability).abs(),
                Outcome::Multiclass { probabilities, .. } => probabilities.iter().zip(&e.probabilities).map(|(a, b)| (a - b).abs()).fold(
                    if probabilities.len() == e.probabilities.len() {
                        0.0
                    } else {
                        f64::INFINITY
                    },
                    f64::max,
                ),
                Outcome::Regression { value } => e.value.map_or(f64::INFINITY, |v| (value - v).abs() / std),
            };
            max_diff = max_diff.max(diff);
            // A regression label is the value as text; its agreement is the value's.
            if p.label != e.label && !matches!(p.outcome, Outcome::Regression { .. }) {
                mismatches += 1;
            }
        }
        let tolerance = b.expected.tolerance;
        Ok(CheckReport {
            bundle_id: b.id().to_string(),
            bundle_dir: b.dir.display().to_string(),
            files_verified: b.manifest.files.len(),
            examples_checked: preds.len(),
            max_abs_probability_diff: max_diff,
            tolerance,
            label_mismatches: mismatches,
            runtime_version: TABSMITH_VERSION.to_string(),
            device: "cpu".to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            load_ms: self.load_ms,
            passed: max_diff <= tolerance && mismatches == 0 && !preds.is_empty(),
        })
    }
}

/// Outcome of a successful [`PredictorHandle::try_update`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UpdateReport {
    /// The bundle that was active before the update.
    pub previous_bundle_id: String,
    /// The bundle that is active now.
    pub active_bundle_id: String,
    /// Example records the new bundle reproduced before it was switched in.
    pub examples_checked: usize,
    /// Human-readable input contract differences between the two bundles.
    pub schema_changes: Vec<String>,
    /// Output contract differences (target, labels, threshold).
    pub output_changes: Vec<String>,
}

/// A shareable, hot-swappable predictor. Updates load and check the new bundle in a
/// separate instance first; on any failure the current predictor stays active.
#[derive(Debug)]
pub struct PredictorHandle {
    current: RwLock<Arc<Predictor>>,
    /// Serialises updates so each one is checked against the predictor it replaces.
    updating: std::sync::Mutex<()>,
}

impl PredictorHandle {
    /// A handle whose active predictor is `predictor`.
    pub fn new(predictor: Predictor) -> Self {
        Self {
            current: RwLock::new(Arc::new(predictor)),
            updating: std::sync::Mutex::new(()),
        }
    }

    /// Loads `dir` with [`Predictor::load`] and makes it the active predictor.
    pub fn load(dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        Ok(Self::new(Predictor::load(dir, opts)?))
    }

    /// The active predictor. Requests holding this `Arc` finish on the model they started with.
    pub fn current(&self) -> Arc<Predictor> {
        self.current.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Predicts with the active predictor ([`Predictor::predict`]).
    pub fn predict(&self, records: &[Record]) -> Result<Vec<Prediction>> {
        self.current().predict(records)
    }

    /// Loads, verifies and self-checks `dir`, then atomically makes it active.
    /// Breaking contract changes are refused unless `opts.allow_contract_change`.
    /// On failure the previous predictor remains active and the error says so.
    pub fn try_update(&self, dir: impl AsRef<Path>, opts: LoadOptions) -> Result<UpdateReport> {
        let _guard = self.updating.lock().unwrap_or_else(|e| e.into_inner());
        let old = self.current();
        let dir = dir.as_ref();
        let fail = |cause: TabsmithError| -> TabsmithError {
            let mut e = TabsmithError::new(
                ErrorCode::PredictorUpdateFailed,
                Phase::Load,
                format!("update to `{}` failed: {}", dir.display(), cause.message),
            )
            .retained(format!("bundle {} remains active", old.bundle_id()))
            .actual(format!("{}", cause.code));
            e.field = cause.field.clone();
            e.next_action = cause.next_action.clone();
            e
        };
        let new = Predictor::load(dir, opts.clone().with_self_check(true)).map_err(fail)?;
        let (changes, breaking_in) = input_changes(old.input_schema(), new.input_schema());
        let (out_changes, breaking_out) = output_changes(old.output_schema(), new.output_schema());
        let breaking: Vec<String> = breaking_in.into_iter().chain(breaking_out).collect();
        if !breaking.is_empty() && !opts.allow_contract_change {
            return Err(fail(
                TabsmithError::new(
                    ErrorCode::BundleIncompatible,
                    Phase::Load,
                    format!("the new bundle breaks the current contract: {}", breaking.join("; ")),
                )
                .next("adapt the application first, then retry with LoadOptions::allow_contract_change(true)"),
            ));
        }
        let report = UpdateReport {
            previous_bundle_id: old.bundle_id().to_string(),
            active_bundle_id: new.bundle_id().to_string(),
            examples_checked: new.bundle.examples.records.len(),
            schema_changes: changes,
            output_changes: out_changes,
        };
        *self.current.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(new);
        Ok(report)
    }

    /// Switches back to a previous (immutable) bundle; same checks as an update,
    /// except that returning to the earlier contract is allowed.
    pub fn rollback_to(&self, dir: impl AsRef<Path>, opts: LoadOptions) -> Result<UpdateReport> {
        self.try_update(dir, opts.allow_contract_change(true))
    }
}

/// All input differences, and the subset that can break an application that
/// works with the old bundle.
fn input_changes(old: &InputSchema, new: &InputSchema) -> (Vec<String>, Vec<String>) {
    use tabsmith_contracts::config::{ExtraColumnPolicy, NanPolicy, UnknownCategoryPolicy};
    let mut all = Vec::new();
    let mut breaking = Vec::new();
    for f in &new.fields {
        match old.field(&f.name) {
            None => {
                let msg = format!("added field `{}` ({})", f.name, f.kind.as_str());
                if f.required {
                    breaking.push(format!("new required field `{}`", f.name));
                }
                all.push(msg);
            }
            Some(o) => {
                if o.kind != f.kind || o.raw_dtype != f.raw_dtype {
                    let msg = format!(
                        "field `{}` changed {} ({}) -> {} ({})",
                        f.name,
                        o.kind.as_str(),
                        o.raw_dtype.as_str(),
                        f.kind.as_str(),
                        f.raw_dtype.as_str()
                    );
                    breaking.push(msg.clone());
                    all.push(msg);
                }
                if o.required != f.required {
                    let msg = format!("field `{}` is now {}", f.name, if f.required { "required" } else { "optional" });
                    if f.required {
                        breaking.push(msg.clone());
                    }
                    all.push(msg);
                }
            }
        }
    }
    for f in &old.fields {
        if new.field(&f.name).is_none() {
            let msg = format!("removed field `{}`", f.name);
            if new.extra_columns == ExtraColumnPolicy::Error {
                breaking.push(format!("{msg} (and extra fields are now rejected)"));
            }
            all.push(msg);
        }
    }
    let mut policy = |name: &str, stricter: bool, from: String, to: String| {
        if from != to {
            let msg = format!("{name} policy {from} -> {to}");
            if stricter {
                breaking.push(msg.clone());
            }
            all.push(msg);
        }
    };
    policy(
        "NaN",
        old.nan == NanPolicy::Missing && new.nan == NanPolicy::Error,
        format!("{:?}", old.nan),
        format!("{:?}", new.nan),
    );
    policy(
        "unknown-category",
        old.unknown_category == UnknownCategoryPolicy::Ignore && new.unknown_category == UnknownCategoryPolicy::Error,
        format!("{:?}", old.unknown_category),
        format!("{:?}", new.unknown_category),
    );
    policy(
        "extra-field",
        old.extra_columns == ExtraColumnPolicy::Ignore && new.extra_columns == ExtraColumnPolicy::Error,
        format!("{:?}", old.extra_columns),
        format!("{:?}", new.extra_columns),
    );
    (all, breaking)
}

fn output_changes(old: &OutputSchema, new: &OutputSchema) -> (Vec<String>, Vec<String>) {
    let mut all = Vec::new();
    let mut breaking = Vec::new();
    if old.target != new.target || old.positive_label != new.positive_label || old.negative_label != new.negative_label {
        let msg = format!(
            "prediction meaning changed: P({} == {:?}) -> P({} == {:?})",
            old.target, old.positive_label, new.target, new.positive_label
        );
        breaking.push(msg.clone());
        all.push(msg);
    }
    if old.threshold != new.threshold {
        all.push(format!("decision threshold {} -> {}", old.threshold, new.threshold));
    }
    (all, breaking)
}
