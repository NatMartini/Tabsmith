//! Application-facing prediction API.

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use rustpy_contracts::config::ExtraColumnPolicy;
use rustpy_contracts::manifest::BundleManifest;
use rustpy_contracts::schema::{InputSchema, OutputSchema};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError, RUSTPY_VERSION};
use serde::Serialize;

use crate::bundle::{Bundle, LoadOptions};
use crate::model::sigmoid;
use crate::transform::{apply_transform, check_table_columns, extra_column_error, missing_column_error, Matrix};
use crate::value::{Record, Table};

/// One prediction.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prediction {
    /// Predicted label (one of the two target labels).
    pub label: String,
    /// P(target == positive label).
    pub probability: f64,
    /// `probability >= threshold`.
    pub is_positive: bool,
}

/// Result of [`Predictor::self_check`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CheckReport {
    pub bundle_id: String,
    pub bundle_dir: String,
    pub files_verified: usize,
    pub examples_checked: usize,
    pub max_abs_probability_diff: f64,
    pub tolerance: f64,
    pub label_mismatches: usize,
    pub runtime_version: String,
    pub device: String,
    pub os: String,
    pub arch: String,
    pub load_ms: f64,
    pub passed: bool,
}

/// A loaded, verified model ready for prediction. Cheap to share behind an `Arc`.
#[derive(Debug, Clone)]
pub struct Predictor {
    bundle: Bundle,
    load_ms: f64,
}

impl Predictor {
    /// Loads and verifies a bundle directory. Nothing is loaded if any check fails.
    pub fn load(dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let start = Instant::now();
        let bundle = Bundle::open(dir, &opts)?;
        let p = Self {
            bundle,
            load_ms: start.elapsed().as_secs_f64() * 1e3,
        };
        if opts.self_check {
            let report = p.self_check()?;
            if !report.passed {
                return Err(RustpyError::new(
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

    pub fn bundle_id(&self) -> &str {
        self.bundle.id()
    }

    pub fn manifest(&self) -> &BundleManifest {
        &self.bundle.manifest
    }

    pub fn input_schema(&self) -> &InputSchema {
        &self.bundle.input_schema
    }

    pub fn output_schema(&self) -> &OutputSchema {
        &self.bundle.output_schema
    }

    pub fn bundle(&self) -> &Bundle {
        &self.bundle
    }

    /// Predicts a batch of records.
    pub fn predict(&self, records: &[Record]) -> Result<Vec<Prediction>> {
        let table = self.records_to_table(records)?;
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
                let v = serde_json::to_value(r).map_err(|e| {
                    RustpyError::new(ErrorCode::SchemaTypeMismatch, Phase::Predict, format!("cannot serialize row: {e}"))
                })?;
                Record::from_json(&v)
            })
            .collect::<Result<Vec<_>>>()?;
        self.predict(&records)
    }

    /// Predicts a columnar table (used by offline batch prediction and evaluation).
    pub fn predict_table(&self, table: &Table) -> Result<Vec<Prediction>> {
        let probs = self.probabilities(table)?;
        let out = &self.bundle.output_schema;
        Ok(probs
            .into_iter()
            .map(|p| {
                let is_positive = p >= out.threshold;
                Prediction {
                    label: if is_positive { out.positive_label.clone() } else { out.negative_label.clone() },
                    probability: p,
                    is_positive,
                }
            })
            .collect())
    }

    /// P(positive) per row of a table.
    pub fn probabilities(&self, table: &Table) -> Result<Vec<f64>> {
        check_table_columns(&self.bundle.input_schema, table, Phase::Predict)?;
        let x = apply_transform(&self.bundle.transform, table, Phase::Predict)?;
        Ok(self.probabilities_matrix(&x))
    }

    /// P(positive) per row of an already-transformed matrix.
    pub fn probabilities_matrix(&self, x: &Matrix) -> Vec<f64> {
        self.bundle
            .model
            .logits(x)
            .into_iter()
            .map(|z| sigmoid(z as f64))
            .collect()
    }

    fn records_to_table(&self, records: &[Record]) -> Result<Table> {
        let schema = &self.bundle.input_schema;
        for (i, r) in records.iter().enumerate() {
            for field in schema.fields.iter().filter(|f| f.required) {
                if !r.contains(&field.name) {
                    let present: Vec<&str> = r.names().collect();
                    return Err(missing_column_error(&field.name, field.kind, &present, Some(i), Phase::Predict));
                }
            }
            if schema.extra_columns == ExtraColumnPolicy::Error {
                for name in r.names() {
                    let known = schema.field(name).is_some() || schema.passthrough_id.as_deref() == Some(name);
                    if !known {
                        return Err(extra_column_error(name, Phase::Predict));
                    }
                }
            }
        }
        let names: Vec<&str> = schema
            .fields
            .iter()
            .filter(|f| f.required || records.iter().any(|r| r.contains(&f.name)))
            .map(|f| f.name.as_str())
            .collect();
        Ok(Table::from_records(records, &names))
    }

    /// Runs the bundled synthetic examples and compares them with the outputs
    /// recorded at build time.
    pub fn self_check(&self) -> Result<CheckReport> {
        let b = &self.bundle;
        let preds = self.predict_json(&serde_json::Value::Array(b.examples.records.clone()))?;
        let mut max_diff = 0f64;
        let mut mismatches = 0usize;
        for (p, e) in preds.iter().zip(&b.expected.predictions) {
            max_diff = max_diff.max((p.probability - e.probability).abs());
            if p.label != e.label {
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
            runtime_version: RUSTPY_VERSION.to_string(),
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
    pub previous_bundle_id: String,
    pub active_bundle_id: String,
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
    pub fn new(predictor: Predictor) -> Self {
        Self {
            current: RwLock::new(Arc::new(predictor)),
            updating: std::sync::Mutex::new(()),
        }
    }

    pub fn load(dir: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        Ok(Self::new(Predictor::load(dir, opts)?))
    }

    /// The active predictor. Requests holding this `Arc` finish on the model they started with.
    pub fn current(&self) -> Arc<Predictor> {
        self.current.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

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
        let fail = |cause: RustpyError| -> RustpyError {
            let mut e = RustpyError::new(
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
                RustpyError::new(
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
    use rustpy_contracts::config::{ExtraColumnPolicy, NanPolicy, UnknownCategoryPolicy};
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
