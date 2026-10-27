//! Building immutable inference bundles from completed runs.

use std::fs;
use std::path::{Path, PathBuf};

use rustpy_contracts::evaluation::EvaluationSubject;
use rustpy_contracts::data::SplitName;
use rustpy_contracts::hash::sha256_file;
use rustpy_contracts::manifest::{
    BundleEvaluation, BundleManifest, BundleProvenance, ExampleExpected, ExampleInputs,
    ExpectedPrediction, FileEntry, RuntimeRequirements, EVALUATION_FILE, EXAMPLES_EXPECTED_FILE, EXAMPLES_INPUT_FILE,
    INPUT_SCHEMA_FILE, MANIFEST_FILE, MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE, OUTPUT_SCHEMA_FILE, TRANSFORM_FILE,
};
use rustpy_contracts::model::MODEL_FORMAT;
use rustpy_contracts::run::{RunRecord, RunStatus};
use rustpy_contracts::schema::FieldKind;
use rustpy_contracts::time::now_rfc3339;
use rustpy_contracts::transform::{FeatureTransform, TRANSFORM_VERSION};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError, BUNDLE_FORMAT_VERSION, RECIPE_ID, RUSTPY_VERSION};
use rustpy_runtime::{CheckReport, LoadOptions, Predictor, Record, Value};
use serde::Serialize;

use crate::fit::FittedArtifacts;
use crate::pipeline::Workspace;
use crate::project::{read_json, write_atomic, write_json};

#[derive(Debug, Clone, Default)]
pub struct BundleOptions {
    /// Output directory (default: `bundles/<run-id>`).
    pub out: Option<PathBuf>,
    /// Build a candidate bundle without a test evaluation (marked `unevaluated-candidate`).
    pub allow_unevaluated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleBuildReport {
    pub path: String,
    pub manifest: BundleManifest,
    pub check: CheckReport,
    /// True when an identical bundle already existed at the path.
    pub already_existed: bool,
}

/// Synthetic example records derived from the transform statistics: typical
/// values, missing values and (when allowed) an unseen category. They contain no
/// training rows.
pub fn synthetic_examples(fitted: &FittedArtifacts) -> Vec<Record> {
    let t = &fitted.transform;
    let mut typical = Record::new();
    let mut high = Record::new();
    let mut missing = Record::new();
    let mut alt = Record::new();
    for f in &t.features {
        match f {
            FeatureTransform::Numeric { name, mean, std, impute, .. } => {
                let (m, s) = if *std == 1.0 && *mean == 0.0 { (*impute, 1.0) } else { (*mean, *std) };
                typical.insert(name.clone(), round6(m));
                high.insert(name.clone(), round6(m + 2.0 * s));
                alt.insert(name.clone(), round6(m - s));
                missing.insert(name.clone(), Value::Null);
            }
            FeatureTransform::Categorical { name, categories, missing_slot, .. } => {
                typical.insert(name.clone(), categories[0].clone());
                high.insert(name.clone(), categories.last().cloned().unwrap_or_default());
                alt.insert(name.clone(), categories.get(1).cloned().unwrap_or_else(|| categories[0].clone()));
                // A missing value is only valid input when it was seen in training or
                // unknown values are ignored; otherwise use a known category.
                let missing_ok = *missing_slot || t.unknown_category == rustpy_contracts::config::UnknownCategoryPolicy::Ignore;
                missing.insert(name.clone(), if missing_ok { Value::Null } else { Value::from(categories[0].clone()) });
            }
        }
    }
    // Fields that are part of the input contract but unused by the model.
    for d in &t.dropped {
        for r in [&mut typical, &mut high, &mut missing, &mut alt] {
            r.insert(d.name.clone(), Value::Null);
        }
    }
    // Categorical values must use the same raw type as the training data where it matters.
    let mut out = vec![typical, high, alt, missing];
    let unknown_ok = t.unknown_category == rustpy_contracts::config::UnknownCategoryPolicy::Ignore;
    if unknown_ok {
        if let Some(cat) = fitted.input_schema.fields.iter().find(|f| f.kind == FieldKind::Categorical && f.required) {
            let mut r = out[0].clone();
            r.insert(cat.name.clone(), "__rustpy_unseen_category__");
            out.push(r);
        }
    }
    // Fix numeric categories (e.g. int-coded) so they are sent as the training type.
    for rec in &mut out {
        for field in &fitted.input_schema.fields {
            if field.kind == FieldKind::Categorical && field.raw_dtype == rustpy_contracts::schema::RawDtype::Int {
                if let Some(Value::Text(s)) = rec.get(&field.name).cloned() {
                    if let Ok(i) = s.parse::<i64>() {
                        rec.insert(field.name.clone(), i);
                    }
                }
            }
            if field.kind == FieldKind::Categorical && field.raw_dtype == rustpy_contracts::schema::RawDtype::Bool {
                if let Some(Value::Text(s)) = rec.get(&field.name).cloned() {
                    if let Ok(b) = s.parse::<bool>() {
                        rec.insert(field.name.clone(), b);
                    }
                }
            }
        }
    }
    out
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

fn file_entry(dir: &Path, rel: &str) -> Result<FileEntry> {
    let p = dir.join(rel);
    let (sha256, bytes) = sha256_file(&p).map_err(|e| RustpyError::io(Phase::Bundle, &p, e))?;
    Ok(FileEntry {
        path: rel.to_string(),
        sha256,
        bytes,
    })
}

impl Workspace {
    /// Builds an immutable bundle from a completed run.
    pub fn build_bundle(&self, run_id: &str, opts: &BundleOptions) -> Result<BundleBuildReport> {
        let run: RunRecord = self.load_run(run_id).map_err(|e| e.in_phase(Phase::Bundle))?;
        if run.status != RunStatus::Completed {
            return Err(RustpyError::new(
                ErrorCode::RunStateInvalid,
                Phase::Bundle,
                format!("run {run_id} is {}; only completed runs can be bundled", run.status.as_str()),
            )
            .run(run_id)
            .next("pick a completed run from `rustpy runs` (or resume this one first)"));
        }
        let model = self.run_weights(&run).map_err(|e| e.in_phase(Phase::Bundle))?;
        let fitted: FittedArtifacts = read_json(
            &self.project.dir("transforms").join(&run.transform_id).join("fitted.json"),
            Phase::Bundle,
        )?;
        let test_eval = self
            .evaluations()
            .into_iter()
            .rev()
            .find(|e| {
                matches!(&e.subject, EvaluationSubject::Run { run_id: r } if r == run_id)
                    && e.protocol.split == SplitName::Test
            });
        if test_eval.is_none() && !opts.allow_unevaluated {
            return Err(RustpyError::new(
                ErrorCode::EvaluationMissing,
                Phase::Bundle,
                format!("run {run_id} has no held-out test evaluation"),
            )
            .run(run_id)
            .next(format!(
                "lock it as the final candidate with `rustpy evaluate --run {run_id} --split test`, or build a candidate bundle with --allow-unevaluated"
            )));
        }
        let metrics = run.metrics.clone().ok_or_else(|| RustpyError::internal(Phase::Bundle, "completed run without metrics"))?;
        let out = opts.out.clone().unwrap_or_else(|| self.project.bundles_dir().join(run_id));
        let parent = out.parent().unwrap_or(Path::new(".")).to_path_buf();
        fs::create_dir_all(&parent).map_err(|e| RustpyError::io(Phase::Bundle, &parent, e))?;
        let tmp = parent.join(format!(
            ".{}.tmp-{}",
            out.file_name().and_then(|n| n.to_str()).unwrap_or("bundle"),
            std::process::id()
        ));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).map_err(|e| RustpyError::io(Phase::Bundle, &tmp, e))?;
        }
        fs::create_dir_all(tmp.join("examples")).map_err(|e| RustpyError::io(Phase::Bundle, &tmp, e))?;
        let cleanup = |e: RustpyError| {
            let _ = fs::remove_dir_all(&tmp);
            e
        };
        let result = (|| -> Result<(BundleManifest, Vec<FileEntry>)> {
            let mut output_schema = fitted.output_schema.clone();
            output_schema.threshold = run.threshold;
            write_json(&tmp.join(INPUT_SCHEMA_FILE), &fitted.input_schema)?;
            write_json(&tmp.join(OUTPUT_SCHEMA_FILE), &output_schema)?;
            write_json(&tmp.join(TRANSFORM_FILE), &fitted.transform)?;
            write_json(&tmp.join(MODEL_CONFIG_FILE), &model.config)?;
            // The run's verified weight file, byte for byte: a run and its bundle share
            // the model hash used by the test-set discipline.
            let weights = self.project.run_dir(run_id).join("model").join(MODEL_WEIGHTS_FILE);
            let bytes = fs::read(&weights).map_err(|e| RustpyError::io(Phase::Bundle, &weights, e))?;
            write_atomic(&tmp.join(MODEL_WEIGHTS_FILE), &bytes)?;
            let evaluation = BundleEvaluation {
                status: if test_eval.is_some() { "evaluated".into() } else { "unevaluated-candidate".into() },
                test: test_eval.clone(),
                valid: metrics.valid.clone(),
                notes: if test_eval.is_some() {
                    vec![]
                } else {
                    vec!["no held-out test evaluation: treat as a candidate only".into()]
                },
            };
            write_json(&tmp.join(EVALUATION_FILE), &evaluation)?;
            let examples = synthetic_examples(&fitted);
            write_json(
                &tmp.join(EXAMPLES_INPUT_FILE),
                &ExampleInputs {
                    records: examples.iter().map(Record::to_json).collect(),
                    note: "synthetic records generated from transform statistics; no training rows".into(),
                },
            )?;
            // Placeholder (right length) so the bundle can be loaded once to compute the
            // expected outputs with the runtime itself; replaced right after.
            write_json(
                &tmp.join(EXAMPLES_EXPECTED_FILE),
                &ExampleExpected {
                    predictions: examples
                        .iter()
                        .map(|_| ExpectedPrediction { label: String::new(), probability: 0.0 })
                        .collect(),
                    tolerance: 1e-6,
                },
            )?;
            let mut files = Vec::new();
            let mut manifest = BundleManifest {
                bundle_format: BUNDLE_FORMAT_VERSION,
                bundle_id: String::new(),
                name: format!("{}-{}", run.candidate, run_id),
                created_at: now_rfc3339(),
                recipe: RECIPE_ID.to_string(),
                rustpy_version: RUSTPY_VERSION.to_string(),
                provenance: BundleProvenance {
                    run_id: run_id.to_string(),
                    candidate: run.candidate.to_string(),
                    snapshot_id: run.snapshot_id.clone(),
                    split_id: run.split_id.clone(),
                    transform_id: run.transform_id.clone(),
                    config_hash: run.config_hash.clone(),
                    evaluation_id: test_eval.as_ref().map(|e| e.evaluation_id.clone()),
                },
                runtime: RuntimeRequirements {
                    min_runtime_version: "0.1.0".into(),
                    transform_version: TRANSFORM_VERSION.into(),
                    model_format: MODEL_FORMAT.into(),
                    devices: vec!["cpu".into()],
                    precision: "fp32".into(),
                },
                files: vec![],
                self_contained: true,
                contains_training_data: false,
                contains_optimizer_state: false,
                evaluation_status: evaluation.status.clone(),
            };
            let write_manifest = |m: &mut BundleManifest, files: &mut Vec<FileEntry>| -> Result<()> {
                files.clear();
                for rel in [
                    INPUT_SCHEMA_FILE,
                    OUTPUT_SCHEMA_FILE,
                    TRANSFORM_FILE,
                    MODEL_CONFIG_FILE,
                    MODEL_WEIGHTS_FILE,
                    EVALUATION_FILE,
                    EXAMPLES_INPUT_FILE,
                    EXAMPLES_EXPECTED_FILE,
                ] {
                    files.push(file_entry(&tmp, rel)?);
                }
                m.files = files.clone();
                m.bundle_id = m.expected_id();
                write_json(&tmp.join(MANIFEST_FILE), m)
            };
            write_manifest(&mut manifest, &mut files)?;
            let pre = Predictor::load(&tmp, LoadOptions::cpu())?;
            let preds = pre.predict(&examples)?;
            write_json(
                &tmp.join(EXAMPLES_EXPECTED_FILE),
                &ExampleExpected {
                    predictions: preds
                        .iter()
                        .map(|p| ExpectedPrediction {
                            label: p.label.clone(),
                            probability: p.probability,
                        })
                        .collect(),
                    tolerance: 1e-6,
                },
            )?;
            write_manifest(&mut manifest, &mut files)?;
            Ok((manifest, files))
        })();
        let (manifest, _) = result.map_err(cleanup)?;
        let check = Predictor::load(&tmp, LoadOptions::cpu().with_self_check(true))
            .and_then(|p| p.self_check())
            .map_err(cleanup)?;
        if out.exists() {
            let existing: Option<BundleManifest> = read_json(&out.join(MANIFEST_FILE), Phase::Bundle).ok();
            let _ = fs::remove_dir_all(&tmp);
            return match existing {
                Some(m) if m.bundle_id == manifest.bundle_id => {
                    let check = Predictor::load(&out, LoadOptions::cpu().with_self_check(true))?.self_check()?;
                    Ok(BundleBuildReport {
                        path: self.project.rel(&out),
                        manifest: m,
                        check,
                        already_existed: true,
                    })
                }
                _ => Err(RustpyError::new(
                    ErrorCode::BundleExists,
                    Phase::Bundle,
                    format!("`{}` already exists and holds a different bundle; bundles are immutable", out.display()),
                )
                .field(out.display().to_string())
                .retained(format!("existing bundle at {}", out.display()))
                .next("choose another directory with --out")),
            };
        }
        if let Err(e) = fs::rename(&tmp, &out) {
            let _ = fs::remove_dir_all(&tmp);
            // Another build may have published the same bundle meanwhile.
            if let Ok(m) = read_json::<BundleManifest>(&out.join(MANIFEST_FILE), Phase::Bundle) {
                if m.bundle_id == manifest.bundle_id {
                    let check = Predictor::load(&out, LoadOptions::cpu().with_self_check(true))?.self_check()?;
                    return Ok(BundleBuildReport {
                        path: self.project.rel(&out),
                        manifest: m,
                        check,
                        already_existed: true,
                    });
                }
            }
            return Err(RustpyError::io(Phase::Bundle, &out, e));
        }
        let mut check = check;
        check.bundle_dir = out.display().to_string();
        Ok(BundleBuildReport {
            path: self.project.rel(&out),
            manifest,
            check,
            already_existed: false,
        })
    }

    /// Bundles under `<project>/bundles`.
    pub fn list_bundles(&self) -> Vec<(PathBuf, Option<BundleManifest>)> {
        let mut v: Vec<(PathBuf, Option<BundleManifest>)> = fs::read_dir(self.project.bundles_dir())
            .map(|it| {
                it.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir() && !p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')))
                    .map(|p| {
                        let m = read_json(&p.join(MANIFEST_FILE), Phase::Bundle).ok();
                        (p, m)
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}
