//! Building immutable inference bundles from completed runs.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tabsmith_contracts::data::SplitName;
use tabsmith_contracts::evaluation::EvaluationSubject;
use tabsmith_contracts::hash::sha256_file;
use tabsmith_contracts::manifest::{
    BundleEvaluation, BundleManifest, BundleProvenance, CONFORMAL_FILE, EVALUATION_FILE, EXAMPLES_EXPECTED_FILE, EXAMPLES_INPUT_FILE,
    ExampleExpected, ExampleInputs, ExpectedPrediction, FileEntry, IMPORTANCE_FILE, INPUT_SCHEMA_FILE, MANIFEST_FILE, MODEL_CONFIG_FILE,
    MODEL_WEIGHTS_FILE, OUTPUT_SCHEMA_FILE, RuntimeRequirements, TRANSFORM_FILE,
};
use tabsmith_contracts::run::{RunRecord, RunStatus};
use tabsmith_contracts::schema::FieldKind;
use tabsmith_contracts::time::now_rfc3339;
use tabsmith_contracts::transform::{FeatureTransform, TRANSFORM_VERSION};
use tabsmith_contracts::{BUNDLE_FORMAT_VERSION, ErrorCode, Phase, RECIPE_ID, Result, TABSMITH_VERSION, TabsmithError};
use tabsmith_runtime::{CheckReport, LoadOptions, Predictor, Record, Value};

use crate::assurance::{Assurance, DEFAULT_ALPHA};
use crate::fit::FittedArtifacts;
use crate::pipeline::Workspace;
use crate::project::{read_json, write_atomic, write_json};

#[derive(Debug, Clone, Default)]
pub struct BundleOptions {
    /// Output directory (default: `bundles/<run-id>`).
    pub out: Option<PathBuf>,
    /// Build a candidate bundle without a test evaluation (marked `unevaluated-candidate`).
    pub allow_unevaluated: bool,
    /// Miscoverage of the conformal intervals and sets (default 0.1: 90 % coverage).
    pub conformal_alpha: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleBuildReport {
    pub path: String,
    pub manifest: BundleManifest,
    pub check: CheckReport,
    /// True when an identical bundle already existed at the path.
    pub already_existed: bool,
    /// The conformal calibration and the permutation importance written into the
    /// bundle (`conformal.json`, `importance.json`), when they could be computed.
    pub assurance: Assurance,
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
            FeatureTransform::Numeric {
                name, mean, std, impute, ..
            } => {
                let (m, s) = if *std == 1.0 && *mean == 0.0 {
                    (*impute, 1.0)
                } else {
                    (*mean, *std)
                };
                typical.insert(name.clone(), round6(m));
                high.insert(name.clone(), round6(m + 2.0 * s));
                alt.insert(name.clone(), round6(m - s));
                missing.insert(name.clone(), Value::Null);
            }
            FeatureTransform::Categorical {
                name,
                categories,
                missing_slot,
                ..
            } => {
                typical.insert(name.clone(), categories[0].clone());
                high.insert(name.clone(), categories.last().cloned().unwrap_or_default());
                alt.insert(name.clone(), categories.get(1).cloned().unwrap_or_else(|| categories[0].clone()));
                // A missing value is only valid input when it was seen in training or
                // unknown values are ignored; otherwise use a known category.
                let missing_ok = *missing_slot || t.unknown_category == tabsmith_contracts::config::UnknownCategoryPolicy::Ignore;
                missing.insert(
                    name.clone(),
                    if missing_ok {
                        Value::Null
                    } else {
                        Value::from(categories[0].clone())
                    },
                );
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
    let unknown_ok = t.unknown_category == tabsmith_contracts::config::UnknownCategoryPolicy::Ignore;
    if unknown_ok
        && let Some(cat) = fitted
            .input_schema
            .fields
            .iter()
            .find(|f| f.kind == FieldKind::Categorical && f.required)
    {
        let mut r = out[0].clone();
        r.insert(cat.name.clone(), "__tabsmith_unseen_category__");
        out.push(r);
    }
    // Fix numeric categories (e.g. int-coded) so they are sent as the training type.
    for rec in &mut out {
        for field in &fitted.input_schema.fields {
            if field.kind == FieldKind::Categorical
                && field.raw_dtype == tabsmith_contracts::schema::RawDtype::Int
                && let Some(Value::Text(s)) = rec.get(&field.name).cloned()
                && let Ok(i) = s.parse::<i64>()
            {
                rec.insert(field.name.clone(), i);
            }
            if field.kind == FieldKind::Categorical
                && field.raw_dtype == tabsmith_contracts::schema::RawDtype::Bool
                && let Some(Value::Text(s)) = rec.get(&field.name).cloned()
                && let Ok(b) = s.parse::<bool>()
            {
                rec.insert(field.name.clone(), b);
            }
        }
    }
    out
}

/// SHA-256 of a bundle's weight file, as listed in its manifest.
pub fn weights_sha(m: &BundleManifest) -> Option<&str> {
    m.files.iter().find(|f| f.path == MODEL_WEIGHTS_FILE).map(|f| f.sha256.as_str())
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

fn file_entry(dir: &Path, rel: &str) -> Result<FileEntry> {
    let p = dir.join(rel);
    let (sha256, bytes) = sha256_file(&p).map_err(|e| TabsmithError::io(Phase::Bundle, &p, e))?;
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
            return Err(TabsmithError::new(
                ErrorCode::RunStateInvalid,
                Phase::Bundle,
                format!("run {run_id} is {}; only completed runs can be bundled", run.status.as_str()),
            )
            .run(run_id)
            .next("pick a completed run from `tabsmith runs` (or resume this one first)"));
        }
        let model = self.run_weights(&run).map_err(|e| e.in_phase(Phase::Bundle))?;
        let fitted: FittedArtifacts = read_json(
            &self.project.dir("transforms").join(&run.transform_id).join("fitted.json"),
            Phase::Bundle,
        )?;
        // The run's own test evaluation or, failing that, one of an identical model:
        // the same weights trained on the same data version with the same threshold
        // (a deterministic re-run). The evidence is about these exact weights.
        let model_sha = Self::run_model_sha(&run);
        let tests: Vec<_> = self
            .evaluations()
            .into_iter()
            .rev()
            .filter(|e| e.protocol.split == SplitName::Test)
            .collect();
        let test_eval = tests
            .iter()
            .find(|e| matches!(&e.subject, EvaluationSubject::Run { run_id: r } if r == run_id))
            .or_else(|| {
                tests.iter().find(|e| {
                    matches!(&e.subject, EvaluationSubject::Run { .. })
                        && !model_sha.is_empty()
                        && e.model_sha256.as_deref() == Some(model_sha.as_str())
                        && e.metrics.binary().is_none_or(|m| m.threshold == run.threshold)
                        && e.model_snapshot_id
                            .as_deref()
                            .is_some_and(|s| self.same_data_version(s, &run.snapshot_id))
                })
            })
            .cloned();
        if test_eval.is_none() && !opts.allow_unevaluated {
            return Err(TabsmithError::new(
                ErrorCode::EvaluationMissing,
                Phase::Bundle,
                format!("run {run_id} has no held-out test evaluation"),
            )
            .run(run_id)
            .next(format!(
                "lock it as the final candidate with `tabsmith evaluate --run {run_id} --split test`, or build a candidate bundle with --allow-unevaluated"
            )));
        }
        let metrics = run
            .metrics
            .clone()
            .ok_or_else(|| TabsmithError::internal(Phase::Bundle, "completed run without metrics"))?;
        // Conformal calibration and importance need the project's data; a bundle is
        // still built without them when the data is gone, and says so.
        let mut assurance_notes = Vec::new();
        let assurance = match self.assurance(&run, &model, opts.conformal_alpha.unwrap_or(DEFAULT_ALPHA)) {
            Ok(a) => a,
            Err(e) => {
                assurance_notes.push(format!("no conformal calibration and no importance: {}", e.message));
                Assurance::default()
            }
        };
        let out = opts.out.clone().unwrap_or_else(|| self.project.bundles_dir().join(run_id));
        let parent = out.parent().unwrap_or(Path::new(".")).to_path_buf();
        fs::create_dir_all(&parent).map_err(|e| TabsmithError::io(Phase::Bundle, &parent, e))?;
        let tmp = parent.join(format!(
            ".{}.tmp-{}",
            out.file_name().and_then(|n| n.to_str()).unwrap_or("bundle"),
            std::process::id()
        ));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).map_err(|e| TabsmithError::io(Phase::Bundle, &tmp, e))?;
        }
        fs::create_dir_all(tmp.join("examples")).map_err(|e| TabsmithError::io(Phase::Bundle, &tmp, e))?;
        let cleanup = |e: TabsmithError| {
            let _ = fs::remove_dir_all(&tmp);
            e
        };
        let result = (|| -> Result<(BundleManifest, Vec<FileEntry>)> {
            let mut output_schema = fitted.output_schema.clone();
            output_schema.threshold = run.threshold;
            write_json(&tmp.join(INPUT_SCHEMA_FILE), &fitted.input_schema)?;
            write_json(&tmp.join(OUTPUT_SCHEMA_FILE), &output_schema)?;
            write_json(&tmp.join(TRANSFORM_FILE), &fitted.transform)?;
            write_json(&tmp.join(MODEL_CONFIG_FILE), model.config())?;
            // The run's verified weight file, byte for byte: a run and its bundle share
            // the model hash used by the test-set discipline.
            let weights = self.project.run_dir(run_id).join("model").join(MODEL_WEIGHTS_FILE);
            let bytes = fs::read(&weights).map_err(|e| TabsmithError::io(Phase::Bundle, &weights, e))?;
            write_atomic(&tmp.join(MODEL_WEIGHTS_FILE), &bytes)?;
            let mut notes = match &test_eval {
                Some(e) => match &e.subject {
                    EvaluationSubject::Run { run_id: other } if other != run_id => vec![format!(
                        "test evaluation {} of run {other}, which produced identical weights on the same data version",
                        e.evaluation_id
                    )],
                    _ => vec![],
                },
                None => vec!["no held-out test evaluation: treat as a candidate only".into()],
            };
            notes.extend(assurance_notes.iter().cloned());
            let evaluation = BundleEvaluation {
                status: if test_eval.is_some() {
                    crate::auto::EVALUATED.into()
                } else {
                    "unevaluated-candidate".into()
                },
                test: test_eval.clone(),
                valid: metrics.valid.clone(),
                notes,
            };
            write_json(&tmp.join(EVALUATION_FILE), &evaluation)?;
            if let Some(c) = &assurance.conformal {
                write_json(&tmp.join(CONFORMAL_FILE), c)?;
            }
            if let Some(i) = &assurance.importance {
                write_json(&tmp.join(IMPORTANCE_FILE), i)?;
            }
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
                        .map(|_| ExpectedPrediction {
                            label: String::new(),
                            probability: 0.0,
                            probabilities: Vec::new(),
                            value: None,
                        })
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
                tabsmith_version: TABSMITH_VERSION.to_string(),
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
                    // A 0.1.0 runtime rejects a tree config with `monotone` (unknown
                    // field); every other bundle of this version still loads there.
                    min_runtime_version: if model.config().trees.as_ref().is_some_and(|t| !t.monotone.is_empty()) {
                        "0.2.0".into()
                    } else {
                        "0.1.0".into()
                    },
                    transform_version: TRANSFORM_VERSION.into(),
                    model_format: model.config().format.clone(),
                    devices: vec!["cpu".into()],
                    precision: "fp32".into(),
                },
                files: vec![],
                self_contained: true,
                contains_training_data: false,
                contains_optimizer_state: false,
                evaluation_status: evaluation.status.clone(),
            };
            let mut listed = vec![
                INPUT_SCHEMA_FILE,
                OUTPUT_SCHEMA_FILE,
                TRANSFORM_FILE,
                MODEL_CONFIG_FILE,
                MODEL_WEIGHTS_FILE,
                EVALUATION_FILE,
                EXAMPLES_INPUT_FILE,
                EXAMPLES_EXPECTED_FILE,
            ];
            if assurance.conformal.is_some() {
                listed.push(CONFORMAL_FILE);
            }
            if assurance.importance.is_some() {
                listed.push(IMPORTANCE_FILE);
            }
            let write_manifest = |m: &mut BundleManifest, files: &mut Vec<FileEntry>| -> Result<()> {
                files.clear();
                for rel in &listed {
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
                            probability: p.probability().unwrap_or(0.0),
                            probabilities: p.probabilities().map(<[f64]>::to_vec).unwrap_or_default(),
                            value: p.value(),
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
                        assurance: assurance.clone(),
                    })
                }
                _ => Err(TabsmithError::new(
                    ErrorCode::BundleExists,
                    Phase::Bundle,
                    format!(
                        "`{}` already exists and holds a different bundle; bundles are immutable",
                        out.display()
                    ),
                )
                .field(out.display().to_string())
                .retained(format!("existing bundle at {}", out.display()))
                .next("choose another directory with --out")),
            };
        }
        if let Err(e) = fs::rename(&tmp, &out) {
            let _ = fs::remove_dir_all(&tmp);
            // Another build may have published the same bundle meanwhile.
            if let Ok(m) = read_json::<BundleManifest>(&out.join(MANIFEST_FILE), Phase::Bundle)
                && m.bundle_id == manifest.bundle_id
            {
                let check = Predictor::load(&out, LoadOptions::cpu().with_self_check(true))?.self_check()?;
                return Ok(BundleBuildReport {
                    path: self.project.rel(&out),
                    manifest: m,
                    check,
                    already_existed: true,
                    assurance: assurance.clone(),
                });
            }
            return Err(TabsmithError::io(Phase::Bundle, &out, e));
        }
        let mut check = check;
        check.bundle_dir = out.display().to_string();
        Ok(BundleBuildReport {
            path: self.project.rel(&out),
            manifest,
            check,
            already_existed: false,
            assurance,
        })
    }

    /// The newest bundle under `<project>/bundles` that holds exactly these weights
    /// (SHA-256 of the weight file), e.g. the bundle of an identical earlier run.
    pub fn bundle_with_weights(&self, sha: &str) -> Option<(PathBuf, BundleManifest)> {
        self.list_bundles()
            .into_iter()
            .filter_map(|(p, m)| m.map(|m| (p, m)))
            .filter(|(_, m)| weights_sha(m) == Some(sha))
            .max_by(|a, b| a.1.created_at.cmp(&b.1.created_at).then(a.0.cmp(&b.0)))
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
