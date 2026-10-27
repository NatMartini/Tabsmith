//! `rustpy train`: the whole workflow in one call — train every candidate, pick
//! one on validation, test it once, build its bundle and, when the project
//! already shipped a model, score that model and the new one on the same unseen
//! test rows.
//!
//! Each step is the ordinary step (`run`, `compare`, `evaluate`, `bundle build`),
//! with the same records and the same refusals; nothing here bypasses a check.

use std::path::PathBuf;

use serde::Serialize;

use rustpy_contracts::data::SplitName;
use rustpy_contracts::evaluation::{BinaryMetrics, EvaluationRecord, EvaluationSubject};
use rustpy_contracts::manifest::BundleManifest;
use rustpy_contracts::run::RunStatus;
use rustpy_contracts::{ErrorCode, Result};
use rustpy_runtime::{LoadOptions, Predictor};

use crate::bundle::{BundleBuildReport, BundleOptions, weights_sha};
use crate::pipeline::{Progress, RunGroupResult, RunOptions, Workspace};
use crate::project::ConfigRequest;

/// Evaluation status of a bundle with a held-out test evaluation.
pub const EVALUATED: &str = "evaluated";

/// The model the project shipped before this call, scored on the new test rows.
#[derive(Debug, Clone, Serialize)]
pub struct PreviousModel {
    pub path: String,
    pub bundle_id: String,
    pub run_id: String,
    /// The previous bundle on the new test rows it never saw (`None` when it could
    /// not be evaluated on the new data, see `note`).
    pub evaluation: Option<EvaluationRecord>,
    /// The new model on exactly the rows of `evaluation`: the like-for-like comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_model_on_same_rows: Option<BinaryMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoReport {
    pub group: RunGroupResult,
    /// Run chosen on the validation split (`None`: nothing completed or met the constraints).
    pub chosen: Option<String>,
    /// Its one test evaluation.
    pub test: Option<EvaluationRecord>,
    /// Why the test split was not used (e.g. already used by another model of this data version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_skipped: Option<String>,
    pub bundle: Option<BundleBuildReport>,
    /// The chosen run reproduced the weights of an earlier run bit for bit (same
    /// data and settings): that run's id. Its evaluation and bundle are reused.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identical_to: Option<String>,
    pub previous: Option<PreviousModel>,
    pub notes: Vec<String>,
}

impl Workspace {
    /// Runs the full workflow on the project's bound data (see the module docs).
    pub fn train_auto(&self, req: &ConfigRequest, opts: &RunOptions, progress: &mut dyn FnMut(Progress)) -> Result<AutoReport> {
        // The bundle `predict` used before this call adds one.
        let previous = self.latest_bundle();
        let group = self.run(req, opts, progress)?;
        let mut report = AutoReport {
            chosen: None,
            test: None,
            test_skipped: None,
            bundle: None,
            identical_to: None,
            previous: None,
            notes: Vec::new(),
            group,
        };
        if report.group.cancelled || report.group.runs.iter().any(|r| r.status != RunStatus::Completed) {
            report
                .notes
                .push("not every run completed; no model was selected (see the runs above)".into());
            return Ok(report);
        }
        let Some(chosen) = report.group.comparison.recommended.clone() else {
            report
                .notes
                .push("no candidate met the selection constraints on the validation split; nothing was tested or bundled".into());
            return Ok(report);
        };
        report.chosen = Some(chosen.clone());
        let chosen_sha = Self::run_model_sha(&self.load_run(&chosen)?);
        progress(Progress::Stage(format!("testing {chosen} once on the held-out test split")));
        match self.evaluate_run(&chosen, SplitName::Test, false, false) {
            Ok(e) => {
                // An identical model (same weights, same rows) gets its earlier evaluation back.
                if let EvaluationSubject::Run { run_id } = &e.subject
                    && run_id != &chosen
                {
                    report.identical_to = Some(run_id.clone());
                }
                report.test = Some(e);
            }
            Err(e) if e.code == ErrorCode::TestSetReuse => {
                report.test_skipped = Some(format!(
                    "{} — the bundle is built as an unevaluated candidate; only new rows give a fresh test split",
                    e.message
                ));
            }
            Err(e) => return Err(e),
        }
        // A bundle with exactly these weights already exists (e.g. `rustpy train`
        // again on unchanged data): keep it instead of building a duplicate.
        let existing = self
            .bundle_with_weights(&chosen_sha)
            .filter(|(_, m)| report.test.is_none() || m.evaluation_status == EVALUATED);
        if let Some((dir, m)) = existing {
            let check = Predictor::load(&dir, LoadOptions::cpu().with_self_check(true))?.self_check()?;
            report.notes.push(format!(
                "{chosen} has the same weights as bundle {} (run {}): nothing changed since then, so that bundle is kept",
                self.project.rel(&dir),
                m.provenance.run_id
            ));
            report.bundle = Some(BundleBuildReport {
                path: self.project.rel(&dir),
                manifest: m,
                check,
                already_existed: true,
            });
        } else {
            progress(Progress::Stage(format!("building the bundle of {chosen}")));
            report.bundle = Some(self.build_bundle(
                &chosen,
                &BundleOptions {
                    out: None,
                    allow_unevaluated: report.test.is_none(),
                },
            )?);
        }
        if report.test.is_none()
            && let Some((dir, m)) = &previous
            && m.evaluation_status == EVALUATED
        {
            report.notes.push(format!(
                "`rustpy predict` and `rustpy serve` keep using the evaluated bundle {}; pass --bundle {} to use the candidate",
                self.project.rel(dir),
                report.bundle.as_ref().map(|b| b.path.as_str()).unwrap_or("<dir>")
            ));
        }
        // Old vs new on the same unseen test rows (only meaningful when the new model was tested).
        if let (Some((dir, m)), Some(_), Some(new_bundle)) = (previous, &report.test, &report.bundle)
            && weights_sha(&m) != Some(chosen_sha.as_str())
        {
            progress(Progress::Stage(format!(
                "scoring the previous bundle {} on the same test rows",
                m.bundle_id
            )));
            let new_dir = self.project.root.join(&new_bundle.path);
            let evaluation = self.evaluate_bundle_against(&dir, Some(&new_dir), SplitName::Test, req, false, false);
            let (evaluation, new_model_on_same_rows, note) = match evaluation {
                Ok((e, new)) => (Some(e), new, None),
                Err(e) => (None, None, Some(format!("[{}] {}", e.code, e.message))),
            };
            report.previous = Some(PreviousModel {
                path: self.project.rel(&dir),
                bundle_id: m.bundle_id.clone(),
                run_id: m.provenance.run_id.clone(),
                evaluation,
                new_model_on_same_rows,
                note,
            });
        }
        Ok(report)
    }

    /// The bundle `predict` and `serve` use by default: the newest bundle in
    /// `<project>/bundles` (by manifest time) with a held-out test evaluation or,
    /// when none has one, the newest bundle.
    pub fn latest_bundle(&self) -> Option<(PathBuf, BundleManifest)> {
        let all: Vec<(PathBuf, BundleManifest)> = self.list_bundles().into_iter().filter_map(|(p, m)| m.map(|m| (p, m))).collect();
        let newest = |evaluated_only: bool| {
            all.iter()
                .filter(|(_, m)| !evaluated_only || m.evaluation_status == EVALUATED)
                .max_by(|a, b| a.1.created_at.cmp(&b.1.created_at).then(a.0.cmp(&b.0)))
                .cloned()
        };
        newest(true).or_else(|| newest(false))
    }

    /// Unevaluated candidate bundles newer than `than` (skipped by [`Self::latest_bundle`]).
    pub fn newer_candidates(&self, than: &BundleManifest) -> Vec<PathBuf> {
        self.list_bundles()
            .into_iter()
            .filter_map(|(p, m)| m.map(|m| (p, m)))
            .filter(|(_, m)| m.evaluation_status != EVALUATED && m.created_at >= than.created_at && m.bundle_id != than.bundle_id)
            .map(|(p, _)| p)
            .collect()
    }
}
