//! `rustpy train`: the whole workflow in one call — train every candidate, pick
//! one on validation, test it once, build its bundle and, when the project
//! already shipped a model, score that model on the same new test rows.
//!
//! Each step is the ordinary step (`run`, `compare`, `evaluate`, `bundle build`),
//! with the same records and the same refusals; nothing here bypasses a check.

use serde::Serialize;

use rustpy_contracts::data::SplitName;
use rustpy_contracts::evaluation::EvaluationRecord;
use rustpy_contracts::manifest::BundleManifest;
use rustpy_contracts::run::RunStatus;
use rustpy_contracts::{ErrorCode, Result};

use crate::bundle::{BundleBuildReport, BundleOptions};
use crate::pipeline::{Progress, RunGroupResult, RunOptions, Workspace};
use crate::project::ConfigRequest;

/// The model the project shipped before this call, scored on the new test rows.
#[derive(Debug, Clone, Serialize)]
pub struct PreviousModel {
    pub path: String,
    pub bundle_id: String,
    pub run_id: String,
    /// `None` when it could not be evaluated on the new data (see `note`).
    pub evaluation: Option<EvaluationRecord>,
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
    pub previous: Option<PreviousModel>,
    pub notes: Vec<String>,
}

impl Workspace {
    /// Runs the full workflow on the project's bound data (see the module docs).
    pub fn train_auto(
        &self,
        req: &ConfigRequest,
        opts: &RunOptions,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<AutoReport> {
        // The newest bundle shipped so far, before this call adds one.
        let previous = self.latest_bundle();
        let group = self.run(req, opts, progress)?;
        let mut report = AutoReport {
            chosen: None,
            test: None,
            test_skipped: None,
            bundle: None,
            previous: None,
            notes: Vec::new(),
            group,
        };
        if report.group.cancelled || report.group.runs.iter().any(|r| r.status != RunStatus::Completed) {
            report.notes.push("not every run completed; no model was selected (see the runs above)".into());
            return Ok(report);
        }
        let Some(chosen) = report.group.comparison.recommended.clone() else {
            report.notes.push(
                "no candidate met the selection constraints on the validation split; nothing was tested or bundled".into(),
            );
            return Ok(report);
        };
        report.chosen = Some(chosen.clone());
        progress(Progress::Stage(format!("testing {chosen} once on the held-out test split")));
        match self.evaluate_run(&chosen, SplitName::Test, false, false) {
            Ok(e) => report.test = Some(e),
            Err(e) if e.code == ErrorCode::TestSetReuse => {
                report.test_skipped = Some(format!(
                    "{} — the bundle is built as an unevaluated candidate; new data (or a new data.seed) gives a fresh test split",
                    e.message
                ));
            }
            Err(e) => return Err(e),
        }
        progress(Progress::Stage(format!("building the bundle of {chosen}")));
        report.bundle = Some(self.build_bundle(
            &chosen,
            &BundleOptions {
                out: None,
                allow_unevaluated: report.test.is_none(),
            },
        )?);
        // Old vs new on the same new test rows (only meaningful when the new model was tested).
        if let (Some((dir, m)), Some(_)) = (previous, &report.test) {
            if m.provenance.run_id != chosen {
                progress(Progress::Stage(format!("scoring the previous bundle {} on the same test rows", m.bundle_id)));
                let evaluation = self.evaluate_bundle(&dir, SplitName::Test, req, false, false);
                let (evaluation, note) = match evaluation {
                    Ok(e) => (Some(e), None),
                    Err(e) => (None, Some(format!("[{}] {}", e.code, e.message))),
                };
                report.previous = Some(PreviousModel {
                    path: self.project.rel(&dir),
                    bundle_id: m.bundle_id.clone(),
                    run_id: m.provenance.run_id.clone(),
                    evaluation,
                    note,
                });
            }
        }
        Ok(report)
    }

    /// The most recently built bundle in `<project>/bundles` (by manifest time).
    pub fn latest_bundle(&self) -> Option<(std::path::PathBuf, BundleManifest)> {
        self.list_bundles()
            .into_iter()
            .filter_map(|(p, m)| m.map(|m| (p, m)))
            .max_by(|a, b| a.1.created_at.cmp(&b.1.created_at).then(a.0.cmp(&b.0)))
    }
}
