//! The `gbdt` candidate end to end: bit-identical when retrained, resumable exactly
//! after a cancel, scored identically by trainer and runtime, bundled and predicted.

use std::path::Path;

use tabsmith_contracts::config::Candidate;
use tabsmith_contracts::model::TREES_FORMAT;
use tabsmith_contracts::run::RunStatus;
use tabsmith_runtime::{LoadOptions, Predictor, Record};
use tabsmith_workflow::bundle::BundleOptions;
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::sample::{SampleVariant, write_defects_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::{ConfigRequest, RunControl, Workspace};

fn project(dir: &Path) -> Workspace {
    init_project(dir, "tabular-binary", false).unwrap();
    write_defects_csv(&dir.join("data/defects.csv"), 2000, 11, SampleVariant::Base).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/defects.csv"),
        target: "defect".into(),
        positive: Some("1".into()),
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        ..Default::default()
    })
    .unwrap();
    ws
}

fn config() -> ConfigRequest {
    ConfigRequest {
        profile: None,
        overrides: vec![
            "train.gbdt.epochs=40".into(),
            "train.gbdt.early_stopping_patience=10".into(),
            "train.gbdt.checkpoint_every_epochs=5".into(),
            "train.gbdt.checkpoint_min_interval_seconds=0".into(),
        ],
        override_tables: vec![],
    }
}

fn gbdt(control: RunControl) -> RunOptions {
    RunOptions {
        candidates: Some(vec![Candidate::Gbdt]),
        control,
        ..Default::default()
    }
}

#[test]
fn gbdt_is_reproducible_resumable_and_bundled() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path());
    let a = ws.run(&config(), &gbdt(RunControl::default()), &mut |_| {}).unwrap().runs.remove(0);
    let b = ws.run(&config(), &gbdt(RunControl::default()), &mut |_| {}).unwrap().runs.remove(0);
    assert_eq!(a.status, RunStatus::Completed);
    assert_eq!(a.artifacts[0].sha256, b.artifacts[0].sha256, "retraining changed the trees");
    assert_eq!(a.history, b.history);
    assert_eq!(a.engine_parity_max_abs_diff, Some(0.0), "trainer and runtime disagree");
    assert_eq!(a.device_plan.backend, "tabsmith-gbdt");
    assert_eq!(a.device_receipt.as_ref().map(|r| r.matches_plan), Some(true));
    let valid = &a.metrics.as_ref().unwrap().valid;
    let (ba, auc) = (valid.get("balanced_accuracy").unwrap(), valid.get("roc_auc").unwrap());
    assert!(ba > 0.65 && auc > 0.75, "{valid:?}");

    // Cancel after 12 rounds, resume in place: the same trees as the uninterrupted run.
    let control = RunControl {
        cancel_after_steps: Some(12),
        ..Default::default()
    };
    let part = ws.run(&config(), &gbdt(control), &mut |_| {}).unwrap().runs.remove(0);
    assert_eq!(part.status, RunStatus::Cancelled);
    assert_eq!(part.steps_done, 12);
    let done = ws
        .resume(&part.run_id, false, &ConfigRequest::default(), &RunControl::default(), &mut |_| {})
        .unwrap();
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!(done.history, a.history, "history differs after resume");
    assert_eq!(done.artifacts[0].sha256, a.artifacts[0].sha256);

    // A warm start continues network weights only.
    let e = ws
        .resume(&a.run_id, true, &ConfigRequest::default(), &RunControl::default(), &mut |_| {})
        .unwrap_err();
    assert_eq!(e.code, tabsmith_contracts::ErrorCode::ResumeIncompatible);

    // Bundled, verified and predicted by the runtime.
    let out = tmp.path().join("bundles/gbdt");
    ws.build_bundle(
        &a.run_id,
        &BundleOptions {
            allow_unevaluated: true,
            out: Some(out.clone()),
        },
    )
    .unwrap();
    let p = Predictor::load(&out, LoadOptions::cpu().with_self_check(true)).unwrap();
    assert_eq!(p.manifest().runtime.model_format, TREES_FORMAT);
    let examples: Vec<Record> = p.bundle().examples.records.iter().map(|v| Record::from_json(v).unwrap()).collect();
    for pred in p.predict(&examples).unwrap() {
        assert!((0.0..=1.0).contains(&pred.probability().unwrap()));
    }
}
