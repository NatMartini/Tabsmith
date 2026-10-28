//! End-to-end checks of the CPU workflow: data -> split -> fitted transform ->
//! two candidates -> comparison -> test evaluation -> bundle -> runtime predictor.

use std::path::Path;

use tabsmith_contracts::ErrorCode;
use tabsmith_contracts::data::SplitName;
use tabsmith_contracts::run::{ResumeLevel, RunStatus};
use tabsmith_runtime::{LoadOptions, Predictor, Record, Value};
use tabsmith_workflow::bundle::BundleOptions;
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::sample::{SampleVariant, write_defects_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::{ConfigRequest, Workspace};

fn quick() -> ConfigRequest {
    ConfigRequest {
        profile: None,
        overrides: vec![
            "train.linear.epochs=6".into(),
            "train.small-mlp.epochs=6".into(),
            "train.small-mlp.hidden=[16]".into(),
        ],
        override_tables: vec![],
    }
}

fn project(dir: &Path, rows: usize) -> Workspace {
    init_project(dir, "tabular-binary", false).unwrap();
    write_defects_csv(&dir.join("data/defects.csv"), rows, 7, SampleVariant::Base).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/defects.csv"),
        target: "defect".into(),
        positive: None,
        split: None,
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        exclude: vec![],
        task: None,
    })
    .unwrap();
    ws
}

#[test]
fn full_cpu_loop_with_consistent_preprocessing() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 3000);
    let req = quick();

    // Plan: inferred roles and dropped features are visible.
    let plan = ws.plan(&req).unwrap();
    let excluded: Vec<&str> = plan.excluded().iter().map(|(n, _)| *n).collect();
    assert!(excluded.contains(&"inspector_note"), "{excluded:?}");
    assert!(
        plan.warnings.iter().any(|w| w.contains("calibration_offset")),
        "{:?}",
        plan.warnings
    );

    // Run the default candidates: linear, small-mlp and gbdt.
    let result = ws.run(&req, &RunOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(result.runs.len(), 3);
    assert!(result.runs.iter().all(|r| r.status == RunStatus::Completed));
    for r in &result.runs {
        let parity = r.engine_parity_max_abs_diff.unwrap();
        assert!(parity < 1e-4, "engine/runtime parity {parity}");
        let auc = r.metrics.as_ref().unwrap().valid.get("roc_auc").unwrap();
        assert!(auc > 0.6, "valid AUC {auc} for {}", r.run_id);
    }
    let best = result.comparison.recommended.clone().expect("a recommendation");

    // R1: a second run reuses snapshot, split and transform.
    let again = ws
        .run(
            &req,
            &RunOptions {
                candidates: Some(vec![tabsmith_contracts::config::Candidate::Linear]),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(again.reused, vec!["snapshot", "split", "transform"]);

    // Test split: first use is recorded; a different run is refused; same run is idempotent.
    let eval = ws.evaluate_run(&best, SplitName::Test, false, false).unwrap();
    assert_eq!(eval.test_use_index, Some(1));
    let same = ws.evaluate_run(&best, SplitName::Test, false, false).unwrap();
    assert_eq!(same.evaluation_id, eval.evaluation_id);
    let other = result.runs.iter().find(|r| r.run_id != best).unwrap();
    let refused = ws.evaluate_run(&other.run_id, SplitName::Test, false, false).unwrap_err();
    assert_eq!(refused.code, ErrorCode::TestSetReuse);
    // Bundle requires a test evaluation of its weights (`other` has none yet).
    let unevaluated = ws.build_bundle(&other.run_id, &BundleOptions::default()).unwrap_err();
    assert_eq!(unevaluated.code, ErrorCode::EvaluationMissing);
    let reused = ws.evaluate_run(&other.run_id, SplitName::Test, true, false).unwrap();
    assert_eq!(reused.test_use_index, Some(2));
    // A deterministic re-run has the same weights: it shares their evaluation.
    let linear_eval = if other.candidate == tabsmith_contracts::config::Candidate::Linear {
        &reused
    } else {
        &eval
    };
    let twin = ws
        .build_bundle(
            &again.runs[0].run_id,
            &BundleOptions {
                out: Some(tmp.path().join("twin")),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        twin.manifest.provenance.evaluation_id.as_deref(),
        Some(linear_eval.evaluation_id.as_str())
    );

    let built = ws.build_bundle(&best, &BundleOptions::default()).unwrap();
    assert!(built.check.passed);
    let bundle_dir = tmp.path().join(&built.path);
    // Immutable: rebuilding yields the same id; a different run into the same dir is refused.
    let rebuilt = ws.build_bundle(&best, &BundleOptions::default()).unwrap();
    assert!(rebuilt.already_existed);
    assert_eq!(rebuilt.manifest.bundle_id, built.manifest.bundle_id);

    // Training-side probabilities == runtime Predictor on raw records (new instance,
    // records built from the raw snapshot rows, columns in shuffled order).
    let run = ws.load_run(&best).unwrap();
    let prepared = ws.prepare_for_run(&run).unwrap();
    let test = ws.split_data(&prepared, SplitName::Test).unwrap();
    let model = ws.run_weights(&run).unwrap();
    let train_side: Vec<f64> = model
        .outputs(&test.x)
        .data
        .into_iter()
        .map(|z| tabsmith_runtime::model::sigmoid(f64::from(z)))
        .collect();
    let predictor = Predictor::load(&bundle_dir, LoadOptions::cpu()).unwrap();
    let table = prepared.table.take(&test.rows);
    let mut names: Vec<&str> = table.names();
    names.reverse();
    let records: Vec<Record> = (0..table.n_rows())
        .map(|i| {
            let mut r = Record::new();
            for n in &names {
                let v = match table.column(n).unwrap().get(i) {
                    tabsmith_runtime::ValueRef::Null => Value::Null,
                    tabsmith_runtime::ValueRef::Bool(b) => Value::Bool(b),
                    tabsmith_runtime::ValueRef::Int(x) => Value::Int(x),
                    tabsmith_runtime::ValueRef::Float(x) => Value::Float(x),
                    tabsmith_runtime::ValueRef::Text(s) => Value::Text(s.to_string()),
                };
                r.insert(*n, v);
            }
            r
        })
        .collect();
    let app_side = predictor.predict(&records).unwrap();
    let max_diff = train_side
        .iter()
        .zip(&app_side)
        .map(|(a, b)| (a - b.probability().unwrap()).abs())
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-9, "train/app probability mismatch {max_diff}");
    // Same metrics as the recorded evaluation.
    let y: Vec<bool> = test.y.binary().unwrap().to_vec();
    let m = tabsmith_workflow::metrics::binary_metrics(
        &app_side.iter().map(|p| p.probability().unwrap()).collect::<Vec<_>>(),
        &y,
        run.threshold,
    );
    assert!((m.balanced_accuracy - eval.metrics.get("balanced_accuracy").unwrap()).abs() < 1e-12);

    // Missing field in an app record is a precise schema error.
    let mut r = records[0].clone();
    let mut bad = Record::new();
    for n in r.names().filter(|n| *n != "vibration_rms") {
        bad.insert(n, r.get(n).unwrap().clone());
    }
    let e = predictor.predict_one(&bad).unwrap_err();
    assert_eq!(e.code, ErrorCode::SchemaMissingColumn);
    assert_eq!(e.field.as_deref(), Some("vibration_rms"));
    // Unknown category with the default policy encodes as all zeros (no error).
    r.insert("supplier", "S99");
    assert!(predictor.predict_one(&r).is_ok());
    // Strict types: a numeric field sent as text is rejected.
    r.insert("temp_mean", "61.0");
    assert_eq!(predictor.predict_one(&r).unwrap_err().code, ErrorCode::SchemaTypeMismatch);

    // Report renders.
    let html = ws.write_report().unwrap();
    assert!(std::fs::read_to_string(html).unwrap().contains(&best));
}

#[test]
fn exact_resume_matches_uninterrupted_training_bitwise() {
    use std::sync::atomic::Ordering;
    use tabsmith_workflow::train::{RunControl, StartFrom, TrainOutcome, train, weights_hash};

    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1500);
    let resolved = ws.resolve(&quick()).unwrap();
    let p = ws.prepare(resolved, false).unwrap();
    let tr = ws.split_data(&p, SplitName::Train).unwrap();
    let va = ws.split_data(&p, SplitName::Valid).unwrap();
    let cfg = p.resolved.config.train.small_mlp.clone();
    let mcfg = tabsmith_workflow::train::model_config(tabsmith_contracts::config::Candidate::SmallMlp, p.fitted.transform.output_dim, &cfg);
    let key = tabsmith_workflow::train::ResumeKey {
        snapshot_id: p.snapshot.snapshot_id.clone(),
        split_id: p.split.split_id.clone(),
        transform_id: p.fitted.transform_id.clone(),
        candidate: tabsmith_contracts::config::Candidate::SmallMlp,
        train_config: cfg.clone(),
        seed: 7,
        threshold: 0.5,
        model_config: mcfg,
        engine: tabsmith_workflow::device::ENGINE.into(),
        tabsmith_version: tabsmith_contracts::TABSMITH_VERSION.into(),
        n_train: tr.x.rows,
        platform: tabsmith_workflow::train::platform_fingerprint(),
        recipe_constants: tabsmith_workflow::train::RECIPE_CONSTANTS.to_string(),
    };
    let inputs = |dir: &str| tabsmith_workflow::train::TrainInputs {
        candidate: tabsmith_contracts::config::Candidate::SmallMlp,
        cfg: &cfg,
        seed: 7,
        metric: tabsmith_contracts::config::SelectionMetric::BalancedAccuracy,
        max_train_seconds: 0,
        output: &p.fitted.output_schema,
        x_train: &tr.x,
        y_train: &tr.y,
        x_valid: &va.x,
        y_valid: &va.y,
        checkpoint_dir: tmp.path().join(dir),
        key: key.clone(),
        device: tabsmith_workflow::device::EngineDevice::Cpu,
    };

    let full = match train(&inputs("full"), &RunControl::default(), StartFrom::Scratch, &mut |_| {}).unwrap() {
        TrainOutcome::Completed { best, state, .. } => (weights_hash(&best), state),
        TrainOutcome::Cancelled { .. } => panic!("not cancelled"),
    };

    // Cancel after epoch 2 (stops at the next step boundary with a checkpoint).
    let control = RunControl::default();
    let flag = control.cancel.clone();
    let first = train(&inputs("part"), &control, StartFrom::Scratch, &mut |s| {
        if s.epoch == 2 {
            flag.store(true, Ordering::SeqCst);
        }
    })
    .unwrap();
    let ck = match first {
        TrainOutcome::Cancelled { checkpoint, state } => {
            assert_eq!(state.epoch, 2);
            checkpoint
        }
        TrainOutcome::Completed { .. } => panic!("should have been cancelled"),
    };
    let resumed = match train(&inputs("part"), &RunControl::default(), StartFrom::Checkpoint(ck), &mut |_| {}).unwrap() {
        TrainOutcome::Completed { best, state, .. } => (weights_hash(&best), state),
        TrainOutcome::Cancelled { .. } => panic!("not cancelled"),
    };
    assert_eq!(full.0, resumed.0, "final weights differ after resume");
    assert_eq!(full.1.history, resumed.1.history, "training history differs after resume");
    assert_eq!(full.1.step, resumed.1.step);

    // A changed configuration is refused for R2 with the differences listed.
    let mut other_key = key.clone();
    other_key.train_config.learning_rate *= 2.0;
    let ck2 = tabsmith_workflow::train::latest_checkpoint(&tmp.path().join("part")).unwrap();
    let device = burn_device();
    let e = tabsmith_workflow::train::load_checkpoint(&ck2, &other_key, &device).err().unwrap();
    assert_eq!(e.code, ErrorCode::ResumeIncompatible);
    assert!(e.message.contains("training configuration"));
}

fn burn_device() -> burn::tensor::Device {
    burn::tensor::Device::flex().autodiff()
}

#[test]
fn resume_and_warm_start_through_workspace() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1500);
    let req = quick();
    let opts = RunOptions {
        candidates: Some(vec![tabsmith_contracts::config::Candidate::SmallMlp]),
        ..Default::default()
    };
    let first = ws.run(&req, &opts, &mut |_| {}).unwrap();
    let run_id = first.runs[0].run_id.clone();
    // Completed runs cannot be resumed exactly...
    let e = ws.resume(&run_id, false, &req, &Default::default(), &mut |_| {}).unwrap_err();
    assert_eq!(e.code, ErrorCode::RunStateInvalid);
    // ...but can seed a warm start, which is labelled as such.
    let warm = ws.resume(&run_id, true, &req, &Default::default(), &mut |_| {}).unwrap();
    assert_eq!(warm.status, RunStatus::Completed);
    let l = warm.lineage.unwrap();
    assert_eq!(l.level, ResumeLevel::WarmStart);
    assert_eq!(l.parent_run_id, run_id);
}

#[test]
fn data_update_creates_new_snapshot_and_old_bundle_stays_usable() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 2000);
    let req = quick();
    let opts = RunOptions {
        candidates: Some(vec![tabsmith_contracts::config::Candidate::Linear]),
        ..Default::default()
    };
    let v1 = ws.run(&req, &opts, &mut |_| {}).unwrap();
    let r1 = v1.runs[0].run_id.clone();
    ws.evaluate_run(&r1, SplitName::Test, false, false).unwrap();
    let b1 = ws.build_bundle(&r1, &BundleOptions::default()).unwrap();

    // New data arrives at the same path: a new snapshot is created automatically.
    write_defects_csv(&tmp.path().join("data/defects.csv"), 2500, 8, SampleVariant::Shifted).unwrap();
    let v2 = ws.run(&req, &opts, &mut |_| {}).unwrap();
    assert_ne!(v2.group.snapshot_id, v1.group.snapshot_id);
    assert!(v2.reused.is_empty(), "nothing can be reused for new data: {:?}", v2.reused);
    // Runs on different snapshots are flagged as not comparable.
    let cmp = ws.compare(&[r1.clone(), v2.runs[0].run_id.clone()]).unwrap();
    assert!(cmp.rows.iter().any(|r| !r.comparable));
    // The old bundle is still loadable and can be re-evaluated on the new data
    // (it has never seen supplier S7: unknown category encodes as zeros).
    let e_old = ws
        .evaluate_bundle(&tmp.path().join(&b1.path), SplitName::Valid, &req, false, false)
        .unwrap();
    assert_eq!(e_old.protocol.snapshot_id, v2.group.snapshot_id);
    // Old runs on the old snapshot remain loadable (the snapshot copy is immutable).
    let old = ws.load_run(&r1).unwrap();
    assert!(ws.prepare_for_run(&old).is_ok());
}
