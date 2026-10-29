//! The multiclass and regression recipes end to end: every candidate trains,
//! the test split is evaluated with the task's metrics, the bundle verifies and
//! predicts what the task promises (class probabilities, values).

use std::path::Path;

use tabsmith_contracts::ErrorCode;
use tabsmith_contracts::config::{Candidate, Task};
use tabsmith_contracts::data::SplitName;
use tabsmith_contracts::evaluation::Metrics;
use tabsmith_contracts::run::RunStatus;
use tabsmith_runtime::{LoadOptions, Outcome, Predictor, Record};
use tabsmith_workflow::bundle::BundleOptions;
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::sample::{SampleVariant, sample_target, write_sample_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::{ConfigRequest, Workspace};

fn project(dir: &Path, task: Task, rows: usize) -> Workspace {
    init_project(dir, task.recipe(), false).unwrap();
    write_sample_csv(&dir.join("data/sample.csv"), rows, 5, SampleVariant::Base, task).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/sample.csv"),
        target: sample_target(task).into(),
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        task: Some(task),
        ..Default::default()
    })
    .unwrap();
    ws
}

fn quick() -> ConfigRequest {
    ConfigRequest {
        profile: None,
        overrides: vec![
            "train.linear.epochs=6".into(),
            "train.small-mlp.epochs=6".into(),
            "train.gbdt.epochs=30".into(),
        ],
        override_tables: vec![],
    }
}

fn examples(p: &Predictor) -> Vec<Record> {
    p.bundle().examples.records.iter().map(|v| Record::from_json(v).unwrap()).collect()
}

#[test]
fn multiclass_trains_evaluates_and_predicts_probabilities() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), Task::Multiclass, 2500);
    let all = ws.run(&quick(), &RunOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(all.runs.len(), 3);
    for r in &all.runs {
        assert_eq!(r.status, RunStatus::Completed, "{}", r.run_id);
        let Metrics::Multiclass(m) = &r.metrics.as_ref().unwrap().valid else {
            panic!("{} has no multiclass metrics", r.run_id)
        };
        assert_eq!(m.class_counts.len(), 4);
        assert!(m.balanced_accuracy > 0.3, "{} barely beats chance: {m:?}", r.run_id);
    }
    // Trees are deterministic for several outputs too.
    let gbdt = RunOptions {
        candidates: Some(vec![Candidate::Gbdt]),
        ..Default::default()
    };
    let again = ws.run(&quick(), &gbdt, &mut |_| {}).unwrap().runs.remove(0);
    let first = all.runs.iter().find(|r| r.candidate == Candidate::Gbdt).unwrap();
    assert_eq!(first.artifacts[0].sha256, again.artifacts[0].sha256);
    assert_eq!(first.engine_parity_max_abs_diff, Some(0.0));

    let chosen = &all.runs[0];
    let test = ws.evaluate_run(&chosen.run_id, SplitName::Test, false, false).unwrap();
    assert!(matches!(test.metrics, Metrics::Multiclass(_)));
    let out = tmp.path().join("bundles/mc");
    ws.build_bundle(
        &chosen.run_id,
        &BundleOptions {
            allow_unevaluated: false,
            out: Some(out.clone()),
            conformal_alpha: None,
        },
    )
    .unwrap();
    let p = Predictor::load(&out, LoadOptions::cpu().with_self_check(true)).unwrap();
    let classes = p.output_schema().classes.clone();
    assert_eq!(classes, ["bearing", "none", "overheat", "seal"]);
    for pred in p.predict(&examples(&p)).unwrap() {
        let Outcome::Multiclass {
            probability,
            probabilities,
        } = &pred.outcome
        else {
            panic!("not a multiclass prediction: {pred:?}")
        };
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let best = probabilities.iter().copied().fold(f64::MIN, f64::max);
        assert_eq!(*probability, best);
        assert_eq!(classes[probabilities.iter().position(|&q| q == best).unwrap()], pred.label);
    }
}

#[test]
fn regression_trains_evaluates_and_predicts_values() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), Task::Regression, 2500);
    let cfg = ws.resolve(&quick()).unwrap().config;
    assert_eq!(cfg.metric().as_str(), "rmse", "the regression default metric");
    let all = ws.run(&quick(), &RunOptions::default(), &mut |_| {}).unwrap();
    for r in &all.runs {
        let Metrics::Regression(m) = &r.metrics.as_ref().unwrap().valid else {
            panic!("{} has no regression metrics", r.run_id)
        };
        assert!(m.r2 > 0.5 && m.rmse > 0.0, "{}: {m:?}", r.run_id);
    }
    // The recommendation has the lowest RMSE.
    let best = all
        .comparison
        .rows
        .iter()
        .filter_map(|r| r.metric_value.map(|v| (v, r.run_id.clone())))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    assert_eq!(all.comparison.recommended.as_deref(), Some(best.1.as_str()));

    let out = tmp.path().join("bundles/rg");
    ws.build_bundle(
        &best.1,
        &BundleOptions {
            allow_unevaluated: true,
            out: Some(out.clone()),
            conformal_alpha: None,
        },
    )
    .unwrap();
    let p = Predictor::load(&out, LoadOptions::cpu().with_self_check(true)).unwrap();
    let scale = p.output_schema().target_scale.unwrap();
    assert!(scale.std > 0.0);
    for pred in p.predict(&examples(&p)).unwrap() {
        let v = pred.value().expect("a value");
        assert!(v.is_finite());
        assert_eq!(pred.label, v.to_string());
        assert_eq!(pred.probability(), None);
    }
}

#[test]
fn a_binary_target_under_the_multiclass_recipe_is_refused_with_advice() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    init_project(dir, Task::Multiclass.recipe(), false).unwrap();
    write_sample_csv(&dir.join("data/d.csv"), 600, 1, SampleVariant::Base, Task::Binary).unwrap();
    let ws = Workspace::open(dir).unwrap();
    let e = ws
        .bind_data(&BindRequest {
            file: dir.join("data/d.csv"),
            target: "defect".into(),
            task: Some(Task::Multiclass),
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::TargetInvalid);
    assert!(e.next_action.as_deref().unwrap_or("").contains("binary"), "{e:?}");
}
