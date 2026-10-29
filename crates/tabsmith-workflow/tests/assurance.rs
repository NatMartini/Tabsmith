//! A bundle carries its conformal calibration and its permutation importance, the
//! runtime reads them, and predictions carry an interval or a prediction set.

use std::path::Path;

use tabsmith_contracts::config::{Candidate, Task};
use tabsmith_contracts::manifest::{CONFORMAL_FILE, IMPORTANCE_FILE};
use tabsmith_runtime::{LoadOptions, Outcome, Predictor, Record};
use tabsmith_workflow::bundle::BundleOptions;
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::sample::{SampleVariant, write_sample_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::{ConfigRequest, RunControl, Workspace};

fn project(dir: &Path, task: Task, target: &str) -> Workspace {
    init_project(dir, "tabular-binary", false).unwrap();
    write_sample_csv(&dir.join("data/sample.csv"), 2000, 11, SampleVariant::Base, task).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/sample.csv"),
        target: target.into(),
        positive: (task == Task::Binary).then(|| "1".into()),
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        task: Some(task),
        ..Default::default()
    })
    .unwrap();
    ws
}

fn quick(candidate: &str) -> ConfigRequest {
    ConfigRequest {
        profile: None,
        overrides: vec![
            format!("train.{candidate}.epochs=8"),
            format!("train.{candidate}.early_stopping_patience=4"),
            format!("train.{candidate}.checkpoint_every_epochs=100"),
        ],
        override_tables: vec![],
    }
}

fn build(ws: &Workspace, cfg: &ConfigRequest, candidate: Candidate, out: &Path) -> Predictor {
    let opts = RunOptions {
        candidates: Some(vec![candidate]),
        ..Default::default()
    };
    let run = ws.run(cfg, &opts, &mut |_| {}).unwrap().runs.remove(0);
    let report = ws
        .build_bundle(
            &run.run_id,
            &BundleOptions {
                allow_unevaluated: true,
                out: Some(out.to_path_buf()),
                conformal_alpha: None,
            },
        )
        .unwrap();
    assert!(
        report.assurance.conformal.is_some() && report.assurance.importance.is_some(),
        "{report:?}"
    );
    assert!(out.join(CONFORMAL_FILE).is_file() && out.join(IMPORTANCE_FILE).is_file());
    assert!(report.manifest.files.iter().any(|f| f.path == CONFORMAL_FILE));
    Predictor::load(out, LoadOptions::cpu().with_self_check(true)).unwrap()
}

fn examples(p: &Predictor) -> Vec<Record> {
    p.bundle().examples.records.iter().map(|r| Record::from_json(r).unwrap()).collect()
}

#[test]
fn a_regression_bundle_carries_intervals_and_importance() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), Task::Regression, "remaining_hours");
    let p = build(&ws, &quick("linear"), Candidate::Linear, &tmp.path().join("bundles/reg"));
    let c = p.bundle().conformal.as_ref().unwrap();
    assert_eq!((c.alpha, c.calibration_split.as_str()), (0.1, "valid"));
    assert!(c.calibration_coverage >= 0.9 && c.quantile > 0.0, "{c:?}");
    assert_eq!(c.method, "split-conformal/absolute-residual");
    for pred in p.predict(&examples(&p)).unwrap() {
        let Outcome::Regression { value } = pred.outcome else {
            panic!("regression")
        };
        let [lo, hi] = pred.interval.expect("an interval");
        assert!(lo <= value && value <= hi && (hi - lo - 2.0 * c.quantile).abs() < 1e-9);
        assert!(pred.prediction_set.is_none());
    }
    let imp = p.bundle().importance.as_ref().unwrap();
    assert_eq!((imp.split.as_str(), imp.repeats), ("valid", 3));
    assert!(imp.fields.windows(2).all(|w| w[0].importance >= w[1].importance), "sorted");
    let inputs: Vec<&str> = p.input_schema().fields.iter().map(|f| f.name.as_str()).collect();
    assert!(imp.fields.iter().all(|f| inputs.contains(&f.name.as_str())));
    assert!(imp.fields[0].importance > 0.0, "{:?}", imp.fields[0]);
    // The JSON form of a prediction shows the interval.
    let json = serde_json::to_value(p.predict_one(&examples(&p)[0]).unwrap()).unwrap();
    assert!(json["interval"].as_array().is_some_and(|a| a.len() == 2), "{json}");
}

#[test]
fn a_classification_bundle_carries_prediction_sets() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), Task::Binary, "defect");
    let p = build(&ws, &quick("gbdt"), Candidate::Gbdt, &tmp.path().join("bundles/bin"));
    let c = p.bundle().conformal.as_ref().unwrap();
    assert_eq!(c.method, "split-conformal/lac");
    assert!(c.calibration_coverage >= 0.9 && (0.0..=1.0).contains(&c.quantile), "{c:?}");
    let schema = p.output_schema().clone();
    let preds = p.predict(&examples(&p)).unwrap();
    for pred in &preds {
        let set = pred.prediction_set.as_ref().expect("a set");
        assert!(
            set.iter().all(|l| *l == schema.negative_label || *l == schema.positive_label),
            "{set:?}"
        );
        assert!(pred.interval.is_none());
        let Outcome::Binary { probability, .. } = pred.outcome else {
            panic!("binary")
        };
        // The set is exactly the labels whose probability clears 1 - quantile.
        assert_eq!(set.contains(&schema.positive_label), probability >= 1.0 - c.quantile);
        assert_eq!(set.contains(&schema.negative_label), 1.0 - probability >= 1.0 - c.quantile);
    }
    assert!(preds.iter().any(|p| !p.prediction_set.as_ref().unwrap().is_empty()));
    // A bundle built with another alpha gets another quantile.
    let run = ws
        .run(
            &quick("gbdt"),
            &RunOptions {
                candidates: Some(vec![Candidate::Gbdt]),
                control: RunControl::default(),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
        .runs
        .remove(0);
    let strict = ws
        .build_bundle(
            &run.run_id,
            &BundleOptions {
                allow_unevaluated: true,
                out: Some(tmp.path().join("bundles/strict")),
                conformal_alpha: Some(0.02),
            },
        )
        .unwrap();
    let s = strict.assurance.conformal.unwrap();
    assert!(s.quantile >= c.quantile && s.alpha == 0.02, "{s:?} vs {c:?}");
}
