//! Checks RustPy against frozen scikit-learn reference values
//! (fixtures/reference/goldens.json, produced by tools/reference/make_goldens.py).

use std::path::PathBuf;

use rustpy_contracts::config::Candidate;
use rustpy_contracts::data::SplitName;
use rustpy_workflow::pipeline::RunOptions;
use rustpy_workflow::setup::{BindRequest, init_project};
use rustpy_workflow::{ConfigRequest, Workspace};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn goldens() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(repo().join("fixtures/reference/goldens.json")).unwrap()).unwrap()
}

fn fixture_project() -> (tempfile::TempDir, Workspace) {
    let tmp = tempfile::tempdir().unwrap();
    init_project(tmp.path(), "tabular-binary", false).unwrap();
    let data = tmp.path().join("data/defects_1500.csv");
    std::fs::copy(repo().join("fixtures/reference/defects_1500.csv"), &data).unwrap();
    let ws = Workspace::open(tmp.path()).unwrap();
    ws.bind_data(&BindRequest {
        file: data,
        target: "defect".into(),
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        ..Default::default()
    })
    .unwrap();
    (tmp, ws)
}

#[test]
fn transform_matches_scikit_learn() {
    let g = goldens();
    let (_tmp, ws) = fixture_project();
    let p = ws.prepare(ws.resolve(&ConfigRequest::default()).unwrap(), false).unwrap();
    let valid = ws.split_data(&p, SplitName::Valid).unwrap();
    let names: Vec<String> = serde_json::from_value(g["transform"]["names"].clone()).unwrap();
    let rows: Vec<Vec<f64>> = serde_json::from_value(g["transform"]["valid_rows"].clone()).unwrap();
    let tol = g["transform"]["tolerance"].as_f64().unwrap();
    assert_eq!(rows.len(), valid.x.rows, "validation rows differ from the golden split");
    let ours = &p.fitted.transform.output_names;
    assert_eq!(ours.len(), names.len(), "feature sets differ: {ours:?} vs {names:?}");
    let mut worst = 0f64;
    for (j, name) in names.iter().enumerate() {
        let k = ours
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("missing feature {name}"));
        for (r, row) in rows.iter().enumerate() {
            worst = worst.max((row[j] - f64::from(valid.x.row(r)[k])).abs());
        }
    }
    assert!(worst < tol, "max |Δ| vs scikit-learn = {worst}");
}

#[test]
fn metrics_match_scikit_learn() {
    let g = goldens();
    let m = &g["metrics"];
    let y: Vec<bool> = m["y"].as_array().unwrap().iter().map(|v| v.as_i64() == Some(1)).collect();
    let p: Vec<f64> = m["p"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let ours = rustpy_workflow::metrics::binary_metrics(&p, &y, m["threshold"].as_f64().unwrap());
    let tol = m["tolerance"].as_f64().unwrap();
    for (k, v) in m["values"].as_object().unwrap() {
        let got = ours.get(k).unwrap();
        assert!((got - v.as_f64().unwrap()).abs() < tol, "{k}: rustpy {got} vs sklearn {v}");
    }
}

#[test]
fn linear_baseline_reaches_the_reference_optimum() {
    let g = goldens();
    let o = &g["linear_objective"];
    let (_tmp, ws) = fixture_project();
    let opts = RunOptions {
        candidates: Some(vec![Candidate::Linear]),
        ..Default::default()
    };
    let run = ws.run(&ConfigRequest::default(), &opts, &mut |_| {}).unwrap().runs.remove(0);
    let model = ws.run_weights(&run).unwrap();
    let p = ws.prepare_for_run(&run).unwrap();
    let tr = ws.split_data(&p, SplitName::Train).unwrap();
    let n = tr.y.len() as f64;
    let pos = tr.y.iter().filter(|&&t| t).count() as f64;
    let (w_pos, w_neg) = (n / (2.0 * pos), n / (2.0 * (n - pos)));
    let l = &model.layers[0];
    let mut loss = 0.0;
    for (i, &t) in tr.y.iter().enumerate() {
        let z: f64 = f64::from(l.bias[0])
            + tr.x
                .row(i)
                .iter()
                .zip(&l.weight)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>();
        let li = z.max(0.0) - if t { z } else { 0.0 } + (-z.abs()).exp().ln_1p();
        loss += li * if t { w_pos } else { w_neg };
    }
    let reg: f64 = l.weight.iter().map(|w| f64::from(*w) * f64::from(*w)).sum();
    let ours = loss / n + 0.5 * o["l2"].as_f64().unwrap() * reg;
    let optimum = o["sklearn_optimum"].as_f64().unwrap();
    let gap = (ours - optimum) / optimum;
    assert!(gap > -1e-6, "below the optimum? {ours} vs {optimum}");
    assert!(
        gap <= o["max_relative_gap"].as_f64().unwrap(),
        "objective {ours} is {gap:.4} above the sklearn optimum {optimum}"
    );
}
