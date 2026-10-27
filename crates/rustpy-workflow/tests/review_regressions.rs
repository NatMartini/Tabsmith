//! Regression tests for defects found by the adversarial review (ids in comments).

use std::fs;
use std::path::Path;

use rustpy_contracts::config::Candidate;
use rustpy_contracts::data::SplitName;
use rustpy_contracts::run::{ResumeLevel, RunStatus};
use rustpy_contracts::ErrorCode;
use rustpy_runtime::{LoadOptions, Predictor, Record, Value};
use rustpy_workflow::bundle::BundleOptions;
use rustpy_workflow::data::{read_table, ReadOptions};
use rustpy_workflow::pipeline::RunOptions;
use rustpy_workflow::sample::{write_defects_csv, SampleVariant};
use rustpy_workflow::setup::{init_project, BindRequest};
use rustpy_workflow::{ConfigRequest, RunControl, Workspace};

fn quick(extra: &[&str]) -> ConfigRequest {
    let mut overrides: Vec<String> = vec![
        "train.linear.epochs=5".into(),
        "train.small-mlp.epochs=5".into(),
        "train.small-mlp.hidden=[12]".into(),
        "train.small-mlp.checkpoint_min_interval_seconds=0".into(),
    ];
    overrides.extend(extra.iter().map(|s| s.to_string()));
    ConfigRequest {
        profile: None,
        overrides,
        override_tables: vec![],
    }
}

fn linear() -> RunOptions {
    RunOptions {
        candidates: Some(vec![Candidate::Linear]),
        ..Default::default()
    }
}

fn mlp() -> RunOptions {
    RunOptions {
        candidates: Some(vec![Candidate::SmallMlp]),
        ..Default::default()
    }
}

fn project(dir: &Path, rows: usize, id: bool) -> Workspace {
    init_project(dir, "tabular-binary", false).unwrap();
    write_defects_csv(&dir.join("data/defects.csv"), rows, 11, SampleVariant::Base).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/defects.csv"),
        target: "defect".into(),
        positive: Some("1".into()),
        group: Some("device_id".into()),
        id: id.then(|| "unit_id".to_string()),
        ..Default::default()
    })
    .unwrap();
    ws
}

fn csv(dir: &Path, name: &str, text: &str) -> std::path::PathBuf {
    let p = dir.join(name);
    fs::write(&p, text).unwrap();
    p
}

/// data-01: blank lines must never become rows of missing values.
#[test]
fn blank_lines_do_not_create_phantom_rows() {
    let d = tempfile::tempdir().unwrap();
    let opts = |p: &Path| ReadOptions::from_config(p, &Default::default()).unwrap();
    let mid = csv(d.path(), "mid.csv", "x,y\n1,a\n\n2,b\n");
    let e = read_table(&mid, &opts(&mid), None).err().unwrap();
    assert!(e.message.contains("line 3 is blank"), "{}", e.message);
    let trailing = csv(d.path(), "trail.csv", "x,y\n1,a\n2,b\n\n\n");
    let t = read_table(&trailing, &opts(&trailing), None).unwrap();
    assert_eq!(t.table.n_rows(), 2);
    // Frozen re-read of the same file agrees.
    let again = read_table(&trailing, &opts(&trailing), Some(&t.raw_dtypes)).unwrap();
    assert_eq!(again.table.n_rows(), 2);
}

/// data-03: a BOM and padded header names survive bind -> plan.
#[test]
fn bom_and_padded_headers_read_consistently() {
    let d = tempfile::tempdir().unwrap();
    let p = csv(d.path(), "bom.csv", "\u{feff}x1, x2\n1.5,a\n2.5,b\n");
    let o = ReadOptions::from_config(&p, &Default::default()).unwrap();
    let first = read_table(&p, &o, None).unwrap();
    let names: Vec<&str> = first.table.names();
    assert_eq!(names, vec!["x1", " x2"]);
    let frozen = read_table(&p, &o, Some(&first.raw_dtypes)).unwrap();
    assert_eq!(frozen.table.names(), names);
}

/// data-10 / data-11: integral float ids are excluded; stray tokens in a numeric
/// column are an error that names them.
#[test]
fn identifier_and_stray_token_heuristics() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    init_project(dir, "tabular-binary", false).unwrap();
    let mut text = String::from("order_id,x,q,y\n");
    for i in 0..60 {
        let q = if i == 5 { "?".to_string() } else { format!("{}.5", i % 7) };
        text.push_str(&format!("{}.0,{},{},{}\n", 70000 + 3 * i, i % 5, q, i % 2));
    }
    fs::write(dir.join("data/ids.csv"), &text).unwrap();
    let ws = Workspace::open(dir).unwrap();
    let e = ws
        .bind_data(&BindRequest {
            file: dir.join("data/ids.csv"),
            target: "y".into(),
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::SchemaTypeMismatch);
    assert_eq!(e.field.as_deref(), Some("q"));
    assert!(e.actual.unwrap_or_default().contains('?'));
    // With "?" declared as missing, order_id (unique integral floats) is excluded.
    let cfg = fs::read_to_string(dir.join("rustpy.toml")).unwrap();
    fs::write(dir.join("rustpy.toml"), cfg.replace("[data]", "[data]\nnull_values = [\"\", \"?\"]")).unwrap();
    let r = ws
        .bind_data(&BindRequest {
            file: dir.join("data/ids.csv"),
            target: "y".into(),
            ..Default::default()
        })
        .unwrap();
    assert!(r.warnings.iter().any(|w| w.contains("order_id")), "{:?}", r.warnings);
}

/// data-02 / train-01 / data-04: re-splitting after a test evaluation, and warm
/// starts across splits, cannot produce a "first use" on rows already used.
#[test]
fn test_discipline_survives_resplits_and_warm_starts() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1500, false);
    // Numeric features only, so the feature layout does not depend on the split and a
    // warm start across splits is possible (the case the leakage check must catch).
    const NUMERIC_ONLY: &str =
        "data.exclude=[\"line\", \"shift\", \"supplier\", \"device_type\", \"firmware\", \"maintenance_flag\"]";
    let req = quick(&[NUMERIC_ONLY]);
    let a = ws.run(&req, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let ev = ws.evaluate_run(&a.run_id, SplitName::Test, false, false).unwrap();
    assert_eq!(ev.test_use_index, Some(1));
    // A run and its own bundle are the same model: idempotent, not "another model".
    let b = ws.build_bundle(&a.run_id, &BundleOptions::default()).unwrap();
    let same = ws
        .evaluate_bundle(&tmp.path().join(&b.path), SplitName::Test, &req, false, false)
        .unwrap();
    assert_eq!(same.evaluation_id, ev.evaluation_id);
    // New seed => new split id; the overlapping test rows are still detected.
    let reseeded = quick(&[NUMERIC_ONLY, "data.seed=99"]);
    let c = ws.run(&reseeded, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let e = ws.evaluate_run(&c.run_id, SplitName::Test, false, false).unwrap_err();
    assert!(
        matches!(e.code, ErrorCode::TestSetReuse | ErrorCode::LeakageDetected),
        "{e}"
    );
    // Warm start from `a` on the re-seeded split: a's training rows are now partly in
    // the new test split, so evaluating the child there is leakage.
    let w = ws
        .resume(&a.run_id, true, &reseeded, &RunControl::default(), &mut |_| {})
        .unwrap();
    assert_eq!(w.lineage.as_ref().unwrap().level, ResumeLevel::WarmStart);
    let e = ws.evaluate_run(&w.run_id, SplitName::Test, true, false).unwrap_err();
    assert_eq!(e.code, ErrorCode::LeakageDetected, "{e}");
}

/// data-05: with an id column, rows keep their split when the data grows, so an
/// old bundle evaluated on the new data is verifiably scored on unseen rows.
#[test]
fn id_based_splits_are_stable_across_data_versions() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let req = quick(&[]);
    let a = ws.run(&req, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    ws.evaluate_run(&a.run_id, SplitName::Test, false, false).unwrap();
    let b = ws.build_bundle(&a.run_id, &BundleOptions::default()).unwrap();
    // Append rows: same first 1200 rows (same seed), then 600 new ones.
    let old = fs::read_to_string(tmp.path().join("data/defects.csv")).unwrap();
    let more = rustpy_workflow::sample::defects_csv(600, 12, SampleVariant::Shifted);
    fs::write(tmp.path().join("data/defects.csv"), format!("{old}{}", more.split_once('\n').unwrap().1)).unwrap();
    // Test split: with stable id/group splits the old test rows are still test rows,
    // never rows the old bundle trained on.
    let r = ws
        .evaluate_bundle(&tmp.path().join(&b.path), SplitName::Test, &req, false, false)
        .unwrap();
    assert_eq!(r.unseen_rows_verified, Some(true), "{:?}", r.notes);
    assert!(r.metrics.n > 0);
    // Validation split: rows the bundle used for early stopping are excluded, not scored.
    let v = ws
        .evaluate_bundle(&tmp.path().join(&b.path), SplitName::Valid, &req, false, false)
        .unwrap();
    assert!(v.notes.iter().any(|n| n.contains("are excluded")), "{:?}", v.notes);
}

/// train-06 / train-11: a clean mid-epoch stop resumes to the identical history and
/// weights; a resumed child that dies before its own checkpoint is resumable.
#[test]
fn mid_epoch_cancel_resume_is_exact_and_children_are_resumable() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1500, true);
    let req = quick(&[]);
    let full = ws.run(&req, &mlp(), &mut |_| {}).unwrap().runs.remove(0);
    let control = RunControl {
        cancel_after_steps: Some(13),
        ..Default::default()
    };
    let part = ws
        .run(&req, &RunOptions { control, ..mlp() }, &mut |_| {})
        .unwrap()
        .runs
        .remove(0);
    assert_eq!(part.status, RunStatus::Cancelled);
    assert_eq!(part.steps_done, 13);
    // Resume with a cancel again (child stops before writing a checkpoint of its own
    // beyond the copied one), then resume the child.
    let control = RunControl {
        cancel_after_steps: Some(14),
        ..Default::default()
    };
    let child = ws.resume(&part.run_id, false, &ConfigRequest::default(), &control, &mut |_| {}).unwrap();
    assert_eq!(child.status, RunStatus::Cancelled);
    let done = ws
        .resume(&child.run_id, false, &ConfigRequest::default(), &RunControl::default(), &mut |_| {})
        .unwrap();
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!(done.history, full.history, "history differs after mid-epoch resume");
    assert_eq!(done.artifacts[0].sha256, full.artifacts[0].sha256);
}

/// train-07: settings that differ from the run's recorded configuration are
/// refused for an exact resume instead of being silently ignored.
#[test]
fn exact_resume_refuses_changed_settings() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let control = RunControl {
        cancel_after_steps: Some(5),
        ..Default::default()
    };
    let part = ws
        .run(&quick(&[]), &RunOptions { control, ..mlp() }, &mut |_| {})
        .unwrap()
        .runs
        .remove(0);
    let e = ws
        .resume(&part.run_id, false, &quick(&["train.small-mlp.learning_rate=0.5"]), &RunControl::default(), &mut |_| {})
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::ResumeIncompatible);
    assert!(e.message.contains("learning_rate"), "{}", e.message);
    // The same settings are fine.
    assert!(ws.resume(&part.run_id, false, &quick(&[]), &RunControl::default(), &mut |_| {}).is_ok());
}

/// train-09: a damaged newest checkpoint falls back to an older valid one.
#[test]
fn damaged_newest_checkpoint_falls_back_to_older() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let control = RunControl {
        cancel_after_steps: Some(20),
        ..Default::default()
    };
    let part = ws
        .run(&quick(&[]), &RunOptions { control, ..mlp() }, &mut |_| {})
        .unwrap()
        .runs
        .remove(0);
    let cks = rustpy_workflow::train::list_checkpoints(&tmp.path().join(".rustpy/runs").join(&part.run_id).join("checkpoints"));
    assert!(cks.len() >= 2, "{cks:?}");
    let newest = cks.last().unwrap();
    let mut bytes = fs::read(newest.join("optim.bpk")).unwrap();
    bytes.push(0);
    fs::write(newest.join("optim.bpk"), bytes).unwrap();
    let r = ws.resume(&part.run_id, false, &ConfigRequest::default(), &RunControl::default(), &mut |_| {}).unwrap();
    assert_eq!(r.status, RunStatus::Completed);
    assert!(r.lineage.unwrap().from_step.unwrap() < 20);
}

/// data-06: missing-value tokens (e.g. -999) behave the same in training, CSV
/// prediction and application records.
#[test]
fn missing_tokens_are_part_of_the_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let cfg = fs::read_to_string(tmp.path().join("rustpy.toml")).unwrap();
    fs::write(tmp.path().join("rustpy.toml"), cfg.replace("[data]", "[data]\nnull_values = [\"\", \"NA\", \"-999\"]")).unwrap();
    let a = ws.run(&quick(&[]), &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let b = ws
        .build_bundle(&a.run_id, &BundleOptions { allow_unevaluated: true, ..Default::default() })
        .unwrap();
    let p = Predictor::load(tmp.path().join(&b.path), LoadOptions::cpu()).unwrap();
    assert!(p.bundle().transform.missing_tokens.contains(&"-999".to_string()));
    let ex = Record::from_json(&p.bundle().examples.records[0]).unwrap();
    let mut with_null = ex.clone();
    with_null.insert("humidity", Value::Null);
    let mut with_token = ex.clone();
    with_token.insert("humidity", -999);
    let a = p.predict_one(&with_null).unwrap().probability;
    let b = p.predict_one(&with_token).unwrap().probability;
    assert_eq!(a, b);
}

/// train-04: comparisons use the selection settings the runs were trained with.
#[test]
fn compare_uses_recorded_selection_settings() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let res = ws.run(&quick(&["selection.max_model_kib=0.5"]), &RunOptions::default(), &mut |_| {}).unwrap();
    let later = ws.compare(&[]).unwrap();
    assert_eq!(later.recommended, res.comparison.recommended);
    let mlp_row = later.rows.iter().find(|r| r.candidate == Candidate::SmallMlp).unwrap();
    assert!(!mlp_row.meets_constraints);
}

/// data-09: grouped splits that cannot approach the fractions are refused.
#[test]
fn grouped_split_with_few_unequal_groups_is_refused_or_warned() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    init_project(dir, "tabular-binary", false).unwrap();
    let mut text = String::from("site,x,y\n");
    for (site, n) in [("a", 400), ("b", 300), ("c", 200), ("d", 100)] {
        for i in 0..n {
            text.push_str(&format!("{site},{},{}\n", i % 13, (i / 3) % 2));
        }
    }
    fs::write(dir.join("data/sites.csv"), text).unwrap();
    let ws = Workspace::open(dir).unwrap();
    let r = ws.bind_data(&BindRequest {
        file: dir.join("data/sites.csv"),
        target: "y".into(),
        group: Some("site".into()),
        ..Default::default()
    });
    match r {
        Err(e) => assert_eq!(e.code, ErrorCode::SplitDegenerate, "{e}"),
        Ok(r) => assert!(r.warnings.iter().any(|w| w.contains("configured")), "{:?}", r.warnings),
    }
}

/// runtime-02: bundles build with unknown_category = "error".
#[test]
fn bundle_builds_with_strict_unknown_categories() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let a = ws
        .run(&quick(&["preprocess.unknown_category=\"error\""]), &linear(), &mut |_| {})
        .unwrap()
        .runs
        .remove(0);
    let b = ws
        .build_bundle(&a.run_id, &BundleOptions { allow_unevaluated: true, ..Default::default() })
        .unwrap();
    assert!(b.check.passed);
}

fn bundle(ws: &Workspace, dir: &Path, extra: &[&str]) -> std::path::PathBuf {
    let a = ws.run(&quick(extra), &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let out = dir.join(format!("bundles/{}", a.run_id));
    ws.build_bundle(&a.run_id, &BundleOptions { allow_unevaluated: true, out: Some(out.clone()) }).unwrap();
    out
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.path().is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()));
        } else {
            fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}

/// runtime-05/08/12: manifest claims, unlisted files and malformed versions are rejected.
#[test]
fn bundle_tampering_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let b = bundle(&ws, tmp.path(), &[]);
    assert!(Predictor::load(&b, LoadOptions::cpu()).is_ok());
    let tamper = |name: &str, f: &dyn Fn(&Path)| -> ErrorCode {
        let t = tmp.path().join(name);
        copy_dir(&b, &t);
        f(&t);
        Predictor::load(&t, LoadOptions::cpu()).unwrap_err().code
    };
    let edit_manifest = |key: &'static str, v: serde_json::Value| {
        move |t: &Path| {
            let mut m: serde_json::Value = serde_json::from_slice(&fs::read(t.join("manifest.json")).unwrap()).unwrap();
            let mut cur = &mut m;
            let parts: Vec<&str> = key.split('.').collect();
            for p in &parts[..parts.len() - 1] {
                cur = cur.get_mut(*p).unwrap();
            }
            cur[parts[parts.len() - 1]] = v.clone();
            fs::write(t.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        }
    };
    assert_eq!(tamper("t1", &edit_manifest("evaluation_status", "evaluated".into())), ErrorCode::BundleCorrupt);
    assert_eq!(tamper("t2", &edit_manifest("runtime.min_runtime_version", "banana".into())), ErrorCode::BundleCorrupt);
    assert_eq!(tamper("t3", &|t: &Path| fs::write(t.join("train.csv"), "x\n1\n").unwrap()), ErrorCode::BundleCorrupt);
    assert_eq!(
        tamper("t4", &|t: &Path| {
            let mut tr: serde_json::Value = serde_json::from_slice(&fs::read(t.join("transform.json")).unwrap()).unwrap();
            tr["future_semantics"] = serde_json::json!(true);
            fs::write(t.join("transform.json"), serde_json::to_vec(&tr).unwrap()).unwrap();
        }),
        ErrorCode::BundleCorrupt
    );
}

/// runtime-04/06: updates that break the input or output contract are refused.
#[test]
fn contract_breaking_updates_are_refused() {
    use rustpy_runtime::PredictorHandle;
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let base = bundle(&ws, tmp.path(), &[]);
    let flipped = bundle(&ws, tmp.path(), &["data.positive_label=\"0\""]);
    let stricter = bundle(&ws, tmp.path(), &["preprocess.extra_columns=\"error\"", "data.exclude=[\"supplier\"]"]);
    let h = PredictorHandle::load(&base, LoadOptions::cpu()).unwrap();
    let before = h.current().bundle_id().to_string();
    let e = h.try_update(&flipped, LoadOptions::cpu()).unwrap_err();
    assert_eq!(e.code, ErrorCode::PredictorUpdateFailed);
    assert!(e.message.contains("prediction meaning changed"), "{}", e.message);
    let e = h.try_update(&stricter, LoadOptions::cpu()).unwrap_err();
    assert!(e.message.contains("removed field `supplier`"), "{}", e.message);
    assert_eq!(h.current().bundle_id(), before);
    let r = h.try_update(&flipped, LoadOptions::cpu().allow_contract_change(true)).unwrap();
    assert!(!r.output_changes.is_empty());
}

/// plan-gaps-12: label mapping for a positive class that sorts first, and text labels.
#[test]
fn positive_label_semantics_hold_for_any_label_order() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1500, true);
    let b1 = bundle(&ws, tmp.path(), &[]);
    let b0 = bundle(&ws, tmp.path(), &["data.positive_label=\"0\""]);
    let p1 = Predictor::load(&b1, LoadOptions::cpu()).unwrap();
    let p0 = Predictor::load(&b0, LoadOptions::cpu()).unwrap();
    assert_eq!(p0.output_schema().positive_label, "0");
    let recs: Vec<Record> = p1.bundle().examples.records.iter().map(|r| Record::from_json(r).unwrap()).collect();
    for (a, b) in p1.predict(&recs).unwrap().iter().zip(p0.predict(&recs).unwrap()) {
        // Same data, opposite positive class: probabilities are (close to) complementary.
        assert!((a.probability + b.probability - 1.0).abs() < 0.2, "{} vs {}", a.probability, b.probability);
        // ...so, away from the threshold, both name the same class.
        if (a.probability - 0.5).abs() > 0.2 {
            assert_eq!(a.label, b.label);
        }
    }
    assert!(p1.predict(&[]).unwrap().is_empty());
}

/// plan-gaps-12: the target cannot be a feature, excluded or given a kind.
#[test]
fn target_cannot_be_excluded_or_typed() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 600, true);
    for o in ["data.exclude=[\"defect\"]", "data.columns.defect=\"numeric\""] {
        let e = ws.plan(&quick(&[o])).unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigInvalid, "{o}: {e}");
    }
}

/// walkthrough-1: without an id column, rows are matched by content, so an old
/// bundle evaluated on appended data never scores rows it trained or selected on,
/// and an evaluation whose rows cannot be matched is refused unless allowed.
#[test]
fn bundle_evaluation_without_id_matches_rows_by_content() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    init_project(dir, "tabular-binary", false).unwrap();
    write_defects_csv(&dir.join("data/defects.csv"), 1200, 11, SampleVariant::Base).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/defects.csv"),
        target: "defect".into(),
        positive: Some("1".into()),
        exclude: vec!["device_id".into(), "unit_id".into()],
        ..Default::default()
    })
    .unwrap();
    let req = quick(&[]);
    let p = ws.prepare(ws.resolve(&req).unwrap(), false).unwrap();
    assert!(p.split.algorithm.contains("content"), "{}", p.split.algorithm);
    let a = ws.run(&req, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    ws.evaluate_run(&a.run_id, SplitName::Test, false, false).unwrap();
    let b = ws.build_bundle(&a.run_id, &BundleOptions::default()).unwrap();
    let old_p = ws.prepare_for_run(&a).unwrap();
    let old_rows = old_p.split.rows(SplitName::Train).len() + old_p.split.rows(SplitName::Valid).len();
    // Append 600 rows and shuffle the file: positions no longer identify rows.
    let old = fs::read_to_string(dir.join("data/defects.csv")).unwrap();
    let more = rustpy_workflow::sample::defects_csv(600, 12, SampleVariant::Shifted);
    let mut lines: Vec<&str> = old.lines().skip(1).chain(more.lines().skip(1)).collect();
    lines.reverse();
    fs::write(dir.join("data/defects.csv"), format!("{}\n{}\n", old.lines().next().unwrap(), lines.join("\n"))).unwrap();
    let bundle_dir = dir.join(&b.path);
    // Validation rows of the old run are recognised by content and excluded.
    let v = ws.evaluate_bundle(&bundle_dir, SplitName::Valid, &req, false, false).unwrap();
    assert_eq!(v.unseen_rows_verified, Some(true), "{:?}", v.notes);
    assert!(v.notes.iter().any(|n| n.contains("are excluded")), "{:?}", v.notes);
    // Test split: verified, and none of the evaluated rows is an old training/validation row.
    let t = ws.evaluate_bundle(&bundle_dir, SplitName::Test, &req, false, false).unwrap();
    assert_eq!(t.unseen_rows_verified, Some(true), "{:?}", t.notes);
    let new_p = ws.prepare(ws.resolve(&req).unwrap(), false).unwrap();
    let exposure = ws.run_exposure(&a, true);
    let keys = exposure[0].1.as_ref().unwrap();
    let seen = keys.matcher(&new_p.snapshot, &new_p.table).unwrap();
    let new_test = new_p.split.rows(SplitName::Test);
    let leaked = new_test.iter().filter(|&&r| seen(r)).count();
    assert_eq!(t.metrics.n as usize + leaked, new_test.len(), "excluded rows = seen rows");
    assert!(leaked < old_rows / 20, "content-keyed split kept almost every old row in its split ({leaked} moved)");

    // A bundle whose source run is unknown here cannot be checked: refused unless allowed.
    let other = tempfile::tempdir().unwrap();
    init_project(other.path(), "tabular-binary", false).unwrap();
    fs::copy(dir.join("data/defects.csv"), other.path().join("data/defects.csv")).unwrap();
    let ws2 = Workspace::open(other.path()).unwrap();
    ws2.bind_data(&BindRequest {
        file: other.path().join("data/defects.csv"),
        target: "defect".into(),
        positive: Some("1".into()),
        exclude: vec!["device_id".into(), "unit_id".into()],
        ..Default::default()
    })
    .unwrap();
    let e = ws2.evaluate_bundle(&bundle_dir, SplitName::Test, &req, false, false).unwrap_err();
    assert_eq!(e.code, ErrorCode::LeakageDetected, "{e}");
    assert!(e.next_action.as_deref().unwrap_or("").contains("--allow-unverified-rows"));
    let ok = ws2.evaluate_bundle(&bundle_dir, SplitName::Test, &req, false, true).unwrap();
    assert_eq!(ok.unseen_rows_verified, None);
    assert!(ok.notes.iter().any(|n| n.contains("not verified")), "{:?}", ok.notes);
}

/// walkthrough-5/4: the test ledger records rows shared with a model of another
/// data version, and `compare` ranks on the newest data version.
#[test]
fn shared_hold_out_and_newest_reference() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path(), 1200, true);
    let req = quick(&[]);
    let a = ws.run(&req, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let b = ws.build_bundle(&a.run_id, &BundleOptions { allow_unevaluated: true, ..Default::default() }).unwrap();
    let old = fs::read_to_string(tmp.path().join("data/defects.csv")).unwrap();
    let more = rustpy_workflow::sample::defects_csv(600, 12, SampleVariant::Shifted);
    fs::write(tmp.path().join("data/defects.csv"), format!("{old}{}", more.split_once('\n').unwrap().1)).unwrap();
    let n = ws.run(&req, &linear(), &mut |_| {}).unwrap().runs.remove(0);
    let new_eval = ws.evaluate_run(&n.run_id, SplitName::Test, false, false).unwrap();
    assert_eq!(new_eval.test_use_index, Some(1));
    let old_eval = ws.evaluate_bundle(&tmp.path().join(&b.path), SplitName::Test, &req, false, false).unwrap();
    assert_eq!(old_eval.test_use_index, Some(1));
    assert_eq!(old_eval.test_shared_with.len(), 1, "{:?}", old_eval.notes);
    assert!(old_eval.test_shared_with[0].contains(&n.run_id));
    assert!(old_eval.notes.iter().any(|x| x.contains("another data version")), "{:?}", old_eval.notes);
    // Listing the old run first must not make it the reference.
    let cmp = ws.compare(&[a.run_id.clone(), n.run_id.clone()]).unwrap();
    assert_eq!(cmp.recommended.as_deref(), Some(n.run_id.as_str()));
    assert!(cmp.notes.iter().any(|x| x.contains("evaluate --bundle")), "{:?}", cmp.notes);
}
