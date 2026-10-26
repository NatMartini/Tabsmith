//! Process-level tests of the `rustpy` binary: the user flow, exact resume after
//! a crash in an independent process, and the preset failure scenarios.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_rustpy");

fn rustpy(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN).current_dir(dir).args(args).output().expect("run rustpy")
}

fn ok(dir: &Path, args: &[&str]) -> String {
    let out = rustpy(dir, args);
    assert!(
        out.status.success(),
        "rustpy {args:?} failed ({:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn json(dir: &Path, args: &[&str]) -> serde_json::Value {
    let mut a = args.to_vec();
    a.push("--json");
    let out = rustpy(dir, &a);
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "rustpy {a:?}: invalid JSON ({e}); stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

/// Expects a failure with `code`, a non-empty next action, and the given exit code.
fn fails(dir: &Path, args: &[&str], code: &str, exit: i32) -> serde_json::Value {
    let mut a = args.to_vec();
    a.push("--json");
    let out = rustpy(dir, &a);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| {
        panic!("rustpy {a:?}: stdout {}\nstderr {}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
    });
    assert_eq!(v["ok"], false, "{a:?} unexpectedly succeeded: {v}");
    assert_eq!(v["error"]["code"], code, "{a:?}: {v}");
    assert!(v["error"]["next_action"].as_str().is_some_and(|s| !s.is_empty()), "{a:?}: no next action: {v}");
    assert_eq!(out.status.code(), Some(exit), "{a:?}: exit code");
    v
}

const QUICK: [&str; 6] = [
    "--set",
    "train.linear.epochs=4",
    "--set",
    "train.small-mlp.epochs=4",
    "--set",
    "train.small-mlp.hidden=[12]",
];

fn new_project(root: &Path, name: &str, rows: usize) -> PathBuf {
    ok(root, &["init", name]);
    let dir = root.join(name);
    ok(&dir, &["sample-data", "data/defects.csv", "--rows", &rows.to_string()]);
    ok(&dir, &["data", "bind", "data/defects.csv", "--target", "defect", "--group", "device_id", "--id", "unit_id"]);
    dir
}

fn args<'a>(base: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = base.to_vec();
    v.extend_from_slice(extra);
    v
}

#[test]
fn user_flow_from_csv_to_predictions() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = new_project(tmp.path(), "proj", 1500);
    let plan = json(&dir, &args(&["plan"], &QUICK));
    assert_eq!(plan["ok"], true);
    assert_eq!(plan["result"]["target"]["positive_label"], "1");
    let run = json(&dir, &args(&["run"], &QUICK));
    let best = run["result"]["comparison"]["recommended"].as_str().unwrap().to_string();
    ok(&dir, &["compare"]);
    ok(&dir, &["evaluate", "--run", &best, "--split", "test"]);
    ok(&dir, &["bundle", "build", "--run", &best]);
    let check = json(&dir, &["bundle", "check", &format!("bundles/{best}")]);
    assert_eq!(check["result"]["check"]["passed"], true);
    ok(&dir, &["predict", "--bundle", &format!("bundles/{best}"), "--input", "data/defects.csv", "--output", "preds.csv"]);
    let preds = std::fs::read_to_string(dir.join("preds.csv")).unwrap();
    assert_eq!(preds.lines().count(), 1501);
    assert!(preds.starts_with("unit_id,label,probability"));
    ok(&dir, &["report"]);
    assert!(dir.join(".rustpy/reports/index.html").is_file());
    ok(&dir, &["doctor"]);
}

#[test]
fn crash_then_resume_equals_uninterrupted_run_bitwise() {
    let tmp = tempfile::tempdir().unwrap();
    let a = new_project(tmp.path(), "a", 1500);
    let b = new_project(tmp.path(), "b", 1500);
    let train = args(&["run", "--candidates", "small-mlp"], &QUICK);
    // Uninterrupted reference in project b.
    let reference = json(&b, &train);
    let ref_sha = reference["result"]["runs"][0]["artifacts"][0]["sha256"].as_str().unwrap().to_string();
    // Project a: the process dies after 20 optimizer steps (inside epoch 3).
    let crash = rustpy(&a, &args(&train, &["--simulate-crash-after-steps", "20"]));
    assert_eq!(crash.status.code(), Some(137));
    let runs = json(&a, &["runs"]);
    assert_eq!(runs["result"][0]["status"], "interrupted");
    assert_eq!(runs["result"][0]["error"]["code"], "E_INTERRUPTED");
    // A second, independent process continues from the last checkpoint.
    let resumed = json(&a, &args(&["resume", "run-001"], &QUICK));
    let r = &resumed["result"];
    assert_eq!(r["status"], "completed");
    assert_eq!(r["lineage"]["level"], "R2");
    assert_eq!(r["lineage"]["parent_run_id"], "run-001");
    assert_eq!(r["artifacts"][0]["sha256"].as_str().unwrap(), ref_sha, "resumed weights differ from the uninterrupted run");
    assert_eq!(r["history"], reference["result"]["runs"][0]["history"]);
}

#[test]
fn preset_failures_are_actionable() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = new_project(tmp.path(), "f", 1200);
    let mut recoverable = 0;

    // 1. Misspelled configuration key.
    fails(&dir, &["plan", "--set", "train.small-mlp.epoch=3"], "E_CONFIG_UNKNOWN_FIELD", 2);
    recoverable += 1;
    // 2. Category budget exceeded.
    fails(&dir, &["plan", "--set", "preprocess.max_categories=3"], "E_CARDINALITY_BUDGET", 3);
    recoverable += 1;
    // 3. Requested GPU backend not available: no silent CPU fallback.
    fails(&dir, &args(&["run", "--set", "execution.candidate_device=\"cuda\""], &QUICK), "E_DEVICE_UNSUPPORTED", 5);
    recoverable += 1;
    // 4. Memory budget (pre-flight "OOM").
    fails(&dir, &args(&["run", "--set", "execution.max_memory_mb=1"], &QUICK), "E_RESOURCE_BUDGET_EXCEEDED", 5);
    recoverable += 1;
    // 5. Time-dependent data cannot fall back to a random split.
    fails(&dir, &["plan", "--set", "data.time_column=\"hours_in_service\""], "E_SPLIT_UNSUPPORTED", 3);
    recoverable += 1;

    let run = json(&dir, &args(&["run", "--candidates", "linear"], &QUICK));
    let id = run["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    ok(&dir, &["evaluate", "--run", &id, "--split", "test"]);
    ok(&dir, &["bundle", "build", "--run", &id]);
    let bundle = format!("bundles/{id}");

    // 6. Test split reuse by another model.
    let run2 = json(&dir, &args(&["run", "--candidates", "small-mlp"], &QUICK));
    let id2 = run2["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    fails(&dir, &["evaluate", "--run", &id2, "--split", "test"], "E_TEST_SET_REUSE", 7);
    recoverable += 1;
    // 7. Prediction input missing a required column.
    std::fs::write(dir.join("missing.jsonl"), "{\"line\": \"A\"}\n").unwrap();
    let out = rustpy(&dir, &["predict", "--bundle", &bundle, "--input", "missing.jsonl", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["error"]["code"], "E_SCHEMA_MISSING_COLUMN");
    recoverable += 1;
    // 8. Data file missing.
    std::fs::rename(dir.join("data/defects.csv"), dir.join("data/moved.csv")).unwrap();
    fails(&dir, &["plan"], "E_DATA_SOURCE_NOT_FOUND", 3);
    std::fs::rename(dir.join("data/moved.csv"), dir.join("data/defects.csv")).unwrap();
    recoverable += 1;
    // 9. Damaged bundle: one flipped byte in the weights.
    let broken = dir.join("broken");
    copy_dir(&dir.join(&bundle), &broken);
    let w = broken.join("model.safetensors");
    let mut bytes = std::fs::read(&w).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&w, bytes).unwrap();
    fails(&dir, &["bundle", "check", "broken"], "E_BUNDLE_CORRUPT", 4);
    recoverable += 1;
    // 10. Incomplete bundle: a file is missing.
    std::fs::remove_file(broken.join("transform.json")).unwrap();
    fails(&dir, &["bundle", "check", "broken"], "E_BUNDLE_MISSING_FILE", 4);
    recoverable += 1;
    // The original bundle is untouched.
    ok(&dir, &["bundle", "check", &bundle]);
    assert_eq!(recoverable, 10);
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let target = to.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}
