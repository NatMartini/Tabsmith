//! Process-level tests of the `tabsmith` binary: the user flow, exact resume after
//! a crash in an independent process, and the preset failure scenarios.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_tabsmith");

fn tabsmith(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN).current_dir(dir).args(args).output().expect("run tabsmith")
}

fn ok(dir: &Path, args: &[&str]) -> String {
    let out = tabsmith(dir, args);
    assert!(
        out.status.success(),
        "tabsmith {args:?} failed ({:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn json(dir: &Path, args: &[&str]) -> serde_json::Value {
    let mut a = args.to_vec();
    a.push("--json");
    let out = tabsmith(dir, &a);
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "tabsmith {a:?}: invalid JSON ({e}); stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

/// Expects a failure with `code`, a non-empty next action, and the given exit code.
fn fails(dir: &Path, args: &[&str], code: &str, exit: i32) -> serde_json::Value {
    let mut a = args.to_vec();
    a.push("--json");
    let out = tabsmith(dir, &a);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| {
        panic!(
            "tabsmith {a:?}: stdout {}\nstderr {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    assert_eq!(v["ok"], false, "{a:?} unexpectedly succeeded: {v}");
    assert_eq!(v["error"]["code"], code, "{a:?}: {v}");
    assert!(
        v["error"]["next_action"].as_str().is_some_and(|s| !s.is_empty()),
        "{a:?}: no next action: {v}"
    );
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
    ok(
        &dir,
        &[
            "data",
            "bind",
            "data/defects.csv",
            "--target",
            "defect",
            "--group",
            "device_id",
            "--id",
            "unit_id",
        ],
    );
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
    ok(
        &dir,
        &[
            "predict",
            "--bundle",
            &format!("bundles/{best}"),
            "--input",
            "data/defects.csv",
            "--output",
            "preds.csv",
        ],
    );
    let preds = std::fs::read_to_string(dir.join("preds.csv")).unwrap();
    assert_eq!(preds.lines().count(), 1501);
    assert!(preds.starts_with("unit_id,label,probability"));
    ok(&dir, &["report"]);
    assert!(dir.join(".tabsmith/reports/index.html").is_file());
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
    let ref_sha = reference["result"]["runs"][0]["artifacts"][0]["sha256"]
        .as_str()
        .unwrap()
        .to_string();
    // Project a: the process dies after 20 optimizer steps (inside epoch 3).
    let crash = tabsmith(&a, &args(&train, &["--simulate-crash-after-steps", "20"]));
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
    assert_eq!(
        r["artifacts"][0]["sha256"].as_str().unwrap(),
        ref_sha,
        "resumed weights differ from the uninterrupted run"
    );
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
    fails(
        &dir,
        &args(&["run", "--set", "execution.candidate_device=\"cuda\""], &QUICK),
        "E_DEVICE_UNSUPPORTED",
        5,
    );
    recoverable += 1;
    // 4. Memory budget (pre-flight "OOM").
    fails(
        &dir,
        &args(&["run", "--set", "execution.max_memory_mb=1"], &QUICK),
        "E_RESOURCE_BUDGET_EXCEEDED",
        5,
    );
    recoverable += 1;
    // 5. Time-dependent data cannot fall back to a random split.
    fails(
        &dir,
        &["plan", "--set", "data.time_column=\"hours_in_service\""],
        "E_SPLIT_UNSUPPORTED",
        3,
    );
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
    let out = tabsmith(&dir, &["predict", "--bundle", &bundle, "--input", "missing.jsonl", "--json"]);
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

#[test]
fn json_usage_errors_flags_and_bind_details() {
    let tmp = tempfile::tempdir().unwrap();
    // Usage errors keep the JSON envelope.
    let out = tabsmith(tmp.path(), &["--json", "run", "--bogus"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("JSON even for usage errors");
    assert_eq!(v["ok"], false);
    assert_eq!(out.status.code(), Some(2));
    // `init .` keeps an existing .gitignore.
    let here = tmp.path().join("here");
    std::fs::create_dir_all(&here).unwrap();
    std::fs::write(here.join(".gitignore"), "/target\n*.secret\n").unwrap();
    ok(&here, &["init", "."]);
    let gi = std::fs::read_to_string(here.join(".gitignore")).unwrap();
    assert!(gi.contains("*.secret") && gi.contains("/.tabsmith/"), "{gi}");
    // An inline `data = {...}` table survives `data bind`.
    let dir = new_project(tmp.path(), "p", 900);
    let cfg = std::fs::read_to_string(dir.join("tabsmith.toml")).unwrap();
    let cfg = cfg.replacen("[data]\n", "[data_old]\n", 1);
    let cfg = cfg.replace("[data_old]", "data = { seed = 1234, exclude = [\"inspector_note\"] }\n[data_old]");
    // Keep only the inline table: drop the old [data] section body.
    let start = cfg.find("[data_old]").unwrap();
    let end = cfg[start..].find("\n[selection]").map(|i| start + i).unwrap();
    std::fs::write(dir.join("tabsmith.toml"), format!("{}{}", &cfg[..start], &cfg[end + 1..])).unwrap();
    ok(
        &dir,
        &[
            "data",
            "bind",
            "data/defects.csv",
            "--target",
            "defect",
            "--group",
            "device_id",
            "--id",
            "unit_id",
        ],
    );
    let eff = json(&dir, &["config"]);
    let seed = eff["result"].as_array().unwrap().iter().find(|r| r[0] == "data.seed").unwrap()[1].clone();
    assert_eq!(seed, "1234");
    // String keys accept bare numbers; flags that would be ignored are refused.
    ok(&dir, &["config", "--set", "data.positive_label=1"]);
    let run = json(&dir, &args(&["run", "--candidates", "linear"], &QUICK));
    let id = run["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    fails(
        &dir,
        &["evaluate", "--run", &id, "--set", "selection.threshold=0.2"],
        "E_CONFIG_INVALID",
        2,
    );
    // The device error names the setting that is actually wrong.
    let v = fails(
        &dir,
        &["plan", "--set", "execution.baseline_device=\"cuda\""],
        "E_DEVICE_UNSUPPORTED",
        5,
    );
    assert_eq!(v["error"]["field"], "execution.baseline_device");
}

#[test]
fn separate_process_predictions_equal_training_side_probabilities() {
    use tabsmith_runtime::model::sigmoid;
    let tmp = tempfile::tempdir().unwrap();
    let dir = new_project(tmp.path(), "par", 1500);
    let run = json(&dir, &args(&["run", "--candidates", "small-mlp"], &QUICK));
    let id = run["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    ok(&dir, &["bundle", "build", "--run", &id, "--allow-unevaluated"]);
    // Application side: a separate `tabsmith predict` process reading the raw CSV.
    let out = json(
        &dir,
        &["predict", "--bundle", &format!("bundles/{id}"), "--input", "data/defects.csv"],
    );
    let preds = out["result"]["predictions"].as_array().unwrap();
    // Training side: the run's snapshot rows through the fitted transform and kept weights.
    let ws = tabsmith_workflow::Workspace::open(&dir).unwrap();
    let r = ws.load_run(&id).unwrap();
    let p = ws.prepare_for_run(&r).unwrap();
    let model = ws.run_weights(&r).unwrap();
    let x = tabsmith_runtime::apply_transform(&p.fitted.transform, &p.table, tabsmith_contracts::Phase::Evaluate).unwrap();
    let ids = p.table.column("unit_id").unwrap();
    let mut by_id = std::collections::HashMap::new();
    for (i, z) in model.outputs(&x).data.into_iter().enumerate() {
        if let tabsmith_runtime::ValueRef::Text(u) = ids.get(i) {
            by_id.insert(u.to_string(), sigmoid(f64::from(z)));
        }
    }
    assert_eq!(preds.len(), by_id.len());
    let mut worst = 0f64;
    for p in preds {
        let u = p["unit_id"].as_str().unwrap();
        worst = worst.max((p["probability"].as_f64().unwrap() - by_id[u]).abs());
    }
    assert!(worst < 1e-12, "train-side vs separate-process predict: max |Δp| = {worst}");
}

/// Cold-start walkthrough findings: relative data paths, `(unset)` settings,
/// comma-separated run lists, `--keep` columns, unseen-category warnings and a
/// predict error that points at `--input`.
#[test]
fn walkthrough_details() {
    let tmp = tempfile::tempdir().unwrap();
    ok(tmp.path(), &["init", "w"]);
    let dir = tmp.path().join("w");
    ok(tmp.path(), &["sample-data", "data.csv", "--rows", "1200"]);
    ok(&dir, &["data", "bind", "../data.csv", "--target", "defect", "--group", "device_id"]);
    let toml = std::fs::read_to_string(dir.join("tabsmith.toml")).unwrap();
    assert!(toml.contains("source = \"../data.csv\""), "{toml}");
    assert!(!toml.contains("Filled in by"), "template hint left behind:\n{toml}");
    let cfg = json(&dir, &["config"]);
    let rows = cfg["result"].as_array().unwrap();
    assert!(rows.iter().any(|r| r[0] == "data.id_column" && r[1] == "(unset)"), "{cfg}");

    let run = json(&dir, &args(&["run"], &QUICK));
    let ids: Vec<String> = run["result"]["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["run_id"].as_str().unwrap().to_string())
        .collect();
    // The default candidates: linear, small-mlp and gbdt.
    let all = ids.join(",");
    let cmp = json(&dir, &["compare", "--runs", &all]);
    assert_eq!(cmp["result"]["rows"].as_array().unwrap().len(), 3, "{cmp}");
    let missing = fails(&dir, &["compare", "--runs", "run-404"], "E_RUN_NOT_FOUND", 6);
    assert_eq!(missing["error"]["phase"], "compare");

    ok(&dir, &["bundle", "build", "--run", &ids[0], "--allow-unevaluated"]);
    let bundle = format!("bundles/{}", ids[0]);
    // A later delivery with a new supplier: predictions warn and name the field.
    ok(&dir, &["sample-data", "later.csv", "--rows", "60", "--seed", "9", "--shifted"]);
    let out = json(&dir, &["predict", "--bundle", &bundle, "--input", "later.csv", "--keep", "unit_id"]);
    let preds = out["result"]["predictions"].as_array().unwrap();
    assert_eq!(preds.len(), 60);
    assert!(preds[0]["unit_id"].is_string(), "{}", preds[0]);
    assert!(
        out["result"]["unseen_categories"].as_array().is_some_and(|u| !u.is_empty()),
        "{out}"
    );
    assert!(preds.iter().any(|p| p["unseen_categories"].is_array()));
    let human = tabsmith(
        &dir,
        &["predict", "--bundle", &bundle, "--input", "later.csv", "--keep", "unit_id,line"],
    );
    assert!(human.status.success());
    let stdout = String::from_utf8_lossy(&human.stdout);
    assert!(stdout.starts_with("unit_id,line,label,probability,unseen_categories"), "{stdout}");
    assert!(String::from_utf8_lossy(&human.stderr).contains("warning: categories not seen in training"));
    // Without key columns, rows are numbered from 1.
    let numbered = json(&dir, &["predict", "--bundle", &bundle, "--input", "later.csv"]);
    assert_eq!(numbered["result"]["predictions"][0]["row"], 1);
    let e = fails(
        &dir,
        &["predict", "--bundle", &bundle, "--input", "nope.csv"],
        "E_DATA_SOURCE_NOT_FOUND",
        3,
    );
    assert_eq!(e["error"]["field"], "--input");
    fails(
        &dir,
        &["predict", "--bundle", &bundle, "--input", "later.csv", "--keep", "no_such"],
        "E_SCHEMA_MISSING_COLUMN",
        3,
    );
}

/// `tabsmith train`: one command from a CSV to a tested bundle; again after new data,
/// with the previous bundle and the new model scored on the same new test rows;
/// `predict` without --bundle uses the newest evaluated bundle.
#[test]
fn one_command_train_and_retrain() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    ok(dir, &["sample-data", "machines.csv", "--rows", "1500"]);
    let first = json(
        dir,
        &args(&["train", "machines.csv", "--target", "defect", "--group", "device_id"], &QUICK),
    );
    assert_eq!(first["ok"], true, "{first}");
    assert!(
        dir.join("tabsmith.toml").is_file(),
        "train creates the project in the current directory"
    );
    let chosen = first["result"]["chosen"].as_str().unwrap().to_string();
    assert_eq!(first["result"]["test"]["test_use_index"], 1);
    assert_eq!(first["result"]["bundle"]["check"]["passed"], true);
    assert!(first["result"]["previous"].is_null());
    // New delivery appended to the same file, then retrain without arguments.
    ok(dir, &["sample-data", "more.csv", "--rows", "900", "--seed", "5", "--shifted"]);
    let more = std::fs::read_to_string(dir.join("more.csv")).unwrap();
    let mut all = std::fs::read_to_string(dir.join("machines.csv")).unwrap();
    all.push_str(more.split_once('\n').unwrap().1);
    std::fs::write(dir.join("machines.csv"), all).unwrap();
    let second = json(dir, &args(&["train"], &QUICK));
    assert_eq!(second["ok"], true, "{second}");
    let prev = &second["result"]["previous"];
    assert_eq!(prev["run_id"], chosen.as_str(), "{second}");
    assert_eq!(prev["evaluation"]["unseen_rows_verified"], true, "{prev}");
    assert_eq!(
        prev["new_model_on_same_rows"]["n"], prev["evaluation"]["metrics"]["n"],
        "same rows for old and new: {prev}"
    );
    let new_bundle = second["result"]["bundle"]["path"].as_str().unwrap().to_string();
    // Again on unchanged data: same weights, the bundle is kept (no failure, no duplicate).
    let third = json(dir, &args(&["train"], &QUICK));
    assert_eq!(third["ok"], true, "{third}");
    assert_eq!(third["result"]["bundle"]["path"], new_bundle.as_str(), "{third}");
    assert!(third["result"]["identical_to"].is_string(), "{third}");
    // A re-split of the same data gives no fresh test split; the candidate it
    // builds does not replace the evaluated bundle as the default.
    let reseeded = json(dir, &args(&["train", "--set", "data.seed=7"], &QUICK));
    assert!(
        reseeded["result"]["test_skipped"].as_str().unwrap().contains("only new rows"),
        "{reseeded}"
    );
    // Predict with the newest bundle by default.
    ok(dir, &["sample-data", "new.csv", "--rows", "5", "--seed", "77"]);
    let p = json(dir, &["predict", "--input", "new.csv"]);
    assert_eq!(p["result"]["predictions"].as_array().unwrap().len(), 5);
    assert_eq!(
        p["result"]["bundle_id"], second["result"]["bundle"]["manifest"]["bundle_id"],
        "{new_bundle}"
    );
    // Data flags without a file are a usage error, not silently ignored.
    fails(dir, &["train", "--target", "defect"], "E_CONFIG_INVALID", 2);
}

/// Minimal HTTP/1.1 client for the serve test: returns (status, JSON body).
fn http(addr: &str, method: &str, path: &str, body: &str) -> (u16, serde_json::Value) {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut text = String::new();
    s.read_to_string(&mut text).unwrap();
    let status: u16 = text.split_whitespace().nth(1).unwrap().parse().unwrap();
    let json = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    (status, serde_json::from_str(json).unwrap_or(serde_json::Value::Null))
}

/// `tabsmith serve`: predictions over HTTP equal `tabsmith predict`; errors are
/// structured; /reload swaps bundles only when they pass their self-check.
#[test]
fn serve_predicts_over_http() {
    use std::io::BufRead;
    let tmp = tempfile::tempdir().unwrap();
    let dir = new_project(tmp.path(), "srv", 1200);
    let run = json(&dir, &args(&["run", "--candidates", "linear"], &QUICK));
    let id = run["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    ok(&dir, &["bundle", "build", "--run", &id, "--allow-unevaluated"]);
    let mut child = Command::new(BIN)
        .current_dir(&dir)
        .args(["serve", "--port", "0", "--allow-reload"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let mut first = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let addr = first
        .trim()
        .strip_prefix("listening on http://")
        .unwrap_or_else(|| panic!("{first}"))
        .to_string();
    let (code, health) = http(&addr, "GET", "/health", "");
    assert_eq!(code, 200);
    assert!(health["bundle_id"].as_str().unwrap().starts_with("b-"));
    let examples = std::fs::read_to_string(dir.join(format!("bundles/{id}/examples/input.json"))).unwrap();
    let records: serde_json::Value = serde_json::from_str(&examples).unwrap();
    let body = serde_json::json!({"records": records["records"]}).to_string();
    let (code, out) = http(&addr, "POST", "/predict", &body);
    assert_eq!(code, 200, "{out}");
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("bundles/{id}/examples/expected.json"))).unwrap()).unwrap();
    let got = out["predictions"].as_array().unwrap();
    let want = expected["predictions"].as_array().unwrap();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g["probability"].as_f64().unwrap() - w["probability"].as_f64().unwrap()).abs() < 1e-6);
    }
    let (code, err) = http(&addr, "POST", "/predict", "{\"line\": \"A\"}");
    assert_eq!(code, 400);
    assert_eq!(err["error"]["code"], "E_SCHEMA_MISSING_COLUMN");
    let (code, err) = http(&addr, "POST", "/reload", "{\"bundle\": \"does/not/exist\"}");
    assert!(code >= 400, "{err}");
    // Outside the bundle folder: refused before anything is read, with the same answer
    // as for a path that does not exist.
    let (code, outside) = http(&addr, "POST", "/reload", "{\"bundle\": \".\"}");
    assert_eq!(code, 400, "{outside}");
    assert_eq!(outside["error"]["code"], "E_CONFIG_INVALID");
    assert_eq!(err["error"]["code"], outside["error"]["code"]);
    let (code, same) = http(
        &addr,
        "POST",
        "/reload",
        &serde_json::json!({"bundle": format!("bundles/{id}")}).to_string(),
    );
    assert_eq!(code, 200, "a bundle in the bundle folder can be reloaded: {same}");
    let (_, still) = http(&addr, "GET", "/health", "");
    assert_eq!(still["bundle_id"], health["bundle_id"], "a failed reload keeps the old bundle");
    child.kill().unwrap();
    let _ = child.wait();
}

/// review-cli: --keep collisions are refused, kept model inputs keep the file's
/// spelling, list arguments tolerate spaces and trailing commas, and JSON inputs
/// list their real keys when a --keep column is missing.
#[test]
fn predict_keep_details() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = new_project(tmp.path(), "keep", 1200);
    let run = json(&dir, &args(&["run", "--candidates", "linear"], &QUICK));
    let id = run["result"]["runs"][0]["run_id"].as_str().unwrap().to_string();
    ok(&dir, &["bundle", "build", "--run", &id, "--allow-unevaluated"]);
    std::fs::write(
        dir.join("in.csv"),
        std::fs::read_to_string(dir.join("data/defects.csv"))
            .unwrap()
            .lines()
            .take(4)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    let raw_temp: Vec<String> = std::fs::read_to_string(dir.join("in.csv"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(9).unwrap().to_string())
        .collect();
    fails(&dir, &["predict", "--input", "in.csv", "--keep", "label"], "E_CONFIG_INVALID", 2);
    let out = ok(&dir, &["predict", "--input", "in.csv", "--keep", "unit_id, temp_mean,"]);
    let rows: Vec<&str> = out.lines().collect();
    assert!(rows[0].starts_with("unit_id,temp_mean,label,probability"), "{out}");
    for (line, raw) in rows[1..].iter().zip(&raw_temp) {
        assert_eq!(line.split(',').nth(1).unwrap(), raw, "temp_mean is copied as written in the file");
    }
    std::fs::write(dir.join("in.jsonl"), "{\"customerId\": 1}\n").unwrap();
    let e = fails(
        &dir,
        &["predict", "--input", "in.jsonl", "--keep", "customer_id"],
        "E_SCHEMA_MISSING_COLUMN",
        3,
    );
    assert!(e["error"]["actual"].as_str().unwrap().contains("customerId"), "{e}");
    let cmp = json(&dir, &["compare", "--runs", &format!("{id},")]);
    assert_eq!(cmp["result"]["rows"].as_array().unwrap().len(), 1);
    // Kept values equal to a missing-value token are copied as written (NA is Namibia).
    let lines: Vec<String> = std::fs::read_to_string(dir.join("in.csv"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    let mut text = format!(
        "country,{}
",
        lines[0]
    );
    for (line, country) in lines[1..].iter().zip(["NA", "DE", "null"]) {
        text.push_str(&format!(
            "{country},{line}
"
        ));
    }
    std::fs::write(dir.join("countries.csv"), text).unwrap();
    let out = ok(&dir, &["predict", "--input", "countries.csv", "--keep", "country,unit_id"]);
    let kept: Vec<&str> = out.lines().skip(1).map(|l| l.split(',').nth(1).unwrap()).collect();
    assert_eq!(kept, ["NA", "DE", "null"], "{out}");
}
