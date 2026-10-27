//! Command implementations: thin presentation over `rustpy-workflow`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustpy_contracts::config::Candidate;
use rustpy_contracts::data::SplitName;
use rustpy_contracts::run::{RunRecord, RunStatus};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::{Device, LoadOptions, Predictor, Record};
use rustpy_workflow::bundle::BundleOptions;
use rustpy_workflow::pipeline::{Progress, RunOptions};
use rustpy_workflow::setup::{init_project, BindRequest};
use rustpy_workflow::{ConfigRequest, RunControl, Workspace};

use crate::output::{duration, f4, opt4, print_json, print_json_status, Table};
use crate::{BundleCommand, Cli, Command, ConfigArgs, DataCommand};

fn workspace(cli_project: &Option<PathBuf>) -> Result<Workspace> {
    match cli_project {
        Some(p) => Workspace::open(p),
        None => Workspace::discover("."),
    }
}

fn config_request(c: &ConfigArgs) -> ConfigRequest {
    ConfigRequest {
        profile: c.profile.clone(),
        overrides: c.overrides.clone(),
        override_tables: vec![],
    }
}

fn parse_split(s: &str) -> Result<SplitName> {
    SplitName::parse(s).ok_or_else(|| {
        RustpyError::new(ErrorCode::ConfigInvalid, Phase::Evaluate, format!("unknown split `{s}`"))
            .field("--split")
            .expected("valid or test")
    })
}

/// Ctrl-C: first press requests a clean stop at the next step boundary (with a
/// checkpoint); a second press exits immediately.
fn install_cancel_handler() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = flag.clone();
    let _ = ctrlc::set_handler(move || {
        if f.swap(true, Ordering::SeqCst) {
            eprintln!("\nsecond Ctrl-C: exiting immediately (the run will show as interrupted)");
            std::process::exit(130);
        }
        eprintln!("\nstopping at the next step boundary and writing a checkpoint… (Ctrl-C again to abort)");
    });
    flag
}

fn progress_printer(json: bool) -> impl FnMut(Progress) {
    move |p| {
        if json {
            return;
        }
        match p {
            Progress::Stage(s) => eprintln!("· {s}"),
            Progress::RunStarted { run_id, candidate, steps_planned } => {
                eprintln!("· {run_id}: training {candidate} (up to {steps_planned} steps)")
            }
            Progress::Epoch { run_id, epoch, epochs, train_loss, valid_loss, valid_metric } => {
                if epoch == 1 || epoch % 5 == 0 || epoch == epochs {
                    eprintln!(
                        "  {run_id} epoch {epoch:>3}/{epochs}: train loss {train_loss:.4}  valid loss {valid_loss:.4}  valid metric {valid_metric:.4}"
                    );
                }
            }
            Progress::RunFinished { run_id, status } => eprintln!("· {run_id}: {}", status.as_str()),
        }
    }
}

pub fn dispatch(cli: Cli) -> Result<u8> {
    let json = cli.json;
    match cli.command {
        Command::Init { name, recipe, sample_data } => {
            let r = init_project(&name, &recipe, sample_data)?;
            if json {
                print_json(&r);
            } else {
                println!("created project {}", r.project);
                println!("  config: {}", r.config);
                match &r.sample_data {
                    Some(d) => {
                        println!("  sample data: {d} (10,000 synthetic rows, bound with target `defect`)");
                        println!("\nnext:\n  cd {}\n  rustpy plan\n  rustpy run", name.display());
                    }
                    None => println!(
                        "\nnext:\n  cd {}\n  rustpy data bind <file.csv|file.parquet> --target <column>\n  rustpy plan",
                        name.display()
                    ),
                }
            }
            Ok(0)
        }
        Command::SampleData { out, rows, seed, shifted } => {
            let variant = if shifted {
                rustpy_workflow::sample::SampleVariant::Shifted
            } else {
                rustpy_workflow::sample::SampleVariant::Base
            };
            rustpy_workflow::sample::write_defects_csv(&out, rows, seed, variant)?;
            if json {
                print_json(&serde_json::json!({"path": out, "rows": rows, "seed": seed, "shifted": shifted}));
            } else {
                println!("wrote {rows} synthetic rows to {}", out.display());
            }
            Ok(0)
        }
        Command::Data(DataCommand::Bind { file, target, positive, split, group, id, exclude }) => {
            let ws = workspace(&cli.project)?;
            let r = ws.bind_data(&BindRequest { file, target, positive, split, group, id, exclude })?;
            if json {
                print_json(&r);
                return Ok(0);
            }
            println!("bound {} ({} rows, {} columns) as snapshot {}", r.source, r.snapshot.n_rows, r.snapshot.columns.len(), r.snapshot.snapshot_id);
            println!(
                "target `{}`: positive {:?} ({} rows), negative {:?} ({} rows)",
                r.target.name, r.target.positive_label, r.target.positives, r.target.negative_label, r.target.negatives
            );
            println!("split {}:", r.split);
            for (name, rows, pos) in &r.split_counts {
                println!("  {name:<5} {rows:>7} rows, {pos} positive");
            }
            for w in &r.warnings {
                println!("note: {w}");
            }
            println!("\nnext: rustpy plan");
            Ok(0)
        }
        Command::Data(DataCommand::Inspect { config }) | Command::Plan { config } => {
            let ws = workspace(&cli.project)?;
            let plan = ws.plan(&config_request(&config))?;
            if json {
                print_json(&plan);
                return Ok(0);
            }
            print_plan(&plan);
            Ok(0)
        }
        Command::Config { config } => {
            let ws = workspace(&cli.project)?;
            let rows = ws.effective_config(&config_request(&config))?;
            if json {
                print_json(&rows);
                return Ok(0);
            }
            let mut t = Table::new(&["setting", "value", "source"]);
            for (k, v, s) in rows {
                t.row(vec![k, v, s.to_string()]);
            }
            print!("{}", t.render());
            Ok(0)
        }
        Command::Run { config, candidates, no_cache, simulate_crash_after_steps, simulate_cancel_after_steps } => {
            let ws = workspace(&cli.project)?;
            let candidates = match candidates {
                Some(list) => Some(
                    list.iter()
                        .map(|c| {
                            Candidate::parse(c).ok_or_else(|| {
                                RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, format!("unknown candidate `{c}`"))
                                    .field("--candidates")
                                    .expected("linear, small-mlp")
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            };
            let cancel = install_cancel_handler();
            let opts = RunOptions {
                candidates,
                no_cache,
                control: RunControl {
                    cancel,
                    crash_after_steps: simulate_crash_after_steps,
                    cancel_after_steps: simulate_cancel_after_steps,
                },
            };
            let mut progress = progress_printer(json);
            let result = ws.run(&config_request(&config), &opts, &mut progress)?;
            // Exit code: 130 when cancelled, else the first failed run's error family.
            let code: u8 = if result.cancelled {
                130
            } else {
                result
                    .runs
                    .iter()
                    .find(|r| r.status != RunStatus::Completed)
                    .map(|r| r.error.as_ref().map_or(1, |e| e.code.exit_code().clamp(1, 255) as u8))
                    .unwrap_or(0)
            };
            if json {
                print_json_status(
                    &serde_json::json!({
                        "group": result.group,
                        "runs": result.runs,
                        "comparison": result.comparison,
                        "reused_stages": result.reused,
                    }),
                    code == 0,
                );
            } else {
                if !result.reused.is_empty() {
                    println!("reused cached stages (R1): {}", result.reused.join(", "));
                }
                print_comparison(&result.comparison);
                for r in &result.runs {
                    if let Some(e) = &r.error {
                        eprint!("{}", e.render());
                    }
                }
                if let Some(best) = &result.comparison.recommended {
                    println!(
                        "\nnext: lock the final candidate and use the test split once:\n  rustpy evaluate --run {best} --split test\n  rustpy bundle build --run {best}"
                    );
                }
            }
            Ok(code)
        }
        Command::Runs => {
            let ws = workspace(&cli.project)?;
            let runs = ws.list_runs()?;
            if json {
                print_json(&runs);
                return Ok(0);
            }
            if runs.is_empty() {
                println!("no runs yet — start with `rustpy run`");
                return Ok(0);
            }
            let mut t = Table::new(&["run", "candidate", "status", "lineage", "snapshot", "steps", "valid bal.acc", "valid AUC", "train time"]);
            for r in &runs {
                t.row(vec![
                    r.run_id.clone(),
                    r.candidate.to_string(),
                    r.status.as_str().into(),
                    lineage(r),
                    r.snapshot_id.clone(),
                    format!("{}/{}", r.steps_done, r.steps_planned),
                    opt4(r.metrics.as_ref().map(|m| m.valid.balanced_accuracy)),
                    opt4(r.metrics.as_ref().and_then(|m| m.valid.roc_auc)),
                    duration(r.timings.train_ms),
                ]);
            }
            print!("{}", t.render());
            for r in runs.iter().filter(|r| matches!(r.status, RunStatus::Interrupted | RunStatus::Cancelled | RunStatus::Failed)) {
                if let Some(e) = &r.error {
                    println!("{}: {} — next: {}", r.run_id, e.message, e.next_action.as_deref().unwrap_or("-"));
                }
            }
            Ok(0)
        }
        Command::Status { run } => {
            let ws = workspace(&cli.project)?;
            let r = ws.load_run(&run)?;
            if json {
                print_json(&r);
                return Ok(0);
            }
            print_run(&r);
            Ok(0)
        }
        Command::Compare { runs } => {
            let ws = workspace(&cli.project)?;
            let c = ws.compare(&runs)?;
            if json {
                print_json(&c);
            } else {
                print_comparison(&c);
            }
            Ok(0)
        }
        Command::Resume { run, warm_start, config, simulate_crash_after_steps, simulate_cancel_after_steps } => {
            let ws = workspace(&cli.project)?;
            let cancel = install_cancel_handler();
            let control = RunControl {
                cancel,
                crash_after_steps: simulate_crash_after_steps,
                cancel_after_steps: simulate_cancel_after_steps,
            };
            let mut progress = progress_printer(json);
            let r = ws.resume(&run, warm_start, &config_request(&config), &control, &mut progress)?;
            let code: u8 = match r.status {
                RunStatus::Completed => 0,
                RunStatus::Cancelled => 130,
                _ => r.error.as_ref().map_or(1, |e| e.code.exit_code().clamp(1, 255) as u8),
            };
            if json {
                print_json_status(&r, code == 0);
            } else {
                print_run(&r);
                if let Some(e) = &r.error {
                    eprint!("{}", e.render());
                }
            }
            Ok(code)
        }
        Command::Evaluate { run, bundle, split, allow_test_reuse, config } => {
            let ws = workspace(&cli.project)?;
            let split = parse_split(&split)?;
            let rec = match (run, bundle) {
                (Some(r), None) => {
                    if config.profile.is_some() || !config.overrides.is_empty() {
                        return Err(RustpyError::new(
                            ErrorCode::ConfigInvalid,
                            Phase::Evaluate,
                            "--set/--profile do not apply to `evaluate --run`: a run is evaluated with its recorded configuration (threshold, preprocessing)",
                        )
                        .field("--set/--profile")
                        .next("drop the flags; to evaluate other settings, train a run with them (`rustpy run --set ...`)"));
                    }
                    ws.evaluate_run(&r, split, allow_test_reuse)?
                }
                (None, Some(b)) => ws.evaluate_bundle(&b, split, &config_request(&config), allow_test_reuse)?,
                _ => {
                    return Err(RustpyError::new(ErrorCode::ConfigInvalid, Phase::Evaluate, "pass exactly one of --run or --bundle")
                        .next("rustpy evaluate --run run-002 --split test"));
                }
            };
            if json {
                print_json(&rec);
                return Ok(0);
            }
            let m = &rec.metrics;
            println!("{} on {} split of snapshot {} ({} rows):", rec.evaluation_id, rec.protocol.split.as_str(), rec.protocol.snapshot_id, m.n);
            let mut t = Table::new(&["metric", "value"]);
            for (k, v) in [
                ("balanced_accuracy", Some(m.balanced_accuracy)),
                ("precision", Some(m.precision)),
                ("recall", Some(m.recall)),
                ("f1", Some(m.f1)),
                ("roc_auc", m.roc_auc),
                ("log_loss", Some(m.log_loss)),
                ("brier", Some(m.brier)),
                ("accuracy", Some(m.accuracy)),
            ] {
                t.row(vec![k.into(), opt4(v)]);
            }
            print!("{}", t.render());
            println!(
                "  confusion at threshold {}: tp {} fp {} tn {} fn {}",
                m.threshold, m.confusion.tp, m.confusion.fp, m.confusion.tn, m.confusion.fn_
            );
            if let Some(l) = &rec.latency {
                println!("  single-row latency (transform + model): p50 {:.0} µs, p95 {:.0} µs", l.single_row_p50_us, l.single_row_p95_us);
            }
            match rec.test_use_index {
                Some(1) => println!("  test split: first use (recorded in the test ledger)"),
                Some(n) => println!("  test split: REUSE #{n} — not an unbiased estimate"),
                None => {}
            }
            for n in &rec.notes {
                println!("  note: {n}");
            }
            Ok(0)
        }
        Command::Bundle(BundleCommand::Build { run, profile, out, allow_unevaluated }) => {
            if profile != "cpu" {
                return Err(RustpyError::new(
                    ErrorCode::DeviceUnsupported,
                    Phase::Bundle,
                    format!("bundle profile `{profile}` is not available in this release"),
                )
                .field("--profile")
                .expected("cpu"));
            }
            let ws = workspace(&cli.project)?;
            let r = ws.build_bundle(&run, &BundleOptions { out, allow_unevaluated })?;
            if json {
                print_json(&r);
                return Ok(0);
            }
            if r.already_existed {
                println!("bundle {} already exists at {} (identical content)", r.manifest.bundle_id, r.path);
            } else {
                println!("built bundle {} at {}", r.manifest.bundle_id, r.path);
            }
            println!(
                "  {} files, evaluation: {}, self-check: {} ({} examples, max |Δp| {:.1e})",
                r.manifest.files.len(),
                r.manifest.evaluation_status,
                if r.check.passed { "passed" } else { "FAILED" },
                r.check.examples_checked,
                r.check.max_abs_probability_diff
            );
            println!("\nuse it from Rust:\n  let predictor = Predictor::load(\"{}\", LoadOptions::cpu())?;", r.path);
            println!("check it on the target machine:\n  rustpy bundle check {}", r.path);
            Ok(0)
        }
        Command::Bundle(BundleCommand::Check { dir, device }) => {
            let dev = Device::parse(&device).ok_or_else(|| {
                RustpyError::new(ErrorCode::DeviceUnsupported, Phase::Load, format!("unknown device `{device}`"))
                    .field("--device")
                    .expected("cpu")
            })?;
            let p = Predictor::load(&dir, LoadOptions::device(dev).with_self_check(true))?;
            let r = p.self_check()?;
            if json {
                print_json_status(&serde_json::json!({"check": r, "manifest": p.manifest()}), r.passed);
            } else {
                println!("bundle {} at {}", r.bundle_id, dir.display());
                println!("  files verified (sha256): {}", r.files_verified);
                println!("  runtime {} on {}/{} ({}), load {:.1} ms", r.runtime_version, r.os, r.arch, r.device, r.load_ms);
                println!("  example predictions: {} checked, max |Δp| {:.1e} (tolerance {:.0e}), label mismatches {}", r.examples_checked, r.max_abs_probability_diff, r.tolerance, r.label_mismatches);
                println!("  inputs: {} fields; output: P({} = {:?}), threshold {}", p.input_schema().fields.len(), p.output_schema().target, p.output_schema().positive_label, p.output_schema().threshold);
                println!("  evaluation: {}", p.manifest().evaluation_status);
                println!("  result: {}", if r.passed { "PASSED on this machine" } else { "FAILED" });
            }
            Ok(if r.passed { 0 } else { 4 })
        }
        Command::Bundle(BundleCommand::List) => {
            let ws = workspace(&cli.project)?;
            let list = ws.list_bundles();
            if json {
                print_json(&list.iter().map(|(p, m)| serde_json::json!({"path": p, "manifest": m})).collect::<Vec<_>>());
                return Ok(0);
            }
            let mut t = Table::new(&["path", "bundle id", "run", "candidate", "evaluation", "created"]);
            for (p, m) in list {
                match m {
                    Some(m) => t.row(vec![
                        ws.project.rel(&p),
                        m.bundle_id,
                        m.provenance.run_id,
                        m.provenance.candidate,
                        m.evaluation_status,
                        m.created_at,
                    ]),
                    None => t.row(vec![ws.project.rel(&p), "(unreadable)".into(), "".into(), "".into(), "".into(), "".into()]),
                }
            }
            print!("{}", t.render());
            Ok(0)
        }
        Command::Predict { bundle, input, output } => predict_file(&bundle, &input, output.as_deref(), json),
        Command::Report => {
            let ws = workspace(&cli.project)?;
            let path = ws.write_report()?;
            if json {
                print_json(&serde_json::json!({"html": path, "json": path.with_file_name("report.json")}));
            } else {
                println!("wrote {}", path.display());
                println!("      {}", path.with_file_name("report.json").display());
            }
            Ok(0)
        }
        Command::Doctor => doctor(&cli.project, json),
    }
}

fn lineage(r: &RunRecord) -> String {
    r.lineage
        .as_ref()
        .map(|l| {
            let level = serde_json::to_value(l.level).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
            format!("{level} of {}", l.parent_run_id)
        })
        .unwrap_or_else(|| "–".into())
}

fn print_plan(plan: &rustpy_workflow::setup::PlanReport) {
    let s = &plan.snapshot;
    println!("data      {} — {} rows × {} columns (snapshot {}, sha256 {})", s.source.path, s.n_rows, s.columns.len(), s.snapshot_id, &s.source.sha256[..12]);
    println!(
        "target    `{}`: positive {:?} = {} rows ({:.1}%), negative {:?} = {} rows",
        plan.target.name,
        plan.target.positive_label,
        plan.target.positives,
        100.0 * plan.target.positives as f64 / (plan.target.positives + plan.target.negatives) as f64,
        plan.target.negative_label,
        plan.target.negatives
    );
    println!("split     {} ({}):", plan.split_strategy, plan.split_id);
    for (name, rows, pos) in &plan.split_counts {
        println!("            {name:<5} {rows:>7} rows, {pos} positive");
    }
    println!("\ncolumns");
    let mut t = Table::new(&["column", "type", "missing", "distinct", "role"]);
    for c in &plan.columns {
        let prof = s.column(&c.name);
        let role = match &c.role {
            rustpy_workflow::data::ColumnRole::Feature { kind } => format!("feature ({})", kind.as_str()),
            rustpy_workflow::data::ColumnRole::Target => "target".into(),
            rustpy_workflow::data::ColumnRole::Id => "row id".into(),
            rustpy_workflow::data::ColumnRole::Group => "group (split key)".into(),
            rustpy_workflow::data::ColumnRole::Excluded { reason } => format!("excluded: {reason}"),
        };
        let role = if c.decided_by == "config" { format!("{role} [config]") } else { role };
        t.row(vec![
            c.name.clone(),
            prof.map(|p| p.engine_dtype.clone()).unwrap_or_default(),
            prof.map(|p| {
                let m = p.null_count + p.nan_count;
                if m == 0 { "0".into() } else { format!("{m} ({} null, {} NaN)", p.null_count, p.nan_count) }
            })
            .unwrap_or_default(),
            prof.map(|p| p.distinct.to_string()).unwrap_or_default(),
            role,
        ]);
    }
    print!("{}", t.render());
    println!("\npreprocessing (fitted on train rows only; transform {}, {} model inputs)", plan.transform_id, plan.feature_dim);
    let mut t = Table::new(&["feature", "encoding"]);
    for f in &plan.features {
        t.row(vec![f.name.clone(), f.encoding.clone()]);
    }
    print!("{}", t.render());
    println!("\ncandidates  ({})", plan.selection);
    let mut t = Table::new(&["candidate", "params", "hidden", "epochs", "batch", "lr", "l2", "class weight", "steps", "device"]);
    for c in &plan.candidates {
        let dev = plan
            .devices
            .iter()
            .find(|(k, _)| *k == c.candidate)
            .map(|(_, d)| format!("{} ({}, requested {})", d.device, d.backend, d.requested))
            .unwrap_or_default();
        t.row(vec![
            c.candidate.to_string(),
            c.parameters.to_string(),
            format!("{:?}", c.hidden),
            c.epochs.to_string(),
            c.batch_size.to_string(),
            c.learning_rate.to_string(),
            c.l2.to_string(),
            c.class_weight.clone(),
            c.steps.to_string(),
            dev,
        ]);
    }
    print!("{}", t.render());
    if let Some((_, d)) = plan.devices.first() {
        println!("\nexecution  precision {}, fallback {}, threads {}, estimated peak {:.0} MB of {} MB budget", d.precision, d.fallback, d.threads, d.estimated_peak_mb, d.memory_budget_mb);
        println!("           data placement: {}", d.data_placement);
        for step in &d.copy_path {
            println!("           copy: {step}");
        }
        for dev in &d.detected {
            println!("           detected: {} [{}] — {:?}{}", dev.name, dev.kind, dev.certification, dev.note.as_ref().map(|n| format!(" ({n})")).unwrap_or_default());
        }
        for n in plan.devices.iter().flat_map(|(_, d)| d.notes.iter()) {
            println!("           note: {n}");
        }
    }
    if !plan.reused.is_empty() {
        println!("\ncached stages reused: {}", plan.reused.join(", "));
    }
    for w in &plan.warnings {
        println!("note: {w}");
    }
    println!("\nnext: rustpy run");
}

fn print_comparison(c: &rustpy_workflow::Comparison) {
    println!("\ncomparison on the validation split (metric: {})", c.metric);
    let mut t = Table::new(&["run", "candidate", "status", &c.metric, "AUC", "F1", "log loss", "params", "p95 latency", "train time", "peak mem", "comparable"]);
    for r in &c.rows {
        t.row(vec![
            r.run_id.clone(),
            r.candidate.to_string(),
            r.status.as_str().into(),
            opt4(r.metric_value),
            opt4(r.valid.as_ref().and_then(|v| v.roc_auc)),
            opt4(r.valid.as_ref().map(|v| v.f1)),
            opt4(r.valid.as_ref().map(|v| v.log_loss)),
            r.parameters.to_string(),
            r.p95_latency_us.map(|l| format!("{l:.0} µs")).unwrap_or_else(|| "–".into()),
            duration(r.train_ms),
            r.peak_rss_mb.map(|m| format!("{m:.0} MB")).unwrap_or_else(|| "–".into()),
            if r.comparable { "yes".into() } else { "no".into() },
        ]);
    }
    print!("{}", t.render());
    for r in &c.rows {
        for n in &r.notes {
            println!("  {}: {n}", r.run_id);
        }
    }
    match &c.recommended {
        Some(r) => println!("recommended: {r} (highest validation {} among comparable runs meeting constraints)", c.metric),
        None => println!("recommended: none (no comparable completed run meets the constraints)"),
    }
    for n in &c.notes {
        println!("note: {n}");
    }
}

fn print_run(r: &RunRecord) {
    println!("{} — {} — {}", r.run_id, r.candidate, r.status.as_str());
    println!("  lineage: {}", lineage(r));
    println!("  snapshot {} · split {} · transform {} · config {}", r.snapshot_id, r.split_id, r.transform_id, &r.config_hash[..12]);
    println!("  steps {}/{} · stop: {} · train {} · data {} · fit {}", r.steps_done, r.steps_planned, r.stop_reason.as_deref().unwrap_or("–"), duration(r.timings.train_ms), duration(r.timings.data_ms), duration(r.timings.fit_transform_ms));
    if !r.reused_stages.is_empty() {
        println!("  reused stages: {}", r.reused_stages.join(", "));
    }
    println!("  device: requested {} → {} {} {} (threads {})", r.device_plan.requested, r.device_plan.backend, r.device_plan.device, r.device_plan.precision, r.device_plan.threads);
    if let Some(rc) = &r.device_receipt {
        println!("  receipt: {} {} {}, peak RSS {}, matches plan: {}", rc.backend, rc.device, rc.precision, rc.peak_rss_mb.map(|m| format!("{m:.0} MB")).unwrap_or_else(|| "–".into()), rc.matches_plan);
    }
    if let Some(m) = &r.metrics {
        println!("  best epoch {} · valid: bal.acc {} AUC {} F1 {} log loss {} · train bal.acc {}", m.best_epoch, f4(m.valid.balanced_accuracy), opt4(m.valid.roc_auc), f4(m.valid.f1), f4(m.valid.log_loss), f4(m.train.balanced_accuracy));
    }
    if let Some(d) = r.engine_parity_max_abs_diff {
        println!("  engine vs runtime max |Δlogit| on validation rows: {d:.2e}");
    }
    for a in &r.artifacts {
        println!("  artifact {} ({} bytes, sha256 {})", a.path, a.bytes, &a.sha256[..12]);
    }
    if let Some(e) = &r.error {
        println!("  error: {} {}", e.code, e.message);
        for k in &e.retained_artifacts {
            println!("  kept: {k}");
        }
        if let Some(n) = &e.next_action {
            println!("  next: {n}");
        }
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn predict_file(bundle: &Path, input: &Path, output: Option<&Path>, json: bool) -> Result<u8> {
    let predictor = Predictor::load(bundle, LoadOptions::cpu())?;
    let ext = input.extension().and_then(|e| e.to_str()).map(str::to_lowercase).unwrap_or_default();
    let id_field = predictor.input_schema().passthrough_id.clone();
    let (preds, ids): (Vec<rustpy_runtime::Prediction>, Vec<Option<String>>) = match ext.as_str() {
        "json" | "jsonl" | "ndjson" => {
            let text = std::fs::read_to_string(input).map_err(|e| RustpyError::io(Phase::Predict, input, e))?;
            let values: Vec<serde_json::Value> = if ext == "json" {
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(serde_json::Value::Array(a)) => a,
                    Ok(v) => vec![v],
                    Err(e) => {
                        return Err(RustpyError::new(ErrorCode::DataReadFailed, Phase::Predict, format!("invalid JSON: {e}")).field(input.display().to_string()));
                    }
                }
            } else {
                text.lines()
                    .filter(|l| !l.trim().is_empty())
                    .enumerate()
                    .map(|(i, l)| {
                        serde_json::from_str(l).map_err(|e| {
                            RustpyError::new(ErrorCode::DataReadFailed, Phase::Predict, format!("line {}: invalid JSON: {e}", i + 1))
                                .field(input.display().to_string())
                        })
                    })
                    .collect::<Result<_>>()?
            };
            let records = values.iter().map(Record::from_json).collect::<Result<Vec<_>>>()?;
            let ids = records
                .iter()
                .map(|r| id_field.as_ref().and_then(|f| r.get(f)).map(value_text))
                .collect();
            (predictor.predict(&records)?, ids)
        }
        _ => {
            let table = rustpy_workflow::data::read_for_prediction(input, predictor.input_schema(), &predictor.bundle().transform)?;
            let ids = match id_field.as_ref().and_then(|f| table.column(f)) {
                Some(col) => (0..table.n_rows())
                    .map(|i| match col.get(i) {
                        rustpy_runtime::ValueRef::Null => Some(String::new()),
                        rustpy_runtime::ValueRef::Text(s) => Some(s.to_string()),
                        rustpy_runtime::ValueRef::Int(v) => Some(v.to_string()),
                        rustpy_runtime::ValueRef::Float(v) => Some(v.to_string()),
                        rustpy_runtime::ValueRef::Bool(v) => Some(v.to_string()),
                    })
                    .collect(),
                None => vec![None; table.n_rows()],
            };
            (predictor.predict_table(&table)?, ids)
        }
    };
    // The id column is used only when the input has it; otherwise rows are numbered.
    let id_header = match &id_field {
        Some(f) if ids.iter().any(Option::is_some) => f.clone(),
        _ => "row".into(),
    };
    if json && output.is_none() {
        // One JSON document on stdout.
        let rows: Vec<serde_json::Value> = preds
            .iter()
            .zip(&ids)
            .enumerate()
            .map(|(i, (p, id))| {
                let idv = id.clone().map(serde_json::Value::String).unwrap_or(serde_json::Value::from(i));
                serde_json::json!({ id_header.as_str(): idv, "label": p.label, "probability": p.probability, "is_positive": p.is_positive })
            })
            .collect();
        print_json(&serde_json::json!({"bundle_id": predictor.bundle_id(), "predictions": rows}));
        return Ok(0);
    }
    // The file format follows --output's extension (.jsonl or CSV).
    let jsonl = output.is_some_and(|o| o.extension().is_some_and(|e| e == "jsonl" || e == "ndjson"));
    let mut text = String::new();
    if jsonl {
        for (i, (p, id)) in preds.iter().zip(&ids).enumerate() {
            let idv = id.clone().map(serde_json::Value::String).unwrap_or(serde_json::Value::from(i));
            text.push_str(&serde_json::json!({ id_header.as_str(): idv, "label": p.label, "probability": p.probability, "is_positive": p.is_positive }).to_string());
            text.push('\n');
        }
    } else {
        text.push_str(&format!("{},label,probability\n", csv_escape(&id_header)));
        for (i, (p, id)) in preds.iter().zip(&ids).enumerate() {
            let idv = id.clone().unwrap_or_else(|| i.to_string());
            text.push_str(&format!("{},{},{:.6}\n", csv_escape(&idv), csv_escape(&p.label), p.probability));
        }
    }
    match output {
        Some(path) => {
            std::fs::write(path, text).map_err(|e| RustpyError::io(Phase::Predict, path, e))?;
            let pos = preds.iter().filter(|p| p.is_positive).count();
            if json {
                print_json(&serde_json::json!({"path": path, "rows": preds.len(), "positive": pos, "bundle_id": predictor.bundle_id()}));
            } else {
                eprintln!("wrote {} predictions ({pos} positive) to {} using bundle {}", preds.len(), path.display(), predictor.bundle_id());
            }
        }
        None => {
            let _ = std::io::stdout().write_all(text.as_bytes());
        }
    }
    Ok(0)
}

fn value_text(v: &rustpy_runtime::Value) -> String {
    match v {
        rustpy_runtime::Value::Null => String::new(),
        rustpy_runtime::Value::Bool(b) => b.to_string(),
        rustpy_runtime::Value::Int(i) => i.to_string(),
        rustpy_runtime::Value::Float(f) => f.to_string(),
        rustpy_runtime::Value::Text(s) => s.clone(),
    }
}

fn doctor(project: &Option<PathBuf>, json: bool) -> Result<u8> {
    use rustpy_workflow::device::{detect_devices, environment, COMPILED_BACKENDS};
    let env = environment();
    let devices = detect_devices();
    let ws = workspace(project);
    let project_status = match &ws {
        Ok(w) => {
            let probe = w.project.state_dir().join(".doctor-write-test");
            let writable = std::fs::create_dir_all(w.project.state_dir()).and_then(|_| std::fs::write(&probe, b"ok")).is_ok();
            let _ = std::fs::remove_file(&probe);
            let cfg = w.project.resolve_config(&ConfigRequest::default());
            serde_json::json!({
                "root": w.project.root,
                "state_writable": writable,
                "config_valid": cfg.is_ok(),
                "config_error": cfg.err().map(|e| e.message),
            })
        }
        Err(e) => serde_json::json!({"found": false, "hint": e.next_action}),
    };
    if json {
        print_json(&serde_json::json!({
            "environment": env,
            "compiled_backends": COMPILED_BACKENDS,
            "devices": devices,
            "project": project_status,
            "python_required": false,
        }));
        return Ok(0);
    }
    println!("rustpy {}  ({}/{})", env.rustpy_version, env.os, env.arch);
    println!("engine      {}", env.engine);
    println!("backends    {}", COMPILED_BACKENDS.join(", "));
    println!("cpu         {} ({} logical)", env.cpu_model.as_deref().unwrap_or("unknown"), env.logical_cpus);
    for d in &devices {
        println!("device      {} [{}] — {:?}{}", d.name, d.kind, d.certification, d.note.as_ref().map(|n| format!(": {n}")).unwrap_or_default());
    }
    println!("python      not required (training and prediction are pure Rust)");
    match ws {
        Ok(w) => {
            println!("project     {}", w.project.root.display());
            println!("            state dir writable: {}", project_status["state_writable"]);
            match w.project.resolve_config(&ConfigRequest::default()) {
                Ok(_) => println!("            rustpy.toml: valid"),
                Err(e) => println!("            rustpy.toml: INVALID — {}", e.message),
            }
        }
        Err(_) => println!("project     none here (create one with `rustpy init <name> --sample-data`)"),
    }
    Ok(0)
}
