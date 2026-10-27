//! Static, offline project report (HTML + JSON). No server, no account.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use rustpy_contracts::data::DataSnapshot;
use rustpy_contracts::evaluation::{EvaluationRecord, EvaluationSubject};
use rustpy_contracts::manifest::BundleManifest;
use rustpy_contracts::run::{RunRecord, RunStatus};
use rustpy_contracts::time::now_rfc3339;
use rustpy_contracts::{Phase, Result, RUSTPY_VERSION};
use rustpy_runtime::{LoadOptions, Predictor};
use serde::Serialize;

use crate::device::ENGINE;
use crate::pipeline::{Comparison, RunGroup, Workspace};
use crate::project::{read_json, write_atomic, write_json, ConfigRequest};

#[derive(Debug, Clone, Serialize)]
pub struct BundleStatus {
    pub path: String,
    pub manifest: Option<BundleManifest>,
    pub check_passed: Option<bool>,
    pub check_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectReport {
    pub generated_at: String,
    pub rustpy_version: String,
    pub engine: String,
    pub project: String,
    pub snapshots: Vec<DataSnapshot>,
    pub runs: Vec<RunRecord>,
    pub groups: Vec<(RunGroup, Comparison)>,
    pub evaluations: Vec<EvaluationRecord>,
    pub bundles: Vec<BundleStatus>,
    /// Per run: config hash and the effective settings that differ from the first
    /// run of its group.
    pub config_diffs: Vec<ConfigDiff>,
    /// Missing values, split warnings, dropped features.
    pub data_issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigDiff {
    pub group_id: String,
    pub run_id: String,
    pub reference_run: String,
    pub config_hash: String,
    pub differences: Vec<String>,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn f4(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.4}")).unwrap_or_else(|| "–".into())
}

fn sparkline(values: &[f64]) -> String {
    if values.len() < 2 {
        return String::new();
    }
    let (w, h) = (160.0, 36.0);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-12);
    let pts: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = i as f64 / (values.len() - 1) as f64 * w;
            let y = h - (v - min) / span * (h - 4.0) - 2.0;
            format!("{x:.1},{y:.1}")
        })
        .collect();
    format!(
        "<svg class=\"spark\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" role=\"img\" aria-label=\"validation loss per epoch\"><polyline fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" points=\"{}\"/></svg>",
        pts.join(" ")
    )
}

impl Workspace {
    pub fn collect_report(&self) -> Result<ProjectReport> {
        let mut snapshots: Vec<DataSnapshot> = fs::read_dir(self.project.dir("snapshots"))
            .map(|it| {
                it.flatten()
                    .filter_map(|e| read_json(&e.path().join("snapshot.json"), Phase::Report).ok())
                    .collect()
            })
            .unwrap_or_default();
        snapshots.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        let runs = self.list_runs()?;
        let cfg = self.project.resolve_config(&ConfigRequest::default())?.config;
        let mut group_files: Vec<PathBuf> = fs::read_dir(self.project.dir("groups"))
            .map(|it| {
                it.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "json"))
                    .collect()
            })
            .unwrap_or_default();
        group_files.sort_by_key(|p| {
            crate::pipeline::id_number(p.file_stem().and_then(|s| s.to_str()).unwrap_or_default())
        });
        let mut groups = Vec::new();
        for gf in group_files {
            let g: RunGroup = read_json(&gf, Phase::Report)?;
            let members: Vec<RunRecord> = runs.iter().filter(|r| g.run_ids.contains(&r.run_id)).cloned().collect();
            let ids: Vec<String> = members.iter().map(|r| r.run_id.clone()).collect();
            let cmp = self.compare(&ids).unwrap_or_else(|_| Comparison {
                metric: cfg.selection.metric.as_str().to_string(),
                protocol: None,
                rows: vec![],
                recommended: None,
                notes: vec![],
            });
            groups.push((g, cmp));
        }
        let bundles = self
            .list_bundles()
            .into_iter()
            .map(|(path, manifest)| {
                let (passed, err) = match Predictor::load(&path, LoadOptions::cpu()).and_then(|p| p.self_check()) {
                    Ok(r) => (Some(r.passed), None),
                    Err(e) => (Some(false), Some(format!("{}: {}", e.code, e.message))),
                };
                BundleStatus {
                    path: self.project.rel(&path),
                    manifest,
                    check_passed: passed,
                    check_error: err,
                }
            })
            .collect();
        // Configuration differences within each group.
        let mut config_diffs = Vec::new();
        for (g, _) in &groups {
            let resolved: Vec<(String, rustpy_contracts::config::ResolvedConfig)> = g
                .run_ids
                .iter()
                .filter_map(|id| {
                    read_json(&self.project.run_dir(id).join("resolved.json"), Phase::Report)
                        .ok()
                        .map(|r| (id.clone(), r))
                })
                .collect();
            let Some((ref_id, reference)) = resolved.first() else { continue };
            let base: std::collections::BTreeMap<String, String> =
                reference.effective_values().into_iter().map(|(k, v, _)| (k, v)).collect();
            for (id, r) in &resolved {
                let differences = r
                    .effective_values()
                    .into_iter()
                    .filter(|(k, v, _)| base.get(k) != Some(v))
                    .map(|(k, v, src)| format!("{k} = {v} ({src}; {ref_id}: {})", base.get(&k).map(String::as_str).unwrap_or("unset")))
                    .collect();
                config_diffs.push(ConfigDiff {
                    group_id: g.group_id.clone(),
                    run_id: id.clone(),
                    reference_run: ref_id.clone(),
                    config_hash: r.config_hash.clone(),
                    differences,
                });
            }
        }
        // Data issues: missing values per snapshot, split warnings, dropped features.
        let mut data_issues = Vec::new();
        for snap in &snapshots {
            for c in &snap.columns {
                let missing = c.null_count + c.nan_count;
                if missing > 0 || c.inf_count > 0 {
                    data_issues.push(format!(
                        "{}: `{}` has {} null, {} NaN, {} infinite of {} rows",
                        snap.snapshot_id, c.name, c.null_count, c.nan_count, c.inf_count, snap.n_rows
                    ));
                }
            }
            for l in &snap.limitations {
                data_issues.push(format!("{}: {l}", snap.snapshot_id));
            }
        }
        if let Ok(it) = fs::read_dir(self.project.dir("splits")) {
            for e in it.flatten() {
                if let Ok(sp) = read_json::<rustpy_contracts::data::SplitManifest>(&e.path(), Phase::Report) {
                    for w in &sp.warnings {
                        data_issues.push(format!("{}: {w}", sp.split_id));
                    }
                }
            }
        }
        if let Ok(it) = fs::read_dir(self.project.dir("transforms")) {
            for e in it.flatten() {
                if let Ok(fa) = read_json::<crate::fit::FittedArtifacts>(&e.path().join("fitted.json"), Phase::Report) {
                    for d in &fa.transform.dropped {
                        data_issues.push(format!("{}: feature `{}` dropped ({})", fa.transform_id, d.name, d.reason));
                    }
                }
            }
        }
        data_issues.sort();
        data_issues.dedup();
        Ok(ProjectReport {
            config_diffs,
            data_issues,
            generated_at: now_rfc3339(),
            rustpy_version: RUSTPY_VERSION.to_string(),
            engine: ENGINE.to_string(),
            project: self.project.root.display().to_string(),
            snapshots,
            runs,
            groups,
            evaluations: self.evaluations(),
            bundles,
        })
    }

    /// Writes `.rustpy/reports/index.html` and `report.json`; returns the HTML path.
    pub fn write_report(&self) -> Result<PathBuf> {
        let r = self.collect_report()?;
        let dir = self.project.dir("reports");
        write_json(&dir.join("report.json"), &r)?;
        let html = render_html(&r);
        let path = dir.join("index.html");
        write_atomic(&path, html.as_bytes())?;
        Ok(path)
    }
}

fn status_class(s: RunStatus) -> &'static str {
    match s {
        RunStatus::Completed => "ok",
        RunStatus::Failed | RunStatus::Interrupted => "bad",
        RunStatus::Cancelled => "warn",
        _ => "muted",
    }
}

pub fn render_html(r: &ProjectReport) -> String {
    let mut h = String::new();
    let _ = write!(
        h,
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RustPy project report</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#ffffff;--ok:#1a7f4b;--bad:#b42318;--warn:#a15c00;--accent:#3355cc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1e1e1c;--ok:#4cc38a;--bad:#ff7b72;--warn:#e3a33b;--accent:#8ea8ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
.meta{{color:var(--muted);font-size:13px}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:var(--card);font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}}
th{{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.02em}}
.ok{{color:var(--ok)}} .bad{{color:var(--bad)}} .warn{{color:var(--warn)}} .muted{{color:var(--muted)}}
.note{{white-space:normal;color:var(--muted);font-size:12px;max-width:420px}}
.spark{{color:var(--accent)}} code{{font-size:12px}}
.callout{{border-left:3px solid var(--accent);padding:8px 12px;background:var(--card);margin:8px 0}}
</style></head><body><main>
<h1>RustPy project report</h1>
<div class="meta">{} · generated {} · rustpy {} · engine {}</div>
<div class="callout">Metrics come from recorded runs and evaluations on this machine. Validation metrics rank candidates; a test evaluation is the final, once-only estimate. Results on different data snapshots or splits are marked not comparable.</div>
"#,
        esc(&r.project),
        esc(&r.generated_at),
        esc(&r.rustpy_version),
        esc(&r.engine)
    );

    h.push_str("<h2>Data snapshots</h2><div class=\"wrap\"><table><tr><th>snapshot</th><th>source</th><th>rows</th><th>columns</th><th>sha256</th><th>storage</th><th>created</th></tr>");
    for s in &r.snapshots {
        let storage = match &s.storage {
            rustpy_contracts::data::SnapshotStorage::Copy { .. } => "copy (replayable)",
            rustpy_contracts::data::SnapshotStorage::Reference => "reference",
        };
        let _ = write!(
            h,
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>",
            esc(&s.snapshot_id),
            esc(&s.source.path),
            s.n_rows,
            s.columns.len(),
            &s.source.sha256[..12],
            storage,
            esc(&s.created_at)
        );
    }
    h.push_str("</table></div>");

    h.push_str("<h2>Runs</h2><div class=\"wrap\"><table><tr><th>run</th><th>candidate</th><th>status</th><th>lineage</th><th>snapshot / split</th><th>valid bal. acc</th><th>valid F1</th><th>valid AUC</th><th>valid log loss</th><th>best epoch</th><th>steps</th><th>train time</th><th>engine Δ</th><th>p95 latency</th><th>valid loss curve</th><th>notes</th></tr>");
    for run in &r.runs {
        let v = run.metrics.as_ref().map(|m| &m.valid);
        let lineage = run
            .lineage
            .as_ref()
            .map(|l| format!("{} of {}", serde_json::to_value(l.level).ok().and_then(|x| x.as_str().map(String::from)).unwrap_or_default(), l.parent_run_id))
            .unwrap_or_else(|| "–".into());
        let curve: Vec<f64> = run.history.iter().map(|e| e.valid_loss).collect();
        let notes = match &run.error {
            Some(e) => format!("{}: {}{}", e.code, e.message, e.next_action.as_ref().map(|n| format!(" — next: {n}")).unwrap_or_default()),
            None => run.stop_reason.clone().map(|s| format!("stopped: {s}")).unwrap_or_default(),
        };
        let _ = write!(
            h,
            "<tr><td><code>{}</code></td><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td><code>{} / {}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td><td>{:.1} s</td><td>{}</td><td>{}</td><td>{}</td><td class=\"note\">{}</td></tr>",
            esc(&run.run_id),
            run.candidate,
            status_class(run.status),
            run.status.as_str(),
            esc(&lineage),
            esc(&run.snapshot_id),
            esc(&run.split_id),
            f4(v.map(|m| m.balanced_accuracy)),
            f4(v.map(|m| m.f1)),
            f4(v.and_then(|m| m.roc_auc)),
            f4(v.map(|m| m.log_loss)),
            run.metrics.as_ref().map(|m| m.best_epoch.to_string()).unwrap_or_else(|| "–".into()),
            run.steps_done,
            run.steps_planned,
            run.timings.train_ms / 1000.0,
            run.engine_parity_max_abs_diff.map(|d| format!("{d:.1e}")).unwrap_or_else(|| "–".into()),
            run.latency.as_ref().map(|l| format!("{:.0} µs", l.single_row_p95_us)).unwrap_or_else(|| "–".into()),
            sparkline(&curve),
            esc(&notes)
        );
    }
    h.push_str("</table></div><p class=\"meta\">engine Δ = max |logit| difference between the training engine and the runtime on validation rows for the kept weights.</p>");

    h.push_str("<h2>Comparisons (validation split)</h2>");
    for (g, c) in &r.groups {
        let _ = write!(
            h,
            "<p><b>{}</b> · {} · snapshot <code>{}</code> · split <code>{}</code> · recommended: <b>{}</b></p><div class=\"wrap\"><table><tr><th>run</th><th>candidate</th><th>{}</th><th>params</th><th>p95 latency</th><th>comparable</th><th>constraints</th><th>notes</th></tr>",
            esc(&g.group_id),
            esc(&g.created_at),
            esc(&g.snapshot_id),
            esc(&g.split_id),
            c.recommended.as_deref().unwrap_or("none"),
            esc(&c.metric)
        );
        for row in &c.rows {
            let _ = write!(
                h,
                "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td class=\"{}\">{}</td><td class=\"note\">{}</td></tr>",
                esc(&row.run_id),
                row.candidate,
                f4(row.metric_value),
                row.parameters,
                row.p95_latency_us.map(|l| format!("{l:.0} µs")).unwrap_or_else(|| "–".into()),
                if row.comparable { "ok" } else { "bad" },
                if row.comparable { "yes" } else { "no" },
                if row.meets_constraints { "ok" } else { "warn" },
                if row.meets_constraints { "met" } else { "not met" },
                esc(&row.notes.join("; "))
            );
        }
        h.push_str("</table></div>");
    }

    h.push_str("<h2>Evaluations</h2><div class=\"wrap\"><table><tr><th>id</th><th>subject</th><th>split</th><th>snapshot</th><th>n</th><th>bal. acc</th><th>precision</th><th>recall</th><th>F1</th><th>AUC</th><th>confusion (tp/fp/tn/fn)</th><th>test use</th><th>notes</th></tr>");
    for e in &r.evaluations {
        let subject = match &e.subject {
            EvaluationSubject::Run { run_id } => format!("run {run_id}"),
            EvaluationSubject::Bundle { bundle_id, path } => format!("bundle {bundle_id} ({path})"),
        };
        let m = &e.metrics;
        let reuse = match e.test_use_index {
            Some(1) => "<span class=\"ok\">first use</span>".to_string(),
            Some(n) => format!("<span class=\"bad\">reuse #{n}</span>"),
            None => "–".into(),
        };
        let _ = write!(
            h,
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}/{}/{}</td><td>{}</td><td class=\"note\">{}</td></tr>",
            esc(&e.evaluation_id),
            esc(&subject),
            e.protocol.split.as_str(),
            esc(&e.protocol.snapshot_id),
            m.n,
            f4(Some(m.balanced_accuracy)),
            f4(Some(m.precision)),
            f4(Some(m.recall)),
            f4(Some(m.f1)),
            f4(m.roc_auc),
            m.confusion.tp,
            m.confusion.fp,
            m.confusion.tn,
            m.confusion.fn_,
            reuse,
            esc(&e.notes.join("; "))
        );
    }
    h.push_str("</table></div>");

    h.push_str("<h2>Bundles</h2><div class=\"wrap\"><table><tr><th>path</th><th>bundle id</th><th>run</th><th>evaluation</th><th>check on this machine</th></tr>");
    for b in &r.bundles {
        let (id, run, status) = match &b.manifest {
            Some(m) => (m.bundle_id.clone(), m.provenance.run_id.clone(), m.evaluation_status.clone()),
            None => ("(unreadable manifest)".into(), "–".into(), "–".into()),
        };
        let check = match (b.check_passed, &b.check_error) {
            (Some(true), _) => "<span class=\"ok\">passed</span>".to_string(),
            (_, Some(e)) => format!("<span class=\"bad\">failed</span> <span class=\"note\">{}</span>", esc(e)),
            _ => "<span class=\"bad\">failed</span>".into(),
        };
        let _ = write!(
            h,
            "<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>",
            esc(&b.path),
            esc(&id),
            esc(&run),
            esc(&status),
            check
        );
    }
    h.push_str("</table></div>");

    h.push_str("<h2>Configuration</h2><div class=\"wrap\"><table><tr><th>group</th><th>run</th><th>config hash</th><th>differences from the group's first run</th></tr>");
    for d in &r.config_diffs {
        let _ = write!(
            h,
            "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td class=\"note\">{}</td></tr>",
            esc(&d.group_id),
            esc(&d.run_id),
            &d.config_hash[..12.min(d.config_hash.len())],
            if d.differences.is_empty() { "—".to_string() } else { esc(&d.differences.join("; ")) }
        );
    }
    h.push_str("</table></div>");

    h.push_str("<h2>Data issues</h2>");
    if r.data_issues.is_empty() {
        h.push_str("<p class=\"meta\">none recorded</p>");
    } else {
        h.push_str("<ul>");
        for i in &r.data_issues {
            let _ = write!(h, "<li class=\"note\">{}</li>", esc(i));
        }
        h.push_str("</ul>");
    }

    h.push_str("<h2>Devices</h2><div class=\"wrap\"><table><tr><th>run</th><th>requested</th><th>used</th><th>precision</th><th>threads</th><th>peak RSS</th><th>matches plan</th><th>detected devices</th></tr>");
    for run in &r.runs {
        let p = &run.device_plan;
        let rc = run.device_receipt.as_ref();
        let detected: Vec<String> = p
            .detected
            .iter()
            .map(|d| format!("{} ({:?})", d.name, d.certification).to_lowercase())
            .collect();
        let _ = write!(
            h,
            "<tr><td><code>{}</code></td><td>{}</td><td>{} / {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"note\">{}</td></tr>",
            esc(&run.run_id),
            esc(&p.requested),
            esc(&p.backend),
            esc(&p.device),
            esc(&p.precision),
            p.threads,
            rc.and_then(|r| r.peak_rss_mb).map(|m| format!("{m:.0} MB")).unwrap_or_else(|| "–".into()),
            rc.map(|r| if r.matches_plan { "yes" } else { "NO" }).unwrap_or("–"),
            esc(&detected.join("; "))
        );
    }
    h.push_str("</table></div></main></body></html>\n");
    h
}
