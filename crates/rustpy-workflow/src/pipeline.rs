//! Orchestration of the stages: plan, run, resume, evaluate, compare.
//! The CLI and the Rust SDK both call these functions; neither has its own logic.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustpy_contracts::config::{Candidate, ProjectConfig, ResolvedConfig, ValueSource};
use rustpy_contracts::data::{DataSnapshot, SplitManifest, SplitName};
use rustpy_contracts::device::DevicePlan;
use rustpy_contracts::evaluation::{BinaryMetrics, ComparisonProtocol, EvaluationRecord, EvaluationSubject, LatencyStats};
use rustpy_contracts::hash::sha256_file;
use rustpy_contracts::run::{ArtifactRef, Lineage, ResumeLevel, RunMetrics, RunRecord, RunStatus, Timings};
use rustpy_contracts::time::now_rfc3339;
use rustpy_contracts::{ErrorCode, METRICS_VERSION, Phase, RUN_FORMAT_VERSION, RUSTPY_VERSION, Result, RustpyError};
use rustpy_runtime::model::{DenseModel, sigmoid};
use rustpy_runtime::{LoadOptions, Matrix, Predictor, Table, apply_transform};
use serde::{Deserialize, Serialize};

use crate::data::{FeaturePlan, load_snapshot, plan_columns, snapshot_source};
use crate::device::{ENGINE, environment, estimate_peak_mb, plan_device, receipt};
use crate::exposure::{ExposureCheck, RowKeys, schema_content_columns};
use crate::fit::{FittedArtifacts, fit_transform, transform_id};
use crate::metrics::binary_metrics;
use crate::project::{ConfigRequest, Project, append_jsonl, read_json, write_json};
use crate::split::{labels, make_split};
use crate::train::{
    ResumeKey, RunControl, StartFrom, TrainInputs, TrainOutcome, TrainState, latest_checkpoint, list_checkpoints, model_config,
    platform_fingerprint, train, verify_checkpoint,
};

/// Ledger entry for one use of a test split (see `check_test_reuse`).
#[derive(Debug, Default)]
struct TestLedger {
    /// The same model was already evaluated on exactly these rows.
    existing: Option<EvaluationRecord>,
    index: u32,
    shared_with: Vec<String>,
    notes: Vec<String>,
}

/// All stages before training, either computed or reused from cache (R1).
#[derive(Debug, Clone)]
pub struct Prepared {
    pub resolved: ResolvedConfig,
    pub snapshot: DataSnapshot,
    pub table: Table,
    pub features: FeaturePlan,
    pub split: SplitManifest,
    pub fitted: FittedArtifacts,
    pub reused: Vec<String>,
    pub data_ms: f64,
    pub fit_ms: f64,
}

/// Transformed rows of one split.
#[derive(Debug, Clone)]
pub struct SplitData {
    pub x: Matrix,
    pub y: Vec<bool>,
    pub rows: Vec<usize>,
}

/// Progress callback events for interactive front-ends.
#[derive(Debug, Clone)]
pub enum Progress {
    Stage(String),
    RunStarted {
        run_id: String,
        candidate: Candidate,
        steps_planned: u64,
    },
    Epoch {
        run_id: String,
        epoch: usize,
        epochs: usize,
        train_loss: f64,
        valid_loss: f64,
        valid_metric: f64,
    },
    RunFinished {
        run_id: String,
        status: RunStatus,
        /// `early_stopping`, `epochs` or `time_budget` for completed runs.
        stop_reason: Option<String>,
        /// Epoch whose weights were kept (lowest validation loss).
        best_epoch: Option<usize>,
        epochs_done: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunGroup {
    pub group_id: String,
    pub created_at: String,
    pub run_ids: Vec<String>,
    /// Candidates this group was asked to train (some may not have started).
    #[serde(default)]
    pub planned: Vec<Candidate>,
    pub snapshot_id: String,
    pub split_id: String,
    pub transform_id: String,
    pub config_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub run_id: String,
    pub candidate: Candidate,
    pub status: RunStatus,
    pub snapshot_id: String,
    pub valid: Option<BinaryMetrics>,
    pub metric_value: Option<f64>,
    pub parameters: usize,
    pub model_kib: f64,
    pub p95_latency_us: Option<f64>,
    /// Peak resident memory while this run trained.
    pub peak_rss_mb: Option<f64>,
    pub train_ms: f64,
    pub best_epoch: Option<usize>,
    pub comparable: bool,
    pub notes: Vec<String>,
    pub meets_constraints: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub metric: String,
    pub protocol: Option<ComparisonProtocol>,
    pub rows: Vec<ComparisonRow>,
    pub recommended: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub candidates: Option<Vec<Candidate>>,
    /// R0: recompute split and transform instead of reusing cached stages.
    pub no_cache: bool,
    pub control: RunControl,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            candidates: None,
            no_cache: false,
            control: RunControl::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunGroupResult {
    pub group: RunGroup,
    pub runs: Vec<RunRecord>,
    pub comparison: Comparison,
    pub reused: Vec<String>,
    pub cancelled: bool,
}

/// Entry point for SDK users: a RustPy project directory.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub project: Project,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

impl Workspace {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            project: Project::open(dir)?,
        })
    }

    pub fn discover(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            project: Project::discover(dir)?,
        })
    }

    pub fn resolve(&self, req: &ConfigRequest) -> Result<ResolvedConfig> {
        let r = self.project.resolve_config(req)?;
        r.config.require_data()?;
        Ok(r)
    }

    // -----------------------------------------------------------------------
    // Stages
    // -----------------------------------------------------------------------

    /// Snapshot -> column roles -> split -> fitted transform, reusing cached stages
    /// whose inputs are unchanged (R1) unless `no_cache` (R0).
    pub fn prepare(&self, resolved: ResolvedConfig, no_cache: bool) -> Result<Prepared> {
        let cfg = &resolved.config;
        let t0 = Instant::now();
        let snap = snapshot_source(&self.project, &cfg.data)?;
        let data_ms = ms(t0);
        let mut reused = Vec::new();
        if snap.reused {
            reused.push("snapshot".to_string());
        }
        self.prepare_from(resolved, snap.snapshot, snap.table, no_cache, reused, data_ms)
    }

    fn prepare_from(
        &self,
        resolved: ResolvedConfig,
        snapshot: DataSnapshot,
        table: Table,
        no_cache: bool,
        mut reused: Vec<String>,
        data_ms: f64,
    ) -> Result<Prepared> {
        let cfg = &resolved.config;
        let features = plan_columns(&snapshot, &table, &cfg.data, cfg.preprocess.max_categories)?;
        let split = make_split(&snapshot.snapshot_id, &table, &features.target, &cfg.data)?;
        let split_path = self.project.dir("splits").join(format!("{}.json", split.split_id));
        if split_path.is_file() && !no_cache {
            let cached: SplitManifest = read_json(&split_path, Phase::Split)?;
            if cached.assignment != split.assignment {
                return Err(RustpyError::internal(
                    Phase::Split,
                    format!("cached split {} differs from the recomputed split", split.split_id),
                ));
            }
            reused.push("split".to_string());
        } else {
            write_json(&split_path, &split)?;
        }
        let t1 = Instant::now();
        let tid = transform_id(
            &split.split_id,
            &cfg.preprocess,
            &features,
            cfg.selection.threshold,
            &cfg.data.null_values,
        );
        let tdir = self.project.dir("transforms").join(&tid);
        let fitted = if tdir.join("fitted.json").is_file() && !no_cache {
            reused.push("transform".to_string());
            read_json::<FittedArtifacts>(&tdir.join("fitted.json"), Phase::Fit)?
        } else {
            let train_rows = split.rows(SplitName::Train);
            let train_table = table.take(&train_rows);
            let fa = fit_transform(
                &train_table,
                &train_rows,
                &features,
                &cfg.preprocess,
                &split.split_id,
                cfg.selection.threshold,
                cfg.data.id_column.as_deref(),
                &cfg.data.null_values,
                Some(cfg.data.csv_delimiter.as_str()),
            )?;
            write_json(&tdir.join("fitted.json"), &fa)?;
            fa
        };
        let fit_ms = ms(t1);
        Ok(Prepared {
            resolved,
            snapshot,
            table,
            features,
            split,
            fitted,
            reused,
            data_ms,
            fit_ms,
        })
    }

    /// Loads exactly the stages a run used (for resume and evaluation).
    pub fn prepare_for_run(&self, run: &RunRecord) -> Result<Prepared> {
        let t0 = Instant::now();
        let (snapshot, table) = load_snapshot(&self.project, &run.snapshot_id).map_err(|e| e.run(run.run_id.clone()))?;
        let data_ms = ms(t0);
        let split: SplitManifest = read_json(&self.project.dir("splits").join(format!("{}.json", run.split_id)), Phase::Split)?;
        let fitted: FittedArtifacts = read_json(
            &self.project.dir("transforms").join(&run.transform_id).join("fitted.json"),
            Phase::Fit,
        )?;
        let cfg_path = self.project.run_dir(&run.run_id).join("resolved.json");
        let resolved: ResolvedConfig = read_json(&cfg_path, Phase::Config)?;
        let features = plan_columns(&snapshot, &table, &resolved.config.data, resolved.config.preprocess.max_categories)?;
        Ok(Prepared {
            resolved,
            snapshot,
            table,
            features,
            split,
            fitted,
            reused: vec!["snapshot".into(), "split".into(), "transform".into()],
            data_ms,
            fit_ms: 0.0,
        })
    }

    pub fn split_data(&self, p: &Prepared, which: SplitName) -> Result<SplitData> {
        let rows = p.split.rows(which);
        let sub = p.table.take(&rows);
        // Training and validation rows are encoded while training; errors name data rows.
        let phase = if which == SplitName::Test { Phase::Evaluate } else { Phase::Train };
        let x = rustpy_runtime::apply_transform_rows(&p.fitted.transform, &sub, phase, &rows)?;
        let y = labels(&sub, &p.features.target);
        Ok(SplitData { x, y, rows })
    }

    pub fn protocol(&self, p: &Prepared, split: SplitName) -> ComparisonProtocol {
        ComparisonProtocol {
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            split,
            target: p.features.target.name.clone(),
            positive_label: p.features.target.positive_label.clone(),
            threshold: format!("{}", p.fitted.output_schema.threshold),
            metrics_version: METRICS_VERSION.to_string(),
        }
    }

    /// Device plans for each candidate; fails before any training if one is unavailable.
    pub fn device_plans(&self, p: &Prepared, candidates: &[Candidate]) -> Result<Vec<(Candidate, DevicePlan)>> {
        let cfg = &p.resolved.config;
        candidates
            .iter()
            .map(|&c| {
                let (request, key) = match c {
                    Candidate::Linear => (cfg.execution.baseline_device, "execution.baseline_device"),
                    Candidate::SmallMlp => (cfg.execution.candidate_device, "execution.candidate_device"),
                };
                let hidden = &cfg.train.for_candidate(c).hidden;
                let est = estimate_peak_mb(p.snapshot.source.bytes, p.snapshot.n_rows, p.fitted.transform.output_dim, hidden);
                plan_device(request, key, &cfg.execution, est).map(|plan| (c, plan))
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Runs
    // -----------------------------------------------------------------------

    pub fn run_path(&self, run_id: &str) -> PathBuf {
        self.project.run_dir(run_id).join("run.json")
    }

    pub fn load_run(&self, run_id: &str) -> Result<RunRecord> {
        let path = self.run_path(run_id);
        if !path.is_file() {
            let known = self.run_ids();
            return Err(
                RustpyError::new(ErrorCode::RunNotFound, Phase::Run, format!("run `{run_id}` does not exist"))
                    .field(run_id)
                    .actual(if known.is_empty() {
                        "no runs yet".into()
                    } else {
                        format!("runs: {}", known.join(", "))
                    })
                    .next("list runs with `rustpy runs`"),
            );
        }
        let mut run: RunRecord = read_json(&path, Phase::Run)?;
        self.refresh_status(&mut run)?;
        Ok(run)
    }

    fn save_run(&self, run: &RunRecord) -> Result<()> {
        write_json(&self.run_path(&run.run_id), run)
    }

    fn event(&self, run_id: &str, event: &str, detail: serde_json::Value) {
        let _ = append_jsonl(
            &self.project.run_dir(run_id).join("events.jsonl"),
            &serde_json::json!({"at": now_rfc3339(), "event": event, "detail": detail}),
        );
    }

    /// Marks runs whose process died as `interrupted`.
    fn refresh_status(&self, run: &mut RunRecord) -> Result<()> {
        if run.status == RunStatus::Running {
            let lock_path = self.project.run_dir(&run.run_id).join("process.lock");
            let alive = match fs::OpenOptions::new().read(true).write(true).open(&lock_path) {
                // Someone still holds the training lock.
                Ok(f) => matches!(f.try_lock(), Err(fs::TryLockError::WouldBlock)),
                // Records from before lock files existed: fall back to the pid.
                Err(_) => run.pid.is_some_and(pid_alive),
            };
            if !alive {
                run.status = RunStatus::Interrupted;
                run.finished_at = Some(now_rfc3339());
                let ck = latest_checkpoint(&self.project.run_dir(&run.run_id).join("checkpoints"));
                let mut e = RustpyError::new(
                    ErrorCode::Interrupted,
                    Phase::Train,
                    "the process running this run ended unexpectedly",
                )
                .run(run.run_id.clone());
                e = match &ck {
                    Some(p) => e
                        .retained(format!("checkpoint {}", self.project.rel(p)))
                        .next(format!("rustpy resume {}", run.run_id)),
                    None => e.next("no checkpoint was written yet; start again with `rustpy run`"),
                };
                run.error = Some(e);
                run.pid = None;
                self.save_run(run)?;
            }
        }
        Ok(())
    }

    pub fn run_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = fs::read_dir(self.project.dir("runs"))
            .map(|it| {
                it.flatten()
                    .filter(|e| e.path().join("run.json").is_file())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        ids.sort_by_key(|id| id_number(id));
        ids
    }

    /// A note when the data snapshot differs from the one the latest run used
    /// (typically: the file at the same path received new content).
    pub fn data_change_note(&self, snapshot: &DataSnapshot) -> Option<String> {
        let runs = self.list_runs().ok()?;
        let latest = runs.iter().max_by_key(|r| id_number(&r.run_id))?;
        if self.same_data_version(&latest.snapshot_id, &snapshot.snapshot_id)
            || self.snapshot_sha(&latest.snapshot_id).as_deref() == Some(snapshot.source.sha256.as_str())
        {
            return None;
        }
        let on_previous: Vec<&str> = runs
            .iter()
            .filter(|r| r.snapshot_id == latest.snapshot_id)
            .map(|r| r.run_id.as_str())
            .collect();
        let mut note = format!(
            "new data version: {} is now snapshot {}, while {} used snapshot {}. Earlier runs and bundles keep their own data version and stay usable; to compare an old model with a new one on the same new rows use `rustpy evaluate --bundle <old bundle> --split test`",
            snapshot.source.path,
            snapshot.snapshot_id,
            on_previous.join(", "),
            latest.snapshot_id
        );
        // Without an id column the split is keyed by every column: a changed column
        // set re-splits the old rows.
        let previous: Option<DataSnapshot> = read_json(
            &self.project.dir("snapshots").join(&latest.snapshot_id).join("snapshot.json"),
            Phase::Data,
        )
        .ok();
        if let (Some(prev), rustpy_contracts::data::RowIdentity::RowIndex) = (previous, &snapshot.row_identity) {
            let names = |s: &DataSnapshot| s.columns.iter().map(|c| c.name.clone()).collect::<BTreeSet<_>>();
            let (old, new) = (names(&prev), names(snapshot));
            if old != new {
                let added: Vec<&String> = new.difference(&old).collect();
                let removed: Vec<&String> = old.difference(&new).collect();
                note.push_str(&format!(
                    ". The file's columns changed (added {added:?}, removed {removed:?}): without data.id_column its rows are split anew, so fewer old test rows stay test rows (rows a model saw are still excluded from its evaluations)"
                ));
            }
        }
        Some(note)
    }

    pub fn list_runs(&self) -> Result<Vec<RunRecord>> {
        self.run_ids().iter().map(|id| self.load_run(id)).collect()
    }

    pub fn latest_group(&self) -> Result<Option<RunGroup>> {
        let dir = self.project.dir("groups");
        let mut names: Vec<String> = fs::read_dir(&dir)
            .map(|it| {
                it.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.ends_with(".json"))
                    .collect()
            })
            .unwrap_or_default();
        names.sort_by_key(|n| id_number(n.trim_end_matches(".json")));
        match names.pop() {
            Some(n) => Ok(Some(read_json(&dir.join(n), Phase::Compare)?)),
            None => Ok(None),
        }
    }

    /// Trains every configured candidate on the same snapshot, split and transform.
    pub fn run(&self, req: &ConfigRequest, opts: &RunOptions, progress: &mut dyn FnMut(Progress)) -> Result<RunGroupResult> {
        let resolved = self.resolve(req)?;
        progress(Progress::Stage("preparing data, split and preprocessing".into()));
        let prepared = self.prepare(resolved, opts.no_cache)?;
        let data_note = self.data_change_note(&prepared.snapshot);
        if let Some(n) = &data_note {
            progress(Progress::Stage(n.clone()));
        }
        let candidates = opts
            .candidates
            .clone()
            .unwrap_or_else(|| prepared.resolved.config.selection.candidates.clone());
        let plans = self.device_plans(&prepared, &candidates)?;
        let train_data = self.split_data(&prepared, SplitName::Train)?;
        let valid_data = self.split_data(&prepared, SplitName::Valid)?;
        let group_id = self.project.allocate_id("groups", "grp")?;
        // The group record exists from the start, so a crash never leaves a run
        // without its comparison group.
        let mut group = RunGroup {
            group_id: group_id.clone(),
            created_at: now_rfc3339(),
            run_ids: Vec::new(),
            planned: candidates.clone(),
            snapshot_id: prepared.snapshot.snapshot_id.clone(),
            split_id: prepared.split.split_id.clone(),
            transform_id: prepared.fitted.transform_id.clone(),
            config_hash: prepared.resolved.config_hash.clone(),
        };
        self.save_group(&group)?;
        let mut runs = Vec::new();
        let mut cancelled = false;
        for (candidate, plan) in plans {
            if cancelled {
                break;
            }
            let run = self.execute_run(
                &prepared,
                &group_id,
                candidate,
                plan,
                &train_data,
                &valid_data,
                &opts.control,
                StartFrom::Scratch,
                None,
                progress,
            )?;
            cancelled = run.status == RunStatus::Cancelled;
            group.run_ids.push(run.run_id.clone());
            self.save_group(&group)?;
            runs.push(run);
        }
        let mut comparison = self.compare_records(&runs, &prepared.resolved.config)?;
        comparison.notes.extend(Self::unstarted_note(&group, &runs));
        comparison.notes.extend(data_note);
        Ok(RunGroupResult {
            group,
            runs,
            comparison,
            reused: prepared.reused.clone(),
            cancelled,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_run(
        &self,
        p: &Prepared,
        group_id: &str,
        candidate: Candidate,
        plan: DevicePlan,
        train_data: &SplitData,
        valid_data: &SplitData,
        control: &RunControl,
        start: StartFrom,
        lineage: Option<(Lineage, Option<TrainState>)>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<RunRecord> {
        let cfg = &p.resolved.config;
        let tcfg = cfg.train.for_candidate(candidate).clone();
        let run_id = self.project.allocate_id("runs", "run")?;
        let run_dir = self.project.run_dir(&run_id);
        fs::create_dir_all(&run_dir).map_err(|e| RustpyError::io(Phase::Train, &run_dir, e))?;
        write_json(&run_dir.join("resolved.json"), &p.resolved)?;
        let mconfig = model_config(candidate, p.fitted.transform.output_dim, &tcfg);
        let batches = train_data.x.rows.div_ceil(tcfg.batch_size) as u64;
        let (lineage, prior_state) = match lineage {
            Some((l, s)) => (Some(l), s),
            None => (None, None),
        };
        let mut run = RunRecord {
            format_version: RUN_FORMAT_VERSION,
            run_id: run_id.clone(),
            group_id: group_id.to_string(),
            candidate,
            status: RunStatus::Planned,
            created_at: now_rfc3339(),
            started_at: None,
            finished_at: None,
            lineage,
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            transform_id: p.fitted.transform_id.clone(),
            config_hash: p.resolved.config_hash.clone(),
            train_config: tcfg.clone(),
            recipe_constants: Some(crate::train::RECIPE_CONSTANTS.to_string()),
            train_seed: cfg.train.seed,
            threshold: cfg.selection.threshold,
            reused_stages: p.reused.clone(),
            device_plan: plan.clone(),
            device_receipt: None,
            steps_done: prior_state.as_ref().map_or(0, |s| s.step),
            steps_planned: batches * tcfg.epochs as u64,
            history: prior_state.as_ref().map(|s| s.history.clone()).unwrap_or_default(),
            metrics: None,
            artifacts: Vec::new(),
            timings: Timings {
                data_ms: p.data_ms,
                fit_transform_ms: p.fit_ms,
                train_ms: 0.0,
                eval_ms: 0.0,
            },
            error: None,
            environment: environment(),
            pid: None,
            engine_parity_max_abs_diff: None,
            latency: None,
            stop_reason: None,
        };
        self.save_run(&run)?;
        // Liveness: this process holds an exclusive OS lock on the file while it
        // trains; the lock disappears with the process however it ends.
        let lock_path = run_dir.join("process.lock");
        let _process_lock = fs::File::create(&lock_path).map_err(|e| RustpyError::io(Phase::Train, &lock_path, e))?;
        _process_lock
            .try_lock()
            .map_err(|e| RustpyError::internal(Phase::Train, format!("cannot lock {}: {e}", lock_path.display())))?;
        run.status = RunStatus::Running;
        run.started_at = Some(now_rfc3339());
        run.pid = Some(std::process::id());
        self.save_run(&run)?;
        self.event(
            &run_id,
            "started",
            serde_json::json!({"candidate": candidate, "device": plan.device}),
        );
        progress(Progress::RunStarted {
            run_id: run_id.clone(),
            candidate,
            steps_planned: run.steps_planned,
        });
        let key = ResumeKey {
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            transform_id: p.fitted.transform_id.clone(),
            candidate,
            train_config: tcfg.clone(),
            seed: cfg.train.seed,
            threshold: cfg.selection.threshold,
            model_config: mconfig.clone(),
            engine: ENGINE.to_string(),
            rustpy_version: RUSTPY_VERSION.to_string(),
            n_train: train_data.x.rows,
            platform: platform_fingerprint(),
            recipe_constants: crate::train::RECIPE_CONSTANTS.to_string(),
        };
        // R2: continue from a private copy of the checkpoint, so this run can itself
        // be resumed even if it dies before writing its own first checkpoint.
        let start = match start {
            StartFrom::Checkpoint(src) => {
                let dst = run_dir.join("checkpoints").join(src.file_name().unwrap_or_default());
                copy_dir(&src, &dst).map_err(|e| RustpyError::io(Phase::Resume, &dst, e))?;
                StartFrom::Checkpoint(dst)
            }
            other => other,
        };
        let inputs = TrainInputs {
            candidate,
            cfg: &tcfg,
            seed: cfg.train.seed,
            threshold: cfg.selection.threshold,
            metric: cfg.selection.metric,
            max_train_seconds: cfg.execution.max_train_seconds,
            x_train: &train_data.x,
            y_train: &train_data.y,
            x_valid: &valid_data.x,
            y_valid: &valid_data.y,
            checkpoint_dir: run_dir.join("checkpoints"),
            key,
        };
        crate::device::reset_peak_rss();
        let t0 = Instant::now();
        let epochs = tcfg.epochs;
        let run_path = self.run_path(&run_id);
        let mut on_epoch = |s: &TrainState| {
            if let Some(h) = s.history.last() {
                progress(Progress::Epoch {
                    run_id: run_id.clone(),
                    epoch: h.epoch,
                    epochs,
                    train_loss: h.train_loss,
                    valid_loss: h.valid_loss,
                    valid_metric: h.valid_metric,
                });
            }
            // Keep run.json's progress current so `rustpy runs` shows it.
            if let Ok(mut r) = read_json::<RunRecord>(&run_path, Phase::Train) {
                r.steps_done = s.step;
                r.history = s.history.clone();
                let _ = write_json(&run_path, &r);
            }
        };
        let outcome = train(&inputs, control, start, &mut on_epoch);
        let mut run: RunRecord = match read_json(&run_path, Phase::Train) {
            Ok(r) => r,
            Err(_) => run.clone(),
        };
        run.timings.train_ms = ms(t0);
        run.pid = None;
        run.finished_at = Some(now_rfc3339());
        run.device_receipt = Some(receipt(&plan));
        match outcome {
            Ok(TrainOutcome::Completed {
                best,
                state,
                parity_max_abs_diff,
            }) => {
                let t1 = Instant::now();
                run.steps_done = state.step;
                run.history = state.history.clone();
                run.stop_reason = state.stop_reason.clone();
                run.engine_parity_max_abs_diff = Some(parity_max_abs_diff);
                // Any failure while publishing results marks the run failed with the real
                // error (never leaves it "running").
                let published = (|| -> Result<Vec<ArtifactRef>> {
                    let model_dir = run_dir.join("model");
                    crate::project::write_atomic(&model_dir.join("model.safetensors"), &best.to_safetensors()?)?;
                    write_json(&model_dir.join("model.config.json"), &best.config)?;
                    let mut artifacts = Vec::new();
                    for name in ["model.safetensors", "model.config.json"] {
                        let path = model_dir.join(name);
                        let (sha, bytes) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Train, &path, e))?;
                        artifacts.push(ArtifactRef {
                            name: name.to_string(),
                            path: self.project.rel(&path),
                            sha256: sha,
                            bytes,
                        });
                    }
                    Ok(artifacts)
                })();
                match published {
                    Ok(artifacts) => {
                        run.artifacts = artifacts;
                        let p_train = probs(&best, &train_data.x);
                        let p_valid = probs(&best, &valid_data.x);
                        run.metrics = Some(RunMetrics {
                            train: binary_metrics(&p_train, &train_data.y, cfg.selection.threshold),
                            valid: binary_metrics(&p_valid, &valid_data.y, cfg.selection.threshold),
                            best_epoch: state.best_epoch,
                        });
                        run.latency = Some(measure_latency(&p.fitted, &best, &p.table, &valid_data.rows));
                        run.timings.eval_ms = ms(t1);
                        run.status = RunStatus::Completed;
                        self.event(
                            &run_id,
                            "completed",
                            serde_json::json!({"steps": state.step, "best_epoch": state.best_epoch}),
                        );
                    }
                    Err(e) => {
                        let mut e = e.run(run_id.clone());
                        if let Some(ck) = latest_checkpoint(&run_dir.join("checkpoints")) {
                            e = e.retained(format!("checkpoint {}", self.project.rel(&ck)));
                        }
                        run.status = RunStatus::Failed;
                        run.error = Some(e.clone());
                        self.event(&run_id, "failed", serde_json::to_value(&e).unwrap_or_default());
                    }
                }
            }
            Ok(TrainOutcome::Cancelled { state, checkpoint }) => {
                run.steps_done = state.step;
                run.history = state.history.clone();
                run.status = RunStatus::Cancelled;
                run.stop_reason = Some("cancelled".into());
                run.error = Some(
                    RustpyError::new(ErrorCode::Cancelled, Phase::Train, format!("cancelled at step {}", state.step))
                        .run(run_id.clone())
                        .retained(format!("checkpoint {}", self.project.rel(&checkpoint)))
                        .next(format!("rustpy resume {run_id}")),
                );
                self.event(&run_id, "cancelled", serde_json::json!({"step": state.step}));
            }
            Err(e) => {
                run.status = RunStatus::Failed;
                run.error = Some(e.clone().run(run_id.clone()));
                self.event(&run_id, "failed", serde_json::to_value(&e).unwrap_or_default());
            }
        }
        self.save_run(&run)?;
        progress(Progress::RunFinished {
            run_id: run_id.clone(),
            status: run.status,
            stop_reason: run.stop_reason.clone(),
            best_epoch: run.metrics.as_ref().map(|m| m.best_epoch),
            epochs_done: run.history.len(),
        });
        Ok(run)
    }

    /// Continues a cancelled/interrupted/failed run exactly from its latest
    /// checkpoint (R2), or starts a warm-started new training from its weights.
    pub fn resume(
        &self,
        run_id: &str,
        warm_start: bool,
        req: &ConfigRequest,
        control: &RunControl,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<RunRecord> {
        let parent = self.load_run(run_id)?;
        if parent.status == RunStatus::Running {
            return Err(RustpyError::new(
                ErrorCode::RunStateInvalid,
                Phase::Resume,
                format!("run {run_id} is still running (pid {:?})", parent.pid),
            )
            .run(run_id)
            .next("wait for it to finish or stop it with Ctrl-C first"));
        }
        if warm_start {
            return self.warm_start(&parent, req, control, progress);
        }
        if parent.status == RunStatus::Completed {
            return Err(RustpyError::new(
                ErrorCode::RunStateInvalid,
                Phase::Resume,
                format!("run {run_id} already completed; there is nothing to resume"),
            )
            .run(run_id)
            .next(format!(
                "start a new run with `rustpy run`, or continue training from its weights with `rustpy resume {run_id} --warm-start`"
            )));
        }
        // An exact resume continues with the run's recorded configuration. Flags that
        // would change it are refused instead of being silently ignored.
        if req.profile.is_some() || !req.overrides.is_empty() || !req.override_tables.is_empty() {
            let recorded: ResolvedConfig = read_json(&self.project.run_dir(run_id).join("resolved.json"), Phase::Resume)?;
            let requested = self.project.resolve_config(req)?;
            let old: std::collections::BTreeMap<String, String> = recorded.effective_values().into_iter().map(|(k, v, _)| (k, v)).collect();
            let changed: Vec<String> = requested
                .effective_values()
                .into_iter()
                .filter(|(k, v, _)| old.get(k) != Some(v))
                .map(|(k, v, _)| format!("{k} = {v} (run used {})", old.get(&k).map(String::as_str).unwrap_or("unset")))
                .collect();
            if !changed.is_empty() {
                return Err(RustpyError::new(
                    ErrorCode::ResumeIncompatible,
                    Phase::Resume,
                    format!(
                        "an exact resume continues with run {run_id}'s recorded configuration, but the requested settings differ: {}",
                        changed.join("; ")
                    ),
                )
                .run(run_id)
                .next(format!(
                    "drop --set/--profile to continue exactly, or start a new training with these settings (`rustpy run`, or `rustpy resume {run_id} --warm-start`)"
                )));
            }
        }
        let p = self.prepare_for_run(&parent)?;
        let cfg = &p.resolved.config;
        if cfg.train.for_candidate(parent.candidate) != &parent.train_config {
            return Err(RustpyError::internal(Phase::Resume, "run configuration record is inconsistent"));
        }
        let plan = self
            .device_plans(&p, &[parent.candidate])?
            .pop()
            .map(|(_, plan)| plan)
            .expect("one candidate");
        let train_data = self.split_data(&p, SplitName::Train)?;
        let valid_data = self.split_data(&p, SplitName::Valid)?;
        let key = ResumeKey {
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            transform_id: p.fitted.transform_id.clone(),
            candidate: parent.candidate,
            train_config: parent.train_config.clone(),
            seed: parent.train_seed,
            threshold: parent.threshold,
            model_config: model_config(parent.candidate, p.fitted.transform.output_dim, &parent.train_config),
            engine: ENGINE.to_string(),
            rustpy_version: RUSTPY_VERSION.to_string(),
            n_train: train_data.x.rows,
            platform: platform_fingerprint(),
            recipe_constants: crate::train::RECIPE_CONSTANTS.to_string(),
        };
        // Newest checkpoint that verifies; damaged ones are skipped (and reported).
        let (ck, skipped) = self.resumable_checkpoint(&parent, &key)?;
        for s in &skipped {
            progress(Progress::Stage(format!("skipped checkpoint {s}")));
        }
        let state: TrainState = read_json(&ck.join("state.json"), Phase::Resume)?;
        let from_step = state.step;
        let lineage = Lineage {
            level: ResumeLevel::ExactResume,
            parent_run_id: run_id.to_string(),
            from_step: Some(from_step),
        };
        // Copy the checkpoint into the new run so the parent stays immutable.
        let run = self.execute_run(
            &p,
            &parent.group_id,
            parent.candidate,
            plan,
            &train_data,
            &valid_data,
            control,
            StartFrom::Checkpoint(ck.clone()),
            Some((lineage, Some(state))),
            progress,
        )?;
        // The continued run belongs to the parent's comparison group.
        let gpath = self.project.dir("groups").join(format!("{}.json", parent.group_id));
        if let Ok(mut g) = read_json::<RunGroup>(&gpath, Phase::Resume) {
            if !g.run_ids.contains(&run.run_id) {
                g.run_ids.push(run.run_id.clone());
                self.save_group(&g)?;
            }
        }
        Ok(run)
    }

    fn save_group(&self, g: &RunGroup) -> Result<()> {
        write_json(&self.project.dir("groups").join(format!("{}.json", g.group_id)), g)
    }

    /// Note about planned candidates that were never trained (e.g. after Ctrl-C).
    fn unstarted_note(g: &RunGroup, runs: &[RunRecord]) -> Option<String> {
        let missing: Vec<&str> = g
            .planned
            .iter()
            .filter(|c| !runs.iter().any(|r| r.candidate == **c))
            .map(|c| c.as_str())
            .collect();
        (!missing.is_empty()).then(|| {
            format!(
                "planned candidates not trained yet: {} (train them with `rustpy run --candidates {}`)",
                missing.join(", "),
                missing.join(",")
            )
        })
    }

    /// Newest verified, compatible checkpoint of `run` or, if it has none (a resumed
    /// child that crashed before its first checkpoint), of its R2 ancestors.
    fn resumable_checkpoint(&self, run: &RunRecord, key: &ResumeKey) -> Result<(PathBuf, Vec<String>)> {
        let mut skipped = Vec::new();
        let mut cur = Some(run.clone());
        while let Some(r) = cur.take() {
            let root = self.project.run_dir(&r.run_id).join("checkpoints");
            for ck in list_checkpoints(&root).into_iter().rev() {
                match verify_checkpoint(&ck) {
                    Ok(()) => {
                        let saved: ResumeKey = read_json(&ck.join("key.json"), Phase::Resume)?;
                        let diffs = saved.differences(key);
                        if !diffs.is_empty() {
                            return Err(RustpyError::new(
                                ErrorCode::ResumeIncompatible,
                                Phase::Resume,
                                format!("cannot continue run {} exactly (R2): {}", run.run_id, diffs.join("; ")),
                            )
                            .run(run.run_id.clone())
                            .retained(format!("run {} and its checkpoints are unchanged", run.run_id))
                            .next(format!(
                                "start a new run (`rustpy run`), or `rustpy resume {} --warm-start` for a new training from its last weights",
                                run.run_id
                            )));
                        }
                        return Ok((ck, skipped));
                    }
                    Err(e) => skipped.push(format!("{} ({})", self.project.rel(&ck), e.message)),
                }
            }
            cur = match &r.lineage {
                Some(l) if l.level == ResumeLevel::ExactResume => self.load_run(&l.parent_run_id).ok(),
                _ => None,
            };
        }
        let mut e = RustpyError::new(
            ErrorCode::CheckpointMissing,
            Phase::Resume,
            format!("run {} has no valid checkpoint to continue from", run.run_id),
        )
        .run(run.run_id.clone())
        .next("start again with `rustpy run` (R0)");
        if !skipped.is_empty() {
            e.code = ErrorCode::CheckpointCorrupt;
            e = e.actual(format!("damaged: {}", skipped.join("; ")));
        }
        Err(e)
    }

    /// Kept weights of a run: its final model, or the best weights of its newest
    /// valid checkpoint when it did not complete.
    fn warm_start_weights(&self, parent: &RunRecord) -> Result<DenseModel> {
        if parent.status == RunStatus::Completed {
            return self.run_weights(parent).map_err(|e| e.in_phase(Phase::Resume));
        }
        let root = self.project.run_dir(&parent.run_id).join("checkpoints");
        for ck in list_checkpoints(&root).into_iter().rev() {
            if verify_checkpoint(&ck).is_ok() && ck.join("best.safetensors").is_file() {
                let key: ResumeKey = read_json(&ck.join("key.json"), Phase::Resume)?;
                let bytes = fs::read(ck.join("best.safetensors")).map_err(|e| RustpyError::io(Phase::Resume, &ck, e))?;
                return DenseModel::from_safetensors(key.model_config, &bytes).map_err(|e| e.in_phase(Phase::Resume));
            }
        }
        Err(RustpyError::new(
            ErrorCode::CheckpointMissing,
            Phase::Resume,
            format!("run {} has no weights to warm-start from (no completed epoch)", parent.run_id),
        )
        .run(parent.run_id.clone())
        .next("start a new run with `rustpy run`"))
    }

    fn warm_start(
        &self,
        parent: &RunRecord,
        req: &ConfigRequest,
        control: &RunControl,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<RunRecord> {
        let weights = self.warm_start_weights(parent)?;
        let resolved = self.resolve(req)?;
        let p = self.prepare(resolved, false)?;
        let parent_fitted: FittedArtifacts = read_json(
            &self.project.dir("transforms").join(&parent.transform_id).join("fitted.json"),
            Phase::Resume,
        )?;
        if parent_fitted.transform.output_names != p.fitted.transform.output_names {
            return Err(RustpyError::new(
                ErrorCode::ResumeIncompatible,
                Phase::Resume,
                "the feature layout changed (different columns or categories), so the old weights do not fit",
            )
            .run(parent.run_id.clone())
            .next("train from scratch with `rustpy run`"));
        }
        if p.split.split_id != parent.split_id {
            progress(Progress::Stage(format!(
                "note: warm start uses split {} (parent {} used {}); evaluations check that the parent's training rows are not evaluated",
                p.split.split_id, parent.run_id, parent.split_id
            )));
        }
        let cfg = &p.resolved.config;
        if model_config(
            parent.candidate,
            p.fitted.transform.output_dim,
            cfg.train.for_candidate(parent.candidate),
        ) != weights.config
        {
            return Err(RustpyError::new(
                ErrorCode::ResumeIncompatible,
                Phase::Resume,
                "the configured architecture differs from the parent run's weights",
            )
            .next("train from scratch with `rustpy run`"));
        }
        let plan = self
            .device_plans(&p, &[parent.candidate])?
            .pop()
            .map(|(_, plan)| plan)
            .expect("one candidate");
        let train_data = self.split_data(&p, SplitName::Train)?;
        let valid_data = self.split_data(&p, SplitName::Valid)?;
        let group_id = self.project.allocate_id("groups", "grp")?;
        let run = self.execute_run(
            &p,
            &group_id,
            parent.candidate,
            plan,
            &train_data,
            &valid_data,
            control,
            StartFrom::WarmStart(weights),
            Some((
                Lineage {
                    level: ResumeLevel::WarmStart,
                    parent_run_id: parent.run_id.clone(),
                    from_step: Some(parent.steps_done),
                },
                None,
            )),
            progress,
        )?;
        let group = RunGroup {
            group_id: group_id.clone(),
            created_at: now_rfc3339(),
            run_ids: vec![run.run_id.clone()],
            planned: vec![parent.candidate],
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            transform_id: p.fitted.transform_id.clone(),
            config_hash: p.resolved.config_hash.clone(),
        };
        self.save_group(&group)?;
        Ok(run)
    }

    /// Loads the kept weights of a completed run, verifying the recorded hashes.
    pub fn run_weights(&self, run: &RunRecord) -> Result<DenseModel> {
        if run.status != RunStatus::Completed {
            return Err(RustpyError::new(
                ErrorCode::RunStateInvalid,
                Phase::Evaluate,
                format!("run {} is {}, not completed", run.run_id, run.status.as_str()),
            )
            .run(run.run_id.clone())
            .next("use a completed run (see `rustpy runs`)"));
        }
        for a in &run.artifacts {
            let path = self.project.root.join(&a.path);
            let ok = sha256_file(&path).map(|(s, b)| s == a.sha256 && b == a.bytes).unwrap_or(false);
            if !ok {
                return Err(RustpyError::new(
                    ErrorCode::CheckpointCorrupt,
                    Phase::Evaluate,
                    format!("artifact `{}` of run {} is missing or modified", a.path, run.run_id),
                )
                .run(run.run_id.clone())
                .next("the run's files were changed outside RustPy; train again with `rustpy run`"));
            }
        }
        let dir = self.project.run_dir(&run.run_id).join("model");
        let config = read_json(&dir.join("model.config.json"), Phase::Evaluate)?;
        let path = dir.join("model.safetensors");
        let bytes = fs::read(&path).map_err(|e| RustpyError::io(Phase::Evaluate, &path, e))?;
        DenseModel::from_safetensors(config, &bytes)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    pub fn evaluations(&self) -> Vec<EvaluationRecord> {
        let mut v: Vec<EvaluationRecord> = fs::read_dir(self.project.dir("evaluations"))
            .map(|it| {
                it.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("eval-") && n.ends_with(".json") && !n.ends_with(".rows.json"))
                    .filter_map(|n| read_json(&self.project.dir("evaluations").join(n), Phase::Evaluate).ok())
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by_key(|e: &EvaluationRecord| id_number(&e.evaluation_id));
        v
    }

    fn evaluation_rows(&self, evaluation_id: &str) -> Option<RowKeys> {
        read_json(
            &self.project.dir("evaluations").join(format!("{evaluation_id}.rows.json")),
            Phase::Evaluate,
        )
        .ok()
    }

    /// Rows a run's weights were trained or selected on: training rows (and
    /// validation rows when `include_valid`), plus those of runs it was
    /// warm-started from. Returns `(run_id, rows)` per source; sources whose
    /// snapshot can no longer be loaded are returned as `None`.
    pub fn run_exposure(&self, run: &RunRecord, include_valid: bool) -> Vec<(String, Option<RowKeys>)> {
        let mut out = Vec::new();
        let mut cur = Some(run.clone());
        let mut guard = 0;
        while let Some(r) = cur.take() {
            guard += 1;
            if guard > 64 {
                break;
            }
            let keys = (|| -> Result<RowKeys> {
                let (snapshot, table) = load_snapshot(&self.project, &r.snapshot_id)?;
                let split: SplitManifest = read_json(&self.project.dir("splits").join(format!("{}.json", r.split_id)), Phase::Evaluate)?;
                let mut rows = split.rows(SplitName::Train);
                if include_valid || r.run_id != run.run_id {
                    // Ancestors' validation rows selected their weights, too.
                    rows.extend(split.rows(SplitName::Valid));
                }
                let content = self.run_content_columns(&r).unwrap_or_default();
                Ok(RowKeys::of(&snapshot, &table, &rows, &content))
            })()
            .ok();
            out.push((r.run_id.clone(), keys));
            cur = match &r.lineage {
                Some(l) if l.level == ResumeLevel::WarmStart => self.load_run(&l.parent_run_id).ok(),
                _ => None,
            };
        }
        out
    }

    /// Content hash of a snapshot's source file, if the snapshot is in this project.
    fn snapshot_sha(&self, snapshot_id: &str) -> Option<String> {
        read_json::<DataSnapshot>(&self.project.dir("snapshots").join(snapshot_id).join("snapshot.json"), Phase::Data)
            .ok()
            .map(|s| s.source.sha256)
    }

    /// Two snapshots are the same data version when their files have the same
    /// content, even if they were read with different options.
    pub(crate) fn same_data_version(&self, a: &str, b: &str) -> bool {
        a == b
            || match (self.snapshot_sha(a), self.snapshot_sha(b)) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            }
    }

    /// Input columns + target of a run's model (its content columns).
    fn run_content_columns(&self, run: &RunRecord) -> Option<Vec<String>> {
        let f: FittedArtifacts = read_json(
            &self.project.dir("transforms").join(&run.transform_id).join("fitted.json"),
            Phase::Evaluate,
        )
        .ok()?;
        Some(schema_content_columns(&f.input_schema, &f.output_schema.target))
    }

    /// Rows among `rows` (positions in `p.table`) that the exposure sources saw,
    /// and the sources whose rows cannot be matched with this data.
    fn seen_rows(exposure: &[(String, Option<RowKeys>)], p: &Prepared, rows: &[usize]) -> (BTreeSet<usize>, Vec<String>) {
        let mut seen = BTreeSet::new();
        let mut unverifiable = Vec::new();
        for (source, keys) in exposure {
            match keys.as_ref().and_then(|k| k.matcher(&p.snapshot, &p.table)) {
                Some(m) => seen.extend(rows.iter().copied().filter(|&r| m(r))),
                None => unverifiable.push(source.clone()),
            }
        }
        (seen, unverifiable)
    }

    /// Verifies that the evaluated rows were not seen by the model.
    fn check_unseen(&self, exposure: &[(String, Option<RowKeys>)], p: &Prepared, rows: &[usize], subject: &str) -> Result<ExposureCheck> {
        let (seen, unverifiable) = Self::seen_rows(exposure, p, rows);
        let check = ExposureCheck {
            overlapping_rows: seen.len(),
            unverifiable,
        };
        if check.overlapping_rows > 0 {
            let sources: Vec<&str> = exposure.iter().map(|(s, _)| s.as_str()).collect();
            return Err(RustpyError::new(
                ErrorCode::LeakageDetected,
                Phase::Evaluate,
                format!(
                    "{} of the {} evaluated rows were used to train or select {subject} (training/validation rows of {})",
                    check.overlapping_rows,
                    rows.len(),
                    sources.join(" <- ")
                ),
            )
            .expected("evaluation rows the model has never seen")
            .actual(format!("{} overlapping rows", check.overlapping_rows))
            .next("evaluate on a split the model never saw; declare data.id_column so rows keep their split across data versions and re-splits, and do not change data.seed after evaluating"));
        }
        Ok(check)
    }

    /// Test-set discipline: rows of a test split may be used to evaluate one model.
    /// Returns an existing evaluation when the same model was already evaluated on
    /// the same rows, otherwise the ledger entry for this use.
    ///
    /// Reuse that is refused: another model trained on the *same data snapshot*
    /// (a competing candidate) evaluated on overlapping test rows. Overlap with
    /// models of another data version (the shared hold-out of an old-vs-new
    /// comparison) is allowed and recorded in `test_shared_with`.
    fn check_test_reuse(&self, model_sha: &str, model_snapshot: &str, rows: &RowKeys, allow_reuse: bool) -> Result<TestLedger> {
        let mut other_models: Vec<(String, String, usize)> = Vec::new();
        let mut other_versions: Vec<String> = Vec::new();
        let mut unverifiable = 0usize;
        for prev in self.evaluations().into_iter().filter(|e| e.protocol.split == SplitName::Test) {
            let Some(prev_rows) = self.evaluation_rows(&prev.evaluation_id) else {
                unverifiable += 1;
                continue;
            };
            let overlap = prev_rows.overlap(rows);
            let same_model = prev.model_sha256.as_deref() == Some(model_sha);
            match overlap {
                Some(n) if same_model && n == rows.len() && n == prev_rows.len() => {
                    return Ok(TestLedger {
                        existing: Some(prev),
                        ..Default::default()
                    });
                }
                Some(n) if n > 0 && !same_model => {
                    let subject = match &prev.subject {
                        EvaluationSubject::Run { run_id } => run_id.clone(),
                        EvaluationSubject::Bundle { bundle_id, .. } => bundle_id.clone(),
                    };
                    let label = format!("{subject} ({})", prev.evaluation_id);
                    let same_version = prev
                        .model_snapshot_id
                        .as_deref()
                        .is_none_or(|s| self.same_data_version(s, model_snapshot));
                    if !same_version {
                        if !other_versions.contains(&label) {
                            other_versions.push(label);
                        }
                    } else if !other_models
                        .iter()
                        .any(|(m, _, _)| Some(m.as_str()) == prev.model_sha256.as_deref())
                    {
                        other_models.push((prev.model_sha256.clone().unwrap_or_default(), label, n));
                    }
                }
                None => unverifiable += 1,
                _ => {}
            }
        }
        let mut notes = Vec::new();
        if !other_models.is_empty() {
            let (_, first, n) = &other_models[0];
            if !allow_reuse {
                return Err(RustpyError::new(
                    ErrorCode::TestSetReuse,
                    Phase::Evaluate,
                    format!(
                        "{n} of these {} test rows were already used to evaluate another model of the same data version, {first}; evaluating more models on them would turn the test set into a selection set",
                        rows.len()
                    ),
                )
                .field("--split test")
                .retained(format!("evaluation {first}"))
                .next("compare candidates on the validation split (`--split valid`); if you accept a biased estimate, pass --allow-test-reuse (recorded in reports and bundles)"));
            }
            notes.push(format!(
                "test rows reused (use #{}): already used for {}; this is no longer an unbiased estimate",
                other_models.len() + 1,
                other_models.iter().map(|(_, s, _)| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        if !other_versions.is_empty() {
            notes.push(format!(
                "some of these test rows were also used for models trained on another data version ({}): a shared hold-out for comparing old and new models; picking among many models this way would make it a selection set",
                other_versions.join(", ")
            ));
        }
        if unverifiable > 0 {
            notes.push(format!(
                "{unverifiable} earlier test evaluation(s) could not be matched row by row with these rows (their input columns differ)"
            ));
        }
        Ok(TestLedger {
            existing: None,
            index: other_models.len() as u32 + 1,
            shared_with: other_versions,
            notes,
        })
    }

    fn save_evaluation(&self, mut rec: EvaluationRecord, rows: &RowKeys) -> Result<EvaluationRecord> {
        let id = self.project.allocate_id("evaluations", "eval")?;
        rec.evaluation_id = id.clone();
        let dir = self.project.dir("evaluations");
        // Rows first: a published record always has its row list.
        write_json(&dir.join(format!("{id}.rows.json")), rows)?;
        write_json(&dir.join(format!("{id}.json")), &rec)?;
        Ok(rec)
    }

    pub(crate) fn run_model_sha(run: &RunRecord) -> String {
        run.artifacts
            .iter()
            .find(|a| a.name == "model.safetensors")
            .map(|a| a.sha256.clone())
            .unwrap_or_default()
    }

    /// Evaluates a completed run on one split of its own snapshot. Test rows may be
    /// used for one model; later uses need `allow_test_reuse` and are recorded.
    /// When the rows seen by runs it was warm-started from cannot be matched with
    /// this data, the evaluation is refused unless `allow_unverified_rows`.
    pub fn evaluate_run(
        &self,
        run_id: &str,
        split: SplitName,
        allow_test_reuse: bool,
        allow_unverified_rows: bool,
    ) -> Result<EvaluationRecord> {
        let run = self.load_run(run_id).map_err(|e| e.in_phase(Phase::Evaluate))?;
        let model = self.run_weights(&run)?;
        let model_sha = Self::run_model_sha(&run);
        let p = self.prepare_for_run(&run)?;
        let data = self.split_data(&p, split)?;
        let content = schema_content_columns(&p.fitted.input_schema, &p.fitted.output_schema.target);
        let rows = RowKeys::of(&p.snapshot, &p.table, &data.rows, &content);
        let mut notes = Vec::new();
        let mut ledger = TestLedger::default();
        if split == SplitName::Test {
            ledger = self.check_test_reuse(&model_sha, &run.snapshot_id, &rows, allow_test_reuse)?;
            if let Some(mut rec) = ledger.existing.take() {
                rec.notes
                    .push("existing evaluation returned; the test set was not used again".into());
                return Ok(rec);
            }
            notes.append(&mut ledger.notes);
        }
        let exposure = self.run_exposure(&run, split == SplitName::Test);
        // Evaluating a run on its own validation split is its selection metric, not a held-out estimate.
        let check = self.check_unseen(&exposure, &p, &data.rows, &format!("run {run_id}"))?;
        let verified = check.unverifiable.is_empty();
        if !verified {
            Self::refuse_unverified(&check.unverifiable, allow_unverified_rows, &format!("run {run_id}"))?;
            notes.push(format!(
                "rows could not be matched with those seen by {} (allowed with --allow-unverified-rows): unseen rows not verified",
                check.unverifiable.join(", ")
            ));
        }
        if split == SplitName::Valid {
            notes.push("validation rows selected this run's weights (early stopping): not a held-out estimate".into());
        }
        let pr = probs(&model, &data.x);
        let rec = EvaluationRecord {
            evaluation_id: String::new(),
            created_at: now_rfc3339(),
            subject: EvaluationSubject::Run {
                run_id: run_id.to_string(),
            },
            protocol: self.protocol(&p, split),
            metrics: binary_metrics(&pr, &data.y, run.threshold),
            test_use_index: (split == SplitName::Test).then_some(ledger.index),
            test_shared_with: ledger.shared_with,
            latency: Some(measure_latency(&p.fitted, &model, &p.table, &data.rows)),
            model_sha256: Some(model_sha),
            model_snapshot_id: Some(run.snapshot_id.clone()),
            unseen_rows_verified: if split == SplitName::Valid {
                None
            } else {
                verified.then_some(true)
            },
            notes,
        };
        self.save_evaluation(rec, &rows)
    }

    /// Refuses an evaluation whose rows cannot be checked against what the model saw.
    fn refuse_unverified(sources: &[String], allow: bool, subject: &str) -> Result<()> {
        if allow {
            return Ok(());
        }
        Err(RustpyError::new(
            ErrorCode::LeakageDetected,
            Phase::Evaluate,
            format!(
                "cannot verify that the evaluated rows are unseen by {subject}: the rows seen by {} cannot be matched with this data",
                sources.join(", ")
            ),
        )
        .expected("rows that can be matched by id column, by snapshot, or by content (the model's input columns + target)")
        .actual("the source run, its snapshot or some of its input columns are not available here")
        .next("evaluate inside the project that trained the model, keep the model's input columns in the data, or pass --allow-unverified-rows to accept a possibly optimistic estimate (recorded)"))
    }

    /// Evaluates a bundle (its frozen transform and weights) on a split of the
    /// *current* data snapshot — e.g. old vs new model on the same new data. Rows
    /// the bundle's source run trained or selected on are excluded (matched by id
    /// column, snapshot position or row content); if they cannot be matched the
    /// evaluation is refused unless `allow_unverified_rows`.
    pub fn evaluate_bundle(
        &self,
        bundle_dir: &Path,
        split: SplitName,
        req: &ConfigRequest,
        allow_test_reuse: bool,
        allow_unverified_rows: bool,
    ) -> Result<EvaluationRecord> {
        self.evaluate_bundle_against(bundle_dir, None, split, req, allow_test_reuse, allow_unverified_rows)
            .map(|(rec, _)| rec)
    }

    /// [`Self::evaluate_bundle`], plus the metrics of the bundle `also` on exactly
    /// the rows used for that evaluation: old vs new model on identical rows. `also`
    /// is scored without a ledger entry, so it must be a model whose own test
    /// evaluation already covers these rows (the new model of `rustpy train`).
    pub(crate) fn evaluate_bundle_against(
        &self,
        bundle_dir: &Path,
        also: Option<&Path>,
        split: SplitName,
        req: &ConfigRequest,
        allow_test_reuse: bool,
        allow_unverified_rows: bool,
    ) -> Result<(EvaluationRecord, Option<BinaryMetrics>)> {
        let predictor = Predictor::load(bundle_dir, LoadOptions::cpu())?;
        let resolved = self.resolve(req)?;
        let p = self.prepare(resolved, false)?;
        let out = predictor.output_schema();
        let t = &p.features.target;
        if out.target != t.name || out.positive_label != t.positive_label || out.negative_label != t.negative_label {
            return Err(RustpyError::new(
                ErrorCode::NotComparable,
                Phase::Evaluate,
                "the bundle predicts a different target definition than the current data",
            )
            .expected(format!(
                "{} with labels {:?}/{:?}",
                out.target, out.negative_label, out.positive_label
            ))
            .actual(format!("{} with labels {:?}/{:?}", t.name, t.negative_label, t.positive_label)));
        }
        let model_sha = predictor
            .manifest()
            .files
            .iter()
            .find(|f| f.path == rustpy_contracts::manifest::MODEL_WEIGHTS_FILE)
            .map(|f| f.sha256.clone())
            .unwrap_or_default();
        let model_snapshot = predictor.manifest().provenance.snapshot_id.clone();
        let mut split_rows = p.split.rows(split);
        let mut notes = vec![format!(
            "bundle built from snapshot {} evaluated on snapshot {}",
            model_snapshot, p.snapshot.snapshot_id
        )];
        // Rows the bundle's source run trained or selected on are excluded, so an
        // old model is never scored on its own training rows.
        let source_run = predictor.manifest().provenance.run_id.clone();
        let subject = format!("bundle {} (run {source_run})", predictor.bundle_id());
        // The run must be the one that produced these weights: a bundle from another
        // project may carry a run id that exists here for a different model.
        let source = self
            .load_run(&source_run)
            .ok()
            .filter(|run| Self::run_model_sha(run) == model_sha && run.snapshot_id == model_snapshot);
        let unverifiable = match source {
            Some(run) => {
                let exposure = self.run_exposure(&run, true);
                let (seen, unverifiable) = Self::seen_rows(&exposure, &p, &split_rows);
                if !seen.is_empty() {
                    let before = split_rows.len();
                    split_rows.retain(|r| !seen.contains(r));
                    if split_rows.is_empty() {
                        return Err(RustpyError::new(
                            ErrorCode::LeakageDetected,
                            Phase::Evaluate,
                            format!("all {before} rows of this split were used to train or select {subject}"),
                        )
                        .next("evaluate on data the bundle never saw (e.g. the test split, or new rows)"));
                    }
                    notes.push(format!(
                        "{} of {before} rows were used to train/select the bundle (run {source_run}) and are excluded; evaluated on {} unseen rows",
                        seen.len(),
                        split_rows.len()
                    ));
                }
                unverifiable
            }
            None => vec![format!("run {source_run} (not the run that produced this bundle in this project)")],
        };
        let verified = unverifiable.is_empty();
        if !verified {
            Self::refuse_unverified(&unverifiable, allow_unverified_rows, &subject)?;
            notes.push(format!(
                "rows could not be matched with those seen by {} (allowed with --allow-unverified-rows): unseen rows not verified",
                unverifiable.join(", ")
            ));
        }
        let content = schema_content_columns(predictor.input_schema(), &out.target);
        let rows = RowKeys::of(&p.snapshot, &p.table, &split_rows, &content);
        let sub = p.table.take(&split_rows);
        let y = labels(&sub, t);
        // Probabilities of a bundle on the evaluated rows. Errors name the row of the
        // data file, not the position among the evaluated rows.
        let score = |pred: &Predictor| -> Result<Vec<f64>> {
            let names: Vec<&str> = pred.input_schema().fields.iter().map(|f| f.name.as_str()).collect();
            let inputs = sub.select(&names);
            rustpy_runtime::transform::check_table_columns(pred.input_schema(), &inputs, Phase::Evaluate)?;
            let x = rustpy_runtime::apply_transform_rows(&pred.bundle().transform, &inputs, Phase::Evaluate, &split_rows)?;
            Ok(pred.probabilities_matrix(&x))
        };
        let also_metrics = || -> Result<Option<BinaryMetrics>> {
            match also {
                Some(dir) => {
                    let other = Predictor::load(dir, LoadOptions::cpu())?;
                    let threshold = other.output_schema().threshold;
                    Ok(Some(binary_metrics(&score(&other)?, &y, threshold)))
                }
                None => Ok(None),
            }
        };
        let mut ledger = TestLedger::default();
        if split == SplitName::Test {
            ledger = self.check_test_reuse(&model_sha, &model_snapshot, &rows, allow_test_reuse)?;
            if let Some(rec) = ledger.existing.take() {
                return Ok((rec, also_metrics()?));
            }
            notes.append(&mut ledger.notes);
        }
        let names: Vec<&str> = predictor.input_schema().fields.iter().map(|f| f.name.as_str()).collect();
        let inputs = sub.select(&names);
        let pr = score(&predictor)?;
        for u in predictor.unseen_categories(&inputs) {
            notes.push(format!(
                "{} of {} rows have a `{}` value the bundle never saw (e.g. {}): encoded as all zeros",
                u.rows,
                split_rows.len(),
                u.field,
                u.examples.join(", ")
            ));
        }
        let mut protocol = self.protocol(&p, split);
        protocol.threshold = format!("{}", out.threshold);
        let rec = EvaluationRecord {
            evaluation_id: String::new(),
            created_at: now_rfc3339(),
            subject: EvaluationSubject::Bundle {
                bundle_id: predictor.bundle_id().to_string(),
                path: self.project.rel(bundle_dir),
            },
            protocol,
            metrics: binary_metrics(&pr, &y, out.threshold),
            test_use_index: (split == SplitName::Test).then_some(ledger.index),
            test_shared_with: ledger.shared_with,
            latency: None,
            model_sha256: Some(model_sha),
            model_snapshot_id: Some(model_snapshot),
            unseen_rows_verified: verified.then_some(true),
            notes,
        };
        let also = also_metrics()?;
        Ok((self.save_evaluation(rec, &rows)?, also))
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    pub fn compare(&self, run_ids: &[String]) -> Result<Comparison> {
        let mut group: Option<RunGroup> = None;
        let ids: Vec<String> = if run_ids.is_empty() {
            match self.latest_group()? {
                Some(g) => {
                    group = Some(g.clone());
                    g.run_ids
                }
                None => {
                    return Err(
                        RustpyError::new(ErrorCode::RunNotFound, Phase::Compare, "there are no runs to compare yet")
                            .next("train candidates with `rustpy run`"),
                    );
                }
            }
        } else {
            run_ids.to_vec()
        };
        let runs: Vec<RunRecord> = ids
            .iter()
            .map(|id| self.load_run(id).map_err(|e| e.in_phase(Phase::Compare)))
            .collect::<Result<_>>()?;
        // Use the selection settings the runs were trained with (profile/--set
        // included), not whatever rustpy.toml says now.
        let recorded: Vec<(String, ProjectConfig)> = runs
            .iter()
            .filter_map(|r| {
                read_json::<ResolvedConfig>(&self.project.run_dir(&r.run_id).join("resolved.json"), Phase::Compare)
                    .ok()
                    .map(|c| (r.run_id.clone(), c.config))
            })
            .collect();
        let cfg = match recorded.first() {
            Some((_, c)) => c.clone(),
            None => self.project.resolve_config(&ConfigRequest::default())?.config,
        };
        let mut cmp = self.compare_records(&runs, &cfg)?;
        let differing: Vec<&str> = recorded
            .iter()
            .filter(|(_, c)| c.selection != cfg.selection)
            .map(|(id, _)| id.as_str())
            .collect();
        if !differing.is_empty() {
            cmp.notes.push(format!(
                "runs {} were trained with different selection settings; ranking uses those of {}",
                differing.join(", "),
                recorded[0].0
            ));
        }
        if let Some(g) = &group {
            cmp.notes.extend(Self::unstarted_note(g, &runs));
        }
        Ok(cmp)
    }

    fn compare_records(&self, runs: &[RunRecord], cfg: &ProjectConfig) -> Result<Comparison> {
        let metric = cfg.selection.metric.as_str();
        let protocol_of = |r: &RunRecord| ComparisonProtocol {
            snapshot_id: r.snapshot_id.clone(),
            split_id: r.split_id.clone(),
            split: SplitName::Valid,
            target: String::new(),
            positive_label: String::new(),
            threshold: format!("{}", r.threshold),
            metrics_version: METRICS_VERSION.to_string(),
        };
        // The reference protocol is that of the newest completed run: on mixed data
        // versions, models are ranked on the latest data, never on an older version
        // just because its run was listed first.
        let reference = runs
            .iter()
            .filter(|r| r.status == RunStatus::Completed)
            .max_by_key(|r| id_number(&r.run_id))
            .map(protocol_of);
        let mut rows = Vec::new();
        for r in runs {
            let proto = protocol_of(r);
            let mut notes = Vec::new();
            let comparable = match &reference {
                Some(refp) => {
                    let d = refp.differences(&proto);
                    if !d.is_empty() {
                        notes.push(format!("not directly comparable: {}", d.join(", ")));
                    }
                    d.is_empty()
                }
                None => false,
            };
            let params = self.model_params(r);
            let model_kib = params as f64 * 4.0 / 1024.0;
            let valid = r.metrics.as_ref().map(|m| m.valid.clone());
            let metric_value = valid.as_ref().and_then(|v| v.get(metric));
            let p95 = r.latency.as_ref().map(|l| l.single_row_p95_us);
            let mut ok = r.status == RunStatus::Completed;
            if let Some(v) = metric_value {
                if cfg.selection.min_valid_metric > 0.0 && v < cfg.selection.min_valid_metric {
                    ok = false;
                    notes.push(format!("{metric} {v:.4} below min_valid_metric {}", cfg.selection.min_valid_metric));
                }
            }
            if cfg.selection.max_model_kib > 0.0 && model_kib > cfg.selection.max_model_kib {
                ok = false;
                notes.push(format!(
                    "model {model_kib:.1} KiB above max_model_kib {}",
                    cfg.selection.max_model_kib
                ));
            }
            if let Some(l) = p95 {
                if cfg.selection.max_p95_latency_us > 0.0 && l > cfg.selection.max_p95_latency_us {
                    ok = false;
                    notes.push(format!(
                        "p95 latency {l:.1} µs above max_p95_latency_us {}",
                        cfg.selection.max_p95_latency_us
                    ));
                }
            }
            if r.status != RunStatus::Completed {
                notes.push(format!("run is {}", r.status.as_str()));
            }
            rows.push(ComparisonRow {
                run_id: r.run_id.clone(),
                candidate: r.candidate,
                status: r.status,
                snapshot_id: r.snapshot_id.clone(),
                valid,
                metric_value,
                parameters: params,
                model_kib,
                p95_latency_us: p95,
                peak_rss_mb: r.device_receipt.as_ref().and_then(|d| d.peak_rss_mb),
                train_ms: r.timings.train_ms,
                best_epoch: r.metrics.as_ref().map(|m| m.best_epoch),
                comparable,
                notes,
                meets_constraints: ok,
            });
        }
        let recommended = rows
            .iter()
            .filter(|r| r.comparable && r.meets_constraints && r.metric_value.is_some())
            .max_by(|a, b| {
                let (x, y) = (a.metric_value.unwrap_or(f64::MIN), b.metric_value.unwrap_or(f64::MIN));
                // Higher metric wins; on a tie (within 1e-9) the smaller model wins.
                if (x - y).abs() <= 1e-9 {
                    b.parameters.cmp(&a.parameters)
                } else {
                    x.total_cmp(&y)
                }
            })
            .map(|r| r.run_id.clone());
        let mut notes = vec![format!(
            "ranking uses the validation split only ({metric}); the test split is not used for selection"
        )];
        if rows.iter().any(|r| !r.comparable) {
            notes.push("rows marked not comparable are shown but excluded from the recommendation".into());
        }
        if let Some(refp) = &reference {
            let older: Vec<&str> = rows
                .iter()
                .filter(|r| r.snapshot_id != refp.snapshot_id)
                .map(|r| r.run_id.as_str())
                .collect();
            if !older.is_empty() {
                notes.push(format!(
                    "{} trained on another data version and cannot be ranked by validation scores of different data; to compare an old model with the new one on the same new rows: rustpy bundle build --run <old> (if needed), then rustpy evaluate --bundle bundles/<old> --split test and rustpy evaluate --run {} --split test",
                    older.join(", "),
                    recommended.as_deref().unwrap_or("<new>")
                ));
            }
        }
        Ok(Comparison {
            metric: metric.to_string(),
            protocol: reference,
            rows,
            recommended,
            notes,
        })
    }

    fn model_params(&self, r: &RunRecord) -> usize {
        read_json::<rustpy_contracts::model::ModelConfig>(
            &self.project.run_dir(&r.run_id).join("model").join("model.config.json"),
            Phase::Compare,
        )
        .map(|c| c.parameter_count())
        .unwrap_or(0)
    }

    /// Effective configuration rows `(path, value, source)` for display.
    pub fn effective_config(&self, req: &ConfigRequest) -> Result<Vec<(String, String, ValueSource)>> {
        Ok(self.project.resolve_config(req)?.effective_values())
    }
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for e in fs::read_dir(from)? {
        let e = e?;
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()))?;
        } else {
            fs::copy(e.path(), to.join(e.file_name()))?;
        }
    }
    Ok(())
}

/// Numeric suffix of an id like `eval-012` / `run-1000` (for ordering).
pub fn id_number(id: &str) -> u64 {
    id.rsplit('-').next().and_then(|n| n.parse().ok()).unwrap_or(0)
}

fn probs(model: &DenseModel, x: &Matrix) -> Vec<f64> {
    model.logits(x).into_iter().map(|z| sigmoid(f64::from(z))).collect()
}

/// Single-row latency through the runtime path (transform + forward), on up to 200 rows.
fn measure_latency(fitted: &FittedArtifacts, model: &DenseModel, table: &Table, rows: &[usize]) -> LatencyStats {
    let sample: Vec<usize> = rows.iter().copied().take(200).collect();
    let sub = table.take(&sample);
    let t0 = Instant::now();
    if let Ok(x) = apply_transform(&fitted.transform, &sub, Phase::Evaluate) {
        let _ = model.logits(&x);
    }
    let batch_ms = ms(t0);
    let mut times = Vec::with_capacity(sample.len());
    for i in 0..sample.len() {
        let one = sub.take(&[i]);
        let t = Instant::now();
        if let Ok(x) = apply_transform(&fitted.transform, &one, Phase::Evaluate) {
            std::hint::black_box(model.logits(&x));
        }
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    let q = |f: f64| times.get(((times.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0.0);
    LatencyStats {
        rows: sample.len(),
        batch_ms,
        single_row_p50_us: q(0.5),
        single_row_p95_us: q(0.95),
    }
}

#[cfg(target_os = "linux")]
fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn pid_alive(_pid: u32) -> bool {
    // Without /proc we cannot tell; treat as alive so a live run is never marked
    // interrupted. Users can still resume once the process is gone.
    true
}
