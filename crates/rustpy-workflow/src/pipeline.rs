//! Orchestration of the stages: plan, run, resume, evaluate, compare.
//! The CLI and the Rust SDK both call these functions; neither has its own logic.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustpy_contracts::config::{Candidate, ProjectConfig, ResolvedConfig, ValueSource};
use rustpy_contracts::data::{DataSnapshot, SplitManifest, SplitName};
use rustpy_contracts::device::DevicePlan;
use rustpy_contracts::evaluation::{
    BinaryMetrics, ComparisonProtocol, EvaluationRecord, EvaluationSubject, LatencyStats,
};
use rustpy_contracts::hash::sha256_file;
use rustpy_contracts::run::{ArtifactRef, Lineage, ResumeLevel, RunMetrics, RunRecord, RunStatus, Timings};
use rustpy_contracts::time::now_rfc3339;
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError, METRICS_VERSION, RUN_FORMAT_VERSION, RUSTPY_VERSION};
use rustpy_runtime::model::{sigmoid, DenseModel};
use rustpy_runtime::{apply_transform, Matrix, Predictor, LoadOptions, Table};
use serde::{Deserialize, Serialize};

use crate::data::{plan_columns, snapshot_source, load_snapshot, FeaturePlan};
use crate::device::{environment, estimate_peak_mb, plan_device, receipt, ENGINE};
use crate::fit::{fit_transform, transform_id, FittedArtifacts};
use crate::metrics::binary_metrics;
use crate::project::{append_jsonl, read_json, write_json, ConfigRequest, Project};
use crate::split::{labels, make_split};
use crate::train::{
    latest_checkpoint, model_config, train, ResumeKey, RunControl, StartFrom, TrainInputs, TrainOutcome, TrainState,
};

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
    RunStarted { run_id: String, candidate: Candidate, steps_planned: u64 },
    Epoch { run_id: String, epoch: usize, epochs: usize, train_loss: f64, valid_loss: f64, valid_metric: f64 },
    RunFinished { run_id: String, status: RunStatus },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunGroup {
    pub group_id: String,
    pub created_at: String,
    pub run_ids: Vec<String>,
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

#[derive(Debug, Clone)]
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
        let tid = transform_id(&split.split_id, &cfg.preprocess, &features, cfg.selection.threshold);
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
        let split: SplitManifest = read_json(
            &self.project.dir("splits").join(format!("{}.json", run.split_id)),
            Phase::Split,
        )?;
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
        let phase = if which == SplitName::Train { Phase::Train } else { Phase::Evaluate };
        let x = apply_transform(&p.fitted.transform, &sub, phase)?;
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
                let request = match c {
                    Candidate::Linear => cfg.execution.baseline_device,
                    Candidate::SmallMlp => cfg.execution.candidate_device,
                };
                let hidden = &cfg.train.for_candidate(c).hidden;
                let est = estimate_peak_mb(p.snapshot.source.bytes, p.snapshot.n_rows, p.fitted.transform.output_dim, hidden);
                plan_device(request, &cfg.execution, est).map(|plan| (c, plan))
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
            return Err(RustpyError::new(ErrorCode::RunNotFound, Phase::Resume, format!("run `{run_id}` does not exist"))
                .field(run_id)
                .actual(if known.is_empty() { "no runs yet".into() } else { format!("runs: {}", known.join(", ")) })
                .next("list runs with `rustpy runs`"));
        }
        let mut run: RunRecord = read_json(&path, Phase::Resume)?;
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
            let alive = run.pid.is_some_and(pid_alive);
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
        ids.sort();
        ids
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
        names.sort();
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
        let candidates = opts
            .candidates
            .clone()
            .unwrap_or_else(|| prepared.resolved.config.selection.candidates.clone());
        let plans = self.device_plans(&prepared, &candidates)?;
        let train_data = self.split_data(&prepared, SplitName::Train)?;
        let valid_data = self.split_data(&prepared, SplitName::Valid)?;
        let group_id = self.project.allocate_id("groups", "grp")?;
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
            runs.push(run);
        }
        let group = RunGroup {
            group_id: group_id.clone(),
            created_at: now_rfc3339(),
            run_ids: runs.iter().map(|r| r.run_id.clone()).collect(),
            snapshot_id: prepared.snapshot.snapshot_id.clone(),
            split_id: prepared.split.split_id.clone(),
            transform_id: prepared.fitted.transform_id.clone(),
            config_hash: prepared.resolved.config_hash.clone(),
        };
        write_json(&self.project.dir("groups").join(format!("{group_id}.json")), &group)?;
        let _ = fs::remove_file(self.project.dir("groups").join(format!("{group_id}.lock")));
        let comparison = self.compare_records(&runs, &prepared.resolved.config)?;
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
        let _ = fs::remove_file(self.project.dir("runs").join(format!("{run_id}.lock")));
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
        run.status = RunStatus::Running;
        run.started_at = Some(now_rfc3339());
        run.pid = Some(std::process::id());
        self.save_run(&run)?;
        self.event(&run_id, "started", serde_json::json!({"candidate": candidate, "device": plan.device}));
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
        let mut run: RunRecord = read_json(&run_path, Phase::Train)?;
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
                let model_dir = run_dir.join("model");
                crate::project::write_atomic(&model_dir.join("model.safetensors"), &best.to_safetensors()?)?;
                write_json(&model_dir.join("model.config.json"), &best.config)?;
                for name in ["model.safetensors", "model.config.json"] {
                    let path = model_dir.join(name);
                    let (sha, bytes) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Train, &path, e))?;
                    run.artifacts.push(ArtifactRef {
                        name: name.to_string(),
                        path: self.project.rel(&path),
                        sha256: sha,
                        bytes,
                    });
                }
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
                self.event(&run_id, "completed", serde_json::json!({"steps": state.step, "best_epoch": state.best_epoch}));
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
        let ck_root = self.project.run_dir(run_id).join("checkpoints");
        let Some(ck) = latest_checkpoint(&ck_root) else {
            return Err(RustpyError::new(
                ErrorCode::CheckpointMissing,
                Phase::Resume,
                format!("run {run_id} has no checkpoint to continue from"),
            )
            .run(run_id)
            .next("start again with `rustpy run` (R0); checkpoints are written at the end of every epoch"));
        };
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
        let state: TrainState = read_json(&ck.join("state.json"), Phase::Resume)?;
        let from_step = state.step;
        let train_data = self.split_data(&p, SplitName::Train)?;
        let valid_data = self.split_data(&p, SplitName::Valid)?;
        let lineage = Lineage {
            level: ResumeLevel::ExactResume,
            parent_run_id: run_id.to_string(),
            from_step: Some(from_step),
        };
        // Validate compatibility before allocating a run id.
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
        };
        let saved: ResumeKey = read_json(&ck.join("key.json"), Phase::Resume)?;
        let diffs = saved.differences(&key);
        if !diffs.is_empty() {
            return Err(RustpyError::new(
                ErrorCode::ResumeIncompatible,
                Phase::Resume,
                format!("cannot continue run {run_id} exactly (R2): {}", diffs.join("; ")),
            )
            .run(run_id)
            .retained(format!("run {run_id} and its checkpoints are unchanged"))
            .next(format!("start a new run (`rustpy run`), or `rustpy resume {run_id} --warm-start` for a new training from its weights")));
        }
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
        Ok(run)
    }

    fn warm_start(
        &self,
        parent: &RunRecord,
        req: &ConfigRequest,
        control: &RunControl,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<RunRecord> {
        let weights = self.run_weights(parent).map_err(|e| {
            e.in_phase(Phase::Resume)
                .next(format!("warm start needs a completed run; run {} is {}", parent.run_id, parent.status.as_str()))
        })?;
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
        let cfg = &p.resolved.config;
        if model_config(parent.candidate, p.fitted.transform.output_dim, cfg.train.for_candidate(parent.candidate))
            != weights.config
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
            snapshot_id: p.snapshot.snapshot_id.clone(),
            split_id: p.split.split_id.clone(),
            transform_id: p.fitted.transform_id.clone(),
            config_hash: p.resolved.config_hash.clone(),
        };
        write_json(&self.project.dir("groups").join(format!("{group_id}.json")), &group)?;
        let _ = fs::remove_file(self.project.dir("groups").join(format!("{group_id}.lock")));
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

    fn ledger_path(&self) -> PathBuf {
        self.project.dir("evaluations").join("ledger.jsonl")
    }

    fn ledger(&self) -> Vec<LedgerEntry> {
        fs::read_to_string(self.ledger_path())
            .map(|s| s.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
            .unwrap_or_default()
    }

    pub fn evaluations(&self) -> Vec<EvaluationRecord> {
        let mut v: Vec<EvaluationRecord> = fs::read_dir(self.project.dir("evaluations"))
            .map(|it| {
                it.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with("eval-"))
                    .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
                    .filter_map(|e| read_json(&e.path(), Phase::Evaluate).ok())
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by(|a, b| a.evaluation_id.cmp(&b.evaluation_id));
        v
    }

    fn check_test_use(&self, split_id: &str, subject: &str, allow_reuse: bool) -> Result<(Option<String>, u32)> {
        let uses: Vec<LedgerEntry> = self.ledger().into_iter().filter(|e| e.split_id == split_id).collect();
        if let Some(prev) = uses.iter().find(|e| e.subject == subject) {
            return Ok((Some(prev.evaluation_id.clone()), prev.use_index));
        }
        if !uses.is_empty() && !allow_reuse {
            let prev = &uses[0];
            return Err(RustpyError::new(
                ErrorCode::TestSetReuse,
                Phase::Evaluate,
                format!(
                    "the test rows of split {split_id} were already used to evaluate {} ({}); evaluating another model on them would turn the test set into a selection set",
                    prev.subject, prev.evaluation_id
                ),
            )
            .field("--split test")
            .retained(format!("evaluation {}", prev.evaluation_id))
            .next("compare candidates on the validation split (`--split valid`); if you accept a biased estimate, pass --allow-test-reuse (recorded in reports and bundles)"));
        }
        Ok((None, uses.len() as u32 + 1))
    }

    fn save_evaluation(&self, mut rec: EvaluationRecord, record_test_use: bool) -> Result<EvaluationRecord> {
        let id = self.project.allocate_id("evaluations", "eval")?;
        rec.evaluation_id = id.clone();
        let dir = self.project.dir("evaluations");
        write_json(&dir.join(format!("{id}.json")), &rec)?;
        let _ = fs::remove_file(dir.join(format!("{id}.lock")));
        if record_test_use {
            let subject = match &rec.subject {
                EvaluationSubject::Run { run_id } => run_id.clone(),
                EvaluationSubject::Bundle { bundle_id, .. } => bundle_id.clone(),
            };
            append_jsonl(
                &self.ledger_path(),
                &LedgerEntry {
                    split_id: rec.protocol.split_id.clone(),
                    subject,
                    evaluation_id: id,
                    use_index: rec.test_use_index.unwrap_or(1),
                    at: rec.created_at.clone(),
                },
            )?;
        }
        Ok(rec)
    }

    /// Evaluates a completed run on one split of its own snapshot. The test split
    /// may be used once per split; later uses need `allow_test_reuse` and are recorded.
    pub fn evaluate_run(&self, run_id: &str, split: SplitName, allow_test_reuse: bool) -> Result<EvaluationRecord> {
        let run = self.load_run(run_id)?;
        let model = self.run_weights(&run)?;
        let mut test_index = None;
        if split == SplitName::Test {
            let (existing, idx) = self.check_test_use(&run.split_id, run_id, allow_test_reuse)?;
            if let Some(eid) = existing {
                let mut rec: EvaluationRecord =
                    read_json(&self.project.dir("evaluations").join(format!("{eid}.json")), Phase::Evaluate)?;
                rec.notes.push("existing evaluation returned; the test set was not used again".into());
                return Ok(rec);
            }
            test_index = Some(idx);
        }
        let p = self.prepare_for_run(&run)?;
        let data = self.split_data(&p, split)?;
        let pr = probs(&model, &data.x);
        let metrics = binary_metrics(&pr, &data.y, run.threshold);
        let mut notes = Vec::new();
        if test_index.is_some_and(|i| i > 1) {
            notes.push(format!(
                "test split reused (use #{}): this is no longer an unbiased estimate",
                test_index.unwrap_or(0)
            ));
        }
        let rec = EvaluationRecord {
            evaluation_id: String::new(),
            created_at: now_rfc3339(),
            subject: EvaluationSubject::Run { run_id: run_id.to_string() },
            protocol: self.protocol(&p, split),
            metrics,
            test_use_index: test_index,
            latency: Some(measure_latency(&p.fitted, &model, &p.table, &data.rows)),
            notes,
        };
        self.save_evaluation(rec, split == SplitName::Test)
    }

    /// Evaluates a bundle (its frozen transform and weights) on a split of the
    /// *current* data snapshot — e.g. old vs new model on the same new data.
    pub fn evaluate_bundle(
        &self,
        bundle_dir: &Path,
        split: SplitName,
        req: &ConfigRequest,
        allow_test_reuse: bool,
    ) -> Result<EvaluationRecord> {
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
            .expected(format!("{} with labels {:?}/{:?}", out.target, out.negative_label, out.positive_label))
            .actual(format!("{} with labels {:?}/{:?}", t.name, t.negative_label, t.positive_label)));
        }
        let mut test_index = None;
        if split == SplitName::Test {
            let (existing, idx) = self.check_test_use(&p.split.split_id, predictor.bundle_id(), allow_test_reuse)?;
            if let Some(eid) = existing {
                return read_json(&self.project.dir("evaluations").join(format!("{eid}.json")), Phase::Evaluate);
            }
            test_index = Some(idx);
        }
        let rows = p.split.rows(split);
        let names: Vec<&str> = predictor.input_schema().fields.iter().map(|f| f.name.as_str()).collect();
        let sub = p.table.take(&rows).select(&names);
        let y = labels(&p.table.take(&rows), t);
        let pr = predictor.probabilities(&sub).map_err(|e| e.in_phase(Phase::Evaluate))?;
        let metrics = binary_metrics(&pr, &y, out.threshold);
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
            metrics,
            test_use_index: test_index,
            latency: None,
            notes: vec![format!(
                "bundle built from snapshot {} evaluated on snapshot {}",
                predictor.manifest().provenance.snapshot_id, p.snapshot.snapshot_id
            )],
        };
        self.save_evaluation(rec, split == SplitName::Test)
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------

    pub fn compare(&self, run_ids: &[String]) -> Result<Comparison> {
        let ids: Vec<String> = if run_ids.is_empty() {
            match self.latest_group()? {
                Some(g) => g.run_ids,
                None => {
                    return Err(RustpyError::new(ErrorCode::RunNotFound, Phase::Compare, "there are no runs to compare yet")
                        .next("train candidates with `rustpy run`"));
                }
            }
        } else {
            run_ids.to_vec()
        };
        let runs: Vec<RunRecord> = ids.iter().map(|id| self.load_run(id)).collect::<Result<_>>()?;
        let cfg = self.project.resolve_config(&ConfigRequest::default())?.config;
        self.compare_records(&runs, &cfg)
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
        let reference = runs.iter().find(|r| r.status == RunStatus::Completed).map(protocol_of);
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
                notes.push(format!("model {model_kib:.1} KiB above max_model_kib {}", cfg.selection.max_model_kib));
            }
            if let Some(l) = p95 {
                if cfg.selection.max_p95_latency_us > 0.0 && l > cfg.selection.max_p95_latency_us {
                    ok = false;
                    notes.push(format!("p95 latency {l:.1} µs above max_p95_latency_us {}", cfg.selection.max_p95_latency_us));
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

/// One use of a test split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub split_id: String,
    pub subject: String,
    pub evaluation_id: String,
    pub use_index: u32,
    pub at: String,
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
    pid == std::process::id() || Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn pid_alive(_pid: u32) -> bool {
    // Without /proc we cannot tell; treat as alive so a live run is never marked
    // interrupted. Users can still resume once the process is gone.
    true
}
