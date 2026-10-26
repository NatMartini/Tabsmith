//! Run records: one training attempt of one candidate.

use serde::{Deserialize, Serialize};

use crate::config::{Candidate, CandidateTrainConfig};
use crate::device::{DevicePlan, DeviceReceipt};
use crate::error::RustpyError;
use crate::evaluation::{BinaryMetrics, LatencyStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Planned,
    Running,
    Completed,
    Failed,
    /// Stopped on user request (Ctrl-C); a checkpoint may be available.
    Cancelled,
    /// The process died while running (detected afterwards).
    Interrupted,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled | RunStatus::Interrupted
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Planned => "planned",
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
            RunStatus::Cancelled => "cancelled",
            RunStatus::Interrupted => "interrupted",
        }
    }
}

/// How a Run relates to an earlier one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResumeLevel {
    /// R0: recomputed from scratch; nothing reused.
    #[serde(rename = "R0")]
    Rerun,
    /// R1: reused completed, hash-compatible stages (snapshot, split, transform).
    #[serde(rename = "R1")]
    StageReuse,
    /// R2: continued from a complete optimizer-step boundary of the parent.
    #[serde(rename = "R2")]
    ExactResume,
    /// New training started from the parent's weights with a fresh optimizer. Not a resume.
    #[serde(rename = "warm_start")]
    WarmStart,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lineage {
    pub level: ResumeLevel,
    pub parent_run_id: String,
    /// Optimizer step of the parent the run continued from (R2 / warm start).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_step: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub step: u64,
    pub train_loss: f64,
    pub valid_loss: f64,
    pub valid_metric: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Timings {
    pub data_ms: f64,
    pub fit_transform_ms: f64,
    pub train_ms: f64,
    pub eval_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub name: String,
    /// Path relative to the project directory.
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub rustpy_version: String,
    pub engine: String,
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    pub logical_cpus: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    pub train: BinaryMetrics,
    pub valid: BinaryMetrics,
    /// Epoch whose weights were kept (lowest validation loss).
    pub best_epoch: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub format_version: u32,
    pub run_id: String,
    /// Runs created by the same `rustpy run` invocation share a group.
    pub group_id: String,
    pub candidate: Candidate,
    pub status: RunStatus,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<Lineage>,
    pub snapshot_id: String,
    pub split_id: String,
    pub transform_id: String,
    pub config_hash: String,
    pub train_config: CandidateTrainConfig,
    pub train_seed: u64,
    pub threshold: f64,
    /// Stages taken from cache instead of being recomputed (R1).
    #[serde(default)]
    pub reused_stages: Vec<String>,
    pub device_plan: DevicePlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_receipt: Option<DeviceReceipt>,
    /// Optimizer steps completed so far.
    pub steps_done: u64,
    pub steps_planned: u64,
    #[serde(default)]
    pub history: Vec<EpochRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<RunMetrics>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default)]
    pub timings: Timings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RustpyError>,
    pub environment: EnvironmentInfo,
    /// Process id while running (used to detect interrupted runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Max |logit difference| between the training engine and the runtime forward pass
    /// on the validation rows, for the kept weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_parity_max_abs_diff: Option<f64>,
    /// Prediction latency of the kept model through the runtime (validation rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<LatencyStats>,
    /// Why training stopped (`epochs`, `early_stopping`, `time_budget`, `cancelled`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}
