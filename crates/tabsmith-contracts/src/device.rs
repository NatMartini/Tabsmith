//! Execution plan (before) and execution receipt (after) for devices.

use serde::{Deserialize, Serialize};

/// Certification level of a device path for a given recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Certification {
    /// The full recipe (train, resume, bundle, predict) is verified on this path.
    Certified,
    /// Hardware was detected but this build has no backend for it.
    NotCompiled,
    /// A backend exists but the recipe has not been verified on it.
    NotCertified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedDevice {
    /// `cpu`, `nvidia-gpu`, `amd-gpu`, `intel-gpu`, `apple-gpu`, `gpu` (unknown vendor)
    pub kind: String,
    pub name: String,
    pub certification: Certification,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Backend that reaches this device in this build (`cuda`, `vulkan`), with the
    /// device's index for `execution.gpu`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Driver version (Vulkan) or compute capability (CUDA), part of the certification key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevicePlan {
    /// What the configuration asked for (`cpu`, `auto-certified`, `cuda`, ...).
    pub requested: String,
    /// Backends compiled into this build.
    pub compiled_backends: Vec<String>,
    pub detected: Vec<DetectedDevice>,
    /// Backend + device that will actually be used.
    pub backend: String,
    pub device: String,
    pub precision: String,
    /// Certification of the selected device path for this recipe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certification: Option<Certification>,
    pub fallback: String,
    /// Where data preparation runs; this is a planned placement, not a fallback.
    pub data_placement: String,
    /// Copies made between reading the file and the first training step.
    pub copy_path: Vec<String>,
    pub threads: usize,
    pub estimated_peak_mb: f64,
    pub memory_budget_mb: u64,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceReceipt {
    pub backend: String,
    pub device: String,
    pub precision: String,
    pub threads: usize,
    /// Peak resident set size during this run (VmHWM, reset at the start of the run
    /// on Linux; includes the data already loaded by the process), if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<f64>,
    pub host_device_transfers: String,
    /// Whether backend/device/precision match the plan.
    pub matches_plan: bool,
}
