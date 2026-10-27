//! Device discovery, execution plans and receipts. No silent fallback: a requested
//! device that is not compiled in or not certified is an error.

use std::fs;

use rustpy_contracts::config::{DeviceRequest, ExecutionConfig, Precision};
use rustpy_contracts::device::{Certification, DetectedDevice, DevicePlan, DeviceReceipt};
use rustpy_contracts::run::EnvironmentInfo;
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError, RUSTPY_VERSION};

/// The training engine compiled into this build.
pub const ENGINE: &str = "burn 0.22.1 / ndarray backend (CPU, single thread)";
pub const BACKEND: &str = "burn-ndarray";

pub const COMPILED_BACKENDS: &[&str] = &["burn-ndarray (cpu)"];

pub fn cpu_model() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
}

pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

pub fn environment() -> EnvironmentInfo {
    EnvironmentInfo {
        rustpy_version: RUSTPY_VERSION.to_string(),
        engine: ENGINE.to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: cpu_model(),
        logical_cpus: logical_cpus(),
    }
}

/// Read-only GPU discovery (Linux sysfs). Nothing is installed or initialised.
pub fn detect_devices() -> Vec<DetectedDevice> {
    let verified = cfg!(all(target_os = "linux", target_arch = "x86_64"));
    let mut out = vec![DetectedDevice {
        kind: "cpu".into(),
        name: cpu_model().unwrap_or_else(|| format!("{} CPU", std::env::consts::ARCH)),
        certification: if verified { Certification::Certified } else { Certification::NotCertified },
        note: Some(if verified {
            format!("{} logical CPUs; recipe verified on linux/x86_64", logical_cpus())
        } else {
            format!(
                "{} logical CPUs; the recipe has only been verified on linux/x86_64, not on {}/{}",
                logical_cpus(),
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        }),
    }];
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        let mut seen = std::collections::BTreeSet::new();
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let vendor = fs::read_to_string(e.path().join("device/vendor")).unwrap_or_default();
            let device = fs::read_to_string(e.path().join("device/device")).unwrap_or_default();
            let key = format!("{}:{}", vendor.trim(), device.trim());
            if !seen.insert(key.clone()) {
                continue;
            }
            let kind = match vendor.trim() {
                "0x10de" => "nvidia-gpu",
                "0x1002" => "amd-gpu",
                "0x8086" => "intel-gpu",
                _ => "gpu",
            };
            out.push(DetectedDevice {
                kind: kind.into(),
                name: format!("{name} (pci {key})"),
                certification: Certification::NotCompiled,
                note: Some("no GPU backend is compiled into this build; GPU certification is pending".into()),
            });
        }
    }
    if fs::metadata("/proc/driver/nvidia/gpus").is_ok() && !out.iter().any(|d| d.kind == "nvidia-gpu") {
        out.push(DetectedDevice {
            kind: "nvidia-gpu".into(),
            name: "NVIDIA GPU (driver present)".into(),
            certification: Certification::NotCompiled,
            note: Some("no CUDA backend is compiled into this build".into()),
        });
    }
    out
}

/// Resolves a device request into a plan. `estimated_peak_mb` is checked against the budget.
pub fn plan_device(
    request: DeviceRequest,
    key: &str,
    exec: &ExecutionConfig,
    estimated_peak_mb: f64,
) -> Result<DevicePlan> {
    let detected = detect_devices();
    let mut notes = Vec::new();
    match exec.precision {
        Precision::Fp32 => {}
        other => {
            return Err(RustpyError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Device,
                format!("precision {other:?} is not certified for tabular-binary@0.1"),
            )
            .field("execution.precision")
            .expected("fp32")
            .next("set execution.precision = \"fp32\"; mixed precision is planned as a separate certification"));
        }
    }
    match request {
        DeviceRequest::Cpu => {}
        DeviceRequest::AutoCertified => {
            let gpus: Vec<&DetectedDevice> = detected.iter().filter(|d| d.kind != "cpu").collect();
            if gpus.is_empty() {
                notes.push("auto-certified selected the CPU (the only certified device in this build)".into());
            } else {
                notes.push(format!(
                    "auto-certified selected the CPU: {} detected GPU(s) have no certified backend in this build",
                    gpus.len()
                ));
            }
        }
        other => {
            let found: Vec<String> = detected.iter().map(|d| format!("{} ({})", d.name, d.kind)).collect();
            return Err(RustpyError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Device,
                format!(
                    "device `{}` is not available: no {} backend is compiled into this build and it is not certified for tabular-binary@0.1",
                    other.as_str(),
                    other.as_str()
                ),
            )
            .field(key)
            .expected("cpu or auto-certified")
            .actual(format!("requested {}; detected: {}", other.as_str(), found.join("; ")))
            .next(format!("set {key} = \"cpu\" (or \"auto-certified\"); RustPy never falls back silently")));
        }
    }
    if estimated_peak_mb > exec.max_memory_mb as f64 {
        return Err(RustpyError::new(
            ErrorCode::ResourceBudgetExceeded,
            Phase::Device,
            format!(
                "estimated peak memory {estimated_peak_mb:.0} MB exceeds execution.max_memory_mb = {}",
                exec.max_memory_mb
            ),
        )
        .field("execution.max_memory_mb")
        .expected(format!("<= {} MB", exec.max_memory_mb))
        .actual(format!("{estimated_peak_mb:.0} MB"))
        .next("raise execution.max_memory_mb if the machine has the memory, or use a smaller dataset / fewer categorical levels"));
    }
    Ok(DevicePlan {
        requested: request.as_str().to_string(),
        compiled_backends: COMPILED_BACKENDS.iter().map(|s| s.to_string()).collect(),
        detected,
        backend: BACKEND.to_string(),
        device: "cpu".to_string(),
        precision: "fp32".to_string(),
        fallback: "error".to_string(),
        data_placement: "cpu (host memory; planned placement, not a fallback)".to_string(),
        copy_path: vec![
            "file -> Polars DataFrame (host)".into(),
            "DataFrame -> raw value columns (host copy)".into(),
            "raw values -> f32 feature matrix via the shared transform (host copy)".into(),
            "feature matrix -> per-batch NdArray tensors (host copy, no device transfer)".into(),
        ],
        threads: 1,
        estimated_peak_mb,
        memory_budget_mb: exec.max_memory_mb,
        notes,
    })
}

/// Resets the process peak-RSS counter (Linux `clear_refs` = 5) so the next
/// `peak_rss_mb` reading covers only what follows. Returns false where unsupported.
pub fn reset_peak_rss() -> bool {
    fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// Peak resident set size of this process (Linux VmHWM), in MB.
pub fn peak_rss_mb() -> Option<f64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

pub fn receipt(plan: &DevicePlan) -> DeviceReceipt {
    let receipt = DeviceReceipt {
        backend: BACKEND.to_string(),
        device: "cpu".to_string(),
        precision: "fp32".to_string(),
        threads: 1,
        peak_rss_mb: peak_rss_mb(),
        host_device_transfers: "none (host backend)".to_string(),
        matches_plan: true,
    };
    DeviceReceipt {
        matches_plan: receipt.backend == plan.backend && receipt.device == plan.device && receipt.precision == plan.precision,
        ..receipt
    }
}

/// Rough peak-memory estimate for one training run, in MB.
pub fn estimate_peak_mb(file_bytes: u64, rows: usize, feature_dim: usize, hidden: &[usize]) -> f64 {
    let raw = file_bytes as f64 * 4.0; // DataFrame + raw columns + transient parse buffers
    let features = rows as f64 * feature_dim as f64 * 4.0 * 3.0; // matrices for train/valid/test + batch copies
    let params: usize = {
        let mut d = feature_dim;
        let mut n = 0;
        for &h in hidden {
            n += d * h + h;
            d = h;
        }
        n + d + 1
    };
    let model = params as f64 * 4.0 * 6.0; // weights, grads, Adam moments, best copy, checkpoint buffer
    let engine_overhead = 64.0 * 1024.0 * 1024.0;
    (raw + features + model + engine_overhead) / (1024.0 * 1024.0)
}
