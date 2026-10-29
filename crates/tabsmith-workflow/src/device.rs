//! Device discovery, execution plans and receipts. No silent fallback: a requested
//! device that is not compiled in, not present, or not able to train at the
//! requested precision is an error.
//!
//! GPU training (Cargo features `vulkan` and `cuda`) runs Burn's CubeCL backends
//! with kernel fusion and without autotuning: autotuning picks kernels by timing, so
//! two processes could run different kernels and train different weights.

use std::fs;

use burn::tensor::{Device, Tensor, TensorData};
use serde::Deserialize;
use tabsmith_contracts::config::{DeviceRequest, ExecutionConfig, Precision};
use tabsmith_contracts::device::{Certification, DetectedDevice, DevicePlan, DeviceReceipt};
use tabsmith_contracts::run::EnvironmentInfo;
use tabsmith_contracts::{ErrorCode, Phase, Result, TABSMITH_VERSION, TabsmithError};

/// The CPU training engine.
pub const ENGINE: &str = "burn 0.22.1 / flex backend (CPU, single thread)";
pub const BACKEND: &str = "burn-flex";

/// Training backends compiled into this build.
pub const COMPILED_BACKENDS: &[&str] = &[
    "burn-flex (cpu)",
    #[cfg(feature = "vulkan")]
    "burn-vulkan (vulkan)",
    #[cfg(feature = "cuda")]
    "burn-cuda (cuda)",
];

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
        tabsmith_version: TABSMITH_VERSION.to_string(),
        engine: ENGINE.to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: cpu_model(),
        logical_cpus: logical_cpus(),
    }
}

// ---------------------------------------------------------------------------
// Engine devices

/// Where training runs, as resolved by [`plan_device`].
#[derive(Debug, Clone, PartialEq)]
pub enum EngineDevice {
    /// Burn's Flex backend on the CPU, single thread.
    Cpu,
    /// Tabsmith's own gradient-boosted trees, pure Rust on the CPU, single thread.
    Trees,
    /// A CUDA device (CubeCL CUDA backend), by ordinal.
    Cuda {
        ordinal: usize,
        name: String,
        compute_capability: (i32, i32),
    },
    /// A GPU through Vulkan (CubeCL wgpu backend). Burn numbers discrete and
    /// integrated GPUs separately; `index` counts within the GPU's own kind.
    Vulkan {
        discrete: bool,
        index: usize,
        name: String,
        driver: String,
    },
}

impl EngineDevice {
    pub fn is_gpu(&self) -> bool {
        !matches!(self, EngineDevice::Cpu | EngineDevice::Trees)
    }

    pub fn backend(&self) -> &'static str {
        match self {
            EngineDevice::Cpu => BACKEND,
            EngineDevice::Trees => "tabsmith-gbdt",
            EngineDevice::Cuda { .. } => "burn-cuda",
            EngineDevice::Vulkan { .. } => "burn-vulkan",
        }
    }

    /// Engine description recorded in runs and in the exact-resume key.
    pub fn engine(&self) -> String {
        match self {
            EngineDevice::Cpu => ENGINE.to_string(),
            EngineDevice::Trees => "tabsmith gbdt (pure Rust, CPU, single thread)".into(),
            EngineDevice::Cuda { .. } => "burn 0.22.1 / cuda backend (CubeCL, kernel fusion, fixed kernels)".into(),
            EngineDevice::Vulkan { .. } => "burn 0.22.1 / vulkan backend (CubeCL wgpu, kernel fusion, fixed kernels)".into(),
        }
    }

    /// Short device label for plans and receipts.
    pub fn label(&self) -> String {
        match self {
            EngineDevice::Cpu | EngineDevice::Trees => "cpu".into(),
            EngineDevice::Cuda { ordinal, name, .. } => format!("cuda:{ordinal} ({name})"),
            EngineDevice::Vulkan { name, .. } => format!("vulkan ({name})"),
        }
    }

    /// What, beyond the CPU platform, decides bit-identical results: the GPU model
    /// and its driver (Vulkan) or compute capability (CUDA).
    pub fn identity(&self) -> Option<String> {
        match self {
            EngineDevice::Cpu | EngineDevice::Trees => None,
            EngineDevice::Cuda {
                name, compute_capability, ..
            } => Some(format!(
                "cuda {name} (compute capability {}.{})",
                compute_capability.0, compute_capability.1
            )),
            EngineDevice::Vulkan { name, driver, .. } => Some(format!("vulkan {name} (driver {driver})")),
        }
    }

    /// CUDA on compute capability 8.0+ runs f32 matrix products as TF32 on tensor
    /// cores (CubeCL's matmul converts f32 operands to TF32 whenever the device supports it).
    pub fn uses_tf32(&self) -> bool {
        matches!(self, EngineDevice::Cuda { compute_capability, .. } if compute_capability.0 >= 8)
    }

    /// The precision this device computes with.
    pub fn precision(&self) -> &'static str {
        if self.uses_tf32() { "tf32-matmul" } else { "fp32" }
    }

    /// Losses are read back every step on the CPU (free); on a GPU each read is a
    /// device synchronisation, so they are read at epoch ends and checkpoints.
    pub fn reads_losses_every_step(&self) -> bool {
        !self.is_gpu()
    }

    /// The Burn device (not yet initialised: that happens on first use).
    pub fn burn_device(&self) -> Result<Device> {
        match self {
            EngineDevice::Cpu | EngineDevice::Trees => Ok(Device::flex()),
            EngineDevice::Cuda { ordinal, .. } => gpu::cuda_device(*ordinal),
            EngineDevice::Vulkan { discrete, index, .. } => gpu::vulkan_device(*discrete, *index),
        }
    }
}

fn not_compiled(backend: &str, key: &str) -> TabsmithError {
    TabsmithError::new(
        ErrorCode::DeviceUnsupported,
        Phase::Device,
        format!("device `{backend}` is not available: no {backend} backend is compiled into this build"),
    )
    .field(key)
    .expected(COMPILED_BACKENDS.join(", "))
    .next(match backend {
        "cuda" | "vulkan" | "wgpu" => format!(
            "install a build with GPU support (`cargo install tabsmith-cli --features {}`), or set {key} = \"cpu\"; Tabsmith never falls back silently",
            if backend == "cuda" { "cuda" } else { "vulkan" }
        ),
        _ => format!("set {key} = \"cpu\" (or \"auto-certified\"); Tabsmith never falls back silently"),
    })
}

// ---------------------------------------------------------------------------
// GPU discovery (compiled only with the GPU features)

/// A GPU this build can train on.
#[derive(Debug, Clone)]
pub struct GpuCandidate {
    pub engine: EngineDevice,
    /// `nvidia-gpu`, `amd-gpu`, `intel-gpu`, `apple-gpu` or `gpu`.
    pub kind: &'static str,
    pub memory_mb: Option<u64>,
    pub certification: Certification,
    pub certified_note: String,
}

/// Runs `f`, turning a panic into an error message without printing it: CUDA and
/// Vulkan loaders panic when their libraries are missing, which here only means
/// "no such device".
#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
fn quietly<T>(f: impl FnOnce() -> T) -> std::result::Result<T, String> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous);
    out.map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown failure".into())
    })
}

#[cfg(feature = "cuda")]
mod cuda_probe {
    use cudarc::driver::{CudaContext, result};

    pub struct CudaInfo {
        pub ordinal: usize,
        pub name: String,
        pub compute_capability: (i32, i32),
        pub memory_mb: u64,
    }

    pub fn devices() -> std::result::Result<Vec<CudaInfo>, String> {
        super::quietly(|| -> std::result::Result<Vec<CudaInfo>, String> {
            result::init().map_err(|e| format!("CUDA driver: {e:?}"))?;
            let count = result::device::get_count().map_err(|e| format!("CUDA device count: {e:?}"))?;
            let mut out = Vec::new();
            for ordinal in 0..count.max(0) as usize {
                let ctx = CudaContext::new(ordinal).map_err(|e| format!("CUDA device {ordinal}: {e:?}"))?;
                out.push(CudaInfo {
                    ordinal,
                    name: ctx.name().unwrap_or_else(|_| format!("CUDA device {ordinal}")),
                    compute_capability: ctx.compute_capability().unwrap_or((0, 0)),
                    memory_mb: ctx.total_mem().map(|b| (b / (1024 * 1024)) as u64).unwrap_or(0),
                });
            }
            Ok(out)
        })
        .and_then(|r| r)
        .map_err(|e| {
            if e.contains("load") || e.contains("nvcuda") || e.contains("libcuda") {
                format!("the NVIDIA CUDA driver library could not be loaded ({e})")
            } else {
                e
            }
        })
    }
}

#[cfg(feature = "vulkan")]
mod vulkan_probe {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    pub struct AdapterInfo {
        pub name: String,
        pub vendor: u32,
        pub discrete: bool,
        /// Index within its kind (discrete or integrated), as Burn counts them.
        pub kind_index: usize,
        pub driver: String,
    }

    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    /// Minimal executor: wgpu's adapter enumeration is a future that resolves at
    /// once on native platforms.
    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut fut = std::pin::pin!(fut);
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    /// Discrete GPUs first, then integrated ones, each in enumeration order.
    pub fn adapters() -> std::result::Result<Vec<AdapterInfo>, String> {
        super::quietly(|| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let infos: Vec<wgpu::AdapterInfo> = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
                .iter()
                .map(|a| a.get_info())
                .collect();
            let mut out = Vec::new();
            for discrete in [true, false] {
                let wanted = if discrete {
                    wgpu::DeviceType::DiscreteGpu
                } else {
                    wgpu::DeviceType::IntegratedGpu
                };
                for (kind_index, info) in infos.iter().filter(|i| i.device_type == wanted).enumerate() {
                    out.push(AdapterInfo {
                        name: info.name.clone(),
                        vendor: info.vendor,
                        discrete,
                        kind_index,
                        driver: if info.driver_info.is_empty() {
                            info.driver.clone()
                        } else {
                            info.driver_info.clone()
                        },
                    });
                }
            }
            out
        })
    }
}

mod gpu {
    use super::*;

    #[cfg(feature = "cuda")]
    pub fn cuda_device(ordinal: usize) -> Result<Device> {
        Ok(Device::cuda(ordinal))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_device(_ordinal: usize) -> Result<Device> {
        Err(not_compiled("cuda", "execution"))
    }

    #[cfg(feature = "vulkan")]
    pub fn vulkan_device(discrete: bool, index: usize) -> Result<Device> {
        use burn::tensor::DeviceKind;
        let kind = if discrete {
            DeviceKind::DiscreteGpu(index)
        } else {
            DeviceKind::IntegratedGpu(index)
        };
        Ok(Device::vulkan(kind))
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn vulkan_device(_discrete: bool, _index: usize) -> Result<Device> {
        Err(not_compiled("vulkan", "execution"))
    }
}

fn vendor_kind(vendor: u32) -> &'static str {
    match vendor {
        0x10de => "nvidia-gpu",
        0x1002 => "amd-gpu",
        0x8086 => "intel-gpu",
        0x106b => "apple-gpu",
        _ => "gpu",
    }
}

// ---------------------------------------------------------------------------
// Certification registry

/// GPU paths certified for this recipe, with their evidence (`reports/gpu/`).
#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
const CERTIFIED_DEVICES: &str = include_str!("certified_devices.json");

#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
#[derive(Debug, Clone, Deserialize)]
struct CertifiedEntry {
    backend: String,
    device: String,
    os: String,
    /// Driver version (Vulkan) or `compute capability X.Y` (CUDA).
    driver: String,
    precision: String,
    certified_on: String,
    evidence: String,
}

#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
#[derive(Debug, Deserialize)]
struct CertifiedFile {
    entries: Vec<CertifiedEntry>,
}

#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
fn certified_entries() -> Vec<CertifiedEntry> {
    serde_json::from_str::<CertifiedFile>(CERTIFIED_DEVICES)
        .map(|f| f.entries)
        .unwrap_or_default()
}

/// Certification of a GPU path: exact match of backend, device, OS, precision and
/// driver (Vulkan) or compute capability (CUDA). The same device with another
/// driver is not certified, and the note says which driver was.
#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
fn certify(backend: &str, device: &str, driver: &str, precision: &str) -> (Certification, String) {
    let os = std::env::consts::OS;
    let describe = |d: &str| {
        if backend == "vulkan" {
            format!("driver {d}")
        } else {
            d.to_string()
        }
    };
    let same_device: Vec<CertifiedEntry> = certified_entries()
        .into_iter()
        .filter(|e| e.backend == backend && e.device == device && e.os == os && e.precision == precision)
        .collect();
    match same_device.iter().find(|e| e.driver == driver) {
        Some(e) => (
            Certification::Certified,
            format!("certified on {} with {} ({})", e.certified_on, describe(&e.driver), e.evidence),
        ),
        None => (
            Certification::NotCertified,
            match same_device.first() {
                Some(e) => format!(
                    "not certified: certified with {}, this machine has {}",
                    describe(&e.driver),
                    describe(driver)
                ),
                None => "not certified for this recipe yet (see tools/gpu_certify.py)".to_string(),
            },
        ),
    }
}

/// GPUs this build can train on (Vulkan adapters, discrete before integrated, then
/// CUDA devices), with the reasons a compiled backend found none.
pub fn gpu_candidates() -> (Vec<GpuCandidate>, Vec<String>) {
    #[allow(unused_mut)]
    let mut out: Vec<GpuCandidate> = Vec::new();
    #[allow(unused_mut)]
    let mut problems: Vec<String> = Vec::new();
    #[cfg(feature = "vulkan")]
    match vulkan_probe::adapters() {
        Ok(adapters) => {
            for a in adapters {
                let (certification, certified_note) = certify("vulkan", &a.name, &a.driver, "fp32");
                out.push(GpuCandidate {
                    kind: vendor_kind(a.vendor),
                    memory_mb: None,
                    engine: EngineDevice::Vulkan {
                        discrete: a.discrete,
                        index: a.kind_index,
                        name: a.name,
                        driver: a.driver,
                    },
                    certification,
                    certified_note,
                });
            }
        }
        Err(e) => problems.push(format!("vulkan: {e}")),
    }
    #[cfg(feature = "cuda")]
    match cuda_probe::devices() {
        Ok(devices) => {
            for d in devices {
                let engine = EngineDevice::Cuda {
                    ordinal: d.ordinal,
                    name: d.name.clone(),
                    compute_capability: d.compute_capability,
                };
                let cc = format!("compute capability {}.{}", d.compute_capability.0, d.compute_capability.1);
                let (certification, certified_note) = certify("cuda", &d.name, &cc, engine.precision());
                out.push(GpuCandidate {
                    engine,
                    kind: vendor_kind(0x10de),
                    memory_mb: Some(d.memory_mb),
                    certification,
                    certified_note,
                });
            }
        }
        Err(e) => problems.push(format!("cuda: {e}")),
    }
    (out, problems)
}

/// Read-only device discovery. Nothing is installed; GPUs are initialised only by
/// the backends compiled into this build.
pub fn detect_devices() -> Vec<DetectedDevice> {
    let verified = cfg!(all(target_os = "linux", target_arch = "x86_64"));
    let mut out = vec![DetectedDevice {
        kind: "cpu".into(),
        name: cpu_model().unwrap_or_else(|| format!("{} CPU", std::env::consts::ARCH)),
        certification: if verified {
            Certification::Certified
        } else {
            Certification::NotCertified
        },
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
        backend: Some("cpu".into()),
        index: None,
        driver: None,
    }];
    let (gpus, problems) = gpu_candidates();
    let mut per_backend: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for g in &gpus {
        let backend = if matches!(g.engine, EngineDevice::Cuda { .. }) {
            "cuda"
        } else {
            "vulkan"
        };
        let index = per_backend.entry(backend).or_insert(0);
        let (name, driver) = match &g.engine {
            EngineDevice::Cuda {
                name, compute_capability, ..
            } => (
                name.clone(),
                format!("compute capability {}.{}", compute_capability.0, compute_capability.1),
            ),
            EngineDevice::Vulkan { name, driver, .. } => (name.clone(), driver.clone()),
            EngineDevice::Cpu | EngineDevice::Trees => continue,
        };
        let mut note = g.certified_note.clone();
        if let Some(mb) = g.memory_mb {
            note = format!("{mb} MB; {note}");
        }
        if g.engine.uses_tf32() {
            note.push_str("; matrix products run in TF32 (set execution.precision = \"tf32\" to train here)");
        }
        out.push(DetectedDevice {
            kind: g.kind.into(),
            name,
            certification: g.certification,
            note: Some(note),
            backend: Some(backend.into()),
            index: Some(*index),
            driver: Some(driver),
        });
        *index += 1;
    }
    for p in problems {
        out.push(DetectedDevice {
            kind: "gpu".into(),
            name: "none usable".into(),
            certification: Certification::NotCompiled,
            note: Some(p),
            backend: None,
            index: None,
            driver: None,
        });
    }
    if gpus.is_empty() {
        out.extend(sysfs_gpus());
    }
    out
}

/// GPUs seen through Linux sysfs when no GPU backend is compiled in.
fn sysfs_gpus() -> Vec<DetectedDevice> {
    let mut out = Vec::new();
    let note = if cfg!(any(feature = "vulkan", feature = "cuda")) {
        "not reachable through the GPU backends of this build"
    } else {
        "no GPU backend is compiled into this build (features `vulkan`, `cuda`)"
    };
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
            let vendor_id = u32::from_str_radix(vendor.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            out.push(DetectedDevice {
                kind: vendor_kind(vendor_id).into(),
                name: format!("{name} (pci {key})"),
                certification: Certification::NotCompiled,
                note: Some(note.into()),
                backend: None,
                index: None,
                driver: None,
            });
        }
    }
    if fs::metadata("/proc/driver/nvidia/gpus").is_ok() && !out.iter().any(|d| d.kind == "nvidia-gpu") {
        out.push(DetectedDevice {
            kind: "nvidia-gpu".into(),
            name: "NVIDIA GPU (driver present)".into(),
            certification: Certification::NotCompiled,
            note: Some(note.into()),
            backend: None,
            index: None,
            driver: None,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Planning

/// What a candidate's training will cost, for the memory budget and for choosing
/// between the CPU and a certified GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkEstimate {
    pub peak_mb: f64,
    /// Floating-point operations of one optimizer step (forward and backward).
    pub flops_per_step: f64,
}

impl WorkEstimate {
    /// Matrix products dominate: ~2 FLOPs per multiply-add forward, twice that backward.
    pub fn new(peak_mb: f64, batch: usize, input_dim: usize, hidden: &[usize]) -> Self {
        let mut macs = 0usize;
        let mut d = input_dim;
        for &h in hidden.iter().chain(std::iter::once(&1)) {
            macs += d * h;
            d = h;
        }
        Self {
            peak_mb,
            flops_per_step: 6.0 * batch as f64 * macs as f64,
        }
    }
}

/// Rough time of one optimizer step, in milliseconds. A GPU step costs about a
/// millisecond of launch and transfer overhead before any arithmetic, which a small
/// model on the CPU never reaches. The rates are fitted to Tabsmith's training loop
/// in reports/gpu/2026-10-27-*: a 3.4 GFLOP step (512x512 network, batch 2048) took
/// 32-35 ms on the CPU, 2.9 ms with CUDA and 4.8 ms with Vulkan on an RTX 5090, and
/// 45 ms on an AMD integrated GPU, which is therefore never worth it.
pub fn step_ms_estimate(device: &EngineDevice, flops: f64) -> f64 {
    match device {
        EngineDevice::Cpu | EngineDevice::Trees => 0.03 + flops / 1.0e8,
        EngineDevice::Cuda { .. } => 1.0 + flops / 1.8e9,
        EngineDevice::Vulkan { discrete: true, .. } => 1.5 + flops / 1.0e9,
        EngineDevice::Vulkan { discrete: false, .. } => 1.5 + flops / 8.0e7,
    }
}

/// A device plan together with the engine device it resolved to.
#[derive(Debug, Clone)]
pub struct PlannedDevice {
    pub plan: DevicePlan,
    pub engine: EngineDevice,
}

fn precision_refusal(key: &str, engine: &EngineDevice) -> TabsmithError {
    TabsmithError::new(
        ErrorCode::DeviceUnsupported,
        Phase::Device,
        format!(
            "{} computes f32 matrix products in TF32 (fewer mantissa bits), but execution.precision = \"fp32\"",
            engine.label()
        ),
    )
    .field(key)
    .expected("a device that computes fp32, or execution.precision = \"tf32\"")
    .actual(engine.identity().unwrap_or_default())
    .next(format!(
        "set execution.precision = \"tf32\" to accept TF32 matrix products, or {key} = \"vulkan\" for strict fp32 on this GPU; Tabsmith never lowers precision silently"
    ))
}

/// Resolves a device request into a plan; fails before any training if the device
/// is unavailable, uncertifiable at the requested precision, or over budget.
pub fn plan_device(request: DeviceRequest, key: &str, exec: &ExecutionConfig, work: &WorkEstimate) -> Result<PlannedDevice> {
    match exec.precision {
        Precision::Fp32 | Precision::Tf32 => {}
        other => {
            return Err(TabsmithError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Device,
                format!("precision {other:?} is not certified for tabular-binary@0.1"),
            )
            .field("execution.precision")
            .expected("fp32 (or tf32 on CUDA)")
            .next("set execution.precision = \"fp32\"; half precision is planned as a separate certification"));
        }
    }
    let detected = detect_devices();
    let (gpus, problems) = if matches!(request, DeviceRequest::Cpu) {
        (Vec::new(), Vec::new())
    } else {
        gpu_candidates()
    };
    let mut notes = Vec::new();
    let pick = |backend: &str| -> Result<GpuCandidate> {
        let of_backend: Vec<&GpuCandidate> = gpus
            .iter()
            .filter(|g| {
                matches!(
                    (&g.engine, backend),
                    (EngineDevice::Cuda { .. }, "cuda") | (EngineDevice::Vulkan { .. }, "vulkan")
                )
            })
            .collect();
        of_backend.get(exec.gpu).map(|g| (*g).clone()).ok_or_else(|| {
            let why = problems.iter().find(|p| p.starts_with(backend)).cloned();
            TabsmithError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Device,
                match &why {
                    Some(p) => format!("device `{backend}` is not available: {p}"),
                    None if of_backend.is_empty() => format!("device `{backend}` is not available: no {backend} GPU was found"),
                    None => format!(
                        "execution.gpu = {} but only {} {backend} GPU(s) were found",
                        exec.gpu,
                        of_backend.len()
                    ),
                },
            )
            .field(key)
            .actual(format!("{} {backend} GPU(s)", of_backend.len()))
            .next(format!(
                "check the GPU and its driver with `tabsmith doctor`, or set {key} = \"cpu\"; Tabsmith never falls back silently"
            ))
        })
    };
    let (engine, certification) = match request {
        DeviceRequest::Cpu => (EngineDevice::Cpu, detected[0].certification),
        DeviceRequest::Cuda if !cfg!(feature = "cuda") => return Err(not_compiled("cuda", key)),
        DeviceRequest::Vulkan | DeviceRequest::Wgpu if !cfg!(feature = "vulkan") => {
            return Err(not_compiled(request.as_str(), key));
        }
        DeviceRequest::Rocm | DeviceRequest::Metal => return Err(not_compiled(request.as_str(), key)),
        DeviceRequest::Cuda => {
            let g = pick("cuda")?;
            if g.engine.uses_tf32() && exec.precision == Precision::Fp32 {
                return Err(precision_refusal(key, &g.engine));
            }
            (g.engine, g.certification)
        }
        DeviceRequest::Vulkan | DeviceRequest::Wgpu => {
            let g = pick("vulkan")?;
            (g.engine, g.certification)
        }
        DeviceRequest::AutoCertified => {
            let cpu_ms = step_ms_estimate(&EngineDevice::Cpu, work.flops_per_step);
            let usable: Vec<&GpuCandidate> = gpus
                .iter()
                .filter(|g| g.certification == Certification::Certified && (!g.engine.uses_tf32() || exec.precision == Precision::Tf32))
                .collect();
            let best = usable.iter().min_by(|a, b| {
                step_ms_estimate(&a.engine, work.flops_per_step).total_cmp(&step_ms_estimate(&b.engine, work.flops_per_step))
            });
            match best {
                Some(g) if step_ms_estimate(&g.engine, work.flops_per_step) * 2.0 < cpu_ms => {
                    notes.push(format!(
                        "auto-certified selected {}: about {:.2} ms per step vs {:.2} ms on the CPU",
                        g.engine.label(),
                        step_ms_estimate(&g.engine, work.flops_per_step),
                        cpu_ms
                    ));
                    (g.engine.clone(), g.certification)
                }
                Some(g) => {
                    notes.push(format!(
                        "auto-certified selected the CPU: about {cpu_ms:.2} ms per step vs {:.2} ms on {} (a step this small is cheaper than one GPU round trip)",
                        step_ms_estimate(&g.engine, work.flops_per_step),
                        g.engine.label()
                    ));
                    (EngineDevice::Cpu, detected[0].certification)
                }
                None => {
                    notes.push(if gpus.is_empty() {
                        "auto-certified selected the CPU (no GPU is usable in this build)".to_string()
                    } else {
                        format!(
                            "auto-certified selected the CPU: {} detected GPU(s) are not certified for this recipe{}",
                            gpus.len(),
                            if exec.precision == Precision::Fp32 && gpus.iter().any(|g| g.engine.uses_tf32()) {
                                " at fp32"
                            } else {
                                ""
                            }
                        )
                    });
                    (EngineDevice::Cpu, detected[0].certification)
                }
            }
        }
    };
    if work.peak_mb > exec.max_memory_mb as f64 {
        return Err(TabsmithError::new(
            ErrorCode::ResourceBudgetExceeded,
            Phase::Device,
            format!(
                "estimated peak memory {:.0} MB exceeds execution.max_memory_mb = {}",
                work.peak_mb, exec.max_memory_mb
            ),
        )
        .field("execution.max_memory_mb")
        .expected(format!("<= {} MB", exec.max_memory_mb))
        .actual(format!("{:.0} MB", work.peak_mb))
        .next("raise execution.max_memory_mb if the machine has the memory, or use a smaller dataset / fewer categorical levels"));
    }
    if engine.is_gpu() {
        let ms = warm_up(&engine, key)?;
        notes.push(format!(
            "{} initialised and compiled its first kernels in {ms:.0} ms",
            engine.label()
        ));
        if certification != Certification::Certified {
            notes.push(format!(
                "{} is not certified for this recipe: results are recorded but not claimed reproducible",
                engine.label()
            ));
        }
    }
    let mut copy_path: Vec<String> = vec![
        "file -> Polars DataFrame (host)".into(),
        "DataFrame -> raw value columns (host copy)".into(),
        "raw values -> f32 feature matrix via the shared transform (host copy)".into(),
    ];
    let data_placement = if engine.is_gpu() {
        copy_path.push("feature matrix -> per-batch tensors on the GPU (host-to-device copy)".into());
        copy_path.push("per epoch: losses and weights back to the host for validation through the runtime".into());
        "host, copied to the GPU per batch (planned placement, not a fallback)"
    } else {
        copy_path.push("feature matrix -> per-batch Flex tensors (host copy, no device transfer)".into());
        "cpu (host memory; planned placement, not a fallback)"
    };
    let plan = DevicePlan {
        requested: request.as_str().to_string(),
        compiled_backends: COMPILED_BACKENDS.iter().map(|s| s.to_string()).collect(),
        detected,
        backend: engine.backend().to_string(),
        device: engine.label(),
        precision: engine.precision().to_string(),
        certification: Some(certification),
        fallback: "error".to_string(),
        data_placement: data_placement.to_string(),
        copy_path,
        threads: engine_threads(&engine),
        estimated_peak_mb: work.peak_mb,
        memory_budget_mb: exec.max_memory_mb,
        notes,
    };
    Ok(PlannedDevice { plan, engine })
}

/// The plan of the `gbdt` candidate: Tabsmith's own trees always train on the CPU.
pub fn plan_trees(exec: &ExecutionConfig, work: &WorkEstimate) -> Result<PlannedDevice> {
    let mut planned = plan_device(DeviceRequest::Cpu, "execution.baseline_device", exec, work)?;
    planned.engine = EngineDevice::Trees;
    planned.plan.requested = "cpu (gbdt trains on the CPU)".into();
    planned.plan.backend = EngineDevice::Trees.backend().into();
    planned.plan.copy_path.pop();
    planned
        .plan
        .copy_path
        .push("feature matrix -> per-feature histogram bins (host copy, built once)".into());
    Ok(planned)
}

/// Initialises a GPU and runs one small matrix product on it, so a missing
/// driver, runtime compiler (NVRTC) or Vulkan loader is reported before training.
fn warm_up(engine: &EngineDevice, key: &str) -> Result<f64> {
    let device = engine.burn_device()?;
    let started = std::time::Instant::now();
    let outcome = quietly(|| {
        let a = Tensor::<2>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]), &device);
        let b = Tensor::<2>::from_data(TensorData::new(vec![0.5f32, -1.0, 2.0, 0.25], [2, 2]), &device);
        a.matmul(b).into_data().try_to_vec::<f32>()
    });
    match outcome {
        Ok(Ok(v)) if v.len() == 4 && v.iter().all(|x| x.is_finite()) => Ok(started.elapsed().as_secs_f64() * 1e3),
        Ok(Ok(v)) => Err(device_failed(engine, key, format!("a test product returned {v:?}"))),
        Ok(Err(e)) => Err(device_failed(engine, key, format!("{e:?}"))),
        Err(panic) => Err(device_failed(engine, key, panic)),
    }
}

fn device_failed(engine: &EngineDevice, key: &str, why: String) -> TabsmithError {
    let hint = if matches!(engine, EngineDevice::Cuda { .. }) {
        "CUDA kernels are compiled at run time by NVRTC: install the CUDA toolkit (12.x or 13.x) or its NVRTC library and make sure it is on the library path"
    } else {
        "update the GPU driver (Vulkan 1.2+)"
    };
    TabsmithError::new(
        ErrorCode::DeviceUnsupported,
        Phase::Device,
        format!("{} could not run a test computation: {why}", engine.label()),
    )
    .field(key)
    .next(format!("{hint}; or set {key} = \"cpu\"; Tabsmith never falls back silently"))
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

/// The threads an engine trains with: the networks run on one thread (D5); the trees
/// build their histograms on every core, with sums merged in a fixed order.
fn engine_threads(engine: &EngineDevice) -> usize {
    match engine {
        EngineDevice::Trees => crate::gbdt::threads(),
        _ => 1,
    }
}

pub fn receipt(planned: &PlannedDevice) -> DeviceReceipt {
    let engine = &planned.engine;
    let receipt = DeviceReceipt {
        backend: engine.backend().to_string(),
        device: engine.label(),
        precision: engine.precision().to_string(),
        threads: engine_threads(engine),
        peak_rss_mb: peak_rss_mb(),
        host_device_transfers: if engine.is_gpu() {
            "per batch: features, labels and weights to the GPU; per epoch: losses and weights to the host".to_string()
        } else {
            "none (host backend)".to_string()
        },
        matches_plan: true,
    };
    let plan = &planned.plan;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda(cc: i32) -> EngineDevice {
        EngineDevice::Cuda {
            ordinal: 0,
            name: "Test GPU".into(),
            compute_capability: (cc, 0),
        }
    }

    #[test]
    fn tf32_follows_compute_capability() {
        assert!(cuda(8).uses_tf32() && cuda(12).uses_tf32());
        assert!(!cuda(7).uses_tf32());
        assert_eq!(cuda(12).precision(), "tf32-matmul");
        assert_eq!(EngineDevice::Cpu.precision(), "fp32");
    }

    #[test]
    fn small_steps_stay_on_the_cpu_and_large_ones_move() {
        // The README's model: 35 inputs, layers 32 and 16, batch 64.
        let small = WorkEstimate::new(100.0, 64, 35, &[32, 16]);
        let gpu = EngineDevice::Vulkan {
            discrete: true,
            index: 0,
            name: "Test GPU".into(),
            driver: "1".into(),
        };
        assert!(step_ms_estimate(&EngineDevice::Cpu, small.flops_per_step) < step_ms_estimate(&gpu, small.flops_per_step));
        let large = WorkEstimate::new(100.0, 4096, 64, &[512, 512]);
        assert!(step_ms_estimate(&gpu, large.flops_per_step) * 2.0 < step_ms_estimate(&EngineDevice::Cpu, large.flops_per_step));
    }

    #[test]
    fn certification_needs_the_exact_device_and_driver() {
        let (c, note) = certify("vulkan", "No Such GPU", "1.0", "fp32");
        assert_eq!(c, Certification::NotCertified);
        assert!(note.contains("not certified"), "{note}");
    }

    #[test]
    fn cpu_plans_never_touch_gpus() {
        let exec = ExecutionConfig::default();
        let p = plan_device(
            DeviceRequest::Cpu,
            "execution.baseline_device",
            &exec,
            &WorkEstimate::new(10.0, 64, 10, &[]),
        )
        .unwrap();
        assert_eq!(p.engine, EngineDevice::Cpu);
        assert_eq!(p.plan.backend, "burn-flex");
        assert_eq!(p.plan.precision, "fp32");
    }

    #[test]
    fn half_precision_is_refused() {
        let exec = ExecutionConfig {
            precision: Precision::Fp16,
            ..ExecutionConfig::default()
        };
        let e = plan_device(DeviceRequest::Cpu, "k", &exec, &WorkEstimate::new(10.0, 64, 10, &[])).unwrap_err();
        assert_eq!(e.code, ErrorCode::DeviceUnsupported);
    }
}
