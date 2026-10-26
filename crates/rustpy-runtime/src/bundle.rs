//! Loading and verifying an inference Bundle. A bundle is either loaded completely
//! and consistently, or not at all.

use std::fs;
use std::path::{Component, Path, PathBuf};

use rustpy_contracts::hash::sha256_file;
use rustpy_contracts::manifest::{
    BundleEvaluation, BundleManifest, ExampleExpected, ExampleInputs, compute_bundle_id, version_at_least,
    EVALUATION_FILE, EXAMPLES_EXPECTED_FILE, EXAMPLES_INPUT_FILE, INPUT_SCHEMA_FILE, MANIFEST_FILE,
    MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE, OUTPUT_SCHEMA_FILE, REQUIRED_FILES, TRANSFORM_FILE,
};
use rustpy_contracts::model::{ModelConfig, MODEL_FORMAT};
use rustpy_contracts::schema::{InputSchema, OutputSchema};
use rustpy_contracts::transform::{FittedTransform, TRANSFORM_VERSION};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError, BUNDLE_FORMAT_VERSION, RUSTPY_VERSION};
use serde::de::DeserializeOwned;

use crate::model::DenseModel;

/// Execution device for prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Wgpu,
    Rocm,
    Metal,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Wgpu => "wgpu",
            Device::Rocm => "rocm",
            Device::Metal => "metal",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda),
            "wgpu" | "vulkan" => Some(Device::Wgpu),
            "rocm" => Some(Device::Rocm),
            "metal" => Some(Device::Metal),
            _ => None,
        }
    }
}

/// Options for [`Bundle::open`] / `Predictor::load`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    pub device: Device,
    /// Verify every file's SHA-256 against the manifest (default: true).
    pub verify_hashes: bool,
    /// Run the bundled example records and compare with the expected outputs (default: false).
    pub self_check: bool,
    /// Allow an update whose input schema requires fields the current bundle does not (default: false).
    pub allow_new_required_fields: bool,
}

impl LoadOptions {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            verify_hashes: true,
            self_check: false,
            allow_new_required_fields: false,
        }
    }

    pub fn device(device: Device) -> Self {
        Self {
            device,
            ..Self::cpu()
        }
    }

    pub fn with_self_check(mut self, yes: bool) -> Self {
        self.self_check = yes;
        self
    }

    pub fn with_hash_verification(mut self, yes: bool) -> Self {
        self.verify_hashes = yes;
        self
    }

    pub fn allow_new_required_fields(mut self, yes: bool) -> Self {
        self.allow_new_required_fields = yes;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::cpu()
    }
}

/// A fully verified bundle.
#[derive(Debug, Clone)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: BundleManifest,
    pub input_schema: InputSchema,
    pub output_schema: OutputSchema,
    pub transform: FittedTransform,
    pub model: DenseModel,
    pub evaluation: BundleEvaluation,
    pub examples: ExampleInputs,
    pub expected: ExampleExpected,
}

fn bundle_err(code: ErrorCode, dir: &Path, msg: impl Into<String>) -> RustpyError {
    RustpyError::new(code, Phase::Load, msg).field(dir.display().to_string())
}

fn read_json<T: DeserializeOwned>(dir: &Path, rel: &str) -> Result<T> {
    let path = dir.join(rel);
    let bytes = fs::read(&path).map_err(|e| {
        bundle_err(
            ErrorCode::BundleMissingFile,
            dir,
            format!("cannot read `{rel}`: {e}"),
        )
        .field(rel)
        .next("the bundle is incomplete; rebuild it with `rustpy bundle build` or restore a complete copy")
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        bundle_err(ErrorCode::BundleCorrupt, dir, format!("`{rel}` is not valid: {e}"))
            .field(rel)
            .next("the bundle file is damaged; rebuild the bundle or restore it from a known-good copy")
    })
}

fn safe_relative(p: &str) -> bool {
    let path = Path::new(p);
    !p.is_empty()
        && path
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

impl Bundle {
    /// Opens and verifies a bundle directory.
    pub fn open(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.join(MANIFEST_FILE).is_file() {
            let code = if dir.is_dir() { ErrorCode::BundleMissingFile } else { ErrorCode::BundleNotFound };
            return Err(bundle_err(code, &dir, format!("no bundle manifest at `{}`", dir.join(MANIFEST_FILE).display()))
                .expected("a directory created by `rustpy bundle build`")
                .next("check the path; list bundles with `rustpy bundle list`"));
        }
        let manifest: BundleManifest = read_json(&dir, MANIFEST_FILE)?;
        Self::check_compatibility(&dir, &manifest, opts)?;

        // Every required file must be declared, present, and (optionally) unmodified.
        for required in REQUIRED_FILES {
            if !manifest.files.iter().any(|f| f.path == *required) {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    &dir,
                    format!("manifest does not declare required file `{required}`"),
                )
                .field(*required));
            }
        }
        for entry in &manifest.files {
            if !safe_relative(&entry.path) {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    &dir,
                    format!("manifest lists an unsafe path `{}`", entry.path),
                ));
            }
            let path = dir.join(&entry.path);
            if !path.is_file() {
                return Err(bundle_err(
                    ErrorCode::BundleMissingFile,
                    &dir,
                    format!("bundle file `{}` is missing", entry.path),
                )
                .field(entry.path.clone())
                .next("the bundle is incomplete; rebuild it with `rustpy bundle build` or restore a complete copy"));
            }
            if opts.verify_hashes {
                let (sha, bytes) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Load, &path, e))?;
                if sha != entry.sha256 || bytes != entry.bytes {
                    return Err(bundle_err(
                        ErrorCode::BundleCorrupt,
                        &dir,
                        format!("bundle file `{}` does not match its manifest hash", entry.path),
                    )
                    .field(entry.path.clone())
                    .expected(format!("sha256 {} ({} bytes)", entry.sha256, entry.bytes))
                    .actual(format!("sha256 {sha} ({bytes} bytes)"))
                    .next("the file was modified or damaged; rebuild the bundle or restore it from a known-good copy"));
                }
            }
        }
        let id = compute_bundle_id(&manifest.files);
        if id != manifest.bundle_id {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "bundle_id does not match the file list (manifest edited?)",
            )
            .expected(id)
            .actual(manifest.bundle_id.clone()));
        }

        let input_schema: InputSchema = read_json(&dir, INPUT_SCHEMA_FILE)?;
        let output_schema: OutputSchema = read_json(&dir, OUTPUT_SCHEMA_FILE)?;
        let transform: FittedTransform = read_json(&dir, TRANSFORM_FILE)?;
        let model_config: ModelConfig = read_json(&dir, MODEL_CONFIG_FILE)?;
        let evaluation: BundleEvaluation = read_json(&dir, EVALUATION_FILE)?;
        let examples: ExampleInputs = read_json(&dir, EXAMPLES_INPUT_FILE)?;
        let expected: ExampleExpected = read_json(&dir, EXAMPLES_EXPECTED_FILE)?;

        transform.check_consistency().map_err(|m| {
            bundle_err(ErrorCode::BundleCorrupt, &dir, format!("transform.json is inconsistent: {m}"))
        })?;
        if transform.version != TRANSFORM_VERSION {
            return Err(bundle_err(
                ErrorCode::BundleIncompatible,
                &dir,
                format!("transform version `{}` is not implemented by this runtime", transform.version),
            )
            .expected(TRANSFORM_VERSION));
        }
        if transform.output_dim != model_config.input_dim {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                format!(
                    "transform produces {} features but the model expects {}",
                    transform.output_dim, model_config.input_dim
                ),
            ));
        }
        for f in &transform.features {
            if input_schema.field(f.name()).is_none() {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    &dir,
                    format!("transform uses `{}` which is not in input.schema.json", f.name()),
                ));
            }
        }
        if output_schema.positive_label == output_schema.negative_label {
            return Err(bundle_err(ErrorCode::BundleCorrupt, &dir, "output labels are not distinct"));
        }
        if examples.records.len() != expected.predictions.len() {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "examples/input.json and examples/expected.json have different lengths",
            ));
        }
        let weights_path = dir.join(MODEL_WEIGHTS_FILE);
        let bytes = fs::read(&weights_path).map_err(|e| RustpyError::io(Phase::Load, &weights_path, e))?;
        let model = DenseModel::from_safetensors(model_config, &bytes)?;

        Ok(Self {
            dir,
            manifest,
            input_schema,
            output_schema,
            transform,
            model,
            evaluation,
            examples,
            expected,
        })
    }

    fn check_compatibility(dir: &Path, m: &BundleManifest, opts: &LoadOptions) -> Result<()> {
        let incompatible = |msg: String| {
            bundle_err(ErrorCode::BundleIncompatible, dir, msg)
                .next("use a runtime version that supports this bundle, or rebuild the bundle with this version")
        };
        if m.bundle_format != BUNDLE_FORMAT_VERSION {
            return Err(incompatible(format!("bundle format {} is not supported", m.bundle_format))
                .expected(BUNDLE_FORMAT_VERSION.to_string())
                .actual(m.bundle_format.to_string()));
        }
        if !version_at_least(RUSTPY_VERSION, &m.runtime.min_runtime_version) {
            return Err(incompatible(format!(
                "bundle needs rustpy-runtime >= {}, this is {RUSTPY_VERSION}",
                m.runtime.min_runtime_version
            )));
        }
        if m.runtime.transform_version != TRANSFORM_VERSION {
            return Err(incompatible(format!(
                "transform `{}` is not implemented by this runtime",
                m.runtime.transform_version
            ))
            .expected(TRANSFORM_VERSION));
        }
        if m.runtime.model_format != MODEL_FORMAT {
            return Err(incompatible(format!("model format `{}` is not supported", m.runtime.model_format))
                .expected(MODEL_FORMAT));
        }
        if m.runtime.precision != "fp32" {
            return Err(incompatible(format!("precision `{}` is not supported", m.runtime.precision)));
        }
        if opts.device != Device::Cpu {
            return Err(RustpyError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Load,
                format!(
                    "device `{}` is not available: this runtime executes bundles on CPU only",
                    opts.device.as_str()
                ),
            )
            .field("device")
            .expected("cpu")
            .actual(opts.device.as_str())
            .next("load with LoadOptions::cpu(); GPU inference is not certified in this release"));
        }
        if !m.runtime.devices.iter().any(|d| d == opts.device.as_str()) {
            return Err(RustpyError::new(
                ErrorCode::DeviceUnsupported,
                Phase::Load,
                format!("bundle was not built for device `{}`", opts.device.as_str()),
            )
            .expected(m.runtime.devices.join(", "))
            .next("rebuild the bundle with a profile for this device"));
        }
        Ok(())
    }

    /// Short identifier for display.
    pub fn id(&self) -> &str {
        &self.manifest.bundle_id
    }
}
