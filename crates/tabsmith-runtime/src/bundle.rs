//! Loading and verifying an inference Bundle. A bundle is either loaded completely
//! and consistently, or not at all.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::de::DeserializeOwned;
use tabsmith_contracts::manifest::{
    BundleEvaluation, BundleManifest, CONFORMAL_FILE, Conformal, EVALUATION_FILE, EXAMPLES_EXPECTED_FILE, EXAMPLES_INPUT_FILE,
    ExampleExpected, ExampleInputs, IMPORTANCE_FILE, INPUT_SCHEMA_FILE, Importance, MANIFEST_FILE, MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE,
    OUTPUT_SCHEMA_FILE, REQUIRED_FILES, TRANSFORM_FILE, version_at_least,
};
use tabsmith_contracts::model::{MODEL_FORMAT, ModelConfig, TREES_FORMAT};
use tabsmith_contracts::schema::{InputSchema, OutputSchema};
use tabsmith_contracts::transform::{FittedTransform, TRANSFORM_VERSION};
use tabsmith_contracts::{BUNDLE_FORMAT_VERSION, ErrorCode, Phase, Result, TABSMITH_VERSION, TabsmithError};

use crate::model::Model;

/// Execution device for prediction.
///
/// The runtime executes models on the CPU only: a bundle's small dense network
/// predicts a record in microseconds, less than one GPU round trip. The other
/// variants exist so that a request for them is refused with
/// [`ErrorCode::DeviceUnsupported`] instead of silently running on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// The CPU (the only device this runtime executes on).
    Cpu,
    /// An NVIDIA GPU through CUDA (refused).
    Cuda,
    /// A GPU through wgpu / Vulkan (refused).
    Wgpu,
    /// An AMD GPU through ROCm (refused).
    Rocm,
    /// An Apple GPU through Metal (refused).
    Metal,
}

impl Device {
    /// The device name used in configuration and manifests (`cpu`, `cuda`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Wgpu => "wgpu",
            Device::Rocm => "rocm",
            Device::Metal => "metal",
        }
    }

    /// Parses a device name; `vulkan` is an alias of `wgpu`.
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
    /// Device to predict on; only [`Device::Cpu`] is accepted.
    pub device: Device,
    /// Verify every file's SHA-256 against the manifest (default: true).
    pub verify_hashes: bool,
    /// Run the bundled example records and compare with the expected outputs (default: false).
    pub self_check: bool,
    /// Allow an update that breaks the current input/output contract (new required
    /// fields, changed field kinds or types, stricter policies, a different target
    /// or label mapping). Default: false.
    pub allow_contract_change: bool,
}

impl LoadOptions {
    /// The defaults: CPU, hashes verified, no self-check, no contract changes.
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            verify_hashes: true,
            self_check: false,
            allow_contract_change: false,
        }
    }

    /// The defaults with another device (which loading then refuses unless it is the CPU).
    pub fn device(device: Device) -> Self {
        Self { device, ..Self::cpu() }
    }

    /// Whether loading also predicts the bundled examples and compares them with
    /// the expected outputs.
    pub fn with_self_check(mut self, yes: bool) -> Self {
        self.self_check = yes;
        self
    }

    /// Whether every file's SHA-256 is checked against the manifest.
    pub fn with_hash_verification(mut self, yes: bool) -> Self {
        self.verify_hashes = yes;
        self
    }

    /// Whether [`PredictorHandle::try_update`](crate::PredictorHandle::try_update)
    /// may switch to a bundle with a different input or output contract.
    pub fn allow_contract_change(mut self, yes: bool) -> Self {
        self.allow_contract_change = yes;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::cpu()
    }
}

/// A fully verified bundle: every file listed in the manifest, checked for its
/// hash, format version and internal consistency.
#[derive(Debug, Clone)]
pub struct Bundle {
    /// The bundle directory.
    pub dir: PathBuf,
    /// `manifest.json`: versions, provenance, evaluation status and file hashes.
    pub manifest: BundleManifest,
    /// `input.schema.json`: the fields a record must provide.
    pub input_schema: InputSchema,
    /// `output.schema.json`: target, labels and decision threshold.
    pub output_schema: OutputSchema,
    /// The preprocessing fitted on the training rows.
    pub transform: FittedTransform,
    /// The model: a dense network or a tree ensemble.
    pub model: Model,
    /// `evaluation.json`: the evidence the bundle was built from.
    pub evaluation: BundleEvaluation,
    /// `examples/input.json`: complete example records.
    pub examples: ExampleInputs,
    /// `examples/expected.json`: the predictions the examples must reproduce.
    pub expected: ExampleExpected,
    /// `conformal.json`, when the bundle carries a conformal calibration: every
    /// prediction then has an interval (regression) or a prediction set.
    pub conformal: Option<Conformal>,
    /// `importance.json`, when the bundle carries the permutation importance of its
    /// input fields.
    pub importance: Option<Importance>,
}

fn bundle_err(code: ErrorCode, dir: &Path, msg: impl Into<String>) -> TabsmithError {
    TabsmithError::new(code, Phase::Load, msg).field(dir.display().to_string())
}

fn parse_json<T: DeserializeOwned>(dir: &Path, rel: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| {
        bundle_err(ErrorCode::BundleCorrupt, dir, format!("`{rel}` is not valid: {e}"))
            .field(rel)
            .next(
                "the bundle file is damaged or comes from an incompatible version; rebuild the bundle or restore it from a known-good copy",
            )
    })
}

fn safe_relative(p: &str) -> bool {
    let path = Path::new(p);
    !p.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Every file under `dir` (relative, `/`-separated); symlinks are reported as errors.
fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) -> Result<()> {
    for e in fs::read_dir(dir).map_err(|e| TabsmithError::io(Phase::Load, dir, e))? {
        let e = e.map_err(|e| TabsmithError::io(Phase::Load, dir, e))?;
        let path = e.path();
        let rel = path.strip_prefix(base).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        let ft = fs::symlink_metadata(&path)
            .map_err(|e| TabsmithError::io(Phase::Load, &path, e))?
            .file_type();
        if ft.is_symlink() {
            return Err(
                bundle_err(ErrorCode::BundleCorrupt, base, format!("bundle contains a symbolic link `{rel}`"))
                    .field(rel)
                    .next("bundles contain regular files only; rebuild the bundle"),
            );
        } else if ft.is_dir() {
            walk(&path, base, out)?;
        } else {
            out.push(rel);
        }
    }
    Ok(())
}

/// Where the files of a bundle come from: a directory, or memory.
trait Source {
    /// Names the bundle in errors.
    fn dir(&self) -> &Path;
    /// The manifest's bytes, or the error for a bundle without one.
    fn manifest(&self) -> Result<Cow<'_, [u8]>>;
    /// Every file present, relative and `/`-separated.
    fn list(&self) -> Result<Vec<String>>;
    /// A file's bytes, or `None` when it is absent or unreadable.
    fn read(&self, rel: &str) -> Option<Cow<'_, [u8]>>;
}

struct DirSource(PathBuf);

impl Source for DirSource {
    fn dir(&self) -> &Path {
        &self.0
    }

    fn manifest(&self) -> Result<Cow<'_, [u8]>> {
        let dir = &self.0;
        if !dir.join(MANIFEST_FILE).is_file() {
            let code = if dir.is_dir() {
                ErrorCode::BundleMissingFile
            } else {
                ErrorCode::BundleNotFound
            };
            return Err(
                bundle_err(code, dir, format!("no bundle manifest at `{}`", dir.join(MANIFEST_FILE).display()))
                    .expected("a directory created by `tabsmith bundle build`")
                    .next("check the path; list bundles with `tabsmith bundle list`"),
            );
        }
        let bytes = fs::read(dir.join(MANIFEST_FILE)).map_err(|e| TabsmithError::io(Phase::Load, dir, e))?;
        Ok(Cow::Owned(bytes))
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut present = Vec::new();
        walk(&self.0, &self.0, &mut present)?;
        Ok(present)
    }

    fn read(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        fs::read(self.0.join(rel)).ok().map(Cow::Owned)
    }
}

struct MemorySource {
    label: PathBuf,
    files: BTreeMap<String, Vec<u8>>,
}

impl Source for MemorySource {
    fn dir(&self) -> &Path {
        &self.label
    }

    fn manifest(&self) -> Result<Cow<'_, [u8]>> {
        self.read(MANIFEST_FILE).ok_or_else(|| {
            bundle_err(
                ErrorCode::BundleMissingFile,
                &self.label,
                format!("no `{MANIFEST_FILE}` among the bundle's files"),
            )
            .field(MANIFEST_FILE)
            .expected("the files of a directory created by `tabsmith bundle build`")
        })
    }

    fn list(&self) -> Result<Vec<String>> {
        Ok(self.files.keys().cloned().collect())
    }

    fn read(&self, rel: &str) -> Option<Cow<'_, [u8]>> {
        self.files.get(rel).map(|b| Cow::Borrowed(b.as_slice()))
    }
}

impl Bundle {
    /// Opens and verifies a bundle directory. Each file is read once; the bytes
    /// that are hashed are the bytes that are parsed.
    pub fn open(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        Self::verify(&DirSource(dir.as_ref().to_path_buf()), opts)
    }

    /// Verifies a bundle whose files are already in memory: fetched over the
    /// network, embedded in a binary, or loaded by a WebAssembly page.
    ///
    /// `files` maps each relative, `/`-separated path (`manifest.json`,
    /// `examples/input.json`, ...) to its bytes, and must hold exactly the manifest
    /// and the files it lists. The checks are those of [`Bundle::open`]; `label`
    /// only names the bundle in errors and in [`Bundle::dir`](Bundle#structfield.dir).
    pub fn from_memory(label: impl Into<PathBuf>, files: BTreeMap<String, Vec<u8>>, opts: &LoadOptions) -> Result<Self> {
        Self::verify(
            &MemorySource {
                label: label.into(),
                files,
            },
            opts,
        )
    }

    fn verify(source: &impl Source, opts: &LoadOptions) -> Result<Self> {
        let dir = source.dir().to_path_buf();
        let manifest_bytes = source.manifest()?;
        let manifest: BundleManifest = parse_json(&dir, MANIFEST_FILE, &manifest_bytes)?;
        Self::check_compatibility(&dir, &manifest, opts)?;

        // Every required file must be declared; every declared path safe; nothing else present.
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
        for f in &source.list()? {
            if f != MANIFEST_FILE && !manifest.files.iter().any(|e| &e.path == f) {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    &dir,
                    format!("bundle contains a file that is not in its manifest: `{f}`"),
                )
                .field(f.clone())
                .next("a bundle holds exactly the files its manifest lists; remove the extra file or rebuild the bundle"));
            }
        }
        let mut contents: BTreeMap<String, Cow<'_, [u8]>> = BTreeMap::new();
        for entry in &manifest.files {
            // Checked before anything is read: a listed `../x` must never reach the file system.
            if !safe_relative(&entry.path) {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    &dir,
                    format!("manifest lists an unsafe path `{}`", entry.path),
                ));
            }
            let bytes = match source.read(&entry.path) {
                Some(b) => b,
                None => {
                    return Err(bundle_err(
                        ErrorCode::BundleMissingFile,
                        &dir,
                        format!("bundle file `{}` is missing", entry.path),
                    )
                    .field(entry.path.clone())
                    .next("the bundle is incomplete; rebuild it with `tabsmith bundle build` or restore a complete copy"));
                }
            };
            if opts.verify_hashes {
                let sha = tabsmith_contracts::hash::sha256_hex(&bytes);
                if sha != entry.sha256 || bytes.len() as u64 != entry.bytes {
                    return Err(bundle_err(
                        ErrorCode::BundleCorrupt,
                        &dir,
                        format!("bundle file `{}` does not match its manifest hash", entry.path),
                    )
                    .field(entry.path.clone())
                    .expected(format!("sha256 {} ({} bytes)", entry.sha256, entry.bytes))
                    .actual(format!("sha256 {sha} ({} bytes)", bytes.len()))
                    .next("the file was modified or damaged; rebuild the bundle or restore it from a known-good copy"));
                }
            }
            contents.insert(entry.path.clone(), bytes);
        }
        let id = manifest.expected_id();
        if id != manifest.bundle_id {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "bundle_id does not match the files and manifest claims (manifest edited?)",
            )
            .expected(id)
            .actual(manifest.bundle_id.clone())
            .next("rebuild the bundle; manifests cannot be edited by hand"));
        }
        let get = |rel: &str| contents.get(rel).map(|b| &b[..]).unwrap_or_default();
        let input_schema: InputSchema = parse_json(&dir, INPUT_SCHEMA_FILE, get(INPUT_SCHEMA_FILE))?;
        let output_schema: OutputSchema = parse_json(&dir, OUTPUT_SCHEMA_FILE, get(OUTPUT_SCHEMA_FILE))?;
        let transform: FittedTransform = parse_json(&dir, TRANSFORM_FILE, get(TRANSFORM_FILE))?;
        let model_config: ModelConfig = parse_json(&dir, MODEL_CONFIG_FILE, get(MODEL_CONFIG_FILE))?;
        let evaluation: BundleEvaluation = parse_json(&dir, EVALUATION_FILE, get(EVALUATION_FILE))?;
        let examples: ExampleInputs = parse_json(&dir, EXAMPLES_INPUT_FILE, get(EXAMPLES_INPUT_FILE))?;
        let expected: ExampleExpected = parse_json(&dir, EXAMPLES_EXPECTED_FILE, get(EXAMPLES_EXPECTED_FILE))?;
        let conformal: Option<Conformal> = match contents.get(CONFORMAL_FILE) {
            Some(b) => Some(parse_json(&dir, CONFORMAL_FILE, b)?),
            None => None,
        };
        if let Some(c) = &conformal
            && !(c.alpha > 0.0 && c.alpha < 1.0 && c.quantile >= 0.0 && c.quantile.is_finite())
        {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "conformal.json holds an impossible alpha or quantile",
            )
            .field(CONFORMAL_FILE));
        }
        let importance: Option<Importance> = match contents.get(IMPORTANCE_FILE) {
            Some(b) => Some(parse_json(&dir, IMPORTANCE_FILE, b)?),
            None => None,
        };

        // Claims in the manifest must agree with the evidence they summarise.
        let test_id = evaluation.test.as_ref().map(|t| t.evaluation_id.clone());
        if manifest.evaluation_status != evaluation.status || manifest.provenance.evaluation_id != test_id {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "the manifest's evaluation claims do not match evaluation.json",
            )
            .expected(format!("{} / {:?}", evaluation.status, test_id))
            .actual(format!("{} / {:?}", manifest.evaluation_status, manifest.provenance.evaluation_id)));
        }
        transform
            .check_consistency()
            .map_err(|m| bundle_err(ErrorCode::BundleCorrupt, &dir, format!("transform.json is inconsistent: {m}")))?;
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
        output_schema
            .check_consistency()
            .map_err(|m| bundle_err(ErrorCode::BundleCorrupt, &dir, format!("output.schema.json is inconsistent: {m}")))?;
        if output_schema.n_outputs() != model_config.n_outputs()
            || model_config.output != tabsmith_contracts::model::output_kind(output_schema.task)
        {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                format!(
                    "the model's `{}` output with {} values does not fit the {} task's output schema",
                    model_config.output,
                    model_config.n_outputs(),
                    output_schema.task
                ),
            ));
        }
        if examples.records.len() != expected.predictions.len() {
            return Err(bundle_err(
                ErrorCode::BundleCorrupt,
                &dir,
                "examples/input.json and examples/expected.json have different lengths",
            ));
        }
        let model = Model::from_safetensors(model_config, get(MODEL_WEIGHTS_FILE))?;

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
            conformal,
            importance,
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
        match version_at_least(TABSMITH_VERSION, &m.runtime.min_runtime_version) {
            Some(true) => {}
            Some(false) => {
                return Err(incompatible(format!(
                    "bundle needs tabsmith-runtime >= {}, this is {TABSMITH_VERSION}",
                    m.runtime.min_runtime_version
                )));
            }
            None => {
                return Err(bundle_err(
                    ErrorCode::BundleCorrupt,
                    dir,
                    format!(
                        "min_runtime_version {:?} is not a major.minor.patch version",
                        m.runtime.min_runtime_version
                    ),
                ));
            }
        }
        if m.runtime.transform_version != TRANSFORM_VERSION {
            return Err(incompatible(format!(
                "transform `{}` is not implemented by this runtime",
                m.runtime.transform_version
            ))
            .expected(TRANSFORM_VERSION));
        }
        if m.runtime.model_format != MODEL_FORMAT && m.runtime.model_format != TREES_FORMAT {
            return Err(incompatible(format!("model format `{}` is not supported", m.runtime.model_format))
                .expected(format!("{MODEL_FORMAT} or {TREES_FORMAT}")));
        }
        if m.runtime.precision != "fp32" {
            return Err(incompatible(format!("precision `{}` is not supported", m.runtime.precision)));
        }
        if opts.device != Device::Cpu {
            return Err(TabsmithError::new(
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
            return Err(TabsmithError::new(
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
