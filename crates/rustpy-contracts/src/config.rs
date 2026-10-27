//! Project configuration (`rustpy.toml`) and its layered resolution.
//!
//! Precedence, lowest to highest:
//! recipe defaults < project file < selected profile < explicit overrides (`--set`, SDK).
//!
//! Every layer is validated on its own (unknown keys and wrong types are errors with
//! the offending key), then merged; the effective value of every leaf and the layer
//! it came from are recorded so a Run can show exactly which settings it used.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, Phase, Result, RustpyError};
use crate::hash::canonical_hash;
use crate::{PROJECT_SCHEMA_VERSION, RECIPE_ID, RECIPE_NAME};

// ---------------------------------------------------------------------------
// Typed configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectConfig {
    pub schema_version: u32,
    pub recipe: String,
    pub data: DataConfig,
    pub preprocess: PreprocessConfig,
    pub selection: SelectionConfig,
    pub train: TrainConfig,
    pub execution: ExecutionConfig,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            recipe: RECIPE_ID.to_string(),
            data: DataConfig::default(),
            preprocess: PreprocessConfig::default(),
            selection: SelectionConfig::default(),
            train: TrainConfig::default(),
            execution: ExecutionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DataFormat {
    Auto,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SplitStrategy {
    /// Random split stratified by the target label. Only valid when rows are independent.
    Stratified,
    /// All rows of one entity (`group_column`) land in the same split.
    Group,
    /// Chronological split. Declared so it can be refused explicitly; not implemented in 0.1.
    Time,
}

impl fmt::Display for SplitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SplitStrategy::Stratified => "stratified",
            SplitStrategy::Group => "group",
            SplitStrategy::Time => "time",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SnapshotMode {
    /// Copy the source file into the project so later Runs can be replayed exactly.
    Copy,
    /// Keep only a reference + content hash; any change of the source invalidates the snapshot.
    Reference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ColumnKindOverride {
    Numeric,
    Categorical,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DataConfig {
    /// Path to the CSV or Parquet file, relative to the project directory.
    pub source: Option<String>,
    pub format: DataFormat,
    /// Name of the binary target column.
    pub target: Option<String>,
    /// Which target value is the positive class (compared as text, e.g. "1", "true", "defect").
    pub positive_label: Option<String>,
    /// Optional stable row identifier; never used as a feature.
    pub id_column: Option<String>,
    /// Entity column for grouped splitting; never used as a feature.
    pub group_column: Option<String>,
    /// Declaring a time column means a chronological split is required (refused in 0.1).
    pub time_column: Option<String>,
    pub exclude: Vec<String>,
    /// Explicit feature kinds; overrides inference.
    pub columns: BTreeMap<String, ColumnKindOverride>,
    pub split: SplitStrategy,
    pub seed: u64,
    pub train_fraction: f64,
    pub valid_fraction: f64,
    pub test_fraction: f64,
    pub snapshot: SnapshotMode,
    pub csv_delimiter: String,
    /// CSV tokens read as missing values.
    pub null_values: Vec<String>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            source: None,
            format: DataFormat::Auto,
            target: None,
            positive_label: None,
            id_column: None,
            group_column: None,
            time_column: None,
            exclude: Vec::new(),
            columns: BTreeMap::new(),
            split: SplitStrategy::Stratified,
            seed: 42,
            train_fraction: 0.7,
            valid_fraction: 0.15,
            test_fraction: 0.15,
            snapshot: SnapshotMode::Copy,
            csv_delimiter: ",".to_string(),
            null_values: vec![String::new(), "NA".to_string(), "null".to_string()],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImputeStrategy {
    Mean,
    Median,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScaleStrategy {
    Standard,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NanPolicy {
    /// NaN is treated exactly like a missing value (imputed / missing category).
    Missing,
    /// NaN is rejected with `E_INVALID_VALUE`.
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UnknownCategoryPolicy {
    /// Unknown categories encode as an all-zero one-hot block (like sklearn `handle_unknown="ignore"`).
    Ignore,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExtraColumnPolicy {
    Ignore,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ZeroVariancePolicy {
    /// Drop the feature and record why; the input field becomes optional.
    Drop,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PreprocessConfig {
    pub numeric_impute: ImputeStrategy,
    pub numeric_scale: ScaleStrategy,
    pub nan: NanPolicy,
    /// Categorical columns with more distinct training values than this are rejected.
    pub max_categories: usize,
    pub unknown_category: UnknownCategoryPolicy,
    pub extra_columns: ExtraColumnPolicy,
    pub zero_variance: ZeroVariancePolicy,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            numeric_impute: ImputeStrategy::Mean,
            numeric_scale: ScaleStrategy::Standard,
            nan: NanPolicy::Missing,
            max_categories: 50,
            unknown_category: UnknownCategoryPolicy::Ignore,
            extra_columns: ExtraColumnPolicy::Ignore,
            zero_variance: ZeroVariancePolicy::Drop,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Candidate {
    Linear,
    SmallMlp,
}

impl Candidate {
    pub fn as_str(self) -> &'static str {
        match self {
            Candidate::Linear => "linear",
            Candidate::SmallMlp => "small-mlp",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "linear" => Some(Candidate::Linear),
            "small-mlp" => Some(Candidate::SmallMlp),
            _ => None,
        }
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMetric {
    BalancedAccuracy,
    F1,
    RocAuc,
    Accuracy,
}

impl SelectionMetric {
    pub fn as_str(self) -> &'static str {
        match self {
            SelectionMetric::BalancedAccuracy => "balanced_accuracy",
            SelectionMetric::F1 => "f1",
            SelectionMetric::RocAuc => "roc_auc",
            SelectionMetric::Accuracy => "accuracy",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SelectionConfig {
    pub candidates: Vec<Candidate>,
    /// Validation metric used to rank candidates (higher is better).
    pub metric: SelectionMetric,
    /// Decision threshold on P(positive); fixed before evaluation.
    pub threshold: f64,
    /// Quality constraint on the validation metric (0 disables it).
    pub min_valid_metric: f64,
    /// Resource constraint: p95 single-row latency in microseconds on this machine (0 disables it).
    pub max_p95_latency_us: f64,
    /// Resource constraint: model weight size in KiB (0 disables it).
    pub max_model_kib: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            candidates: vec![Candidate::Linear, Candidate::SmallMlp],
            metric: SelectionMetric::BalancedAccuracy,
            threshold: 0.5,
            min_valid_metric: 0.0,
            max_p95_latency_us: 0.0,
            max_model_kib: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClassWeight {
    /// Positive and negative examples weighted inversely to their training frequency.
    Balanced,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CandidateTrainConfig {
    /// Hidden layer widths (must be empty for `linear`).
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// L2 penalty added to the loss: `l2 * sum(weights^2) / 2` (biases excluded).
    pub l2: f64,
    pub class_weight: ClassWeight,
    /// Stop when validation loss has not improved for this many epochs (0 disables).
    pub early_stopping_patience: usize,
    pub checkpoint_every_epochs: usize,
    /// Minimum wall-clock seconds between epoch checkpoints (0 = every eligible epoch).
    /// A checkpoint is always written when training stops or is cancelled. Placement
    /// does not change results, only how much work a crash can lose.
    pub checkpoint_min_interval_seconds: f64,
}

impl Default for CandidateTrainConfig {
    fn default() -> Self {
        Self::small_mlp()
    }
}

impl CandidateTrainConfig {
    pub fn linear() -> Self {
        Self {
            hidden: Vec::new(),
            // Chosen against the scikit-learn lbfgs optimum of the same objective
            // (see reports/evidence/reference): objective within ~0.2%.
            epochs: 60,
            batch_size: 256,
            learning_rate: 0.02,
            l2: 1e-4,
            class_weight: ClassWeight::Balanced,
            early_stopping_patience: 10,
            checkpoint_every_epochs: 1,
            checkpoint_min_interval_seconds: 2.0,
        }
    }

    pub fn small_mlp() -> Self {
        Self {
            hidden: vec![32, 16],
            epochs: 60,
            batch_size: 128,
            learning_rate: 0.005,
            l2: 1e-4,
            class_weight: ClassWeight::Balanced,
            early_stopping_patience: 8,
            checkpoint_every_epochs: 1,
            checkpoint_min_interval_seconds: 2.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    /// Seed for weight initialisation and minibatch order.
    pub seed: u64,
    pub linear: CandidateTrainConfig,
    #[serde(rename = "small-mlp")]
    pub small_mlp: CandidateTrainConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            linear: CandidateTrainConfig::linear(),
            small_mlp: CandidateTrainConfig::small_mlp(),
        }
    }
}

impl TrainConfig {
    pub fn for_candidate(&self, c: Candidate) -> &CandidateTrainConfig {
        match c {
            Candidate::Linear => &self.linear,
            Candidate::SmallMlp => &self.small_mlp,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeviceRequest {
    Cpu,
    /// Pick the best device that is certified for this recipe in this build.
    AutoCertified,
    Cuda,
    Wgpu,
    Vulkan,
    Rocm,
    Metal,
}

impl DeviceRequest {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceRequest::Cpu => "cpu",
            DeviceRequest::AutoCertified => "auto-certified",
            DeviceRequest::Cuda => "cuda",
            DeviceRequest::Wgpu => "wgpu",
            DeviceRequest::Vulkan => "vulkan",
            DeviceRequest::Rocm => "rocm",
            DeviceRequest::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FallbackPolicy {
    /// Never fall back silently: an unavailable device is an error.
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionConfig {
    pub baseline_device: DeviceRequest,
    pub candidate_device: DeviceRequest,
    pub fallback: FallbackPolicy,
    pub precision: Precision,
    /// Refuse plans whose estimated peak memory exceeds this budget.
    pub max_memory_mb: u64,
    /// Wall-clock training budget per candidate in seconds (0 = unlimited).
    pub max_train_seconds: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            baseline_device: DeviceRequest::Cpu,
            candidate_device: DeviceRequest::AutoCertified,
            fallback: FallbackPolicy::Error,
            precision: Precision::Fp32,
            max_memory_mb: 4096,
            max_train_seconds: 0,
        }
    }
}

/// Top-level document of `rustpy.toml`: the config plus named profiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ProjectFileDoc {
    schema_version: u32,
    recipe: String,
    data: DataConfig,
    preprocess: PreprocessConfig,
    selection: SelectionConfig,
    train: TrainConfig,
    execution: ExecutionConfig,
    profiles: BTreeMap<String, toml::Table>,
}

impl Default for ProjectFileDoc {
    fn default() -> Self {
        let c = ProjectConfig::default();
        Self {
            schema_version: c.schema_version,
            recipe: c.recipe,
            data: c.data,
            preprocess: c.preprocess,
            selection: c.selection,
            train: c.train,
            execution: c.execution,
            profiles: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Layers and resolution
// ---------------------------------------------------------------------------

/// Where an effective configuration value came from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "name")]
pub enum ValueSource {
    Default,
    Project,
    Profile(String),
    Override,
}

impl fmt::Display for ValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueSource::Default => f.write_str("recipe default"),
            ValueSource::Project => f.write_str("rustpy.toml"),
            ValueSource::Profile(p) => write!(f, "profile `{p}`"),
            ValueSource::Override => f.write_str("override"),
        }
    }
}

/// One validated configuration layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigLayer {
    pub source: ValueSource,
    pub table: toml::Table,
}

/// A parsed `rustpy.toml`: the project layer and its (validated) profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectFile {
    pub project: ConfigLayer,
    pub profiles: BTreeMap<String, toml::Table>,
}

impl ProjectFile {
    /// Parses and validates the text of a `rustpy.toml`.
    pub fn parse(text: &str, origin: &str) -> Result<Self> {
        if let Err(e) = toml::from_str::<ProjectFileDoc>(text) {
            return Err(toml_error(e.message(), origin, Some(text), e.span()));
        }
        let mut table: toml::Table = toml::from_str(text).map_err(|e| toml_error(e.message(), origin, Some(text), e.span()))?;
        let mut profiles = BTreeMap::new();
        if let Some(p) = table.remove("profiles") {
            let toml::Value::Table(p) = p else {
                return Err(config_invalid("`profiles` must be a table", "profiles"));
            };
            for (name, value) in p {
                let toml::Value::Table(t) = value else {
                    return Err(config_invalid(
                        format!("profile `{name}` must be a table"),
                        format!("profiles.{name}"),
                    ));
                };
                validate_layer(&t, &format!("profile `{name}` in {origin}"))?;
                profiles.insert(name, t);
            }
        }
        Ok(Self {
            project: ConfigLayer {
                source: ValueSource::Project,
                table,
            },
            profiles,
        })
    }

    pub fn profile_layer(&self, name: &str) -> Result<ConfigLayer> {
        match self.profiles.get(name) {
            Some(t) => Ok(ConfigLayer {
                source: ValueSource::Profile(name.to_string()),
                table: t.clone(),
            }),
            None => {
                let known: Vec<_> = self.profiles.keys().cloned().collect();
                Err(RustpyError::new(
                    ErrorCode::ConfigInvalid,
                    Phase::Config,
                    format!("profile `{name}` is not defined in rustpy.toml"),
                )
                .field(format!("profiles.{name}"))
                .expected(if known.is_empty() {
                    "no profiles are defined".to_string()
                } else {
                    format!("one of: {}", known.join(", "))
                })
                .next(format!("add a [profiles.{name}] table to rustpy.toml or drop --profile")))
            }
        }
    }
}

fn config_invalid(msg: impl Into<String>, field: impl Into<String>) -> RustpyError {
    RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, msg).field(field)
}

fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let before = &text[..offset.min(text.len())];
    let line = before.matches('\n').count() + 1;
    let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
    (line, col)
}

fn toml_error(message: &str, origin: &str, text: Option<&str>, span: Option<std::ops::Range<usize>>) -> RustpyError {
    let location = match (text, span) {
        (Some(t), Some(s)) => {
            let (l, c) = line_col(t, s.start);
            format!("{origin}:{l}:{c}")
        }
        _ => origin.to_string(),
    };
    let unknown = message.starts_with("unknown field") || message.starts_with("unknown variant");
    let code = if message.starts_with("unknown field") {
        ErrorCode::ConfigUnknownField
    } else {
        ErrorCode::ConfigInvalid
    };
    let mut err = RustpyError::new(code, Phase::Config, format!("{location}: {message}")).field(location);
    if unknown {
        let hint = closest_expected(message)
            .map(|k| format!("did you mean `{k}`? "))
            .unwrap_or_default();
        err = err.next(format!("{hint}fix the spelling; docs/configuration.md lists every setting"));
    } else {
        err = err.next("fix the value; see docs/configuration.md for the allowed values and types");
    }
    err
}

/// For serde's "unknown field `x`, expected one of `a`, `b`" messages: the
/// expected name closest to `x` (edit distance at most a third of its length).
fn closest_expected(message: &str) -> Option<String> {
    let mut quoted = message.split('`').skip(1).step_by(2);
    let given = quoted.next()?;
    let (best, d) = quoted.map(|c| (c, edit_distance(given, c))).min_by_key(|(_, d)| *d)?;
    (d > 0 && d <= given.chars().count().div_ceil(3).max(1)).then(|| best.to_string())
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != *cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Validates a partial layer: unknown keys and wrong types are rejected.
pub fn validate_layer(table: &toml::Table, origin: &str) -> Result<()> {
    let check = |t: &toml::Table| toml::Value::Table(t.clone()).try_into::<ProjectConfig>().map(|_| ());
    match check(table) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Find the offending key: validate each leaf on its own.
            let mut leaves = Vec::new();
            collect_leaves(table, "", &mut leaves);
            for (path, value) in leaves {
                if let Err(le) = check(&single_leaf(&path, value)) {
                    let mut err = toml_error(le.message(), &format!("{origin}, key `{path}`"), None, None);
                    err.field = Some(path);
                    return Err(err);
                }
            }
            Err(toml_error(e.message(), origin, None, None))
        }
    }
}

fn single_leaf(path: &str, value: toml::Value) -> toml::Table {
    let mut leaf = value;
    for part in path.split('.').rev() {
        let mut t = toml::Table::new();
        t.insert(part.to_string(), leaf);
        leaf = toml::Value::Table(t);
    }
    match leaf {
        toml::Value::Table(t) => t,
        _ => unreachable!("path has at least one part"),
    }
}

/// Parses `a.b.c=value` into a single-leaf layer table. The value is read as a TOML
/// value when possible (`5`, `0.1`, `true`, `["linear"]`, `"text"`), otherwise as a string.
pub fn parse_override(spec: &str) -> Result<toml::Table> {
    let Some((key, raw)) = spec.split_once('=') else {
        return Err(
            config_invalid(format!("override `{spec}` must look like key.path=value"), spec.to_string())
                .next("example: --set train.small-mlp.epochs=20"),
        );
    };
    let key = key.trim();
    let raw = raw.trim();
    if key.is_empty() || key.split('.').any(str::is_empty) {
        return Err(config_invalid(format!("override key `{key}` is empty or malformed"), key));
    }
    let value = match toml::from_str::<toml::Table>(&format!("v = {raw}")) {
        Ok(mut t) => t.remove("v").unwrap_or_else(|| toml::Value::String(raw.to_string())),
        Err(_) => toml::Value::String(raw.to_string()),
    };
    let is_string = value.is_str();
    let table = single_leaf(key, value);
    match validate_layer(&table, &format!("--set {spec}")) {
        Ok(()) => Ok(table),
        Err(e) => {
            // `--set data.positive_label=1`: a string key given a bare number/bool.
            if !is_string && e.code == ErrorCode::ConfigInvalid {
                let as_text = single_leaf(key, toml::Value::String(raw.to_string()));
                if validate_layer(&as_text, "").is_ok() {
                    return Ok(as_text);
                }
            }
            let mut e = e;
            e.field = Some(key.to_string());
            Err(e)
        }
    }
}

/// Effective configuration with per-leaf provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedConfig {
    pub config: ProjectConfig,
    /// Dotted key path -> layer that supplied the effective value.
    pub sources: BTreeMap<String, ValueSource>,
    /// Canonical hash of `config`; equal configs from CLI and SDK give equal hashes.
    pub config_hash: String,
}

impl ResolvedConfig {
    /// Effective `(path, value, source)` triples, sorted by path.
    pub fn effective_values(&self) -> Vec<(String, String, ValueSource)> {
        let table = toml::Table::try_from(&self.config).expect("config serializes");
        let mut leaves = Vec::new();
        collect_leaves(&table, "", &mut leaves);
        let mut out: Vec<(String, String, ValueSource)> = leaves
            .into_iter()
            .map(|(path, value)| {
                let source = self.sources.get(&path).cloned().unwrap_or(ValueSource::Default);
                (path, value.to_string(), source)
            })
            .collect();
        // Optional settings that are not set do not appear in TOML; list them too.
        let data = serde_json::to_value(&self.config.data).expect("config serializes");
        if let Some(obj) = data.as_object() {
            for (k, v) in obj {
                if v.is_null() {
                    out.push((format!("data.{k}"), "(unset)".into(), ValueSource::Default));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

fn collect_leaves(table: &toml::Table, prefix: &str, out: &mut Vec<(String, toml::Value)>) {
    for (k, v) in table {
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match v {
            toml::Value::Table(t) if !t.is_empty() => collect_leaves(t, &path, out),
            other => out.push((path, other.clone())),
        }
    }
}

fn merge_into(
    base: &mut toml::Table,
    layer: &toml::Table,
    prefix: &str,
    source: &ValueSource,
    sources: &mut BTreeMap<String, ValueSource>,
) {
    for (k, v) in layer {
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(bt)), toml::Value::Table(lt)) => {
                merge_into(bt, lt, &path, source, sources);
            }
            (_, toml::Value::Table(lt)) => {
                let mut fresh = toml::Table::new();
                merge_into(&mut fresh, lt, &path, source, sources);
                base.insert(k.clone(), toml::Value::Table(fresh));
            }
            (_, value) => {
                base.insert(k.clone(), value.clone());
                sources.insert(path, source.clone());
            }
        }
    }
}

/// Recipe defaults as a table (the lowest layer).
pub fn defaults_table() -> toml::Table {
    toml::Table::try_from(ProjectConfig::default()).expect("defaults serialize")
}

/// Merges layers over the recipe defaults and validates the result.
pub fn resolve(layers: &[ConfigLayer]) -> Result<ResolvedConfig> {
    let mut merged = defaults_table();
    let mut sources = BTreeMap::new();
    let mut leaves = Vec::new();
    collect_leaves(&merged, "", &mut leaves);
    for (path, _) in leaves {
        sources.insert(path, ValueSource::Default);
    }
    for layer in layers {
        validate_layer(&layer.table, &layer.source.to_string())?;
        merge_into(&mut merged, &layer.table, "", &layer.source, &mut sources);
    }
    let config: ProjectConfig = toml::Value::Table(merged)
        .try_into()
        .map_err(|e: toml::de::Error| toml_error(e.message(), "merged configuration", None, None))?;
    config.validate().map_err(|mut e| {
        // Say which layer supplied the offending value.
        if let Some(src) = e.field.as_ref().and_then(|f| sources.get(f.as_str())) {
            e.message = format!("{} (value from {src})", e.message);
        }
        if e.next_action.is_none() {
            e.next_action = Some(format!(
                "fix the value in the layer named above; docs/configuration.md lists every setting{}",
                e.expected.as_ref().map(|x| format!(" (expected {x})")).unwrap_or_default()
            ));
        }
        e
    })?;
    let config_hash = canonical_hash(&config);
    Ok(ResolvedConfig {
        config,
        sources,
        config_hash,
    })
}

impl ProjectConfig {
    /// Structural and semantic checks that do not need the data.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != PROJECT_SCHEMA_VERSION {
            return Err(
                config_invalid(format!("unsupported schema_version {}", self.schema_version), "schema_version")
                    .expected(PROJECT_SCHEMA_VERSION.to_string()),
            );
        }
        let recipe_ok = self.recipe == RECIPE_ID || self.recipe == RECIPE_NAME;
        if !recipe_ok {
            return Err(config_invalid(format!("unknown recipe `{}`", self.recipe), "recipe")
                .expected(RECIPE_ID)
                .next("RustPy 0.1 implements only the tabular-binary recipe"));
        }
        let d = &self.data;
        let fr = [d.train_fraction, d.valid_fraction, d.test_fraction];
        if fr.iter().any(|f| !(f.is_finite() && *f > 0.0 && *f < 1.0)) {
            return Err(config_invalid(
                "split fractions must each be strictly between 0 and 1",
                "data.train_fraction / data.valid_fraction / data.test_fraction",
            )
            .actual(format!("{fr:?}")));
        }
        if (fr.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            return Err(config_invalid("split fractions must sum to 1", "data.*_fraction")
                .actual(format!("{} (sum of {fr:?})", fr.iter().sum::<f64>())));
        }
        if d.csv_delimiter.len() != 1 {
            return Err(
                config_invalid("csv_delimiter must be exactly one ASCII character", "data.csv_delimiter")
                    .actual(format!("{:?}", d.csv_delimiter)),
            );
        }
        if let Some(target) = &d.target {
            for (name, col) in [
                ("data.id_column", &d.id_column),
                ("data.group_column", &d.group_column),
                ("data.time_column", &d.time_column),
            ] {
                if col.as_deref() == Some(target.as_str()) {
                    return Err(config_invalid(format!("the target column cannot also be {name}"), name));
                }
            }
            if d.exclude.iter().any(|c| c == target) {
                return Err(config_invalid("the target column is listed in data.exclude", "data.exclude"));
            }
            if d.columns.contains_key(target) {
                return Err(config_invalid(
                    "the target column cannot be given a feature kind in [data.columns]",
                    format!("data.columns.{target}"),
                ));
            }
        }
        if self.selection.candidates.is_empty() {
            return Err(config_invalid("at least one candidate is required", "selection.candidates"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in &self.selection.candidates {
            if !seen.insert(*c) {
                return Err(config_invalid(format!("candidate `{c}` is listed twice"), "selection.candidates"));
            }
        }
        let t = self.selection.threshold;
        if !(t.is_finite() && t > 0.0 && t < 1.0) {
            return Err(config_invalid("threshold must be strictly between 0 and 1", "selection.threshold").actual(t.to_string()));
        }
        for (name, v) in [
            ("selection.min_valid_metric", self.selection.min_valid_metric),
            ("selection.max_p95_latency_us", self.selection.max_p95_latency_us),
            ("selection.max_model_kib", self.selection.max_model_kib),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(config_invalid(format!("{name} must be >= 0"), name).actual(v.to_string()));
            }
        }
        if self.preprocess.max_categories < 2 {
            return Err(config_invalid("max_categories must be at least 2", "preprocess.max_categories"));
        }
        for (name, c) in [("train.linear", &self.train.linear), ("train.small-mlp", &self.train.small_mlp)] {
            if c.epochs == 0 {
                return Err(config_invalid("epochs must be >= 1", format!("{name}.epochs")));
            }
            if c.batch_size == 0 {
                return Err(config_invalid("batch_size must be >= 1", format!("{name}.batch_size")));
            }
            if !(c.learning_rate.is_finite() && c.learning_rate > 0.0) {
                return Err(config_invalid("learning_rate must be > 0", format!("{name}.learning_rate")));
            }
            if !(c.l2.is_finite() && c.l2 >= 0.0) {
                return Err(config_invalid("l2 must be >= 0", format!("{name}.l2")));
            }
            if !(c.checkpoint_min_interval_seconds.is_finite() && c.checkpoint_min_interval_seconds >= 0.0) {
                return Err(config_invalid(
                    "checkpoint_min_interval_seconds must be >= 0",
                    format!("{name}.checkpoint_min_interval_seconds"),
                ));
            }
            if c.checkpoint_every_epochs == 0 {
                return Err(config_invalid(
                    "checkpoint_every_epochs must be >= 1",
                    format!("{name}.checkpoint_every_epochs"),
                ));
            }
        }
        if !self.train.linear.hidden.is_empty() {
            return Err(config_invalid("the linear candidate has no hidden layers", "train.linear.hidden").expected("[]"));
        }
        let h = &self.train.small_mlp.hidden;
        if h.is_empty() || h.len() > 4 || h.iter().any(|w| *w == 0 || *w > 4096) {
            return Err(config_invalid(
                "small-mlp needs 1 to 4 hidden layers with widths between 1 and 4096",
                "train.small-mlp.hidden",
            )
            .actual(format!("{h:?}")));
        }
        if self.execution.max_memory_mb == 0 {
            return Err(config_invalid("max_memory_mb must be > 0", "execution.max_memory_mb"));
        }
        Ok(())
    }

    /// Checks that the data section is complete enough to plan or run.
    pub fn require_data(&self) -> Result<(&str, &str, &str)> {
        let d = &self.data;
        let missing = |field: &str| {
            RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, format!("`data.{field}` is not set"))
                .field(format!("data.{field}"))
                .next("bind a dataset: rustpy data bind <file.csv|file.parquet> --target <column> --positive <value>")
        };
        let source = d.source.as_deref().ok_or_else(|| missing("source"))?;
        let target = d.target.as_deref().ok_or_else(|| missing("target"))?;
        let positive = d.positive_label.as_deref().ok_or_else(|| missing("positive_label"))?;
        Ok((source, target, positive))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
schema_version = 1
recipe = "tabular-binary@0.1"

[data]
source = "data/train.csv"
target = "defect"
positive_label = "1"

[train.small-mlp]
epochs = 12

[profiles.quick.train.small-mlp]
epochs = 3
"#;

    #[test]
    fn layering_and_sources() {
        let pf = ProjectFile::parse(FILE, "rustpy.toml").unwrap();
        let quick = pf.profile_layer("quick").unwrap();
        let ov = ConfigLayer {
            source: ValueSource::Override,
            table: parse_override("train.linear.epochs=9").unwrap(),
        };
        let r = resolve(&[pf.project.clone(), quick, ov]).unwrap();
        assert_eq!(r.config.train.small_mlp.epochs, 3);
        assert_eq!(r.config.train.linear.epochs, 9);
        assert_eq!(r.config.train.small_mlp.hidden, vec![32, 16]);
        assert_eq!(r.sources["train.small-mlp.epochs"], ValueSource::Profile("quick".into()));
        assert_eq!(r.sources["train.linear.epochs"], ValueSource::Override);
        assert_eq!(r.sources["data.target"], ValueSource::Project);
        assert_eq!(r.sources["data.seed"], ValueSource::Default);

        let r2 = resolve(&[pf.project]).unwrap();
        assert_eq!(r2.config.train.small_mlp.epochs, 12);
        assert_ne!(r.config_hash, r2.config_hash);
    }

    #[test]
    fn unknown_field_is_reported_with_location() {
        let text = "[train.small-mlp]\nepoch = 3\n";
        let err = ProjectFile::parse(text, "rustpy.toml").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConfigUnknownField);
        assert!(err.message.contains("epoch"), "{}", err.message);
        assert!(err.message.contains("rustpy.toml:2"), "{}", err.message);
    }

    #[test]
    fn override_typos_and_types_are_rejected() {
        let e = parse_override("train.small-mlp.epoch=3").unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigUnknownField);
        let e = parse_override("train.small-mlp.epochs=lots").unwrap_err();
        assert_eq!(e.code, ErrorCode::ConfigInvalid);
        assert!(parse_override("selection.candidates=[\"linear\"]").is_ok());
        assert!(parse_override("data.target=defect").is_ok());
    }

    #[test]
    fn semantic_validation() {
        let mut c = ProjectConfig::default();
        c.data.train_fraction = 0.8;
        assert!(c.validate().is_err());
        let mut c = ProjectConfig::default();
        c.train.linear.hidden = vec![3];
        assert!(c.validate().is_err());
        assert!(ProjectConfig::default().validate().is_ok());
        assert!(ProjectConfig::default().require_data().is_err());
    }

    #[test]
    fn unknown_keys_suggest_the_closest_name() {
        let e = ProjectFile::parse("[data]\ntrain_fracton = 0.6\n", "rustpy.toml").unwrap_err();
        assert!(
            e.next_action.as_deref().unwrap_or("").contains("did you mean `train_fraction`"),
            "{e}"
        );
        assert_eq!(closest_expected("unknown field `zzzzzz`, expected one of `source`, `target`"), None);
    }

    #[test]
    fn effective_values_cover_every_leaf() {
        let r = resolve(&[]).unwrap();
        let vals = r.effective_values();
        assert!(vals.iter().any(|(p, _, _)| p == "train.small-mlp.hidden"));
        assert!(vals.iter().any(|(p, v, _)| p == "data.id_column" && v == "(unset)"));
        assert!(vals.iter().all(|(_, _, s)| *s == ValueSource::Default));
    }
}
