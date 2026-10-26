//! Inference Bundle manifest.
//!
//! ```text
//! bundle/
//!   manifest.json        this file: versions, provenance, file hashes
//!   input.schema.json    named, typed input fields and policies
//!   output.schema.json   labels, probability meaning, threshold
//!   transform.json       frozen standard transform state
//!   model.config.json    engine-neutral architecture
//!   model.safetensors    F32 weights
//!   evaluation.json      evaluation evidence the bundle was built from
//!   examples/input.json  synthetic example records (no training rows)
//!   examples/expected.json outputs the build produced for those records
//! ```

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;

pub const MANIFEST_FILE: &str = "manifest.json";
pub const INPUT_SCHEMA_FILE: &str = "input.schema.json";
pub const OUTPUT_SCHEMA_FILE: &str = "output.schema.json";
pub const TRANSFORM_FILE: &str = "transform.json";
pub const MODEL_CONFIG_FILE: &str = "model.config.json";
pub const MODEL_WEIGHTS_FILE: &str = "model.safetensors";
pub const EVALUATION_FILE: &str = "evaluation.json";
pub const EXAMPLES_INPUT_FILE: &str = "examples/input.json";
pub const EXAMPLES_EXPECTED_FILE: &str = "examples/expected.json";

/// Files every bundle of format 1 must contain.
pub const REQUIRED_FILES: &[&str] = &[
    INPUT_SCHEMA_FILE,
    OUTPUT_SCHEMA_FILE,
    TRANSFORM_FILE,
    MODEL_CONFIG_FILE,
    MODEL_WEIGHTS_FILE,
    EVALUATION_FILE,
    EXAMPLES_INPUT_FILE,
    EXAMPLES_EXPECTED_FILE,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Oldest `rustpy-runtime` version that can load this bundle.
    pub min_runtime_version: String,
    pub transform_version: String,
    pub model_format: String,
    /// Devices the bundle was checked for at build time.
    pub devices: Vec<String>,
    pub precision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleProvenance {
    pub run_id: String,
    pub candidate: String,
    pub snapshot_id: String,
    pub split_id: String,
    pub transform_id: String,
    pub config_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub bundle_format: u32,
    /// Content id: hash over the sorted file entries (excludes the manifest itself).
    pub bundle_id: String,
    pub name: String,
    pub created_at: String,
    pub recipe: String,
    pub rustpy_version: String,
    pub provenance: BundleProvenance,
    pub runtime: RuntimeRequirements,
    pub files: Vec<FileEntry>,
    /// True when every file needed for prediction is inside the bundle.
    pub self_contained: bool,
    pub contains_training_data: bool,
    pub contains_optimizer_state: bool,
    /// `evaluated` when built from a test evaluation; `unevaluated-candidate` otherwise.
    pub evaluation_status: String,
}

/// Content of `evaluation.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleEvaluation {
    /// `evaluated` or `unevaluated-candidate`.
    pub status: String,
    /// Held-out test evaluation of the source run (absent for unevaluated candidates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<crate::evaluation::EvaluationRecord>,
    /// Validation metrics recorded by the source run.
    pub valid: crate::evaluation::BinaryMetrics,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// `examples/input.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExampleInputs {
    /// Synthetic records built from transform statistics; never copies of training rows.
    pub records: Vec<serde_json::Value>,
    pub note: String,
}

/// One expected prediction in `examples/expected.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedPrediction {
    pub label: String,
    pub probability: f64,
}

/// `examples/expected.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExampleExpected {
    pub predictions: Vec<ExpectedPrediction>,
    /// Absolute tolerance on probabilities.
    pub tolerance: f64,
}

/// Bundle id = sha256 over `path\0sha256\n` lines of the sorted file entries.
pub fn compute_bundle_id(files: &[FileEntry]) -> String {
    let mut sorted: Vec<&FileEntry> = files.iter().collect();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    let mut text = String::new();
    for f in sorted {
        text.push_str(&f.path);
        text.push('\0');
        text.push_str(&f.sha256);
        text.push('\n');
    }
    format!("b-{}", &sha256_hex(text.as_bytes())[..16])
}

/// Compares dotted `major.minor.patch` versions (missing parts count as 0).
pub fn version_at_least(have: &str, need: &str) -> bool {
    fn parse(v: &str) -> Vec<u64> {
        v.split(['.', '-', '+'])
            .take(3)
            .map(|p| p.parse::<u64>().unwrap_or(0))
            .collect()
    }
    let (mut h, mut n) = (parse(have), parse(need));
    h.resize(3, 0);
    n.resize(3, 0);
    h >= n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_id_is_order_independent() {
        let a = FileEntry { path: "a".into(), sha256: "1".into(), bytes: 1 };
        let b = FileEntry { path: "b".into(), sha256: "2".into(), bytes: 1 };
        assert_eq!(compute_bundle_id(&[a.clone(), b.clone()]), compute_bundle_id(&[b, a]));
    }

    #[test]
    fn versions() {
        assert!(version_at_least("0.1.0", "0.1.0"));
        assert!(version_at_least("0.2.0", "0.1.9"));
        assert!(!version_at_least("0.1.0", "0.1.1"));
        assert!(version_at_least("1.0", "0.9.9"));
    }
}
