//! Binary classification metrics and evaluation records.

use serde::{Deserialize, Serialize};

use crate::data::SplitName;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub tp: u64,
    pub fp: u64,
    pub tn: u64,
    #[serde(rename = "fn")]
    pub fn_: u64,
}

/// Metrics at a fixed decision threshold, plus threshold-free metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryMetrics {
    pub metrics_version: String,
    pub n: u64,
    pub n_positive: u64,
    pub n_negative: u64,
    pub threshold: f64,
    pub confusion: ConfusionMatrix,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    /// Precision of the positive class (0 when nothing is predicted positive).
    pub precision: f64,
    /// Recall / sensitivity of the positive class.
    pub recall: f64,
    pub specificity: f64,
    pub f1: f64,
    /// Mean binary cross-entropy of the predicted probabilities (clipped to [ε, 1 − ε] with ε = f64::EPSILON, as scikit-learn).
    pub log_loss: f64,
    pub brier: f64,
    /// Area under the ROC curve (ties averaged); `None` when only one class is present.
    pub roc_auc: Option<f64>,
}

impl BinaryMetrics {
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "accuracy" => Some(self.accuracy),
            "balanced_accuracy" => Some(self.balanced_accuracy),
            "precision" => Some(self.precision),
            "recall" => Some(self.recall),
            "specificity" => Some(self.specificity),
            "f1" => Some(self.f1),
            "log_loss" => Some(self.log_loss),
            "brier" => Some(self.brier),
            "roc_auc" => self.roc_auc,
            _ => None,
        }
    }
}

/// The conditions under which two results may be compared directly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComparisonProtocol {
    pub snapshot_id: String,
    pub split_id: String,
    pub split: SplitName,
    pub target: String,
    pub positive_label: String,
    /// Threshold formatted with full precision (string keeps the struct `Eq`).
    pub threshold: String,
    pub metrics_version: String,
}

impl ComparisonProtocol {
    /// Names the fields in which two protocols differ.
    pub fn differences(&self, other: &Self) -> Vec<String> {
        let mut d = Vec::new();
        if self.snapshot_id != other.snapshot_id {
            d.push(format!("data snapshot ({} vs {})", self.snapshot_id, other.snapshot_id));
        }
        if self.split_id != other.split_id {
            d.push(format!("split ({} vs {})", self.split_id, other.split_id));
        }
        if self.split != other.split {
            d.push(format!("evaluation split ({} vs {})", self.split.as_str(), other.split.as_str()));
        }
        if self.target != other.target || self.positive_label != other.positive_label {
            d.push("target definition".into());
        }
        if self.threshold != other.threshold {
            d.push(format!("threshold ({} vs {})", self.threshold, other.threshold));
        }
        if self.metrics_version != other.metrics_version {
            d.push("metric definitions".into());
        }
        d
    }
}

/// What was evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvaluationSubject {
    Run { run_id: String },
    Bundle { bundle_id: String, path: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    pub rows: usize,
    pub batch_ms: f64,
    pub single_row_p50_us: f64,
    pub single_row_p95_us: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub evaluation_id: String,
    pub created_at: String,
    pub subject: EvaluationSubject,
    pub protocol: ComparisonProtocol,
    pub metrics: BinaryMetrics,
    /// For test-split evaluations: 1 for the first model of this data version
    /// evaluated on these test rows, >1 when competing models of the same data
    /// version already used them (results are then no longer an unbiased estimate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_use_index: Option<u32>,
    /// Earlier test evaluations of models trained on *another* data version that
    /// used some of the same rows (the shared hold-out of an old-vs-new comparison),
    /// as `subject (evaluation id)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test_shared_with: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<LatencyStats>,
    /// SHA-256 of the evaluated weights (`model.safetensors`); a run and its bundle share it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    /// Data snapshot the evaluated model was trained on (its "generation").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_snapshot_id: Option<String>,
    /// `Some(true)`: the evaluated rows were verified to be unseen by the model
    /// (training rows, and validation rows for a test evaluation, of the model and of
    /// runs it was warm-started from). `None`: rows could not be matched, so this
    /// could not be verified (such evaluations need `allow_unverified_rows`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unseen_rows_verified: Option<bool>,
    #[serde(default)]
    pub notes: Vec<String>,
}
