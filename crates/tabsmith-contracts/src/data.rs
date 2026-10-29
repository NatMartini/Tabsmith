//! Data snapshots and split manifests.

use serde::{Deserialize, Serialize};

use crate::config::SplitStrategy;
use crate::schema::RawDtype;

/// File format of a data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    Csv,
    Parquet,
}

/// Identity of the source file at bind time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRef {
    /// Path as written in the configuration.
    pub path: String,
    /// Absolute path at bind time (informational; may differ on other machines).
    pub resolved_path: String,
    pub format: SourceFormat,
    pub sha256: String,
    pub bytes: u64,
}

/// How the snapshot content is kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SnapshotStorage {
    /// The source bytes were copied into the project; the snapshot is fully replayable.
    Copy { file: String },
    /// Only the reference + hash is kept; replay requires the unchanged source.
    Reference,
}

/// How rows are identified in split manifests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowIdentity {
    /// Values of a unique id column.
    IdColumn { column: String },
    /// Position in the snapshot content (stable only for this exact content hash).
    RowIndex,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

/// Per-column profile computed over the whole snapshot (not used for fitting).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    /// `None` when the column type is not supported (e.g. dates, nested types).
    pub raw_dtype: Option<RawDtype>,
    /// Engine type name as read (for diagnostics).
    pub engine_dtype: String,
    pub null_count: u64,
    pub nan_count: u64,
    pub inf_count: u64,
    /// Number of distinct non-missing values (exact).
    pub distinct: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric: Option<NumericSummary>,
    /// Most frequent values (text form) with counts; at most 10.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_values: Vec<(String, u64)>,
    /// Float columns: every finite value is an integer (possible encoded identifier).
    #[serde(default)]
    pub all_integral: bool,
    /// Text columns: number of values that parse as a number.
    #[serde(default)]
    pub numeric_text: u64,
    /// Text columns that are mostly numeric: examples of the values that are not.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub non_numeric_examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSnapshot {
    pub snapshot_id: String,
    pub created_at: String,
    pub source: SourceRef,
    pub storage: SnapshotStorage,
    pub n_rows: usize,
    pub columns: Vec<ColumnProfile>,
    pub row_identity: RowIdentity,
    /// Human-readable limitations (e.g. "row identity is positional").
    #[serde(default)]
    pub limitations: Vec<String>,
}

impl DataSnapshot {
    pub fn column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Split membership code per row.
pub const SPLIT_TRAIN: u8 = b'0';
pub const SPLIT_VALID: u8 = b'1';
pub const SPLIT_TEST: u8 = b'2';

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitName {
    Train,
    Valid,
    Test,
}

impl SplitName {
    pub fn code(self) -> u8 {
        match self {
            SplitName::Train => SPLIT_TRAIN,
            SplitName::Valid => SPLIT_VALID,
            SplitName::Test => SPLIT_TEST,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SplitName::Train => "train",
            SplitName::Valid => "valid",
            SplitName::Test => "test",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "train" => Some(SplitName::Train),
            "valid" | "validation" => Some(SplitName::Valid),
            "test" => Some(SplitName::Test),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitCounts {
    pub rows: usize,
    /// Binary: rows of the positive class.
    pub positives: usize,
    /// Binary: rows of the negative class.
    pub negatives: usize,
    /// Multiclass: rows per class, in class order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub groups: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitManifest {
    pub split_id: String,
    pub snapshot_id: String,
    pub strategy: SplitStrategy,
    pub seed: u64,
    pub fractions: [f64; 3],
    pub target: String,
    pub positive_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_column: Option<String>,
    /// The column a chronological split ordered the rows by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_column: Option<String>,
    /// One ASCII digit per snapshot row: '0' train, '1' valid, '2' test.
    pub assignment: String,
    pub train: SplitCounts,
    pub valid: SplitCounts,
    pub test: SplitCounts,
    /// Split algorithm version; part of the cache key.
    pub algorithm: String,
    /// E.g. fractions of a hash-based split that differ from the configured ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl SplitManifest {
    pub fn rows(&self, split: SplitName) -> Vec<usize> {
        let code = split.code();
        self.assignment
            .bytes()
            .enumerate()
            .filter_map(|(i, b)| (b == code).then_some(i))
            .collect()
    }

    pub fn counts(&self, split: SplitName) -> &SplitCounts {
        match split {
            SplitName::Train => &self.train,
            SplitName::Valid => &self.valid,
            SplitName::Test => &self.test,
        }
    }
}
