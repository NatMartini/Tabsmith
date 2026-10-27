//! Input and output contracts of a trained tabular-binary model.

use serde::{Deserialize, Serialize};

use crate::config::{ExtraColumnPolicy, NanPolicy, UnknownCategoryPolicy};

/// Physical value type observed for a column in the training data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawDtype {
    Float,
    Int,
    Bool,
    Text,
}

impl RawDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            RawDtype::Float => "float",
            RawDtype::Int => "int",
            RawDtype::Bool => "bool",
            RawDtype::Text => "text",
        }
    }
}

/// How a feature is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldKind {
    Numeric,
    Categorical,
}

impl FieldKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FieldKind::Numeric => "numeric",
            FieldKind::Categorical => "categorical",
        }
    }
}

/// One named input field of the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputField {
    pub name: String,
    pub kind: FieldKind,
    /// Type seen in training; offline CSV prediction re-reads the column with this type.
    pub raw_dtype: RawDtype,
    /// Whether the field must be present in every record. Fields dropped by the
    /// transform (e.g. zero variance) are accepted but not required.
    pub required: bool,
    /// Whether missing values (null / NaN when `nan = "missing"`) are accepted.
    pub nullable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Contract for prediction inputs (`input.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSchema {
    pub fields: Vec<InputField>,
    /// A field absent from a record is always an error in 0.1 (an explicit null is a missing value).
    pub missing_column: String,
    pub extra_columns: ExtraColumnPolicy,
    pub unknown_category: UnknownCategoryPolicy,
    pub nan: NanPolicy,
    /// Infinite numeric values are always rejected in 0.1.
    pub infinite: String,
    /// CSV delimiter of the training data (used by offline CSV prediction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csv_delimiter: Option<String>,
    /// Optional id column copied to prediction outputs when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passthrough_id: Option<String>,
}

impl InputSchema {
    pub fn field(&self, name: &str) -> Option<&InputField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Contract for prediction outputs (`output.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSchema {
    pub target: String,
    pub negative_label: String,
    pub positive_label: String,
    /// `probability` is P(target == positive_label).
    pub probability: String,
    pub threshold: f64,
    /// Decision rule: positive iff probability >= threshold.
    pub decision: String,
}

impl OutputSchema {
    pub fn new(target: &str, negative: &str, positive: &str, threshold: f64) -> Self {
        Self {
            target: target.to_string(),
            negative_label: negative.to_string(),
            positive_label: positive.to_string(),
            probability: format!("P({target} == {positive:?})"),
            threshold,
            decision: format!("{positive:?} if probability >= {threshold} else {negative:?}"),
        }
    }
}
