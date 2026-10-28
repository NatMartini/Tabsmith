//! Input and output contracts of a trained tabular-binary model.

use serde::{Deserialize, Serialize};

use crate::config::{ExtraColumnPolicy, NanPolicy, Task, UnknownCategoryPolicy};

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

/// Regression targets are learned standardised: the model outputs
/// `(value - mean) / std` of the training rows' targets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetScale {
    pub mean: f64,
    pub std: f64,
}

/// Contract for prediction outputs (`output.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSchema {
    pub target: String,
    /// How the model's outputs are read (bundles without it are binary).
    #[serde(default)]
    pub task: Task,
    /// Binary: the label below the threshold.
    #[serde(default)]
    pub negative_label: String,
    /// Binary: the label at or above the threshold.
    #[serde(default)]
    pub positive_label: String,
    /// Binary: `probability` is P(target == positive_label).
    #[serde(default)]
    pub probability: String,
    /// Binary: decision threshold on the probability.
    #[serde(default)]
    pub threshold: f64,
    /// How an output becomes the predicted label or value.
    pub decision: String,
    /// Multiclass: the classes, in the order of the model's outputs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
    /// Regression: how a model output becomes a value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_scale: Option<TargetScale>,
}

impl OutputSchema {
    /// Binary: P(positive) and a threshold.
    pub fn new(target: &str, negative: &str, positive: &str, threshold: f64) -> Self {
        Self {
            target: target.to_string(),
            task: Task::Binary,
            negative_label: negative.to_string(),
            positive_label: positive.to_string(),
            probability: format!("P({target} == {positive:?})"),
            threshold,
            decision: format!("{positive:?} if probability >= {threshold} else {negative:?}"),
            classes: Vec::new(),
            target_scale: None,
        }
    }

    /// Multiclass: one probability per class (softmax); the label is the most
    /// probable class, the first in `classes` order on a tie.
    pub fn multiclass(target: &str, classes: Vec<String>) -> Self {
        Self {
            target: target.to_string(),
            task: Task::Multiclass,
            negative_label: String::new(),
            positive_label: String::new(),
            probability: format!("P({target} == label)"),
            threshold: 0.0,
            decision: "the class with the highest probability".into(),
            classes,
            target_scale: None,
        }
    }

    /// Regression: value = mean + std * output.
    pub fn regression(target: &str, scale: TargetScale) -> Self {
        Self {
            target: target.to_string(),
            task: Task::Regression,
            negative_label: String::new(),
            positive_label: String::new(),
            probability: String::new(),
            threshold: 0.0,
            decision: format!("{target} = {} + {} * output", scale.mean, scale.std),
            classes: Vec::new(),
            target_scale: Some(scale),
        }
    }

    /// Outputs per row the model must produce.
    pub fn n_outputs(&self) -> usize {
        match self.task {
            Task::Multiclass => self.classes.len(),
            Task::Binary | Task::Regression => 1,
        }
    }

    /// Whether the fields of this task are set consistently.
    pub fn check_consistency(&self) -> Result<(), String> {
        match self.task {
            Task::Binary if self.positive_label == self.negative_label => Err("output labels are not distinct".into()),
            Task::Binary if !(self.threshold > 0.0 && self.threshold < 1.0) => Err("threshold must be inside (0, 1)".into()),
            Task::Multiclass => {
                let unique: std::collections::BTreeSet<&String> = self.classes.iter().collect();
                if self.classes.len() < 3 || unique.len() != self.classes.len() {
                    return Err("a multiclass output needs 3 or more distinct classes".into());
                }
                Ok(())
            }
            Task::Regression => match self.target_scale {
                Some(s) if s.mean.is_finite() && s.std.is_finite() && s.std > 0.0 => Ok(()),
                _ => Err("a regression output needs a finite target scale".into()),
            },
            Task::Binary => Ok(()),
        }
    }
}
