//! Fitted state of the standard tabular transform.
//!
//! The state is learned from the training split only and frozen; the single
//! implementation that applies it lives in `tabsmith-runtime` and is used both by
//! training/evaluation and by application prediction.

use serde::{Deserialize, Serialize};

use crate::config::{NanPolicy, UnknownCategoryPolicy};

/// Implementation version of the standard transform. A bundle declares the
/// version it needs; a runtime refuses versions it does not implement.
///
/// v2: missing-value tokens and typed categoricals are part of the transform.
pub const TRANSFORM_VERSION: &str = "tabsmith.std-tabular.v2";

/// What a numeric feature does with a missing value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingRule {
    /// Replace it with `impute`: Tabsmith's own models.
    #[default]
    Impute,
    /// Refuse it: a model imported from a library whose trees send a missing value
    /// left at some splits and right at others, which no single value reproduces.
    Refuse,
}

impl MissingRule {
    /// Whether this is the default, which a transform leaves out.
    pub fn is_impute(&self) -> bool {
        *self == MissingRule::Impute
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureTransform {
    /// `z = (x - mean) / std`, missing `x` replaced by `impute` first (or refused, by
    /// [`MissingRule::Refuse`]).
    Numeric {
        name: String,
        impute: f64,
        mean: f64,
        std: f64,
        /// Index of the single output column.
        offset: usize,
        /// `impute` (left out) or `refuse`. A runtime that does not know this field
        /// refuses the bundle rather than impute what it must refuse.
        #[serde(default, skip_serializing_if = "MissingRule::is_impute")]
        missing: MissingRule,
    },
    /// One-hot over `categories` (training order), plus a trailing slot for
    /// missing values when missing values were seen in training. Unseen values
    /// encode as all zeros (`unknown = ignore`) or are rejected (`unknown = error`).
    Categorical {
        name: String,
        categories: Vec<String>,
        missing_slot: bool,
        offset: usize,
        /// Raw type seen in training; values of another type are rejected rather
        /// than silently mapped (e.g. the number 12 for the text code "012").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_dtype: Option<crate::schema::RawDtype>,
    },
}

impl FeatureTransform {
    pub fn name(&self) -> &str {
        match self {
            FeatureTransform::Numeric { name, .. } | FeatureTransform::Categorical { name, .. } => name,
        }
    }

    pub fn width(&self) -> usize {
        match self {
            FeatureTransform::Numeric { .. } => 1,
            FeatureTransform::Categorical {
                categories, missing_slot, ..
            } => categories.len() + usize::from(*missing_slot),
        }
    }

    pub fn offset(&self) -> usize {
        match self {
            FeatureTransform::Numeric { offset, .. } | FeatureTransform::Categorical { offset, .. } => *offset,
        }
    }
}

/// A field that is part of the input contract but contributes no feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DroppedField {
    pub name: String,
    pub reason: String,
}

/// Bundle contract types reject unknown fields: a runtime that does not know a
/// field cannot honour its semantics, so it must refuse rather than ignore it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedTransform {
    pub version: String,
    pub features: Vec<FeatureTransform>,
    pub dropped: Vec<DroppedField>,
    pub output_dim: usize,
    pub output_names: Vec<String>,
    pub nan: NanPolicy,
    pub unknown_category: UnknownCategoryPolicy,
    /// Tokens that mean "missing" (the project's `data.null_values`): text values
    /// equal to a token, and numbers equal to a numeric token (e.g. "-999"), are
    /// treated as missing on every path, exactly as when the CSV was read.
    #[serde(default)]
    pub missing_tokens: Vec<String>,
    /// Split manifest the statistics were learned from (training rows only).
    pub fitted_on_split: String,
    pub fitted_rows: usize,
}

impl FittedTransform {
    /// Whether the numbers' `mean` and `std` come from training rows. An imported model's
    /// transform has none: it uses each number as it is (mean 0, std 1).
    pub fn has_statistics(&self) -> bool {
        self.fitted_rows > 0
    }

    /// Checks internal consistency (offsets contiguous, names match widths).
    pub fn check_consistency(&self) -> Result<(), String> {
        let mut expected = 0usize;
        for f in &self.features {
            if f.offset() != expected {
                return Err(format!(
                    "feature `{}` starts at {} but {} was expected",
                    f.name(),
                    f.offset(),
                    expected
                ));
            }
            if let FeatureTransform::Numeric { std, mean, impute, .. } = f
                && !(std.is_finite() && *std > 0.0 && mean.is_finite() && impute.is_finite())
            {
                return Err(format!("feature `{}` has invalid statistics", f.name()));
            }
            expected += f.width();
        }
        if expected != self.output_dim || self.output_names.len() != self.output_dim {
            return Err(format!(
                "output_dim {} does not match features ({expected}) / names ({})",
                self.output_dim,
                self.output_names.len()
            ));
        }
        Ok(())
    }
}
