//! Fitted state of the standard tabular transform.
//!
//! The state is learned from the training split only and frozen; the single
//! implementation that applies it lives in `rustpy-runtime` and is used both by
//! training/evaluation and by application prediction.

use serde::{Deserialize, Serialize};

use crate::config::{NanPolicy, UnknownCategoryPolicy};

/// Implementation version of the standard transform. A bundle declares the
/// version it needs; a runtime refuses versions it does not implement.
pub const TRANSFORM_VERSION: &str = "rustpy.std-tabular.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeatureTransform {
    /// `z = (x - mean) / std`, missing `x` replaced by `impute` first.
    Numeric {
        name: String,
        impute: f64,
        mean: f64,
        std: f64,
        /// Index of the single output column.
        offset: usize,
    },
    /// One-hot over `categories` (training order), plus a trailing slot for
    /// missing values when missing values were seen in training. Unseen values
    /// encode as all zeros (`unknown = ignore`) or are rejected (`unknown = error`).
    Categorical {
        name: String,
        categories: Vec<String>,
        missing_slot: bool,
        offset: usize,
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
                categories,
                missing_slot,
                ..
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
pub struct DroppedField {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedTransform {
    pub version: String,
    pub features: Vec<FeatureTransform>,
    pub dropped: Vec<DroppedField>,
    pub output_dim: usize,
    pub output_names: Vec<String>,
    pub nan: NanPolicy,
    pub unknown_category: UnknownCategoryPolicy,
    /// Split manifest the statistics were learned from (training rows only).
    pub fitted_on_split: String,
    pub fitted_rows: usize,
}

impl FittedTransform {
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
            if let FeatureTransform::Numeric { std, mean, impute, .. } = f {
                if !(std.is_finite() && *std > 0.0 && mean.is_finite() && impute.is_finite()) {
                    return Err(format!("feature `{}` has invalid statistics", f.name()));
                }
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
