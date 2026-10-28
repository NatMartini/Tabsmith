//! Lightweight Tabsmith prediction runtime.
//!
//! Depends only on `tabsmith-contracts`, `serde`, `serde_json` and `safetensors`:
//! no training engine, no data query engine, no Python.
//!
//! ```no_run
//! use tabsmith_runtime::{LoadOptions, Predictor, Record};
//!
//! // A bundle built by `tabsmith train` (here from the synthetic sample data).
//! let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
//! // Every input field of the bundle is required (see its input.schema.json, or
//! // complete records in examples/input.json); `None` is a missing value, imputed
//! // as in training.
//! let record = Record::new()
//!     .with("line", "B")
//!     .with("temp_mean", 71.4)
//!     .with("hours_in_service", 1200)
//!     .with("calibration_offset", None::<f64>)
//!     /* … the other input fields … */;
//! let prediction = predictor.predict_one(&record)?;
//! println!("{} (p = {:.3})", prediction.label, prediction.probability().unwrap_or_default());
//! # Ok::<(), tabsmith_runtime::TabsmithError>(())
//! ```
#![warn(missing_docs)]

pub mod bundle;
pub mod model;
pub mod predictor;
pub mod transform;
pub mod value;

pub use bundle::{Bundle, Device, LoadOptions};
pub use predictor::{CheckReport, Outcome, Prediction, Predictor, PredictorHandle, UpdateReport, decode};
pub use tabsmith_contracts::{ErrorCode, Phase, TabsmithError};
pub use transform::{
    CoerceError, Matrix, MissingTokens, UnseenCategories, apply_transform, apply_transform_rows, check_category_type, coerce_category,
    coerce_numeric, unseen_categories,
};
pub use value::{Column, ColumnData, Record, Table, Value, ValueRef};
