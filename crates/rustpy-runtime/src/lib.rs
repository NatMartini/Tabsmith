//! Lightweight RustPy prediction runtime.
//!
//! Depends only on `rustpy-contracts`, `serde`, `serde_json` and `safetensors`:
//! no training engine, no data query engine, no Python.
//!
//! ```no_run
//! use rustpy_runtime::{LoadOptions, Predictor, Record};
//!
//! // A bundle built by `rustpy train` (here from the synthetic sample data).
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
//! println!("{} (p = {:.3})", prediction.label, prediction.probability);
//! # Ok::<(), rustpy_runtime::RustpyError>(())
//! ```

pub mod bundle;
pub mod model;
pub mod predictor;
pub mod transform;
pub mod value;

pub use bundle::{Bundle, Device, LoadOptions};
pub use predictor::{CheckReport, Prediction, Predictor, PredictorHandle, UpdateReport};
pub use rustpy_contracts::{ErrorCode, Phase, RustpyError};
pub use transform::{
    CoerceError, Matrix, MissingTokens, UnseenCategories, apply_transform, apply_transform_rows, check_category_type, coerce_category,
    coerce_numeric, unseen_categories,
};
pub use value::{Column, ColumnData, Record, Table, Value, ValueRef};
