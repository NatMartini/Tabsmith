//! Lightweight RustPy prediction runtime.
//!
//! Depends only on `rustpy-contracts`, `serde`, `serde_json` and `safetensors`:
//! no training engine, no data query engine, no Python.
//!
//! ```no_run
//! use rustpy_runtime::{LoadOptions, Predictor, Record};
//!
//! let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
//! let record = Record::new()
//!     .with("temperature_mean", 71.4)
//!     .with("line", "B")
//!     .with("hours_in_service", 1200);
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
pub use transform::{apply_transform, check_category_type, coerce_category, coerce_numeric, CoerceError, Matrix, MissingTokens};
pub use value::{Column, ColumnData, Record, Table, Value, ValueRef};
