//! RustPy workflow core.
//!
//! Turns a project configuration into reproducible stages — data snapshot, split,
//! fitted transform, training runs, evaluations and inference bundles — with
//! one implementation shared by the CLI and the Rust SDK.

pub mod bundle;
pub mod data;
pub mod device;
pub mod fit;
pub mod metrics;
pub mod project;
pub mod rng;
pub mod sample;
pub mod setup;
pub mod report;
pub mod split;
pub mod train;

pub use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
pub mod pipeline;

pub use pipeline::{Comparison, Prepared, Progress, RunGroupResult, RunOptions, Workspace};
pub use project::{ConfigRequest, Project};
pub use train::RunControl;
