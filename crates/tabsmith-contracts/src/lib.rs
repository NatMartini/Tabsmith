//! Stable cross-stage contracts for Tabsmith.
//!
//! This crate deliberately has no dependency on the training engine (Burn) or the
//! data query engine (Polars). Everything that crosses a stage boundary — project
//! configuration, data snapshots, split manifests, fitted transform state, Run
//! records, evaluation records and the inference Bundle manifest — is defined here
//! with an explicit format version.

pub mod config;
pub mod data;
pub mod device;
pub mod error;
pub mod evaluation;
pub mod hash;
pub mod manifest;
pub mod model;
pub mod run;
pub mod schema;
pub mod time;
pub mod transform;

pub use error::{ErrorCode, Phase, Result, TabsmithError};

/// Version of the Tabsmith crates (shared by all workspace crates).
pub const TABSMITH_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The only recipe implemented in this release.
pub const RECIPE_NAME: &str = "tabular-binary";
/// Recipe contract version. Bumped when the recipe's semantics change.
pub const RECIPE_VERSION: &str = "0.1";
/// `recipe` value written to `tabsmith.toml`.
pub const RECIPE_ID: &str = "tabular-binary@0.1";

/// Version of the `tabsmith.toml` schema.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;
/// Version of the on-disk Run record format.
pub const RUN_FORMAT_VERSION: u32 = 1;
/// Version of the inference Bundle layout.
pub const BUNDLE_FORMAT_VERSION: u32 = 1;
/// Version of the metric definitions used in evaluation records.
pub const METRICS_VERSION: &str = "tabsmith.binary-metrics.v1";
