//! One structured error object shared by the CLI, the SDK and JSON output.
//!
//! Every error says what happened (`code`, `message`), where (`phase`, `run_id`,
//! `field`), what was expected and what was found, which artifacts are still
//! valid, and what the user can do next.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;

pub type Result<T, E = RustpyError> = std::result::Result<T, E>;

macro_rules! error_codes {
    ($($variant:ident => $text:literal, $exit:literal;)*) => {
        /// Stable error codes. The string form (e.g. `E_SCHEMA_MISSING_COLUMN`) is part
        /// of the public contract and never reused for a different meaning.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum ErrorCode {
            $(#[serde(rename = $text)] $variant,)*
        }

        impl ErrorCode {
            pub const ALL: &'static [ErrorCode] = &[$(ErrorCode::$variant,)*];

            pub fn as_str(self) -> &'static str {
                match self { $(ErrorCode::$variant => $text,)* }
            }

            /// Process exit code used by the CLI for this error.
            pub fn exit_code(self) -> i32 {
                match self { $(ErrorCode::$variant => $exit,)* }
            }
        }
    };
}

error_codes! {
    ConfigNotFound => "E_CONFIG_NOT_FOUND", 2;
    ConfigUnknownField => "E_CONFIG_UNKNOWN_FIELD", 2;
    ConfigInvalid => "E_CONFIG_INVALID", 2;
    DataSourceNotFound => "E_DATA_SOURCE_NOT_FOUND", 3;
    DataSourceChanged => "E_DATA_SOURCE_CHANGED", 3;
    DataReadFailed => "E_DATA_READ_FAILED", 3;
    DataUnsupported => "E_DATA_UNSUPPORTED", 3;
    SchemaMissingColumn => "E_SCHEMA_MISSING_COLUMN", 3;
    SchemaExtraColumn => "E_SCHEMA_EXTRA_COLUMN", 3;
    SchemaTypeMismatch => "E_SCHEMA_TYPE_MISMATCH", 3;
    InvalidValue => "E_INVALID_VALUE", 3;
    UnknownCategory => "E_UNKNOWN_CATEGORY", 3;
    CardinalityBudget => "E_CARDINALITY_BUDGET", 3;
    TargetInvalid => "E_TARGET_INVALID", 3;
    SplitUnsupported => "E_SPLIT_UNSUPPORTED", 3;
    SplitDegenerate => "E_SPLIT_DEGENERATE", 3;
    LeakageDetected => "E_LEAKAGE_DETECTED", 3;
    DeviceUnsupported => "E_DEVICE_UNSUPPORTED", 5;
    ResourceBudgetExceeded => "E_RESOURCE_BUDGET_EXCEEDED", 5;
    RunNotFound => "E_RUN_NOT_FOUND", 6;
    RunStateInvalid => "E_RUN_STATE_INVALID", 6;
    ResumeIncompatible => "E_RESUME_INCOMPATIBLE", 6;
    CheckpointMissing => "E_CHECKPOINT_MISSING", 6;
    CheckpointCorrupt => "E_CHECKPOINT_CORRUPT", 6;
    TrainingDiverged => "E_TRAINING_DIVERGED", 6;
    Cancelled => "E_CANCELLED", 130;
    Interrupted => "E_INTERRUPTED", 6;
    NotComparable => "E_NOT_COMPARABLE", 7;
    TestSetReuse => "E_TEST_SET_REUSE", 7;
    EvaluationMissing => "E_EVALUATION_MISSING", 7;
    BundleNotFound => "E_BUNDLE_NOT_FOUND", 4;
    BundleMissingFile => "E_BUNDLE_MISSING_FILE", 4;
    BundleCorrupt => "E_BUNDLE_CORRUPT", 4;
    BundleIncompatible => "E_BUNDLE_INCOMPATIBLE", 4;
    BundleExists => "E_BUNDLE_EXISTS", 4;
    PredictorUpdateFailed => "E_PREDICTOR_UPDATE_FAILED", 4;
    Io => "E_IO", 1;
    Internal => "E_INTERNAL", 1;
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The workflow stage in which an error was raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Config,
    Data,
    Split,
    Fit,
    Plan,
    Train,
    Resume,
    Evaluate,
    Compare,
    Bundle,
    Load,
    Predict,
    Device,
    Report,
    Internal,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        f.write_str(&s)
    }
}

/// The single structured error type of RustPy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RustpyError {
    pub code: ErrorCode,
    pub phase: Phase,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Field, column, config key or file path the error is about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    /// Artifacts that are still valid after this error (checkpoints, previous bundle, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retained_artifacts: Vec<String>,
    /// The concrete next step, usually a command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_action: Option<String>,
    /// Stable short id derived from the error content, for correlating logs and reports.
    pub diagnostic_id: String,
}

impl RustpyError {
    pub fn new(code: ErrorCode, phase: Phase, message: impl Into<String>) -> Self {
        let mut e = Self {
            code,
            phase,
            message: message.into(),
            run_id: None,
            field: None,
            expected: None,
            actual: None,
            retained_artifacts: Vec::new(),
            next_action: None,
            diagnostic_id: String::new(),
        };
        e.refresh_id();
        e
    }

    fn refresh_id(&mut self) {
        let key = format!(
            "{}|{}|{}|{}",
            self.code.as_str(),
            self.phase,
            self.message,
            self.field.as_deref().unwrap_or("")
        );
        self.diagnostic_id = format!("d-{}", &sha256_hex(key.as_bytes())[..10]);
    }

    pub fn field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self.refresh_id();
        self
    }

    pub fn expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    pub fn actual(mut self, actual: impl Into<String>) -> Self {
        self.actual = Some(actual.into());
        self
    }

    pub fn retained(mut self, artifact: impl Into<String>) -> Self {
        self.retained_artifacts.push(artifact.into());
        self
    }

    pub fn next(mut self, action: impl Into<String>) -> Self {
        self.next_action = Some(action.into());
        self
    }

    pub fn run(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    pub fn in_phase(mut self, phase: Phase) -> Self {
        self.phase = phase;
        self.refresh_id();
        self
    }

    pub fn io(phase: Phase, path: impl AsRef<std::path::Path>, err: std::io::Error) -> Self {
        let path = path.as_ref().display().to_string();
        Self::new(ErrorCode::Io, phase, format!("I/O error on `{path}`: {err}")).field(path)
    }

    pub fn internal(phase: Phase, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, phase, message)
            .next("this is a bug in RustPy; please report it with the diagnostic id")
    }

    /// Multi-line, human-oriented rendering used by the CLI.
    pub fn render(&self) -> String {
        let mut out = format!("error[{}]: {}\n", self.code, self.message);
        out.push_str(&format!("  phase:    {}\n", self.phase));
        if let Some(run) = &self.run_id {
            out.push_str(&format!("  run:      {run}\n"));
        }
        if let Some(field) = &self.field {
            out.push_str(&format!("  field:    {field}\n"));
        }
        if let Some(expected) = &self.expected {
            out.push_str(&format!("  expected: {expected}\n"));
        }
        if let Some(actual) = &self.actual {
            out.push_str(&format!("  actual:   {actual}\n"));
        }
        for kept in &self.retained_artifacts {
            out.push_str(&format!("  kept:     {kept}\n"));
        }
        if let Some(next) = &self.next_action {
            out.push_str(&format!("  next:     {next}\n"));
        }
        out.push_str(&format!("  diagnostic: {}\n", self.diagnostic_id));
        out
    }
}

impl fmt::Display for RustpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(field) = &self.field {
            write!(f, " (field: {field})")?;
        }
        if let Some(next) = &self.next_action {
            write!(f, "; next: {next}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RustpyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_prefixed() {
        let mut seen = std::collections::BTreeSet::new();
        for code in ErrorCode::ALL {
            assert!(code.as_str().starts_with("E_"));
            assert!(seen.insert(code.as_str()), "duplicate code {}", code.as_str());
        }
    }

    #[test]
    fn error_roundtrips_through_json() {
        let e = RustpyError::new(ErrorCode::SchemaMissingColumn, Phase::Predict, "missing")
            .field("temp")
            .expected("numeric")
            .retained("bundle a")
            .next("add column");
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("E_SCHEMA_MISSING_COLUMN"));
        let back: RustpyError = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
        assert!(e.render().contains("next:"));
        assert!(e.diagnostic_id.starts_with("d-"));
    }
}
