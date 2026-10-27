//! `rustpy` — data to a verified model bundle for Rust applications.

mod commands;
mod output;
mod serve;

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "rustpy",
    version,
    about = "Train, evaluate and ship tabular binary classifiers for Rust applications",
    long_about = "RustPy turns a CSV/Parquet file into a verified model bundle that a Rust application loads with \
rustpy-runtime. One command:\n\n  rustpy train customers.csv --target churned\n  rustpy predict --input new.csv\n\nStep by step:\n\n  rustpy init defect-model --sample-data\n  cd defect-model\n  rustpy plan\n  rustpy run\n  \
rustpy compare\n  rustpy evaluate --run run-002 --split test\n  rustpy bundle build --run run-002\n  rustpy bundle check bundles/run-002"
)]
pub struct Cli {
    /// Project directory (default: the nearest directory with rustpy.toml).
    #[arg(long, global = true, value_name = "DIR")]
    pub project: Option<std::path::PathBuf>,
    /// Machine-readable JSON output (errors included).
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Args, Debug, Clone, Default)]
pub struct ConfigArgs {
    /// Named profile from [profiles.<name>] in rustpy.toml.
    #[arg(long)]
    pub profile: Option<String>,
    /// Override one setting, e.g. --set train.small-mlp.epochs=20 (repeatable).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub overrides: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a new project directory with rustpy.toml.
    Init {
        /// Project directory to create.
        name: std::path::PathBuf,
        /// Recipe (only tabular-binary is available).
        #[arg(long, default_value = "tabular-binary")]
        recipe: String,
        /// Also write a synthetic sample dataset and bind it.
        #[arg(long)]
        sample_data: bool,
    },
    /// Write the synthetic equipment-defect dataset (CC0) to a CSV file.
    SampleData {
        out: std::path::PathBuf,
        #[arg(long, default_value_t = 10_000)]
        rows: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Generate the shifted "later delivery" variant (new supplier, drift).
        #[arg(long)]
        shifted: bool,
    },
    /// Bind and inspect data.
    #[command(subcommand)]
    Data(DataCommand),
    /// Show the resolved plan: data, columns, split, preprocessing, candidates, devices.
    Plan {
        #[command(flatten)]
        config: ConfigArgs,
    },
    /// Show every effective setting and where it came from.
    Config {
        #[command(flatten)]
        config: ConfigArgs,
    },
    /// Train the configured candidates (one Run each) on the same split and transform.
    Run {
        #[command(flatten)]
        config: ConfigArgs,
        /// Comma-separated subset of candidates (default: selection.candidates).
        #[arg(long, value_delimiter = ',')]
        candidates: Option<Vec<String>>,
        /// R0: recompute split and preprocessing instead of reusing cached stages.
        #[arg(long)]
        no_cache: bool,
        /// Test hook: abort the process after N optimizer steps to simulate a crash.
        #[arg(long, hide = true)]
        simulate_crash_after_steps: Option<u64>,
        /// Test hook: request a clean cancel (like Ctrl-C) after N optimizer steps.
        #[arg(long, hide = true)]
        simulate_cancel_after_steps: Option<u64>,
    },
    /// The whole workflow in one command: bind FILE (if given), train every
    /// candidate, pick one on validation, test it once, build its bundle — and
    /// score the previously shipped bundle on the same new test rows.
    /// Creates rustpy.toml in the current directory if there is no project yet.
    Train {
        /// CSV/Parquet file. Omit to retrain on the bound file (e.g. after new data arrived).
        file: Option<std::path::PathBuf>,
        /// Target column (required the first time).
        #[arg(long)]
        target: Option<String>,
        /// Target value that means "positive" (inferred for 0/1, false/true and no/yes labels).
        #[arg(long)]
        positive: Option<String>,
        /// stratified (default) or group.
        #[arg(long)]
        split: Option<String>,
        /// Entity column (customer, device, patient): its rows never cross splits.
        #[arg(long)]
        group: Option<String>,
        /// Unique row id column (kept out of the features, copied to predictions).
        #[arg(long)]
        id: Option<String>,
        /// Columns to exclude from the features (comma-separated).
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,
        /// Comma-separated subset of candidates (default: selection.candidates).
        #[arg(long, value_delimiter = ',')]
        candidates: Option<Vec<String>>,
        #[command(flatten)]
        config: ConfigArgs,
    },
    /// List runs and their status.
    Runs,
    /// Show one run in detail.
    Status { run: String },
    /// Compare runs on the validation split (default: the latest `rustpy run`).
    Compare {
        /// Runs to compare (space- or comma-separated); default: the latest `rustpy run`.
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        runs: Vec<String>,
    },
    /// Continue an interrupted/cancelled run exactly (R2), or warm-start a new one.
    Resume {
        run: String,
        /// Start a new training from the run's weights (fresh optimizer; not an exact resume).
        #[arg(long)]
        warm_start: bool,
        #[command(flatten)]
        config: ConfigArgs,
        #[arg(long, hide = true)]
        simulate_crash_after_steps: Option<u64>,
        #[arg(long, hide = true)]
        simulate_cancel_after_steps: Option<u64>,
    },
    /// Evaluate a run (or a bundle on the current data) on a split.
    Evaluate {
        #[arg(long, conflicts_with = "bundle")]
        run: Option<String>,
        #[arg(long)]
        bundle: Option<std::path::PathBuf>,
        /// valid or test. The test split is for the final candidate only.
        #[arg(long, default_value = "valid")]
        split: String,
        /// Allow evaluating another model on already-used test rows (recorded as reuse).
        #[arg(long)]
        allow_test_reuse: bool,
        /// Evaluate even when the rows the model saw cannot be matched with this data
        /// (recorded; the estimate may be optimistic).
        #[arg(long)]
        allow_unverified_rows: bool,
        #[command(flatten)]
        config: ConfigArgs,
    },
    /// Build, check and list inference bundles.
    #[command(subcommand)]
    Bundle(BundleCommand),
    /// Predict a CSV/Parquet/JSON/JSONL file with a bundle.
    Predict {
        /// Bundle directory (default: the project's most recently built bundle).
        #[arg(long)]
        bundle: Option<std::path::PathBuf>,
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output file (.csv or .jsonl); default: CSV on stdout.
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        /// Input columns copied verbatim to the output to identify rows
        /// (comma-separated, e.g. --keep customer_id,month). The bundle's id
        /// column is always copied when the input has it.
        #[arg(long, value_delimiter = ',')]
        keep: Vec<String>,
    },
    /// Serve a bundle over HTTP: POST /predict, GET /schema, GET /health.
    Serve {
        /// Bundle directory (default: the project's most recently built bundle).
        #[arg(long)]
        bundle: Option<std::path::PathBuf>,
        /// Address to bind (use 0.0.0.0 to accept connections from other machines).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Worker threads.
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Enable POST /reload {"bundle": "<dir>"} (the new bundle must load and self-check).
        #[arg(long)]
        allow_reload: bool,
    },
    /// Write a static HTML + JSON report of the project to .rustpy/reports/.
    Report,
    /// Check the environment: version, engine, compiled backends, detected devices.
    Doctor,
}

#[derive(Subcommand, Debug)]
pub enum DataCommand {
    /// Point the project at a data file and record target/split settings in rustpy.toml.
    Bind {
        file: std::path::PathBuf,
        #[arg(long)]
        target: String,
        /// Target value that means "positive" (inferred for 0/1, false/true and no/yes labels).
        #[arg(long)]
        positive: Option<String>,
        /// stratified (independent rows) or group (repeated entities).
        #[arg(long)]
        split: Option<String>,
        /// Entity column for grouped splitting (implies --split group).
        #[arg(long)]
        group: Option<String>,
        /// Unique row id column (never used as a feature).
        #[arg(long)]
        id: Option<String>,
        /// Columns to exclude from the features (comma-separated).
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,
    },
    /// Show column types, missing values and label distribution of the bound data.
    Inspect {
        #[command(flatten)]
        config: ConfigArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum BundleCommand {
    /// Build an immutable bundle from a completed, test-evaluated run.
    Build {
        #[arg(long)]
        run: String,
        /// Device the bundle is built for (only cpu in this release).
        #[arg(long, alias = "profile", default_value = "cpu")]
        device: String,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        #[arg(long)]
        allow_unevaluated: bool,
    },
    /// Verify a bundle on this machine: hashes, versions, schema, example predictions.
    Check {
        dir: std::path::PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// List bundles in this project.
    List,
}

fn main() -> ExitCode {
    let json = std::env::args().any(|a| a == "--json");
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            use clap::error::ErrorKind;
            if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) || !json {
                e.exit();
            }
            // Usage errors in JSON mode use the same envelope as every other error.
            let err = rustpy_contracts::RustpyError::new(
                rustpy_contracts::ErrorCode::ConfigInvalid,
                rustpy_contracts::Phase::Config,
                e.kind().to_string(),
            )
            .actual(e.to_string().lines().next().unwrap_or_default().to_string())
            .next("see `rustpy --help` or `rustpy <command> --help`");
            output::print_error(&err, true);
            return ExitCode::from(2);
        }
    };
    match commands::dispatch(cli) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            output::print_error(&e, json);
            ExitCode::from(e.code.exit_code().clamp(1, 255) as u8)
        }
    }
}
