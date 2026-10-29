//! `tabsmith` — data to a verified model bundle for Rust applications.

mod commands;
mod output;
mod serve;

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "tabsmith",
    version,
    about = "Train, evaluate and ship tabular binary classifiers for Rust applications",
    long_about = "Tabsmith turns a CSV/Parquet file into a verified model bundle that a Rust application loads with \
tabsmith-runtime. One command:\n\n  tabsmith train customers.csv --target churned\n  tabsmith predict --input new.csv\n\nStep by step:\n\n  tabsmith init defect-model --sample-data\n  cd defect-model\n  tabsmith plan\n  tabsmith run\n  \
tabsmith compare\n  tabsmith evaluate --run run-002 --split test\n  tabsmith bundle build --run run-002\n  tabsmith bundle check bundles/run-002"
)]
pub struct Cli {
    /// Project directory (default: the nearest directory with tabsmith.toml).
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
    /// Named profile from [profiles.<name>] in tabsmith.toml.
    #[arg(long)]
    pub profile: Option<String>,
    /// Override one setting, e.g. --set train.small-mlp.epochs=20 (repeatable).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub overrides: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a new project directory with tabsmith.toml.
    Init {
        /// Project directory to create.
        name: std::path::PathBuf,
        /// Recipe: tabular-binary, tabular-multiclass or tabular-regression.
        #[arg(long, default_value = "tabular-binary")]
        recipe: String,
        /// Also write a synthetic sample dataset and bind it.
        #[arg(long)]
        sample_data: bool,
    },
    /// Write the synthetic equipment dataset (CC0) to a CSV file.
    SampleData {
        /// CSV file to write.
        out: std::path::PathBuf,
        /// Which target to write: binary (`defect`), multiclass (`failure_mode`) or
        /// regression (`remaining_hours`).
        #[arg(long, default_value = "binary")]
        task: String,
        /// Number of rows.
        #[arg(long, default_value_t = 10_000)]
        rows: usize,
        /// Seed of the generator: the same seed writes the same file.
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
    /// score the previously shipped bundle and the new model on the same unseen test rows.
    /// Creates tabsmith.toml in the current directory if there is no project yet.
    Train {
        /// CSV/Parquet file. Omit to retrain on the bound file (e.g. after new data arrived).
        file: Option<std::path::PathBuf>,
        /// Target column (required the first time).
        #[arg(long)]
        target: Option<String>,
        /// Target value that means "positive" (binary; inferred for 0/1, false/true and no/yes labels).
        #[arg(long)]
        positive: Option<String>,
        /// binary (two labels), multiclass (3 to 100 labels) or regression (a number);
        /// sets the project's recipe.
        #[arg(long)]
        task: Option<String>,
        /// stratified (default), group or time.
        #[arg(long)]
        split: Option<String>,
        /// Entity column (customer, device, patient): its rows never cross splits.
        #[arg(long)]
        group: Option<String>,
        /// Time column (a timestamp, a date, a counter): the earliest rows train and the
        /// latest are the test (implies --split time).
        #[arg(long)]
        time: Option<String>,
        /// Unique row id column (kept out of the features, copied to predictions).
        #[arg(long)]
        id: Option<String>,
        /// Columns to exclude from the features (comma-separated; added to the existing data.exclude).
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
    /// Compare runs on the validation split (default: the latest `tabsmith run`).
    Compare {
        /// Runs to compare (space- or comma-separated); default: the latest `tabsmith run`.
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
        /// Run to evaluate (e.g. run-002).
        #[arg(long, conflicts_with = "bundle")]
        run: Option<String>,
        /// Bundle directory to evaluate on the project's current data instead of a run.
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
        /// Bundle directory (default: the project's newest evaluated bundle, or its newest bundle when none was evaluated).
        #[arg(long)]
        bundle: Option<std::path::PathBuf>,
        /// Records to predict: .csv, .parquet, .json (one record or an array) or .jsonl (one per line).
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
        /// Bundle directory (default: the project's newest evaluated bundle, or its newest bundle when none was evaluated).
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
        /// Enable POST /reload {"bundle": "<dir>"} for bundles in the served bundle's folder (the new bundle must load and self-check).
        #[arg(long)]
        allow_reload: bool,
    },
    /// Write a static HTML + JSON report of the project to .tabsmith/reports/.
    Report,
    /// Check the environment: version, engine, compiled backends, detected devices.
    Doctor,
}

#[derive(Subcommand, Debug)]
pub enum DataCommand {
    /// Point the project at a data file and record target/split settings in tabsmith.toml.
    Bind {
        /// CSV or Parquet file.
        file: std::path::PathBuf,
        /// Target column: the labels (classification) or the number (regression) to predict.
        #[arg(long)]
        target: String,
        /// Target value that means "positive" (binary; inferred for 0/1, false/true and no/yes labels).
        #[arg(long)]
        positive: Option<String>,
        /// binary, multiclass or regression; sets the project's recipe.
        #[arg(long)]
        task: Option<String>,
        /// stratified (independent rows), group (repeated entities) or time (rows in time order).
        #[arg(long)]
        split: Option<String>,
        /// Entity column for grouped splitting (implies --split group).
        #[arg(long)]
        group: Option<String>,
        /// Time column for a chronological split (implies --split time).
        #[arg(long)]
        time: Option<String>,
        /// Unique row id column (never used as a feature).
        #[arg(long)]
        id: Option<String>,
        /// Columns to exclude from the features (comma-separated; added to the existing data.exclude).
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
        /// Completed run to package (e.g. run-002).
        #[arg(long)]
        run: String,
        /// Device the bundle is built for (only cpu in this release).
        #[arg(long, alias = "profile", default_value = "cpu")]
        device: String,
        /// Bundle directory to create (default: bundles/<run-id> in the project).
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        /// Build from a run that was not evaluated on the test split; the bundle is
        /// marked `unevaluated-candidate`.
        #[arg(long)]
        allow_unevaluated: bool,
    },
    /// Verify a bundle on this machine: hashes, versions, schema, example predictions.
    Check {
        /// Bundle directory.
        dir: std::path::PathBuf,
        /// Device to check the bundle for (only cpu in this release).
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
            let err = tabsmith_contracts::TabsmithError::new(
                tabsmith_contracts::ErrorCode::ConfigInvalid,
                tabsmith_contracts::Phase::Config,
                e.kind().to_string(),
            )
            .actual(e.to_string().lines().next().unwrap_or_default().to_string())
            .next("see `tabsmith --help` or `tabsmith <command> --help`");
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
