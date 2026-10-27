<div align="center">

# RustPy

**Train a model from a CSV and ship it inside your Rust application — one tool, no Python.**

[![CI](https://github.com/__OWNER__/__REPO__/actions/workflows/ci.yml/badge.svg)](https://github.com/__OWNER__/__REPO__/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/__OWNER__/__REPO__?include_prereleases)](https://github.com/__OWNER__/__REPO__/releases)
[![crates.io](https://img.shields.io/crates/v/rustpy-runtime.svg?label=rustpy-runtime)](https://crates.io/crates/rustpy-runtime)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[中文说明](README.zh-CN.md) · [Install](#install) · [Quick start](#quick-start) · [RustPy vs Python](#rustpy-vs-python) · [Docs](#documentation)

</div>

```console
$ rustpy sample-data machines.csv
$ rustpy train machines.csv --target defect --group device_id
· data machines.csv: 10000 rows × 25 columns, target `defect` (positive "1": 1289 rows, negative "0": 8711 rows)
· run-001: training linear (up to 1740 steps)
· run-001: completed (early stopping after epoch 39; weights of best epoch 29 kept)
· run-002: training small-mlp (up to 3480 steps)
· run-002: completed (early stopping after epoch 13; weights of best epoch 5 kept)

comparison on the validation split (metric: balanced_accuracy)
  run      candidate  balanced_accuracy  AUC     F1      params  p95 latency  train time
  run-001  linear     0.7644             0.8385  0.4602  36      1 µs         147 ms
  run-002  small-mlp  0.7665             0.8427  0.4843  1697    1 µs         133 ms

chosen on validation: run-002 (small-mlp)
test split (1386 rows, used once): balanced accuracy 0.7729  AUC 0.8250  F1 0.4624  precision 0.3325  recall 0.7588
bundle: bundles/run-002 (b-5472199517c4e8d7, evaluated; self-check passed)

$ rustpy predict --input new.csv --keep unit_id   # CSV of predictions
$ rustpy serve                                    # HTTP API on :8080
```

*(Real output of the Quick start below, abridged: progress lines and some table
columns are left out. Full output in the [acceptance report](reports/implementation-acceptance.md#one-command-rustpy-train).)*

RustPy turns a CSV or Parquet file into a **tested, self-contained model bundle**
that any Rust program can load with a small dependency, `rustpy-runtime`.
It covers the whole job, not just the model: it splits the data without leaking it,
learns the preprocessing from the training rows only, trains a strong linear baseline
and a small neural network, compares them fairly, uses the test set once, and packages
the winner with its preprocessing, schema and evidence. When new data arrives,
it retrains and tells you whether the new model beats the one you shipped,
on rows neither of them has seen.

Release 0.1 does **tabular binary classification on CPU** (churn, defect, fraud,
lead scoring, …). See [status](#status) for what is not done yet.

## Why RustPy

- **One command, safe defaults.** `rustpy train data.csv --target y` does split → preprocess → train two models → compare on validation → test once → bundle. Every step is recorded and reproducible.
- **Ship without Python.** Your application depends on `rustpy-runtime` only (no Burn, no Polars, no Python). The example app is a 1.6 MB binary.
- **Fast where it matters in production.** Single-record prediction in microseconds, about a millisecond from process start to first prediction, about 12 MB of memory ([numbers below](#rustpy-vs-python)).
- **Same quality as scikit-learn.** Preprocessing matches scikit-learn's to about 2e-7, the linear model lands within about 0.1 % of scikit-learn's optimum of the same objective, and metrics match to 1e-12 ([measured](reports/implementation-acceptance.md#reference-check-against-scikit-learn); CI guards 1e-5, 1 % and 1e-12 with [frozen goldens](fixtures/reference/)).
- **Mistakes are hard to make.** No train/test leakage by construction, one use of the test set per data version, entity-aware splits, refusal instead of silent fallback, integrity-checked bundles, and errors that say what to do next.
- **Updates you can trust.** New data → new runs; the old bundle is scored on the same unseen rows; applications swap bundles only after the new one passes its self-check, and can roll back.

## Install

Prebuilt binaries (Linux x86_64/arm64, macOS Intel/Apple Silicon, Windows):

```sh
# Linux / macOS
curl -fsSL https://github.com/__OWNER__/__REPO__/releases/latest/download/install.sh | sh
# Windows (PowerShell)
irm https://github.com/__OWNER__/__REPO__/releases/latest/download/install.ps1 | iex
```

With Cargo (Rust 1.95+):

```sh
cargo install rustpy-cli          # builds the `rustpy` binary
cargo binstall rustpy-cli         # or download the prebuilt one
```

From source:

```sh
git clone https://github.com/__OWNER__/__REPO__ rustpy && cd rustpy
cargo install --path crates/rustpy-cli --locked
```

## Quick start

No data at hand? Generate a realistic synthetic dataset (CC0) and train on it:

```sh
rustpy sample-data machines.csv                       # 10,000 rows, target `defect`
rustpy train machines.csv --target defect --group device_id
rustpy sample-data new.csv --rows 200 --seed 7       # some new rows
rustpy predict --input new.csv --keep unit_id --output predictions.csv
rustpy report                                         # HTML report in .rustpy/reports/
```

`--group device_id` keeps all rows of one device in the same split (use it for
customers, patients, devices — anything with repeated rows). If every row has a
unique key, pass `--id <column>` instead.

### Use the model in your Rust application

```toml
[dependencies]
rustpy-runtime = "0.1"
```

```rust
use rustpy_runtime::{LoadOptions, Predictor, Record};

fn main() -> Result<(), rustpy_runtime::RustpyError> {
    // The bundle `rustpy train` printed (Quick start: the machine-defect sample data).
    let model = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
    let reading = Record::new()
        .with("line", "B")
        .with("shift", "night")
        .with("supplier", "S3")
        .with("hours_in_service", 12_000)
        .with("temp_mean", 71.4)
        .with("vibration_rms", 1.42)
        .with("calibration_offset", None::<f64>) // missing values are imputed as in training
        /* … every other input field of the bundle … */;
    let p = model.predict_one(&reading)?;
    println!("{} (p = {:.3})", p.label, p.probability);
    Ok(())
}
```

Every input field of the bundle is required (`None` marks a missing value); the
bundle's `input.schema.json` lists them, and `examples/input.json` holds complete
records.

Or predict from your own `#[derive(Serialize)]` struct with `predict_serialize`.
Errors carry a code, the field, what was expected and what to do next; `{:?}` prints
the full report. Hot-swapping a model in a running service: `PredictorHandle::try_update`
switches only after the new bundle loads and passes its self-check (and refuses a
changed input/output contract unless you allow it); `rollback_to` goes back.
See [examples/predict-app](examples/predict-app).

### Serve it over HTTP

```sh
rustpy serve --port 8080              # the newest evaluated bundle, or --bundle <dir>
curl -s localhost:8080/predict --data-binary @bundles/run-002/examples/input.json
# {"bundle_id":"b-…","predictions":[{"is_positive":false,"label":"0","probability":0.2474},…]}
```

Endpoints: `POST /predict` (one record, an array, or `{"records": [...]}`),
`GET /schema`, `GET /health`, and `POST /reload` with `--allow-reload`.

### When new data arrives

Append to (or replace) the file and run `rustpy train` again:

```console
$ rustpy train
· new data version: machines.csv is now snapshot s-dc9fc115c6c7, while run-001, run-002, … used snapshot s-0a8635d0d212. …
…
chosen on validation: run-006 (small-mlp)
test split (1844 rows, used once): balanced accuracy 0.7567  AUC 0.8282  F1 0.4688  precision 0.3355  recall 0.7778
previous bundle bundles/run-002 (run-002) on the 1844 test rows it never saw: balanced accuracy 0.7737  AUC 0.8334  F1 0.4936  …
new model run-006 on the same 1844 rows: balanced accuracy 0.7567  AUC 0.8282  F1 0.4688  …
  → about the same: the difference (-0.0170) is below the ±0.023 resolution of 1844 rows
  `rustpy predict` uses the newest evaluated bundle; to keep serving the previous one pass --bundle bundles/run-002
bundle: bundles/run-006 (b-c9e177f6b39cdb5e, evaluated; self-check passed)
```

*(Abridged output of the [acceptance run](reports/implementation-acceptance.md#one-command-rustpy-train),
after 3,000 rows from a shifted distribution were appended; your run ids may differ.)*
Here the shipped model is still slightly better on the same unseen rows, and the
difference is within the noise, so you can keep serving it. Run `rustpy train`
again on unchanged data and it keeps the existing bundle instead of building a copy.

`predict` warns about categories the model has never seen (a new month, product or code).

## RustPy vs Python

The same task done with the standard Python stack (pandas + scikit-learn
`ColumnTransformer` + `LogisticRegression` / `MLPClassifier`), on the same rows, split,
features and training settings, same machine. Reproduce with
[`tools/bench/bench.py`](tools/bench/bench.py); full results and logs in
[reports/benchmarks](reports/benchmarks/2026-10-27-linux-x86_64/).

<!-- BENCH:START -->
Machine: AMD Ryzen 9 9950X3D2 16-Core Processor, 32 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. Data: 10000 rows (synthetic equipment-defect sample), same split, features and settings on both sides. Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4.

| | RustPy linear | scikit-learn LogisticRegression | RustPy small-mlp | scikit-learn MLPClassifier |
|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 |
| Epochs / solver iterations | 39 | 25 | 13 | 13 |
| Training time | 0.15 s | 0.03 s | 0.14 s | 0.15 s |
| Single-record prediction in an app (p50) | 2.5 µs | 2004.0 µs | 2.8 µs | 2069.9 µs |
| Single-record prediction (p99) | 4.3 µs | 2587.8 µs | 5.3 µs | 3397.3 µs |
| Batch prediction in process | 825,063 rows/s | 805,501 rows/s | 685,760 rows/s | 353,438 rows/s |
| Cold start: new process, load model, predict 1 record | 1 ms | 637 ms | 1 ms | 639 ms |
| Peak memory of that process | 12 MB | 160 MB | 12 MB | 161 MB |
| Model artifact | 0.0 MB | 0.0 MB | 0.0 MB | 0.1 MB |

| Deployment footprint | RustPy | Python |
|---|---|---|
| What the application ships | 1.6 MB binary (depends on rustpy-runtime only) | 354.2 MB environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |
| Training tool | 28.7 MB `rustpy` binary | the same environment |
| Train both candidates from the CSV | 0.38 s for the whole `rustpy run` process (read, profile, split, preprocess, train, validate, checkpoints), peak 44 MB | 0.19 s for read_csv + both fits (imports and process start not included) |
<!-- BENCH:END -->

What the table says: quality is on par (the small MLP did better than
`MLPClassifier` here), and training takes a fraction of a second on both sides,
with scikit-learn's lbfgs fitting the linear model faster than RustPy's minibatch
Adam. The difference is in **shipping the model**: single-record prediction in an
application takes about 2.5 µs instead of 2 ms, a new process predicts after 1 ms
instead of 0.64 s, in 12 MB instead of 160 MB, and the application is a 1.6 MB
binary instead of a 354 MB Python environment.

### What you would otherwise write yourself

| | RustPy | Typical Python project |
|---|---|---|
| Preprocessing fitted on training rows only, applied identically at inference | built in, one implementation | your code (a frequent source of leakage and train/serve skew) |
| Entity-aware split (no customer in both train and test) | `--group` | your code |
| Test set used once per data version, recorded | enforced | discipline |
| Old vs new model on the same unseen rows after a data update | automatic in `rustpy train` | your code |
| Model + preprocessing + schema + label mapping + evidence in one verified artifact | the bundle | pickle/joblib + conventions |
| Exact, bit-identical resume after a crash | `rustpy resume` | rarely available |
| Actionable errors with a stable code and the next command | every error | stack traces |

## How it works

```
rustpy-contracts   config layering, errors, schemas, run/bundle/evaluation records
rustpy-runtime     bundle verification, the shared transform, model execution, Predictor   ← your app depends on this
rustpy-workflow    snapshots, splits, fitting, training (Burn, CPU), evaluation, bundles, reports
rustpy-cli         the `rustpy` binary (train, run, compare, evaluate, bundle, predict, serve, report, …)
```

Every step has a step-by-step command too (`rustpy plan`, `run`, `compare`,
`evaluate`, `bundle build`, `resume`, …) — see `rustpy --help` and
[the docs](#documentation).

## Status

RustPy 0.1 is young. What is solid and tested (89 automated tests and a doctest,
including process-level ones that drive the real binary): the CPU workflow above,
exact resume, bundles and the runtime, and reference checks against scikit-learn.
The [acceptance report](reports/implementation-acceptance.md) has measured
evidence for each of these.

Not done yet, by design or for lack of hardware:

- **GPU**: no CUDA or other GPU backend is compiled in. `auto-certified` selects the CPU and says so; asking for `cuda` is an error, never a silent fallback.
- **Platforms**: certified on Linux x86_64, where the acceptance run above was made. The full test suite also passes on Windows x86_64 and runs in CI on macOS (Apple Silicon); both are reported as "not certified" by design in 0.1. The Linux arm64 and macOS Intel binaries are built but not tested.
- **Tasks**: binary classification only. Regression and multi-class are next; time-based splits are refused rather than faked.
- **Users**: no formal user study yet — feedback and issues are very welcome.

The [implementation backlog](docs/implementation-backlog.md) lists the status of
every planned item.

## Documentation

[Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) ·
[Bundle format](docs/bundle-format.md) · [Errors](docs/errors.md) ·
[Extension walkthrough (LLM task)](docs/extension-walkthrough-llm.md) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Reports](reports/)

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. The bundled sample dataset generator produces synthetic data (CC0).
