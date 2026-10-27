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
$ rustpy train customers.csv --target churned
· data customers.csv: 2000 rows × 10 columns, target `churned` (positive "yes": 412 rows, negative "no": 1588 rows)
· run-001: training linear … completed (19 epochs; weights of best epoch 9 kept)
· run-002: training small-mlp … completed (23 epochs; weights of best epoch 15 kept)

comparison on the validation split (metric: balanced_accuracy)
  run      candidate  balanced_accuracy  AUC     F1      params  p95 latency  train time
  run-001  linear     0.7861             0.8653  0.6012  13      9 µs         61 ms
  run-002  small-mlp  0.8043             0.8790  0.6230  1121    11 µs        240 ms

chosen on validation: run-002 (small-mlp)
test split (300 rows, used once): balanced accuracy 0.7967  AUC 0.8712  F1 0.6154
bundle: bundles/run-002 (b-3f9c2a51d0e4b7a1, evaluated; self-check passed)

$ rustpy predict --input new_customers.csv --keep customer_id   # CSV of predictions
$ rustpy serve                                                  # HTTP API on :8080
```

*(Illustrative output; your numbers depend on your data.)*

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
- **Fast where it matters in production.** Single-record prediction in microseconds, a few milliseconds from process start to first prediction, a few MB of memory ([numbers below](#rustpy-vs-python)).
- **Same quality as scikit-learn.** Preprocessing reproduces scikit-learn's to 2e-7; the linear model lands within 0.15 % of scikit-learn's optimum of the same objective; metrics match to 1e-12.
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

From source: `git clone … && cargo install --path crates/rustpy-cli`.

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
    let model = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
    let p = model.predict_one(
        &Record::new()
            .with("plan", "pro")
            .with("tenure_months", 18)
            .with("monthly_fee", 42.5)
            .with("support_calls", None::<f64>), // missing values are imputed as in training
    )?;
    println!("{} (p = {:.3})", p.label, p.probability);
    Ok(())
}
```

Or predict from your own `#[derive(Serialize)]` struct with `predict_serialize`.
Errors carry a code, the field, what was expected and what to do next; `{:?}` prints
the full report. Hot-swapping a model in a running service: `PredictorHandle::try_update`
switches only after the new bundle loads and passes its self-check (and refuses a
changed input/output contract unless you allow it); `rollback_to` goes back.
See [examples/predict-app](examples/predict-app).

### Serve it over HTTP

```sh
rustpy serve --port 8080
curl -s localhost:8080/predict -d '{"plan":"pro","tenure_months":18,"monthly_fee":42.5,"support_calls":null}'
# {"bundle_id":"b-…","predictions":[{"label":"no","probability":0.1234,"is_positive":false}]}
```

Endpoints: `POST /predict` (one record, an array, or `{"records": [...]}`),
`GET /schema`, `GET /health`, and `POST /reload` with `--allow-reload`.

### When new data arrives

Append to (or replace) the file and run `rustpy train` again:

```console
$ rustpy train
· new data version: customers.csv is now snapshot s-…, while run-001, run-002 used snapshot s-…
…
chosen on validation: run-004 (small-mlp)
test split (420 rows, used once): balanced accuracy 0.8120 …
previous bundle bundles/run-002 (run-002) on the same test rows (390 rows it never saw): balanced accuracy 0.7710 …
  → the new model is better by 0.0410 balanced accuracy
bundle: bundles/run-004 (…)
```

`predict` warns about categories the model has never seen (a new month, product or code).

## RustPy vs Python

The same task done with the standard Python stack (pandas + scikit-learn
`ColumnTransformer` + `LogisticRegression` / `MLPClassifier`), on the same rows, split,
features and training settings, same machine. Reproduce with
[`tools/bench/bench.py`](tools/bench/bench.py); full results in
[reports/benchmarks](reports/benchmarks/).

<!-- BENCH:START -->
<!-- BENCH:END -->

What the table says: for training, RustPy and scikit-learn are on par in speed and
quality. The difference is in **shipping the model**: no interpreter to start, no
DataFrame per request, no 500 MB environment to deploy.

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

RustPy 0.1 is young. What is solid and tested (90+ automated tests, including
process-level ones): the CPU workflow above, exact resume, bundles and the runtime,
reference checks against scikit-learn.

Not done yet, by design or for lack of hardware:

- **GPU**: no CUDA or other GPU backend is compiled in. `auto-certified` selects the CPU and says so; asking for `cuda` is an error, never a silent fallback.
- **Platforms**: tested on Linux x86_64. macOS and Windows builds are produced by CI; the CPU path is marked "not certified" there until the full test suite has run on them.
- **Tasks**: binary classification only. Regression and multi-class are next; time-based splits are refused rather than faked.
- **Users**: no formal user study yet — feedback and issues are very welcome.

See the [implementation backlog](docs/implementation-backlog.md) for the full,
honest status of every planned item, and the [acceptance report](reports/implementation-acceptance.md)
for measured evidence.

## Documentation

[Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) ·
[Bundle format](docs/bundle-format.md) · [Errors](docs/errors.md) ·
[Extension walkthrough (LLM task)](docs/extension-walkthrough-llm.md) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. The bundled sample dataset generator produces synthetic data (CC0).
