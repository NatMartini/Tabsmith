# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, minor versions may contain breaking changes.

## [Unreleased]

## [0.1.0] - Unreleased

First public release: an early CPU prototype for one job, tabular binary
classification, from a local CSV/Parquet file to a verified model loaded by a
Rust application. The tested guarantees and the known gaps are listed in the
README.

### Added

- **CLI workflow** (`rustpy`): `init` (optionally with a synthetic CC0 sample
  dataset), `data bind` / `data inspect`, `plan`, `config`, `run`, `runs`,
  `status`, `compare`, `resume`, `evaluate`, `bundle build` / `check` / `list`,
  `predict` (CSV, Parquet, JSON and JSONL input), `report` and `doctor`. One
  `rustpy.toml` with profiles and `--set` overrides, each effective value
  recording its source; `--json` output and exit codes per error family.
- **Shared preprocessing**: statistics are fitted on training rows only, and the
  same frozen transform encodes training, evaluation, CLI prediction and in-app
  prediction.
- **Leakage-safe splits**: stratified and grouped (`--group`) splits; rows keep
  their split as the data file grows (keyed by `--id` or by row content); a test
  split is used once, by the final candidate, and any reuse is refused unless
  explicitly allowed and recorded.
- **Models on the CPU**: a linear (logistic) baseline and a small MLP, trained
  with Burn 0.22.1 (ndarray backend, FP32) on the same split and transform, and
  compared on the validation split.
- **Exact resume**: an interrupted or cancelled run continues to the same weights
  and history as an uninterrupted run (same binary, same CPU features);
  `resume --warm-start` starts a new run from existing weights instead.
- **Bundles**: immutable, content-addressed directories with weights,
  preprocessing, input/output schemas, evaluation evidence and synthetic
  examples, verified file by file (SHA-256) before anything is loaded.
- **Runtime-only loading**: `rustpy-runtime` loads bundles without Burn or
  Polars (`Predictor`, `Record`, `predict_serialize`); `examples/predict-app`
  shows a complete application.
- **Update and rollback**: `PredictorHandle::try_update` swaps in a new bundle
  only after it loads and passes its self-check, refuses breaking input/output
  contract changes unless allowed, and reports every change; `rollback_to`
  returns to an earlier bundle.
- **Reports**: an offline HTML and JSON report of the project; the comparison
  shows quality, latency and per-run peak memory (Linux) side by side.
- **Errors**: `RustpyError` with a stable code, phase, field, expected/actual
  values and the next step to take.
- **Distribution**: dual MIT/Apache-2.0 license, CI on Linux, macOS and Windows,
  prebuilt binaries for five targets with `install.sh`, `install.ps1` and
  cargo-binstall metadata, and crates prepared for crates.io
  (`cargo install rustpy-cli` installs the `rustpy` command).

### Known limitations

- No GPU backend is compiled in or certified; explicitly requesting a GPU device
  is an error (no silent fallback to the CPU).
- Before this release only Linux x86_64 had been built and tested by the
  maintainers; macOS and Windows are covered by CI from this release on.
- Regression, multiclass, time-based splits and tree models are not implemented.

[Unreleased]: https://github.com/__OWNER__/__REPO__/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/__OWNER__/__REPO__/releases/tag/v0.1.0
