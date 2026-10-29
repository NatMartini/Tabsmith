# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, minor versions may contain breaking changes.

## [Unreleased]

Bundles that carry monotone constraints need this version of the runtime
(`min_runtime_version` 0.2.0); every other bundle still loads with 0.1.0.

### Added

- **What the compiler knows about a compiled model**: `model!` generates the Kani
  proof harnesses itself (`kani` option; loop bounds from the model's shape), the
  bounds of the raw outputs over every input or over a box of feature values
  (`OUTPUT_BOUNDS`, `output_bounds`, `feature_box`; sound `const fn`s in
  `tabsmith-embedded`), the probability or value bounds derived from them, the
  arithmetic of one prediction (`WORST_CASE`), and a compile-time check of the trees'
  monotone constraints. `Model` implements the `tabsmith_embedded::Model` trait, as
  the runtime's `Predictor` does.
- **Monotone constraints** for the trees (`train.gbdt.trees.monotone`, by numeric
  column): kept by construction, recorded in the bundle, checked by the bundle loader
  and by the compiler.
- **Chronological splits** (`--time <column>` / `split = "time"`): the earliest rows
  train, the latest are the test, rows with the same time stay together; a declared
  time column with any other split is still refused.
- **Conformal prediction**: every bundle built with its project's data carries
  `conformal.json`, calibrated on the validation split (alpha 0.1), and every
  prediction then has an `interval` (regression) or a `prediction_set`
  (classification), in the runtime, the CLI output and the compiled model
  (`CONFORMAL`, `interval`, `prediction_set`).
- **Permutation importance** of every input field (`importance.json`, the drop of
  the selection metric on the validation split when the field is shuffled), listed
  by `tabsmith bundle build`.
- **Drift monitoring**: `tabsmith serve` answers `GET /stats` with the inputs seen
  since the bundle was loaded against the training statistics in the bundle, and
  the fields that drifted.
- **The same trees on every operating system**: the trees' training and the losses
  take `exp` and `log` from `libm`, like inference, instead of the platform C library
  (measured: identical weights and history on Windows and Linux, `reports/reproducibility/`).
- **Trees on every core**: binning, histograms and the application of a grown tree run
  in parallel; sums run in fixed chunks of 16,384 rows merged in order, so the trees
  do not depend on the core count.

### Changed

- Retraining the trees on data that 0.1.0 trained gives the same trees except in two
  places: a node with more than 16,384 rows now sums its histogram in chunks (last
  bits of its leaf values), and the starting score uses `libm`'s `log` (last bit). The
  device plan reports the trees' thread count instead of 1.
- `E_SPLIT_UNSUPPORTED` now means "a time column is declared but the split is not
  `time`"; the message says so.
- The workspace version is 0.2.0 (unreleased); a bundle with monotone constraints
  records `min_runtime_version` 0.2.0.

## [0.1.0] - 2026-10-28

First public release: supervised learning on one table (binary classification,
multiclass classification and regression) from a local CSV/Parquet file to a
verified model loaded by a Rust application. The tested guarantees and the known
gaps are listed in the README; the reasons behind the design are in
`docs/design-decisions.md`, and `docs/technical-report.md` describes the whole
system with its evidence.

### Added

- **One command** (`tabsmith train data.csv --target y`): split, preprocess,
  train every candidate, choose on validation, test once and bundle. Run again
  after new data arrives and it scores the previous bundle and the new model on
  the same unseen test rows; run it again on unchanged data and it keeps the
  existing bundle.
- **Serving**: `tabsmith serve` exposes a bundle over HTTP/JSON (`POST /predict`,
  `GET /schema`, `GET /health`, and `POST /reload` with `--allow-reload`, which
  swaps only to a bundle that passes its self-check).
- **CLI workflow** (`tabsmith`): `init` (optionally with a synthetic CC0 sample
  dataset), `sample-data`, `data bind` / `data inspect`, `plan`, `config`,
  `run`, `runs`, `status`, `compare`, `resume`, `evaluate`, `bundle build` /
  `check` / `list`, `predict` (CSV, Parquet, JSON and JSONL input; defaults to
  the newest evaluated bundle; `--keep` copies key columns verbatim), `report`
  and `doctor`. One `tabsmith.toml` with profiles and `--set` overrides, each
  effective value recording its source; `--json` output and exit codes per
  error family.
- **Shared preprocessing**: statistics are fitted on training rows only, and the
  same frozen transform encodes training, evaluation, CLI prediction and in-app
  prediction.
- **Leakage-safe splits**: stratified and grouped (`--group`) splits; rows keep
  their split as the data file grows (keyed by `--id` or by row content); a test
  split is used once, by the final candidate, and any reuse is refused unless
  explicitly allowed and recorded.
- **Three tasks**: the recipes `tabular-binary@0.1`, `tabular-multiclass@0.1`
  (3 to 100 classes) and `tabular-regression@0.1` (`--task` on `train` and
  `data bind`; `sample-data --task` writes a matching synthetic target). Metrics,
  selection (`selection.metric`: classification metrics, or `rmse` / `mae` / `r2`),
  bundles and predictions follow the task: class probabilities for multiclass,
  values in the target's units for regression.
- **Three candidates**: a linear baseline and a small MLP, trained with Burn 0.22.1
  (Flex CPU backend, single thread, FP32), and `gbdt`, Tabsmith's own
  deterministic histogram gradient-boosted trees (model format
  `tabsmith.trees.v1`, executed by the runtime with no numerical difference). All
  train on the same split and transform and are compared on the validation split.
- **GPU training (opt-in)**: Cargo features `vulkan` and `cuda` (`gpu` on the CLI)
  train the networks with Burn's CubeCL backends, with kernel fusion and without
  autotuning (which made CUDA nondeterministic across processes). CUDA computes
  f32 matrix products in TF32 on compute capability 8.0+, so it requires
  `execution.precision = "tf32"`; Vulkan computes fp32. `execution.gpu` picks the
  GPU; `auto-certified` uses only GPUs that `tools/gpu_certify.py` certified
  (determinism, exact resume, CPU agreement, bundle) and only when a cost model
  expects a 2x speed-up.
- **Exact resume**: an interrupted or cancelled run continues to the same weights
  and history as an uninterrupted run (same binary, same CPU features);
  `resume --warm-start` starts a new run from existing weights instead.
- **Bundles**: immutable, content-addressed directories with weights,
  preprocessing, input/output schemas, evaluation evidence and synthetic
  examples, verified file by file (SHA-256) before anything is loaded.
- **Runtime-only loading**: `tabsmith-runtime` loads bundles without Burn or
  Polars (`Predictor`, `Record`, `predict_serialize`, and `from_memory` for bundles
  that are not on disk); `examples/predict-app` shows a complete application, and
  `examples/wasm-predict` runs the unchanged runtime in a web page (WebAssembly).
- **Models compiled into code**: `tabsmith-macros` (`model!`) turns a bundle into a
  Rust module while the application builds. The bundle is verified exactly as the
  runtime verifies it; the input contract becomes types, so a misspelled field or
  category does not compile; the weights become constants; and inference is `no_std`,
  allocates nothing, and predicts the same bits as the runtime. The kernels moved into
  `tabsmith-embedded`, which both use; `exp` comes from `libm`, so predictions are the
  same on every platform (tested on Linux, macOS and Windows in CI).
  `examples/embedded-predict` builds for a Cortex-M4F, and Kani proves that its
  prediction never panics, for every input.
- **Update and rollback**: `PredictorHandle::try_update` swaps in a new bundle
  only after it loads and passes its self-check, refuses breaking input/output
  contract changes unless allowed, and reports every change; `rollback_to`
  returns to an earlier bundle.
- **Reports**: an offline HTML and JSON report of the project; the comparison
  shows quality, latency and per-run peak memory (Linux) side by side.
- **Errors**: `TabsmithError` with a stable code, phase, field, expected/actual
  values and the next step to take.
- **Assurance**: `unsafe_code = "forbid"` in every workspace crate; property tests
  (proptest) for predictions, metrics, splits, epoch orders and damaged bundles;
  a cargo-fuzz target for the bundle loader (`fuzz/`); clippy and rustdoc warnings
  fail CI, GPU feature builds included.
- **Distribution**: dual MIT/Apache-2.0 license, CI on Linux, macOS and Windows,
  prebuilt binaries for five targets with `install.sh`, `install.ps1` and
  cargo-binstall metadata, and crates prepared for crates.io
  (`cargo install tabsmith-cli` installs the `tabsmith` command).

### Known limitations

- GPU paths are certified only on the hardware they were run on (see
  `reports/gpu/`); the prebuilt binaries are CPU-only, and GPU training needs a
  build with `--features gpu`. CUDA cannot train in strict fp32 (no switch in the
  matmul library). `gbdt` trains on the CPU only.
- The CPU path is certified on Linux x86_64 only. The full test suite also
  passes on Windows x86_64; macOS builds are produced and tested by CI; the
  Linux arm64 and macOS Intel binaries are built but not tested.
- Time-based splits, ranking, survival and multi-table data are not implemented.
  The trees have no native categorical splits, monotone constraints or sampling.
- The Kani proofs show that prediction never panics, for the two models of
  `examples/embedded-predict`; they say nothing about numerical accuracy. Kani does not
  run on Windows.
- macOS is checked only by CI's Apple Silicon runners; nobody has run Tabsmith on a Mac by hand.

[Unreleased]: https://github.com/NatMartini/Tabsmith/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NatMartini/Tabsmith/releases/tag/v0.1.0
