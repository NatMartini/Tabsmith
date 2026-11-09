# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, minor versions may contain breaking changes.

## [Unreleased]

A rule about what a model predicts over a region of inputs can be proved now, or refuted with
an input, and a compiled model can carry rules that the compiler checks. A compiled tree model
is proved to be the runtime's model, for every input, and its bounds are proved to hold. Every
compiled model can predict while the crate compiles, and does, on the bundle's examples.
Models trained by XGBoost and LightGBM become bundles, and XGBoost's margins come out bit for bit.

### Added

- **Models from XGBoost and LightGBM: `tabsmith import model.json --out DIR`.** XGBoost's JSON
  and LightGBM's text become a bundle that `predict`, `serve`, `audit`, `model!` and the browser
  use like any other. XGBoost's margins come out bit for bit: thresholds become the f32 just
  below XGBoost's (it sends `x < s` left, Tabsmith `x <= t`), floats are parsed from their text
  straight to f32, and the leaves are added in f32 in XGBoost's tree order from its starting
  margin. On 100,000 rows each of models with 300 and 1,200 trees, all 270,980 rows the bundles
  accepted gave XGBoost's margins to the last bit (`reports/import/2026-11-06-windows-x86_64/`).
  LightGBM's f64 leaves are held as f32, so its raw scores come within a bound the import
  computes and reports (2.3e-5 for 300 trees; 7e-7 measured), for inputs that are f32 values.
  A field whose splits send a missing value left at some and right at others refuses missing
  values rather than predict differently; elsewhere the value that takes the missing value's way
  at every split is imputed, which stays exact. Categorical splits, other links, random forests
  and dart are refused with the reason. `--labels`, `--features`, `--target` and `--threshold`
  name what the model file does not ([docs/import.md](docs/import.md)).
- **glibc's `logf`, ported.** A binary XGBoost model starts from `-logf(1/p - 1)`, and C libraries
  round `logf` differently: on the 2,139,095,039 positive finite f32 inputs, the correctly rounded
  value differs from glibc 2.43 on 416,909 and the Windows CRT on 501,924, so XGBoost itself
  starts some models from different margins on Linux and Windows. The import carries a port of
  Arm's `logf` (glibc, musl) that equals glibc on every one of them, and stores the margin, so a
  bundle starts from Linux XGBoost's bits on every platform. For 12 of 1,596 random base scores
  near the class balance glibc is not correctly rounded; with the correctly rounded start, 1,498 of
  12,000 margins of models with those base scores would be one ulp off.
- Bundle format: a tree model may say `"sum": "f32"` (leaves added as XGBoost adds them), and a
  numeric feature `"missing": "refuse"`. Both are left out when default, so every existing bundle
  keeps its bytes and its id; runtimes that do not know them refuse such bundles. Status
  `imported`, with no `valid` metrics in `evaluation.json`.
- `kernels::Trees::outputs_in_f32` and `bounds_in_f32`, `kernels::numeric` with `refuse_missing`,
  `InputError::Missing`; the runtime, the compiled code (both forms of trees), the compiler's
  constant evaluation, `output_bounds` and the rule engine add an `f32` model's leaves in f32. Two
  Kani lemmas show that a larger f32 sum or a larger leaf never gives a smaller f32 sum.
- `tabsmith_workflow::import`: `import_model`, `ImportOptions`, `ImportReport`, `ImportedModel`.
- `fixtures/import/`: small XGBoost 3.4.1 and LightGBM 4.7.0 models with rows and the libraries'
  outputs; `tests/import.rs` compares every row. `fixtures/bundles/imported-*`: two of them as
  bundles, compiled by every `model!` test (code against arrays at every node, the compiler, the
  runtime on random records, bounds, the same bits on every platform) and checked against
  XGBoost's margins (`tests/imported.rs`).

- **Rules over regions of inputs: `tabsmith audit`.** A rule is one line,
  `conditions => claim`, for example
  `age in 17..=25, capital_gain == 0 => probability <= 0.99` or `_ within 3 sd => value <= 600000`.
  Each rule ends proved, with a counterexample, or undecided. The search cuts the region into
  boxes of feature values in which a category's slots stay one-hot, so it never reports an input
  with two categories at once. For trees it is exact: it ends with a proof or a counterexample
  unless the budget, counted in boxes so that every machine gives the same answer, runs out. For
  networks, interval bounds prove and predicted points refute. A counterexample is a record that
  `predict_one` predicts, reported only when that prediction breaks the rule; a proof is checked
  again before it is reported. Exit 0 when every rule holds, 8 when one fails, 9 when one is
  undecided. `--stability k --input FILE` asks of every record whether its label survives every
  move of its numbers by up to `k` standard deviations, and shows an input that changes it where
  one exists. On Adult's 155 trees, proofs of 24,587 to 338,266 boxes take 0.8 to 8 s, and 50 of
  200 records can change their label within 0.05 standard deviations
  (`reports/rules/2026-11-06-windows-x86_64/`).
- `Predictor::audit`, `audit_near` and `stability_rule`, and the module `tabsmith_runtime::audit`:
  `Rule` (parsed from the text), `Budget`, `Audit`, `Verdict`, `Certificate`.
- **`model!(…, rule(conditions => claim))`**: the macro audits the rule while the crate compiles;
  a rule that fails is a compile error that shows an input breaking it, and an undecided one is
  too. A rule that holds comes with its proof, and the compiler walks the proof again with the
  module's own `output_bounds`, in its own floating point, when that takes at most about 3 s
  (a million tree nodes times boxes); `RULES` lists the rules.
- `tabsmith_embedded::rules`: the `const fn` checker of a proof (`Walk`, `narrow`, `is_empty`,
  `Claim`). Two Kani proofs show that a split keeps every input of a box in the part its value
  selects and that such a part is never taken for empty, for every box, row and threshold
  (8 s and 163 MB together; CI runs them).
- A test compares the search with every row of 400 random small tree ensembles, one row per cell
  of their thresholds.

- **The whole prediction is a `const fn`**, for every kind of model and both forms of trees:
  `features`, `outputs`, `decode`, `predict`, `interval` and `prediction_set`, and every kernel
  they call. While the crate compiles, the compiler predicts the bundle's examples and must get
  the bits tabsmith-runtime predicted, or the crate does not compile; an application's own
  constants can be predictions. A refused input in a constant is a compile error. Compile time
  does not change measurably, even at Covertype's 169,248 nodes
  (`reports/const/2026-11-06-windows-x86_64/`).
- `kernels::exp`: libm 0.2.16's algorithm (FreeBSD's e_exp.c), operation for operation, as a
  `const fn`. It gives libm's bits for every input that is an `f32`, all 2^32 checked, and for
  3.2 billion random `f64`. tabsmith-embedded now has no dependencies; libm stays as the
  reference its tests compare against.
- `PROBABILITY_BOUNDS` and `VALUE_BOUNDS` are computed by the compiler with the kernels `decode`
  calls. The computed logistic function never decreases from one `f32` logit to the next, all
  2^32 checked, so no input leaves the probability bounds; before, that held only "up to the
  rounding of `exp`".
- `tests/const_eval.rs`: the compiler and the program give the same bits on rows the compiler
  makes up, NaN and infinities among them, for every fixture and both forms of trees.

- **Two Kani proofs per tree** with `model!(…, kani)`. `<name>_tree_<t>_is_its_walk`: the tree's
  code gives the same bits as the walk through its arrays, the walk the runtime takes, for every
  row of features, NaN and infinities included. `<name>_tree_<t>_stays_in_its_range`: every row
  of every box of feature values reaches a leaf inside the range that `output_bounds` adds up for
  the box. With the existing proofs that prediction never panics, they form a chain from the
  runtime to the bounds; `reports/kani/2026-11-05-wsl-linux-x86_64/` shows each link and what
  proves it. Each proof takes one tree, and the range is proved on the code, where every index is
  a constant: through the arrays, one tree took 82 s and over 7 GB instead of 5 s and 130 MB.
  Trees kept as arrays get their code inside the proofs only. A walk proof grows with the whole
  ensemble, about 7 s and 1.3 GB per tree at 854 nodes and 32 s and 5.7 GB at 3,610, so a large
  model takes hours; `cargo kani --harness` proves a subset.
- `Trees::leaf_range` in tabsmith-embedded: the smallest and largest leaf one tree can reach in a
  box. `Trees::bounds` adds them up, with the same bits as before.
- **Three rounding lemmas, proved with Kani in tabsmith-embedded**, which carry the ranges to the
  sum: a larger sum stays larger when the same leaf is added, a larger leaf gives a larger sum,
  and rounding to `f32` keeps the order. `cargo kani -p tabsmith-embedded` runs them, and so does
  `tools/verify.sh --all`; CI does not, because they do not depend on the code and one of them
  takes 9 to 13 minutes and 5 to 10 GB.
- `examples/embedded-predict/src/mutants.rs`: four proofs that must fail, each with one deliberate
  mistake (another tree, a threshold one unit in the last place off, a strict comparison, the
  NaN rule of 0.4.0), so that the generated proofs cannot pass by checking nothing.

### Changed

- New error code `E_RULE_INVALID` (exit 2) and phase `audit`: a rule that cannot be read, or that
  does not fit the bundle.
- New error codes `E_IMPORT_INVALID` and `E_IMPORT_UNSUPPORTED` (exit 4) and phase `import`.
- `tabsmith audit` refuses `within k sd` and `--stability` on a bundle without training
  statistics (an imported one), which has no standard deviations to measure with.
- `serde_json` is built with `raw_value`, for the import's text-to-f32 parsing; no new crate.
- **Security review of the repository (2026-11-07).** Every GitHub Action of the three workflows
  is pinned to a commit, with its version as a comment (Dependabot keeps the pins current); the
  `msrv` job checks the form of the versions it reads from the manifests and passes them to its
  shell steps through environment variables, not as script text. `write_atomic` creates its
  temporary file with `create_new`, and a bundle build or import creates its temporary directory
  with `create_dir`, so a link planted at the predictable temporary name is never written through
  (a test plants one). The import refuses counts a model file declares but does not back, before
  anything is sized by them: at most 1,048,576 features, no more classes than trees, and
  LightGBM's `max_feature_idx` cannot overflow; `import::describe` reads a model without writing,
  and a property test feeds it randomly damaged copies of the test models. `cargo audit` with the
  advisory database of 2026-11-03 reports no vulnerability beyond the recorded exceptions.
- CI's `kani` job runs 69 proofs instead of nine: 459 s on the development machine instead of
  81 s, none above 1.8 GB.
- `self_check()` passes the examples through `black_box`, so that the device computes them and
  the optimiser cannot.
- Compiled models predict 8 to 37 % faster against 0.4.0, mostly in `features` and `decode`,
  where `exp` can now be inlined; the walk through trees kept as arrays is 3 to 6 % slower.
  Every prediction has the same bits as before (`reports/const/`).

### Fixed

- **A NaN result has the same bits everywhere.** IEEE 754 leaves the sign of a NaN that an
  invalid operation creates to the processor: `inf - inf` in a network's last layer gave `-NaN`
  on x86 and `+NaN` on ARM and in the compiler. Every kernel that can return a NaN now returns
  one, `kernels::NAN_F32` or `NAN_F64`. Only inputs that overflow a network's arithmetic get
  there; trees never produce a NaN. The test of the compiler against the program found it.

- **A NaN lower bound admits NaN rows**, as `output_bounds` always documented. In 0.4.0 a box
  with `lo[j]` NaN and `hi[j]` below a split's threshold reached no leaf of that tree, although a
  NaN row goes right at every split. Writing the documented rule down as the proofs' assumption
  found it. Only boxes with a NaN lower bound and no NaN upper bound change; `OUTPUT_BOUNDS` and
  every other box give the same bits as before.

## [0.4.0] - 2026-11-03

The trees train 1.5 to 4.9 times faster, between LightGBM and XGBoost, in safe Rust, with a model
that no thread count or optimisation can change by a byte. Their weights change
once, because the arithmetic does; bundles built before keep predicting the same bits.

### Changed

- **The trees add integers.** A row's gradient and hessian become integers at a scale fixed per
  tree, and a histogram cell is a sum of integers, which has no rounding and no order. Every
  decision is a pure function of those sums, so threads, row order and how the sums are computed
  cannot change the model ([D15](docs/design-decisions.md#d15-the-trees-add-integers-on-workers-that-spin)).
  Test metrics stay on par (`reports/training/`). A checkpoint of an earlier version is refused,
  as its arithmetic differs.
- **Thresholds come from counts**, of values in cells of their binary form, refined where a cell
  holds more than its share of the rows. They read less cleanly (`3.0000305` for a value that
  occurs once) and split the same way. A value that repeats gets a threshold of exactly itself, so
  one-hot splits stay at `x <= 0.0`.
- **Sections run on workers that spin** for about a millisecond before they park, in about 190
  lines of safe Rust (`pool.rs`). Under rayon a section costs 20 to 80 µs on Windows and 300 to
  1,400 µs in WSL 2; here a microsecond or two. The trees now scale to about 8 threads, where they
  stopped at about 4.
- **Exclusive one-hot columns share a pass**, and a pass adds its rows to two tables in turn.
- **A round computes each row's probabilities once.** The training loss in a run's history is now
  measured on a fixed sample of about 4,096 rows; the validation loss, which decides early
  stopping, still covers every row.
- `gbdt.rs` became `gbdt/`: `bins.rs`, `rows.rs`, `grow.rs`, `mod.rs` and `tests.rs`.
- A short run checkpoints at its end, not also after 50 rounds: the interval counts from the start.
- The acceptance report is made again, on this version; the one before came from 0.1.0. Its
  trees train in 78 ms instead of 797 ms, and after new data they are the model chosen.

### Added

- **The first comparison with LightGBM and XGBoost**, on one machine and the same rows and settings
  (`reports/training/`, `tools/datasets/compare_gbm.py`).
- **CI compares the trees' bytes** on Linux, macOS and Windows, for all three tasks
  (`tasks.rs::the_trees_have_the_same_bytes_on_every_system`).

### Fixed

- A property test failed about once in fifteen runs when a random edit hit the bundle's build time,
  which the bundle's id leaves out on purpose.
- The dataset reports give each download's address and SHA-256. Their steps to reproduce
  pointed to links that were not there.

## [0.3.0] - 2026-11-01

Speed, with every bit kept. The trees train 11 to 36 times faster, and a batch of records
predicts 7 to 11 times faster than scikit-learn's, with the same results at any thread count.
`model!` now turns each tree into code, which is faster and smaller, lets the compiler run the
model, and lets Kani prove all the trees at once. The bundle format is unchanged, and existing
bundles predict the same bits.

### Added

- **`model!` turns trees into code.** Each tree of a compiled model is now a `const fn` of
  nested `if`s, up to 32,768 nodes by default. On real models the outputs take 2.0 to 2.8
  times less time than walking arrays, and flash on a Cortex-M4F shrinks by about a third,
  with every prediction the same to the bit (`reports/inference/`). A release build takes
  7.6 s instead of 2.1 s at 32,450 nodes, which is why the default stops there. `outputs`
  becomes a `const fn`, so the compiler can run the model. The options `tree_code` and
  `tree_arrays` override the default.
- **Kani proves all the trees at once**, for trees as code: `<name>_outputs_are_total`, in
  4.8 s and 104 MB for 14 trees, where one walk through the arrays takes 10 s and 1.7 GB and
  all 14 do not fit in 14 GB (`reports/kani/`). Trees kept as arrays still have
  `<name>_tree_walk_is_total`.
- `outputs_batch` on every compiled model. Above 32,768 nodes it runs tree by tree over blocks
  of rows, up to 2.6 times faster than row by row.
- `tools/publish.cmd`: a double-click on Windows pushes to GitHub, waits for CI on that commit,
  tags, and publishes to crates.io. Run it again after a failure, and what is done is skipped.
- `LoadOptions::threads` and `LoadOptions::with_threads`: the threads a large batch may use.
  A public field, so code that builds `LoadOptions` with a struct literal and no `..` must add
  it.

### Changed

- **The trees train 11 to 36 times faster**, with the same test metrics to four decimals
  on the three public datasets (`reports/datasets/`): Covertype takes 29 s instead of
  513 s, Adult 0.5 s instead of 17 s, California 0.2 s instead of 2.1 s. They now take
  about as long as scikit-learn's `HistGradientBoosting`. Histograms are built feature by
  feature on up to 12 threads, one child's histogram is the parent's minus its sibling's,
  a feature's dominant bin follows from the node's totals, and a tree needs no allocation
  per node. `RAYON_NUM_THREADS` overrides the thread count.
- **The trees stay deterministic on any number of threads**, without chunks: each feature
  sums its rows in row order on one thread. Measured weights are the same at 1 to 32
  threads and on Windows and Linux (`reports/reproducibility/`).
- Retraining gives slightly different trees from 0.2.0 where rounding decides a split or
  a tie: on the public datasets, California's weights are unchanged and Adult's and
  Covertype's differ, with the same test metrics. The bundle format is unchanged and every
  bundle still loads. A gbdt checkpoint of 0.2.0 cannot be resumed.
- A whole `tabsmith train` is 2 to 8 times faster (Covertype with all three candidates:
  71 s instead of 603 s). Row fingerprints and permutation importance use several threads,
  a training round computes only the selected metric, and `run.json` is rewritten at most
  once a second during training instead of after every round or epoch, which also speeds
  up the networks on small problems. After a crash `tabsmith runs` can show up to a second
  less progress; resume reads the checkpoints, as before.
- **Batch prediction runs on several threads.** `Predictor::predict`, `predict_table` and
  `outputs` use up to 16 threads from 4,096 rows on, on plain scoped threads with no dependency
  added: 8 times the one-thread rate on 100,000 records. The predictions are the same at any
  thread count, in the same order, and so is the error when a record is wrong. WebAssembly stays
  on one thread.
- Turning records into a table merges each record's sorted fields with the schema's, which
  makes `predict` on one thread about 40 % faster. Against scikit-learn on the README's problem,
  a batch of records now runs 7 to 11 times faster, up from about even.
- The runtime goes tree by tree over blocks of rows from 32,768 nodes on, 1.6 to 2.8 times faster
  on large ensembles, with the same bits.
- `tabsmith-workflow` depends on `rayon` directly (it was already in its tree through
  Polars) and on `tabsmith-embedded`. `tabsmith-runtime` gains no dependency.

### Fixed

- The trees' device plan reports their thread count; it said 1 while the receipt gave the
  real count.

## [0.2.0] - 2026-10-29

The model as a Rust program, taken further: generated proofs, compile-time bounds,
monotone constraints, chronological splits, conformal prediction, importance, drift
monitoring, trees that are the same on every operating system and use every core, and
the evidence for each (`reports/`). Bundles that carry monotone constraints need this
version of the runtime (`min_runtime_version` 0.2.0); every other bundle still loads
with 0.1.0.

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
- A bundle with monotone constraints records `min_runtime_version` 0.2.0.

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

[Unreleased]: https://github.com/NatMartini/Tabsmith/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NatMartini/Tabsmith/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NatMartini/Tabsmith/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NatMartini/Tabsmith/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NatMartini/Tabsmith/releases/tag/v0.1.0
