# Design decisions

[中文摘要](#中文摘要)

Each entry below records one decision that shapes the code base. It gives the context,
what was decided, what the decision costs, and the alternatives that were rejected. Each
entry also names the code and tests that hold the decision in place, so reversing it means
changing them on purpose. The entries are ordered from the bundle outwards: what an
application receives, then how it is produced, then how the code is kept honest.

| # | Decision | Held in place by |
|---|---|---|
| [D1](#d1-contracts-are-a-crate-versioned-and-strict) | Contracts are a crate of their own, versioned and strict | `tabsmith-contracts`, `deny_unknown_fields`, version checks in the runtime |
| [D2](#d2-one-preprocessing-implementation-owned-by-the-runtime) | One preprocessing implementation, owned by the runtime | `apply_transform`, scikit-learn goldens |
| [D3](#d3-the-runtime-executes-models-itself) | The runtime executes models itself; the training engine is not its dependency | engine Δ per run, CI dependency check, wasm build |
| [D4](#d4-bundles-verify-themselves) | Bundles verify themselves, and applications swap them only after they do | manifest hashes, self-check, `PredictorHandle` |
| [D5](#d5-determinism-is-a-contract) | Determinism is a contract | own PRNG, single-thread engines, bitwise resume tests |
| [D6](#d6-the-test-split-is-used-once) | The test split is used once, and a ledger enforces it | test ledger, group splits |
| [D7](#d7-stages-are-content-addressed) | Stages are content-addressed | snapshot, split and transform ids |
| [D8](#d8-errors-are-data-and-nothing-falls-back-silently) | Errors are data, and nothing falls back silently | `TabsmithError`, `docs/errors.md` |
| [D9](#d9-the-task-is-part-of-the-type-system) | The task is part of the type system | `Task`, `Targets`, `Metrics`, `Outcome` |
| [D10](#d10-gradient-boosted-trees-in-rust-not-bound-from-c) | Gradient-boosted trees written in Rust, not bound from C++ | `gbdt/`, `tabsmith.trees.v1` |
| [D11](#d11-gpu-training-is-opt-in-planned-and-certified) | GPU training is opt-in, planned, and certified per device | `DevicePlan`, `certified_devices.json`, `tools/gpu_certify.py` |
| [D12](#d12-flex-instead-of-ndarray) | Burn's Flex backend instead of NdArray | `device::ENGINE` |
| [D13](#d13-no-unsafe-and-the-untrusted-input-is-fuzzed) | No `unsafe`, and the untrusted input is fuzzed | workspace lints, `fuzz/`, proptest |
| [D15](#d15-the-trees-add-integers-on-workers-that-spin) | The trees add integers, on workers that spin | `gbdt/`, `pool.rs`, `tasks.rs::the_trees_have_the_same_bytes_on_every_system` |
| [D14](#d14-models-compile-into-rust-code-with-the-runtimes-kernels) | A model can be compiled into Rust code, and it shares the runtime's kernels | `tabsmith-embedded`, `model!`, equivalence and portability tests |
| [D16](#d16-the-trees-are-proved-one-at-a-time-on-their-code) | The trees are proved one at a time, on their code | `model!`'s `kani` option, `Trees::leaf_range`, `mutants.rs`, the lemmas in `kernels.rs` |
| [D17](#d17-the-whole-prediction-is-a-const-fn) | The whole prediction is a `const fn`, and the compiler checks it on the examples | `kernels::exp`, the `const` kernels, the generated self-check, `tests/const_eval.rs` |
| [D18](#d18-rules-are-proved-over-regions-with-categories-kept-one-hot) | Rules are proved over regions of inputs, with categories kept one-hot, and every answer is checked on its own terms | `tabsmith_embedded::rules`, `tabsmith_runtime::audit`, `model!(…, rule(…))`, the brute-force test in `audit/search.rs` |

---

## D1. Contracts are a crate, versioned and strict

**Context.** Six stages hand records to each other through files: snapshot, split,
transform, run, evaluation and bundle. A bundle is read by applications that may run a
different Tabsmith version from the one that built it.

**Decision.** Every record that crosses a stage boundary is a type in
`tabsmith-contracts`, a small crate (serde, serde_json, toml, thiserror, sha2) with no
training or data engine in it. Every format has an
explicit version: `bundle_format`, `TRANSFORM_VERSION` (`tabsmith.std-tabular.v2`),
`TREES_FORMAT` (`tabsmith.trees.v1`), and the metrics version. The bundle types carry
`#[serde(deny_unknown_fields)]`. The runtime refuses any version it does not implement.

**Cost.** Adding a field to a bundle file is a format change, and old runtimes refuse new
bundles instead of reading what they can. That friction is deliberate.

**Rejected.** Schemas that ignore unknown fields, as in Protobuf or Avro. Suppose a new
field changes the semantics: for example the missing-value tokens added in transform v2.
An old reader that skips the field still predicts, just differently. A refusal is better
than a quietly different number.

**Code.** `crates/tabsmith-contracts/src/` (`transform.rs`, `model.rs`, `manifest.rs`,
`schema.rs`).

## D2. One preprocessing implementation, owned by the runtime

**Context.** Training/serving skew happens when the preprocessing is written twice, once
for training (for example with dataframe expressions) and once for serving. The two
copies drift.

**Decision.** `tabsmith_runtime::transform::apply_transform` is the only code that turns
raw values into model inputs. Training, evaluation, `tabsmith predict` and
`Predictor::predict` all call it with the frozen `FittedTransform`. Fitting reuses the
runtime's coercion functions, so value semantics are shared too:
- a number sent as text is an error;
- `12` and `12.0` are the same category;
- `NaN` is a missing value, and ±inf is rejected.

The dependency points from the workflow to the runtime, never the other way.

**Cost.** The runtime owns semantics that only training needs to *fit*. Fitting stays in
the workflow; only applying lives in the runtime.

**Evidence.** The fitted transform matches scikit-learn's `SimpleImputer` +
`StandardScaler` + `OneHotEncoder` to about 2e-7. CI guards this with frozen goldens
(`reference_goldens.rs::transform_matches_scikit_learn`). Predictions from a separate
process equal the training side's
(`cli.rs::separate_process_predictions_equal_training_side_probabilities`).

## D3. The runtime executes models itself

**Context.** An application that loads a model should not compile a deep-learning
framework, a dataframe library or a GPU stack.

**Decision.** A bundle describes its model in an engine-neutral form:
- `model.config.json`: dense layers with activations, or a tree ensemble;
- `model.safetensors`: the weights.

`tabsmith-runtime` executes both with small FP32 kernels of its own. They live in the
`no_std` crate `tabsmith-embedded`, so that models compiled into code share them (D14).
Its direct dependencies are `tabsmith-contracts`, `tabsmith-embedded` (which has none since
D17), serde, serde_json and safetensors. Every run
measures the largest difference between the training engine's outputs and the runtime's
on the validation rows ("engine Δ"), and records it in the run:
- networks: about 1e-7;
- trees: exactly 0, because leaf values are rounded to f32 before they are summed, on both
  sides.

**Cost.** There are two forward implementations to keep equal. The engine Δ of every run
is what makes a divergence visible.

**What it bought.**
- The example application is a small binary with no Burn or Polars in its dependency tree
  (a CI step fails if either appears).
- The runtime compiled to WebAssembly without a single change: `examples/wasm-predict`
  verifies and predicts inside a browser tab.
- The same kernels, and for trees the same arithmetic in generated code, run a bundle
  compiled into Rust code, down to a microcontroller (D14).

**Rejected.**
- Burn as a runtime dependency: compile time, binary size and GPU dependencies.
- ONNX: another runtime with its own operator-set versioning, and its tree-ensemble
  operators would have to reproduce the trainer's exact arithmetic.

## D4. Bundles verify themselves

**Context.** A bundle travels: it is copied, unpacked, served, and replaced while an
application is running.

**Decision.** On load, the runtime checks four things:
1. the SHA-256 of every file against the manifest (the bundle id is a hash over the files);
2. every format version (D1);
3. the internal consistency of transform, output schema and model;
4. optionally, a **self-check**: the bundle carries example records and the predictions
   the trainer recorded for them, and the runtime replays them.

The example records are synthetic, built from transform statistics, so a bundle never
contains training rows. `PredictorHandle::try_update` swaps in a new bundle only after its
self-check passes and its input contract matches the active one (`rollback_to` goes back).

**Cost.** Loading reads and hashes every file. That takes a few milliseconds for the
sample bundles, including in WebAssembly.

**Evidence.**
- `review_regressions.rs`: `bundle_tampering_is_rejected` and `contract_breaking_updates_are_refused`;
- `properties.rs::damaged_bundles_are_refused_without_panicking`;
- the fuzz target `fuzz/fuzz_targets/bundle_from_memory.rs` (D13).

## D5. Determinism is a contract

**Context.** Exact resume after a crash, the test ledger (D6, which identifies a model by
the hash of its weights) and reproducible reports all need "same inputs, same model",
bit for bit.

**Decision.**
- **Randomness.** Splits, weight initialisation and minibatch order come from Tabsmith's
  own versioned PRNG (SplitMix64 with Lemire's bounded sampling, `rng.rs`). Each stream
  is derived from `(seed, label)`. The `rand` crate documents that `StdRng` may change
  between releases, so it is not used.
- **Minibatch order** is a pure function of `(seed, epoch)`. There is no dropout or other
  stochastic layer.
- **CPU training** runs on a single-threaded backend (D12), so reductions always happen
  in the same order. Inference over many rows (validation after every epoch,
  evaluation) does use every core: the runtime's rows are split between threads. Each
  row is computed exactly as on one thread, so the results are bit-identical.
- **Trees** add integers. A row's gradient and hessian become integers at a scale fixed
  per tree, and a histogram cell is a sum of integers, which has no rounding and no order.
  So a cell holds the same value on any number of threads, whichever rows a thread takes,
  and one child's histogram is exactly its parent's minus its sibling's. Every decision is
  a pure function of those sums; ties between splits are broken by `(feature, bin)`, and
  ties between leaves by creation order ([D15](#d15-the-trees-add-integers-on-workers-that-spin)).
  Their `exp` and `log` (gradients, starting score, losses) come from the pure-Rust `libm`,
  not the platform C library, so the same data gives the same trees on every operating
  system: `tasks.rs::the_trees_have_the_same_bytes_on_every_system` pins the bytes, and CI
  runs it on Linux, macOS and Windows. The networks' training is reproducible on one machine
  but not across operating systems (Burn's CPU backend), and the report says so.
- **Checkpoints** hold everything: model record, Adam state, loop state and best weights.
  Resuming from one equals the uninterrupted run bitwise.
- **GPU training** runs without autotuning. CubeCL's autotuner picks kernels by timing.
  In a probe on an RTX 5090 this made two CUDA processes train different weights, while
  each process was consistent with itself.

**Cost.** The networks' CPU training uses one core. At the scale Tabsmith targets (the README
benchmark's 10,000 rows train each candidate in well under a second) that is the right
trade. GPU training without autotuning gives up some kernel speed.

**Evidence.**
- `end_to_end.rs::exact_resume_matches_uninterrupted_training_bitwise`,
  `cli.rs::crash_then_resume_equals_uninterrupted_run_bitwise` (a real process abort)
  and `review_regressions.rs::mid_epoch_cancel_resume_is_exact_and_children_are_resumable`;
- `tests/gbdt.rs::gbdt_is_reproducible_resumable_and_bundled`;
- `reports/reproducibility/2026-10-29-windows-vs-linux/` (the trees across operating systems);
- `gpu.rs::every_usable_gpu_trains_deterministically_and_close_to_the_cpu`;
- `properties.rs::epoch_order_is_a_reproducible_permutation`.

## D6. The test split is used once

**Context.** Every look at the test split that influences a choice turns it into
validation data. With retraining, it is easy to look many times without noticing.

**Decision.**
- **Selection** ranks candidates on the validation split only.
- **Test evaluations** are recorded in a ledger keyed by the model's weight hash and the
  data version. A second model scored on the same test rows needs an explicit
  `--allow-test-reuse`, and the reuse is recorded (`test_use_index`, `test_shared_with`).
- **New data.** When data changes, the old bundle and the new model are compared on rows
  neither of them was trained on, and the record says so (`unseen_rows_verified`).
- **Splits** keep entities together: with `--group device_id`, no device has rows on both
  sides. Classification splits are stratified. A binary split must have both classes in
  train, validation and test. A multiclass split must have every class in train, and
  warns when one is missing from validation or test. A regression split refuses a part
  with fewer than two rows.

**Evidence.**
- `review_regressions.rs`: `test_discipline_survives_resplits_and_warm_starts`,
  `old_and_new_models_are_compared_on_the_same_rows` and
  `id_based_splits_are_stable_across_data_versions`;
- `properties.rs::splits_partition_rows_and_respect_groups`.

## D7. Stages are content-addressed

**Decision.** Snapshot, split and transform ids are hashes of everything that determines
them. For example, a split's id covers the snapshot, strategy, seed, fractions, target,
group column and algorithm version. Unchanged inputs reuse a stage, and a change
invalidates exactly the stages that depend on it. Runs are numbered in sequence and record
every id they used, plus the hash of the resolved configuration.

**Cost.** Every stage writes its identity inputs down. In return, `tabsmith train` on
unchanged data does no repeated work, and reports can show why two runs are comparable
(same split id) or not.

## D8. Errors are data, and nothing falls back silently

**Decision.** Every failure is a `TabsmithError`. It has:
- a stable code (`E_SCHEMA_MISSING_COLUMN`, …, all listed in `docs/errors.md`);
- the phase in which it happened;
- where they apply, the offending field and the expected and actual values;
- a **next action** in plain words.

With `--json`, the same structure is printed for machines. When Tabsmith cannot do what
was asked, it refuses instead of doing something nearby:
- a GPU that is not compiled in or not present;
- fp32 requested on a GPU path that would compute in TF32;
- an unseen category under `unknown_category = "error"`;
- a target that looks binary under the multiclass recipe.

**Cost.** Users meet more errors than with a tool that "just works" by guessing. Each
error says what to do, and the CLI tests check that it does
(`cli.rs::preset_failures_are_actionable`).

## D9. The task is part of the type system

**Context.** Release 0.1 was binary-only. Adding multiclass classification and regression
could have spread `if task == …` checks, and optional fields valid for one task only,
through every stage.

**Decision.** The recipe fixes a `Task` (`tabular-{binary,multiclass,regression}@0.1`).
Every task-dependent value is an enum whose variants carry exactly that task's data:
- `Targets`: labels or values;
- `Metrics`: internally tagged by `task` in JSON;
- the runtime's `Outcome`: flattened into `Prediction`, so clients still see flat JSON.

The selection metric is validated against the task (`SelectionMetric::applies_to`).
Regression targets are learned standardised, and the output schema carries the scale.

**Cost.** Every stage now matches on the task. The compiler checks that the matches are
exhaustive, which is the point.

**Evidence.**
- The binary path is unchanged: the scikit-learn goldens and the bitwise-resume tests
  passed without being touched.
- `tasks.rs` trains every candidate for the other two tasks and checks their bundles.

## D10. Gradient-boosted trees in Rust, not bound from C++

**Context.** Gradient-boosted trees are the strongest general baseline for tabular data.
LightGBM and XGBoost would mean three things:
- a C++ toolchain in the build;
- multi-threaded floating-point sums, whose order depends on the thread count;
- a model format the runtime would have to depend on or re-implement.

**Decision.** A histogram GBDT in about 1,700 lines of safe Rust (`gbdt/`):
- split points come from counts of the training rows' values, up to `max_bins` per feature
  (255 by default, at most 256);
- trees grow best-first, with leaf, depth and minimum-rows limits and an L2 penalty on
  leaf values;
- the loss is the task's: class-weighted log loss, softmax cross-entropy with one tree per
  class and round, or squared error on the standardised target;
- early stopping watches the validation loss.

The model is stored as flat arrays in `tabsmith.trees.v1`: feature, threshold, left,
right, value, with each child after its parent. The runtime walks it without recursion.
One boosting round is one "epoch" of the shared training state, so checkpoints,
cancellation and exact resume came for free.

**Cost.** The trees do less than LightGBM: categorical features arrive one-hot from the
shared transform, and there are no sampling schemes (monotone constraints exist, kept by
construction as in XGBoost, and checked again by the loader and the compiler). Speed against
LightGBM and XGBoost is measured in `reports/training/`.

**Evidence.**
- engine Δ = 0 for trees;
- `tests/gbdt.rs::gbdt_is_reproducible_resumable_and_bundled`;
- `gbdt/tests.rs`.

## D11. GPU training is opt-in, planned, and certified

**Context.** A probe on this project's training loop measured three things:
- **Speed.** GPUs help only well beyond the README-sized problems. On the README's
  problem, every GPU path is slower than the CPU end to end (1.0–1.8 s against 0.11 s).
  For a 512×512 network with batch 2048, Tabsmith's loop on an RTX 5090 is 10.8× (CUDA)
  and 6.6× (Vulkan) faster per step than the CPU engine. An AMD integrated GPU is 0.8×,
  slower than the CPU.
- **Precision.** CUDA silently computes f32 matrix products in TF32 on tensor cores, a
  difference of up to 3e-2 from fp32. The matmul library offers no switch to turn this
  off.
- **Stability.** An integrated AMD GPU lost its device under autotuning.

**Decision.**
- **Opt-in.** GPU backends are Cargo features (`vulkan`, `cuda`, both via `gpu` on the
  CLI). The default build has no GPU dependency at all.
- **Planned.** Before training, a `DevicePlan` records what was requested, what is
  compiled in, what was detected (read-only probing behind `catch_unwind`), what was
  selected, and the precision. After training, a `DeviceReceipt` records what actually
  ran.
- **Precision is explicit.** CUDA requires `execution.precision = "tf32"`. Requesting
  fp32 on a TF32 path is an error, not a downgrade. Vulkan computes fp32.
- **Certified.** `device = "auto-certified"` considers only GPUs listed in
  `certified_devices.json`. An entry matches exactly on backend, device name, OS,
  precision, and driver version or compute capability. A GPU is used only if a cost model
  predicts it to be at least twice as fast as the CPU for this problem; the model's rates
  are fitted to the certification timings. Entries come from
  `tools/gpu_certify.py`, which trains on the CPU and on the GPU and checks four things:
  determinism across processes, agreement with the CPU, exact resume, and the bundle. The
  evidence goes to `reports/gpu/`.

**Cost.** A user with an uncertified GPU must name it explicitly. That is the intent: an
unverified GPU path is an experiment, not a default.

## D12. Flex instead of NdArray

**Context.** Burn 0.22 deprecates its NdArray CPU backend in favour of Flex.

**Decision.** CPU training runs on Flex with a single thread and no rayon. On this
project's loop, Flex was about 1.4× faster than NdArray and matched it to 9e-8.

**Cost.** Weights trained with Flex differ from the NdArray-era runs in the last bits,
so numbers in reports made before the switch can differ in the fourth decimal. The
determinism guarantees (D5) hold within the new engine. The engine name is part of every
run's environment record.

## D13. No `unsafe`, and the untrusted input is fuzzed

**Decision.**
- **No unsafe code.** `unsafe_code = "forbid"` applies to every workspace crate. The
  two crates that cannot use it are outside the workspace: the fuzz target, where
  libFuzzer's entry point is unsafe, and `examples/wasm-predict`, where `#[wasm_bindgen]`
  generates unsafe ABI glue.
- **Fuzzing.** The one input an application cannot trust is a bundle. The fuzz target
  feeds arbitrary file sets to `Bundle::from_memory`, with hash checks both on and off so
  that mutations reach the parsers.
- **Property tests.** proptest checks the invariants that example tests sample too
  sparsely: total, batch-independent predictions; order-free metrics; partitioning splits;
  reproducible epoch orders; and damaged bundles refused without a panic.
- **Docs and lints.** The runtime is `#![warn(missing_docs)]`. CI denies clippy warnings
  in all builds, GPU features included, and denies rustdoc warnings.

## D14. Models compile into Rust code, with the runtime's kernels

**Context.** Loading a bundle at run time needs files, an allocator and the standard
library. It also finds a mismatch between an application and the model's input contract
only when the application runs. Firmware has no file system, and a service would rather
not compile a mistake.

**Decision.** `tabsmith_macros::model!(name, "path")` compiles a bundle into a module
while the application builds:
- It loads the bundle with `Predictor::load`, so it verifies exactly what the runtime
  verifies, and a broken or edited bundle is a compile error.
- The input contract becomes types. `Input` has an enum for each categorical field and
  an `Option<f64>` for each number, and each task has its own prediction type. A
  misspelled field or a category the model never saw does not compile.
- Weights become constants, written as `f32::from_bits` so that no decimal round trip
  touches them. Tree ensembles get constant assertions on their structure.
- Each tree becomes a `const fn` of nested `if`s, up to 32,768 nodes by default. Its numbers
  are decimal literals, so that the expansion reads as the tree it is. That is exact (Rust
  prints the shortest decimal that reads back as the same bits, and rustc reads it back
  correctly rounded), and a test holds every number to it. Larger ensembles keep the arrays,
  because of the build time, unless the invocation says `tree_code`.
- A network's `predict` calls the kernels in `tabsmith-embedded`, a `no_std` crate without
  allocation that the runtime calls too, and a generated tree repeats the arithmetic of the
  trees' kernel. `exp` comes from `libm` (pure Rust) instead of the platform's C library.
- The macro records what the runtime predicted for the bundle's examples, and
  `self_check()` replays them on the target.

**Cost.**
- The kernels' order of operations is now part of the format. The runtime and every
  compiled model depend on it, and changing it changes every prediction.
- The trees have two implementations of the walk: the kernel and the generated code.
  `tests/tree_code.rs` keeps them equal at every node, and it was itself checked by changing
  the generator on purpose: a threshold or a leaf by one representable number, `<` for `<=`,
  swapped children, trees added to the wrong output. All six failed the tests.
- Generated trees take longer to build: at 32,450 nodes, 7.6 s against 2.1 s in a release
  build. That is why the default stops at 32,768 nodes.
- `Input::from_values` repeats the runtime's coercion of raw values for the compiled
  path. Property tests keep the two equal.
- An application that uses `model!` compiles `tabsmith-runtime` as a build-time
  dependency of the macro.

**Evidence.**
- `crates/tabsmith-macros/tests/equivalence.rs` covers nine bundles, every task and
  model kind. The compiled model and the runtime return the same bits on each bundle's
  examples and on 3,000 random records per bundle. The records include missing values,
  unseen categories, NaN, infinity and values of the wrong type. A record that one of
  them refuses, the other refuses too.
- `tests/portable.rs` hashes 20,000 predictions per bundle and compares the hash with
  the value recorded on Windows x86_64. CI runs it on Linux x86_64 and macOS arm64 as
  well.
- Compile-fail doctests check a misspelled field, a text where a category belongs, and
  a label that does not exist.
- `examples/embedded-predict` builds for a Cortex-M4F (`thumbv7em-none-eabihf`) and for
  WebAssembly. Kani proofs show that its prediction never panics, for every input.
  `predict` is `decode(outputs(features(input)?))`, and each stage is proved total for
  every value of its argument type, a superset of what the stage before it returns.
  With Kani 0.68 on Linux the nine took 81 s together, each under 2 GB of memory: six for
  the two models and three for the same trees as arrays. Since D16 the trees carry two
  more proofs each, and the example runs 69. The harnesses are generated by
  `model!` (its `kani` option), with loop bounds from the model's shape, so every compiled
  model carries the same proofs. Generated trees are proved all at once, in 4.8 s and
  104 MB, because every index is a constant. Trees kept as arrays are proved one walk at a
  time, through any tree: all 14 walks in one proof ran out of memory with 14 GB, and a CI
  runner has 16 GB. Kani's additional floating-point checks are off, for three reasons:
  - they are not about panics;
  - CBMC treats any division by an infinity as a possible NaN, which flags `exp(-inf)`
    in `libm`;
  - without them Kani slices the network's arithmetic away, and its layers take 25 s
    instead of not finishing in 32 minutes.

  The evidence is in `reports/kani/`.
- On one Windows x86_64 machine, a compiled prediction takes 0.06–0.33 µs, against
  3.6–4.2 µs for the runtime predicting from a `Record`
  (`examples/compiled_vs_runtime.rs`).
- On real models and rows, generated trees run 2.0–2.8× faster than the arrays, and take a
  third less flash on a Cortex-M4F (`reports/inference/`).

**Rejected.**
- A `build.rs` that writes source code: the same result with a worse interface, a
  generated file to include and no errors at the call site.
- Embedding the bundle with `include_bytes!` and parsing it at start-up: this needs an
  allocator, and errors appear at run time.
- A C code generator or ONNX export: another implementation of the arithmetic in another
  language, outside the tests and the proofs, so predictions would no longer be held to the
  runtime's bits.
- Tree by tree for every batch: it takes twice as long on the smallest model and gains
  nothing below about 20,000 nodes, so the runtime and `outputs_batch` switch at 32,768
  nodes, where it gains 1.6× or more in the arrays.

## D15. The trees add integers, on workers that spin

**Context.** Measured on one machine against LightGBM and XGBoost, on the same rows and settings
(`reports/training/`), the trees of version 0.3 took 2.3 to 5.9 times as long as LightGBM at each
one's best thread count, with quality on par. Three causes, each measured:
- **Threads barely helped.** Under rayon, a parallel section after a short idle gap costs 20 to
  80 µs on Windows and 300 to 1,400 µs in WSL 2, where waking a sleeping virtual CPU goes through
  the hypervisor. A boosting round has hundreds of sections, most of them short.
- **One-hot columns took a pass each.** 44 of Covertype's 54 columns and 99 of Adult's 105 features.
- **Floating-point sums stood in the way of the fixes.** A float sum depends on its order, so a
  `(feature, bin)` had to see its rows in one fixed order. Summing several features in one pass, or
  a row into either of two tables, changes that order, and with it the weights.

**Decision.**
- A row's gradient and hessian become `i32` at a scale fixed per tree, chosen from a bound on them
  so that the largest is at most 2^30. A cell of a histogram holds `i64` sums. Up to 2^32 rows fit,
  so no data can overflow a cell.
- Thresholds are a pure function of counts of values in cells of their ordered form: 256 cells,
  and a cell with more than its share of the rows is refined into 256 more, down to single values
  (`bins.rs`). Rows held in several places can sum their counts and get the same thresholds.
- Sections run on workers that spin for about a millisecond before they park, so one starts in a
  microsecond or two (`pool.rs`, about 190 lines of safe Rust). What the items of a
  section share are relaxed atomics, which compile to plain loads and stores on x86-64 and AArch64
  and let the compiler see that no data race is possible.
- Features whose non-dominant rows never meet share a pass, as in LightGBM's exclusive feature
  bundling (`plan_passes`). The rows of a pass go to two tables in turn, so that a row does not wait
  for the store of the row before it.
- A round computes each row's probabilities once, not per tree, and its training loss on a fixed
  sample of about 4,096 rows. The validation loss, which decides early stopping, still covers
  every validation row, through a fixed-point sum.

**Why nothing but the data can change the model.** By induction over trees and splits:
1. A row's gradient and hessian depend only on that row and the trees so far. Its score is the sum
   of the leaves it reached, in tree order.
2. A cell is a sum of integers over the rows of a node. Integer addition has no rounding and no
   order, so the cell does not depend on which thread adds which row, in what order, or in which pass.
3. Every decision is a pure function of the cells and of nothing else: the largest gain, first in
   `(feature, bin)` order; the leaf with the largest gain, first in creation order; a leaf's value.
   Those float operations are IEEE's and `libm`'s, the same on every system.

So threads, row order and passes can change only the time, and the same holds for any other way
of computing the sums. Tests train with each of them changed and compare bytes, and CI compares
bytes across three operating systems. The single-threaded path is the reference, and every faster
path must match it byte for byte, the discipline of simulators such as [Days](https://days.sh).

**Cost.**
- Precision: a row is rounded to 2^-30 of the largest in its tree. In a sweep the test metrics do
  not move until about 8 bits (`reports/training/`).
- Thresholds are boundaries of cells, not values from the data, so they read less cleanly:
  `3.0000305` where the data has `3`. A value that repeats gets a threshold of exactly itself, which
  keeps a one-hot split at `x <= 0.0`.
- `order`, `gh` and the scores are atomics, so the code says `.load(Relaxed)` where a C++ trainer
  would index a pointer. Auto-vectorisation is off on those loops; nothing there vectorised anyway.
- A spinning worker burns up to a millisecond of CPU per pause, while training only.
- The training loss in the history is an estimate, on the same rows every round.

**Rejected.**
- Keeping `f64` and reproducible summation: slower per addition, and no help for merging sums from
  several places in general.
- LightGBM's quantized training, with 2 or 3 bits per gradient: it gains little here (a cell is 64
  bits either way) and on Covertype's 400 rounds it reached accuracy 0.8463 against 0.8903.
- Raising rayon's thresholds until sections are long enough: most of a round's work is in short ones.
- `unsafe` shared slices (D13).

**Evidence.**
- `gbdt/tests.rs`: `the_model_does_not_depend_on_threads_or_passes`,
  `the_model_does_not_depend_on_the_order_of_the_rows`,
  `shortcuts_give_exactly_what_summing_every_row_gives`, `thresholds_do_not_depend_on_where_the_rows_are`;
- `tasks.rs::the_trees_have_the_same_bytes_on_every_system`;
- `pool.rs`: every item once, a late worker never mixes two sections, parking, panics;
- `reports/training/` (the comparison and the cost of a section), `reports/reproducibility/`.

---

## D16. The trees are proved one at a time, on their code

**Context.** Since 0.2 the Kani proofs said that prediction never panics, and tests said that the
compiled trees predict the runtime's bits and keep their bounds. Tests check rows; a proof covers
every row. Two claims were missing: the generated code of a tree is the walk the runtime takes,
and no row of a box leaves the bounds `output_bounds` gives it. Trials showed what a model checker
can carry. With a symbolic tree index, the range proof of 14 trees ran out of memory at 13.3 GB.
Through the arrays' walk, the range proof of one tree took 82 s and over 7 GB, because every index
of a walk depends on the comparisons before it.

**Decision.**
- `model!(…, kani)` generates two proofs per tree, each with the tree's index as a constant:
  `_is_its_walk` (the code gives the bits of the walk, for every row) and `_stays_in_its_range`
  (every row of every box reaches a leaf of the code inside `Trees::leaf_range` of the box).
- The range is proved on the code, where every index is a constant, in about 5 s and 130 MB. Trees
  kept as arrays get their code too, inside the proofs only, and the walk proof links the two.
- The bounds analysis gives each tree's range through `Trees::leaf_range`, so the proofs check the
  function that `bounds` calls, not a copy of it.
- From the trees to their sum: three lemmas in `tabsmith-embedded`, proved with Kani, state that
  adding a smaller leaf to a smaller sum gives a smaller sum and that rounding to `f32` keeps the
  order. Each compares one operand at a time; comparing two different sums at once is beyond what
  the solver finishes in reasonable time. They depend on IEEE 754 and not on the code, so they run
  in `tools/verify.sh --all`, not in CI.
- Four proofs that must fail (`#[kani::should_panic]`), each with one deliberate mistake, keep the
  generated ones from passing vacuously.

**Cost.**
- Two proofs per tree. A range proof does not depend on the size of the ensemble, but a walk proof
  does, because the walk indexes the arrays of every tree: about 7 s and 1.3 GB per tree at 854
  nodes and 32 s and 5.7 GB at 3,610. A model of tens of thousands of nodes takes hours and more
  memory than a CI runner has; `cargo kani --harness` runs a subset.
- The order of the additions in `outputs` is the kernel's by construction and by test
  (`tests/tree_code.rs`), not by proof: a proof would put both sides' `f64` additions into one
  formula.
- A tree deeper than 64 levels has no code and gets neither proof.

**Rejected.**
- All trees in one proof, with a symbolic index: out of memory.
- The range proved on the walk: an order of magnitude slower and larger, per tree.
- Bounds checked by Kani over the sum of the trees: every `f64` addition becomes part of the
  formula, the cost that the lemmas pay once instead.

**Evidence.**
- `reports/kani/2026-11-05-wsl-linux-x86_64/`: every harness, the lemmas, the other fixtures, the
  mutants, and the bug in the NaN rule that stating the range's assumption found;
- `kernels.rs` tests: `the_bounds_add_up_the_leaf_ranges_of_the_trees`, and the NaN lower bound in
  `tree_bounds_over_everything_are_the_sums_of_the_extreme_leaves`.

---

## D17. The whole prediction is a `const fn`

**Context.** Since 0.3, trees compiled as code made `outputs` a `const fn`. A prediction also
needs the features, the decision and `exp`, and trees kept as arrays and networks were not
`const` at all. libm's `exp` is not a `const fn`. A model the compiler can run is one the
compiler can check, against the runtime's bits, before the application exists.

**Decision.**
- Every kernel of a prediction is a `const fn`: `dot`, `dense`, the tree walk and its sum,
  standardisation, `numeric`, `exp`, the logistic function, softmax, argmax and `unscale`.
- `exp` is libm 0.2.16's algorithm (FreeBSD's e_exp.c), ported operation for operation, with
  its notice. tabsmith-embedded has no dependencies; libm is the reference of its tests.
- The generated `features`, `outputs`, `decode`, `predict`, `interval` and `prediction_set`
  are `const fn`s, for every model and both forms of trees. A constant assertion in the
  generated module predicts every example of the bundle and compares the bits with the
  runtime's, so a model the compiler computes differently does not compile.
- Every kernel that can return a NaN returns one quiet positive NaN. IEEE 754 leaves the sign
  of a NaN from an invalid operation to the processor; `inf - inf` in a network's last layer
  gave `-NaN` on x86 and `+NaN` in the compiler and on ARM.
- `PROBABILITY_BOUNDS` and `VALUE_BOUNDS` come from the compiler, with `decode`'s kernels.

**Why the bits stay the same.** rustc's constant evaluator implements IEEE 754 arithmetic in
software, correctly rounded, as the hardware does. The port of `exp` repeats libm's operations
in libm's order, so it gives libm's bits on every input, and the check over all 2^32 `f32`
inputs (every input of a binary model's logistic function) and 3.2 billion random `f64` finds
no difference. The loops of the kernels became `while` loops over the same indices in the same
order.

**Cost.**
- A second `exp` to keep. If libm ever changes its `exp`, Tabsmith's does not follow; that is
  deliberate, since every bundle's examples pin the bits.
- `while` loops instead of iterators, and `dot` cuts its second slice to the first's length
  so that the compiler checks no index. Measured against 0.4.0, compiled predictions got 8 to
  37 % faster, and the walk through arrays 3 to 6 % slower (`reports/const/`).
- The check of the examples runs in every build: 0.2 s more for `cargo check` at 169,248
  nodes, nothing measurable at 9,151 and 18,823.

**Rejected.**
- A `const` copy of each kernel next to the fast one: two implementations to keep in step,
  where one `const` implementation measured as fast.
- Keeping libm and making only the trees `const`: the decision and the networks would stay
  out of the compiler's reach.
- Making the NaN canonical after every operation: only what a kernel returns leaves it, and a
  ReLU sends NaN to 0.

**Evidence.**
- `reports/const/2026-11-06-windows-x86_64/`: the check of `exp` and the logistic function over
  every `f32`, on Windows and Linux; speed against 0.4.0; compile time;
- `tests/const_eval.rs`: the compiler and the program on rows the compiler makes up, NaN and
  infinities among them, for every fixture and both forms of trees;
- `kernels.rs` tests: `exp_gives_the_bits_of_libm`,
  `the_compiler_computes_the_bits_the_program_computes`,
  `every_nan_a_kernel_returns_has_the_same_bits`;
- the generated self-check, in every build of every compiled model.

---

## D18. Rules are proved over regions, with categories kept one-hot

**Context.** `output_bounds` answers what a model can output over a box of features. It adds up
each tree's extremes, which different inputs reach, so it is sound but loose, and it says
nothing about a region an application cares about: "nobody under 25 without capital income is
predicted rich with 99 % confidence", "a district with a median income under 2 is never valued
above $700,000". Tests answer that for the rows they try. A proof answers it for every input,
and when it fails, an input that breaks it is the answer. A prototype on 2026-11-01 showed two
traps. A category is a run of one-hot slots, and a box that treats the slots as numbers holds
rows with two categories at once: that run reported that 87 % of Adult's labels flip within
0.05 standard deviations, which was false. And a counterexample in feature space is worth
something only when an input encodes to it.

**Decision.**
- A rule is one line of text, `conditions => claim`, the same in `tabsmith audit`,
  `Predictor::audit` and `model!(…, rule(…))`. The conditions limit input fields in their own
  units; `_ within k sd` limits every other numeric field. The claim is on the probability, the
  label or the value, read as `decode` reads the outputs.
- The search cuts the region into boxes of features in which a category's slots stay one-hot:
  a slot at 1 sets its category's other slots to 0, and all slots at 0 is allowed only where the
  bundle ignores unseen categories (`rules::narrow`, `rules::is_empty`). A box closes when it
  holds no input or when its bounds settle the claim (`Claim::holds_within`).
- For trees it is exact. It splits where the widest tree's range is still open, at that tree's
  own threshold. A box where every tree reaches leaves of one value is one prediction, so the
  search ends with a proof or a counterexample unless the budget runs out. Every 64 boxes a
  greedy dive follows each tree toward the leaves that break the claim.
- For networks it uses `dense_bounds`, which can close boxes but never refute, so it also
  predicts the centre of each box and the corner a linearisation points to, and splits the
  feature whose sensitivity times width is largest.
- Each answer is checked on its own terms. A counterexample is a record, predicted by
  `predict_one`, and reported only when that prediction breaks the claim. A proof is a
  certificate, the splits in the order of a walk, and `rules::Walk` checks it again with the
  kernels' bounds before it is reported. `model!(…, rule(…))` emits the certificate and makes
  the compiler walk it with the module's `output_bounds`, whose per-tree ranges D16 proves; a rule
  that fails or stays undecided does not compile.
- The budget is counted in boxes, so the same rule gets the same answer on every machine; a time
  limit is optional.
- `tabsmith audit --stability k` asks, for each record of a file, whether its label survives every
  move of its numbers by up to `k` standard deviations, categories kept. Fields that held whole
  numbers have the ends of their ranges rounded inward.

**Why the answers are right.**
- A split keeps every input of a box in the part its value selects, and that part is never
  taken for empty: Kani proves both for every box, row and finite threshold of a domain with
  numbers and both kinds of category (`rules::proofs`). A walk visits both parts of every split,
  so the closed boxes cover the region.
- Per tree, the leaves a box reaches lie in `Trees::leaf_range`, which D16 proves for the trees of
  a compiled model, and the sums keep their order through rounding (D16's lemmas).
- The computed logistic function never decreases between `f32` logits (all 2^32 checked, D17)
  and `mean + std * z` rounds monotonically, so a bound on one side settles a binary or
  regression claim exactly. A multiclass label needs a margin of 10⁻⁶ between outputs unless
  the box's outputs are one point; above it no rounding of the softmax reverses the order.
- A test compares the tree search with every row of 400 random small ensembles, one row per
  cell of their thresholds: every proof holds on every row, every counterexample is a row that
  breaks the claim.

**Cost.**
- Search time grows with the ensemble and with how close the claim is to the truth: rules on the
  14 trees of the defects model take milliseconds; on Adult's 155 trees, proofs of 24,587 to
  338,266 boxes take 0.8 to 8 s, and a claim near the true maximum stays undecided after a
  million boxes. Checking a proof again costs about as much as finding it.
- The compiler is about 4,500 times slower than native code: about 3 µs per tree node per box
  (`reports/rules/`). A proof above a million of those, about 3 s, is checked when the macro
  expands rather than by the compiler, and the module's documentation says so.
- Networks: interval bounds are loose, so tight claims stay undecided, and an unbounded field
  makes a network extrapolate; such rules need ranges such as `_ within 3 sd`.
- "Every input" is what `predict` accepts, not what is realistic: an age of 37.4, $1 of capital
  gain, a district of 14 households with 9,602 rooms. Boxes cannot express correlations between
  fields, and a dataset's own one-hot columns (Covertype's soil types) are numbers to the model.

**Rejected.**
- Boxes over slots as plain numbers: they hold inputs that do not exist, and reported
  counterexamples for them.
- A MILP or SMT solver, as in the literature on tree ensembles: a large native dependency, and
  no certificate the compiler can check.
- Sampling: it finds counterexamples and proves nothing.
- A wall-clock budget by default: a rule could compile on one machine and not on another.

**Evidence.**
- `reports/rules/2026-11-06-windows-x86_64/`: rules on Adult, California Housing and Covertype,
  label stability, and the compiler's cost;
- `audit/search.rs` tests: `the_tree_search_agrees_with_every_row_of_small_ensembles`,
  `kept_ranges_and_the_search_give_the_bits_of_the_kernel`;
- `tests/audit.rs` in tabsmith-runtime: every proof on the fixtures holds on 300 random inputs of
  its region, every witness lies in it and is predicted as reported;
- `tests/rules.rs` in tabsmith-macros: proofs the compiler checks for trees, arrays, multiclass
  and a network; `rules::proofs` for Kani.

## D19. An imported model computes the library's arithmetic, or is refused

**Context.** Most gradient-boosted models in use were trained by XGBoost or LightGBM in Python.
To serve one from Rust today is to bind the C++ library, or to convert the model with a tool
that agrees with the library "to about 1e-6", and then trust that the conversion handled missing
values, thresholds and the starting score the way the library does. Tabsmith's runtime already
predicts tree ensembles, compiles them into code and proves things about them. The question was
how close an import can get, and what to do where it cannot. A study on 2026-11-06 walked saved
models by hand in several precisions against the libraries' own predictions: XGBoost adds f32
leaves in f32, in tree order, starting from an f32 margin, and sends `x < s` left; LightGBM
adds f64 leaves in f64 and sends `x <= t` left.

**Decision.**
- Reproduce the library's arithmetic, not its numbers to a tolerance. Thresholds become
  `s.next_down()`, which splits every f32 input as `x < s` does; floats are parsed from their
  decimal text straight to f32, never through f64 (`raw_value`); a tree model may say
  `"sum": "f32"`, and the runtime, the compiled code, the compiler's constant evaluation, the
  bounds and the rule engine all add its leaves that way (`Trees::outputs_in_f32`,
  `bounds_in_f32`).
- Carry the C library function the margin depends on. XGBoost starts a binary model from
  `-logf(1/p - 1)`; the import uses a port of Arm's `logf` (glibc, musl), checked against
  glibc on all 2,139,095,039 positive finite f32 inputs, because the correctly rounded value differs from glibc
  on 416,909 of them and the Windows CRT on 501,924. The bundle stores the result, so it starts
  from Linux XGBoost's bits everywhere.
- Where exactness is impossible, bound it and say so. LightGBM's leaves are f64 and the format's
  are f32: the import computes a bound for each model and reports it, and thresholds are rounded
  down so that every f32 input takes LightGBM's branch.
- Where neither works, refuse. A missing value that the splits of one field send both ways has
  no single imputation: that field refuses missing values (`MissingRule::Refuse`) instead of
  getting a prediction the library would not give. Categorical splits, other links, random
  forests and dart are refused with the reason (`E_IMPORT_UNSUPPORTED`).
- The bundle says what it is: status `imported`, no metrics, the model file's SHA-256 and the
  closeness in its notes, no training statistics (so `within k sd` is refused). New fields are
  left out when default, so no existing bundle changes a byte, and old runtimes refuse new ones.

**Rejected.**
- Calling XGBoost or LightGBM through their C APIs: a native dependency, no `no_std`, no
  `const`, no proofs, and the Rust replaces nothing.
- Leaves as f64 in the format, for an exact LightGBM: every kernel, proof and generated module
  would carry a second leaf type for one library's last bits; the bound is reported instead.
- The correctly rounded `logf`: it is more accurate and not what XGBoost computes; 1,498 of
  12,000 margins of the models in the evidence would have come out one ulp off.
- Imputing a missing value where the splits disagree (the mean, the most common way): a
  prediction the library would not give, with nothing to show it.
- A per-split missing-value direction in the format: exact, but every walk, bound, proof and the
  rule engine's boxes would need a third way through a split; left for later.

**Evidence.**
- `reports/import/2026-11-06-windows-x86_64/`: 270,980 rows of three XGBoost models (300 and
  1,200 trees) bit for bit, 12 models with base scores where glibc's `logf` is not correctly
  rounded, LightGBM within its bounds, and the `logf` probe on Linux and Windows;
- `tests/import.rs` in tabsmith-workflow: the fixtures against XGBoost 3.4.1 and LightGBM 4.7.0,
  every row compared and every refused row refused;
- `tests/imported.rs` and the fixtures `imported-*` in the `model!` tests: compiled code, arrays
  and the compiler give XGBoost's margins; `import::logf` tests pin glibc's rounding where it
  differs from the correctly rounded value.

---

## 中文摘要

| # | 决策 | 要点 |
|---|---|---|
| D1 | 契约独立成 crate，带版本且严格 | 所有跨阶段记录都是 `tabsmith-contracts` 中带格式版本的类型；模型包类型拒绝未知字段；运行时拒绝未实现的版本——拒绝胜过悄悄给出不同的数字 |
| D2 | 预处理只有一份实现，归运行时所有 | 训练、评估、离线预测和应用预测都调用 `apply_transform`；与 scikit-learn 一致到约 2e-7 |
| D3 | 运行时自己执行模型 | 包内是与引擎无关的模型描述；运行时用 `tabsmith-embedded` 中的 FP32 内核执行（编译进代码的模型也用这套内核）；每个 Run 记录 engine Δ（网络约 1e-7，树为 0）；因此运行时无需修改即可编译为 WebAssembly |
| D4 | 模型包自我校验 | 每个文件的 SHA-256、版本、一致性和示例预测自检；热切换只在自检通过后进行 |
| D5 | 确定性是契约 | 自带 PRNG（SplitMix64）、小批次顺序只取决于 (seed, epoch)、网络用单线程引擎；树按特征并行，每个特征在一个线程上按行序求和，因此线程数不影响结果（1 到 32 线程实测哈希一致）；树的 `exp`/`log` 来自纯 Rust 的 `libm`，同样的数据在任何操作系统上得到同样的树（Windows 与 Linux 实测哈希一致）；GPU 关闭 autotune；续训与不中断训练逐位相同 |
| D6 | 测试集只用一次 | 选择只看验证集；测试使用记入台账；分组切分保证同一实体不跨越训练/测试 |
| D7 | 阶段按内容寻址 | 快照、切分、变换的 id 是其输入的哈希；输入不变即复用 |
| D8 | 错误是数据，绝不悄悄回退 | 稳定错误码 + 阶段 + 字段 + 下一步建议；做不到就拒绝 |
| D9 | 任务进入类型系统 | `Task`、`Targets`、`Metrics`、`Outcome` 枚举；二分类路径保持不变 |
| D10 | 用 Rust 自写梯度提升树 | 直方图 GBDT 约 600 行安全 Rust；树格式由运行时执行；检查点与精确续训直接复用 |
| D11 | GPU 训练：可选、事先规划、按设备认证 | CUDA 必须显式接受 TF32；`auto-certified` 只使用经 `tools/gpu_certify.py` 认证的设备 |
| D12 | 用 Flex 代替 NdArray | 约 1.4 倍速度，与 NdArray 相差 9e-8 |
| D13 | 禁止 `unsafe`，对不可信输入做模糊测试 | 全工作区 `forbid(unsafe_code)`；对 `Bundle::from_memory` 做 cargo-fuzz；proptest 检查不变量 |
| D14 | 模型可以编译进 Rust 代码，并与运行时共用内核 | `model!` 在编译期按运行时的规则校验模型包，把输入契约生成为类型、权重生成为常量；推理不分配内存、不依赖 `std`；与运行时逐位相同，跨平台也逐位相同（CI 在三个平台上验证）。编译器还知道模型的更多性质：由 rustc 常量求值得到的可靠输出范围（树精确，网络用区间算术并为舍入留余量）、由此导出的概率/数值范围、单次预测的最大运算量、单调约束的编译期检查；`kani` 选项让宏自己生成 Kani 证明（循环上界取自模型形状），当时 9 个证明 81 秒通过，D16 之后示例共有 69 个。每棵树生成为嵌套 `if` 的 `const fn`（默认到 32,768 个节点），比数组快 2–2.8 倍、在 Cortex-M4F 上少占三分之一 flash，所有树能放进一个证明，由逐节点测试保证比特相同 |
| D15 | 树的求和用整数，工作线程自旋等待 | 梯度和 Hessian 按每棵树的比例转成整数，直方图是整数和，没有舍入也没有顺序，所以线程数、行序和合并方式都改变不了模型的任何一个字节；阈值只取决于计数；并行段用安全 Rust 写的自旋线程池 |
| D16 | 树逐棵证明，并在代码上证明 | `kani` 选项为每棵树生成两个证明，树的下标是常量：代码与运行时的遍历逐位相同；任何区域里的任何一行都落在 `Trees::leaf_range` 内。范围在代码上证明（下标全是常量，约 5 秒、130 MB），在遍历上要 82 秒、超过 7 GB；数组形式的树只在证明里生成代码。从单棵树到求和靠 `tabsmith-embedded` 里三条用 Kani 证明的舍入引理（不在 CI 里）。四个必须失败的证明防止证明变得空洞。代价：遍历证明随整个集成的节点数增长（854 个节点约 7 秒，3,610 个约 32 秒）；`outputs` 的加法顺序靠构造和测试，没有证明 |
| D17 | 整个预测都是 `const fn`，编译器在示例上检查它 | 预测用到的每个内核都是 `const fn`；`exp` 是 libm 0.2.16 的算法（FreeBSD 的 e_exp.c）逐个运算的移植，对全部 2^32 个 f32 输入与 libm 逐位相同，tabsmith-embedded 因此没有任何依赖。生成的 `features`、`outputs`、`decode`、`predict` 对所有模型和两种树形式都是 `const fn`，生成的模块里有一条常量断言：编译器自己预测模型包的每个示例，必须得到运行时的位，否则不能编译。能返回 NaN 的内核都返回同一个 NaN（x86 上 `inf - inf` 给 `-NaN`，编译器和 ARM 给 `+NaN`）。编译后的预测快了 8% 到 37%，数组遍历慢了 3% 到 6%，编译期检查在 16.9 万个节点上只多 0.2 秒 |
| D18 | 规则在输入区域上证明，类别保持独热，每个答案都按自己的标准复核 | 规则是一行文本 `条件 => 结论`，在 `tabsmith audit`、`Predictor::audit` 和 `model!(…, rule(…))` 中相同；搜索把区域切成特征盒子，类别的槽位保持独热（一个槽位为 1 时同类别其他槽位为 0，全为 0 只在模型包忽略未见类别时允许）。树是精确的：每棵树只到达同一个值的叶子时盒子就是一个预测，所以结果要么是证明、要么是反例，除非预算用完；网络用区间界证明、用盒子中心和线性化指向的角点找反例。反例是用 `predict_one` 真实预测过的记录；证明是证书，报告前用 `rules::Walk` 和内核的界复核，`model!` 还让编译器再走一遍；规则不成立或未定都不能编译。预算按盒子数计，所以每台机器答案相同。Kani 证明切分把盒子里的每个输入都留在它的值选择的一侧、且这一侧不会被当成空；一个测试在 400 个随机小集成的每个阈值格子上逐行对照。代价：搜索时间随集成大小和结论离真实值的远近增长（Adult 的 155 棵树上，24,587 到 338,266 个盒子的证明要 0.8 到 8 秒）；编译器约每个节点每个盒子 3 微秒，超过一百万（约 3 秒）的证明改在宏展开时复核；“每个输入”指 `predict` 接受的输入，不只是现实的输入 |
| D19 | 导入的模型复现库的算术，做不到就拒绝 | XGBoost 的阈值变成紧挨其下方的 f32（`s.next_down()`），对每个 f32 输入与 `x < s` 切分相同；浮点数从十进制文本直接解析成 f32，不经过 f64；模型可以声明 `"sum": "f32"`，运行时、编译后的代码、编译器的常量求值、界和规则引擎都按这种方式相加叶子。二分类起点 `-logf(1/p - 1)` 用移植自 Arm 的 glibc `logf`，在全部 2,139,095,039 个正有限 f32 上与 glibc 相同（正确舍入值与 glibc 有 416,909 个不同，Windows CRT 有 501,924 个），模型包存下结果，任何平台都从 Linux 上 XGBoost 的位开始。LightGBM 的叶子是 f64，无法精确，就为每个模型算出并报告误差界，阈值向下舍入使每个 f32 输入走同一分支。一个字段的切分把缺失值两边都送时没有单一的填补值，该字段拒收缺失值；类别切分、其他链接函数、随机森林、dart 都会说明原因并拒绝。证据：三个 XGBoost 模型（300 和 1,200 棵树）的 270,980 行逐位一致；被拒绝的做法包括调用 C API、格式里存 f64 叶子、用正确舍入的 `logf`（证据中 12,000 个 margin 会有 1,498 个差一个 ulp）、在方向不一致时仍然填补，以及（暂时）每个切分自带缺失方向 |
