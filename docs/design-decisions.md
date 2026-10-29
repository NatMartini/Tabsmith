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
| [D10](#d10-gradient-boosted-trees-in-rust-not-bound-from-c) | Gradient-boosted trees written in Rust, not bound from C++ | `gbdt.rs`, `tabsmith.trees.v1` |
| [D11](#d11-gpu-training-is-opt-in-planned-and-certified) | GPU training is opt-in, planned, and certified per device | `DevicePlan`, `certified_devices.json`, `tools/gpu_certify.py` |
| [D12](#d12-flex-instead-of-ndarray) | Burn's Flex backend instead of NdArray | `device::ENGINE` |
| [D13](#d13-no-unsafe-and-the-untrusted-input-is-fuzzed) | No `unsafe`, and the untrusted input is fuzzed | workspace lints, `fuzz/`, proptest |
| [D14](#d14-models-compile-into-rust-code-with-the-runtimes-kernels) | A model can be compiled into Rust code, and it shares the runtime's kernels | `tabsmith-embedded`, `model!`, equivalence and portability tests |

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
Its direct dependencies are `tabsmith-contracts`, `tabsmith-embedded` (whose only
dependency is `libm`), serde, serde_json and safetensors. Every run
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
- The same kernels run a bundle compiled into Rust code, down to a microcontroller (D14).

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
- **Trees** sum in row order on one thread. Ties between splits are broken by
  `(feature, bin)`, and ties between leaves by creation order.
- **Checkpoints** hold everything: model record, Adam state, loop state and best weights.
  Resuming from one equals the uninterrupted run bitwise.
- **GPU training** runs without autotuning. CubeCL's autotuner picks kernels by timing.
  In a probe on an RTX 5090 this made two CUDA processes train different weights, while
  each process was consistent with itself.

**Cost.** CPU training uses one core. At the scale Tabsmith targets (the README
benchmark's 10,000 rows train each candidate in well under a second) that is the right
trade. GPU training without autotuning gives up some kernel speed.

**Evidence.**
- `end_to_end.rs::exact_resume_matches_uninterrupted_training_bitwise`,
  `cli.rs::crash_then_resume_equals_uninterrupted_run_bitwise` (a real process abort)
  and `review_regressions.rs::mid_epoch_cancel_resume_is_exact_and_children_are_resumable`;
- `gbdt.rs::gbdt_is_reproducible_resumable_and_bundled`;
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

**Decision.** A histogram GBDT in about 600 lines of safe Rust (`gbdt.rs`):
- split points come from the training rows, up to `max_bins` per feature (255 by
  default, at most 256);
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
shared transform, and there are no monotone constraints or sampling schemes. Speed against
scikit-learn's `HistGradientBoostingClassifier` is measured in the benchmark report.

**Evidence.**
- engine Δ = 0 for trees;
- `gbdt.rs::gbdt_is_reproducible_resumable_and_bundled`;
- the unit tests inside `gbdt.rs`.

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
- `predict` calls the kernels in `tabsmith-embedded`, a `no_std` crate without
  allocation that the runtime calls too. `exp` comes from `libm` (pure Rust) instead of
  the platform's C library.
- The macro records what the runtime predicted for the bundle's examples, and
  `self_check()` replays them on the target.

**Cost.**
- The kernels' order of operations is now part of the format. The runtime and every
  compiled model depend on it, and changing it changes every prediction.
- `Input::from_values` repeats the runtime's coercion of raw values for the compiled
  path. Property tests keep the two equal.
- An application that uses `model!` compiles `tabsmith-runtime` as a build-time
  dependency of the macro.

**Evidence.**
- `crates/tabsmith-macros/tests/equivalence.rs` covers seven bundles, every task and
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
  WebAssembly. Six Kani proofs show that its prediction never panics, for every input.
  `predict` is `decode(outputs(features(input)?))`, and each stage is proved total for
  every value of its argument type, a superset of what the stage before it returns.
  With Kani 0.68 on Linux the six take 69 s together, each under 2 GB of memory. The
  trees' `outputs` is proved one walk at a time, through any tree: all 14 walks in one
  proof ran out of memory with 14 GB, and a CI runner has 16 GB. Kani's
  additional floating-point checks are off, for three reasons:
  - they are not about panics;
  - CBMC treats any division by an infinity as a possible NaN, which flags `exp(-inf)`
    in `libm`;
  - without them Kani slices the network's arithmetic away, and its layers take 26 s
    instead of not finishing in 32 minutes.

  The evidence is in `reports/kani/`.
- On one Windows x86_64 machine, a compiled prediction takes 0.13–0.59 µs, against
  3.5–4.2 µs for the runtime predicting from a `Record`
  (`examples/compiled_vs_runtime.rs`).

**Rejected.**
- A `build.rs` that writes source code: the same result with a worse interface, a
  generated file to include and no errors at the call site.
- Embedding the bundle with `include_bytes!` and parsing it at start-up: this needs an
  allocator, and errors appear at run time.
- A C code generator or ONNX export: another implementation of the arithmetic, so
  predictions would no longer match the runtime bit for bit.

---

## 中文摘要

| # | 决策 | 要点 |
|---|---|---|
| D1 | 契约独立成 crate，带版本且严格 | 所有跨阶段记录都是 `tabsmith-contracts` 中带格式版本的类型；模型包类型拒绝未知字段；运行时拒绝未实现的版本——拒绝胜过悄悄给出不同的数字 |
| D2 | 预处理只有一份实现，归运行时所有 | 训练、评估、离线预测和应用预测都调用 `apply_transform`；与 scikit-learn 一致到约 2e-7 |
| D3 | 运行时自己执行模型 | 包内是与引擎无关的模型描述；运行时用 `tabsmith-embedded` 中的 FP32 内核执行（编译进代码的模型也用这套内核）；每个 Run 记录 engine Δ（网络约 1e-7，树为 0）；因此运行时无需修改即可编译为 WebAssembly |
| D4 | 模型包自我校验 | 每个文件的 SHA-256、版本、一致性和示例预测自检；热切换只在自检通过后进行 |
| D5 | 确定性是契约 | 自带 PRNG（SplitMix64）、小批次顺序只取决于 (seed, epoch)、单线程引擎、树按行序求和；GPU 关闭 autotune；续训与不中断训练逐位相同 |
| D6 | 测试集只用一次 | 选择只看验证集；测试使用记入台账；分组切分保证同一实体不跨越训练/测试 |
| D7 | 阶段按内容寻址 | 快照、切分、变换的 id 是其输入的哈希；输入不变即复用 |
| D8 | 错误是数据，绝不悄悄回退 | 稳定错误码 + 阶段 + 字段 + 下一步建议；做不到就拒绝 |
| D9 | 任务进入类型系统 | `Task`、`Targets`、`Metrics`、`Outcome` 枚举；二分类路径保持不变 |
| D10 | 用 Rust 自写梯度提升树 | 直方图 GBDT 约 600 行安全 Rust；树格式由运行时执行；检查点与精确续训直接复用 |
| D11 | GPU 训练：可选、事先规划、按设备认证 | CUDA 必须显式接受 TF32；`auto-certified` 只使用经 `tools/gpu_certify.py` 认证的设备 |
| D12 | 用 Flex 代替 NdArray | 约 1.4 倍速度，与 NdArray 相差 9e-8 |
| D13 | 禁止 `unsafe`，对不可信输入做模糊测试 | 全工作区 `forbid(unsafe_code)`；对 `Bundle::from_memory` 做 cargo-fuzz；proptest 检查不变量 |
| D14 | 模型可以编译进 Rust 代码，并与运行时共用内核 | `model!` 在编译期按运行时的规则校验模型包，把输入契约生成为类型、权重生成为常量；推理不分配内存、不依赖 `std`；与运行时逐位相同，跨平台也逐位相同（CI 在三个平台上验证） |
