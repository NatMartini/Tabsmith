# Tabsmith: Verifiable Tabular Machine Learning in Rust

**Technical report** · Kevin Li · October 2026 · Tabsmith 0.1.0

> **Abstract.** Tabsmith is a tool, written in Rust, that turns a CSV or Parquet file into
> a *model bundle*: a small, self-describing directory that applications load with a
> dependency-light runtime crate. The work treats the usual failure modes of applied
> machine learning as engineering defects rather than matters of discipline:
> - **training/serving skew**: preprocessing written twice;
> - **leakage**: the test set consulted repeatedly, or entities straddling splits;
> - **irreproducibility**: nondeterministic training;
> - **silent degradation**: unchecked artifacts, fallbacks the user never sees.
>
> Each is prevented by construction and enforced by a test. This report describes the
> architecture (versioned contracts, one preprocessing implementation, an
> engine-independent runtime, and a procedural macro that compiles a bundle into `no_std`
> Rust code with the runtime's own arithmetic), the determinism guarantees (bitwise-exact
> resume, and why GPU training breaks them unless autotuning is disabled and TF32 is made
> explicit), and an in-house gradient-boosted tree learner. The evidence:
> - parity with scikit-learn: preprocessing to 2.3e-7, metrics below 1e-15;
> - single-record prediction latency 800–950× lower than an equivalent scikit-learn
>   deployment, and a 1 ms cold start instead of 0.6 s;
> - certification of GPU paths on real hardware;
> - fuzzing of the bundle loader;
> - the unchanged runtime compiled to WebAssembly;
> - compiled models that predict the same bits as the runtime on Linux, macOS and
>   Windows, 7–27× faster, and build for a Cortex-M4F microcontroller.

> **摘要。** Tabsmith 是一个用 Rust 编写的工具：把 CSV/Parquet 文件变成可校验的“模型包”，应用程序只需依赖一个轻量的
> 运行时 crate 即可加载和预测。本文把应用机器学习中常见的失败方式视为工程缺陷，而不是靠使用者自律来避免：
> 训练与服务不一致、数据泄漏、不可复现，以及静默退化。每一种都由构造方式加以排除，并由测试守护。报告介绍了
> 体系结构（带版本的契约、唯一的预处理实现、与训练引擎无关的运行时，以及在编译期把模型包变成 `no_std` Rust 代码、
> 并沿用运行时同一套算术的过程宏）、确定性保证（逐位精确的续训，以及为何 GPU 训练必须关闭 autotune 并显式声明
> TF32），以及自研的梯度提升树。证据包括：与 scikit-learn 的数值一致性、单条预测延迟低 800–950 倍、GPU 路径在真实
> 硬件上的认证、对模型包加载器的模糊测试、运行时无需修改即可编译为 WebAssembly，以及编译进代码的模型：在 Linux、
> macOS 和 Windows 上与运行时逐位相同，快 7–27 倍，并能为 Cortex-M4F 微控制器构建。

---

## 1. Introduction

Most applied machine learning on tabular data follows the same path. A notebook loads a
CSV, splits it, standardises and one-hot encodes the columns, fits a few models, picks
one, and pickles it. A service later re-implements the preprocessing, loads the pickle
into a Python process, and serves predictions. Each step is easy; the failures are in
the seams between steps. Sculley et al. [1] call this *hidden technical debt*. Kaufman et
al. [2] catalogue how *leakage* inflates offline results. The cures are well known, but
they are usually written as advice in checklists [3], not built into tools:
- fit preprocessing on training rows only;
- keep entities on one side of the split;
- look at the test set once;
- version everything;
- verify artifacts.

Tabsmith asks what a tool looks like when those rules are properties of its types and
data formats, and when a test fails if any of them breaks. Rust suits the question for
four reasons:
- **Enums with exhaustive matching** let the compiler check that every stage handles
  every task.
- **The crate system** makes "the runtime does not depend on the training engine" a fact
  CI can check with `cargo tree`.
- **Without a garbage collector or interpreter**, the runtime is a small library with
  microsecond latency, and it compiles to WebAssembly.
- **Procedural macros** can read a bundle while the application compiles. The model's
  input contract then becomes types that the compiler checks, and inference needs no
  standard library.

### Contributions

1. **An architecture** in which every stage boundary is a versioned, strictly parsed
   contract, and in which one preprocessing implementation serves training and
   prediction (§2, §3).
2. **A determinism discipline** under which interrupted training resumes bitwise-exactly
   on the CPU. An analysis of two sources of GPU nondeterminism in the Burn/CubeCL stack
   (timing-based autotuning, and implicit TF32 tensor-core arithmetic) and the policy
   adopted for each (§4).
3. **A deterministic histogram GBDT** in about 600 lines of safe Rust. It shares
   checkpointing and exact resume with the neural candidates, and its model format the
   runtime executes with zero numerical difference (§5).
4. **A verification-first runtime** (§6) and an evaluation against scikit-learn
   covering numerical parity, accuracy, latency, footprint and robustness (§7).
5. **Models compiled into code** (§6.1). A procedural macro turns a bundle into a
   `no_std` Rust module whose types are the model's input contract. The module shares
   the runtime's kernels, so it predicts the same bits as the runtime, on every platform;
   tests check this on seven bundles and three platforms.

### Scope

Release 0.1 covers supervised learning on one table: binary classification, multiclass
classification and regression. It trains three candidate families (a linear model, a
small multilayer perceptron, and gradient-boosted trees) on the CPU, and optionally the
networks on a GPU. It is aimed at datasets that fit in memory (thousands to millions of
rows), which is where most tabular problems are.

## 2. System overview

The code base is a Cargo workspace of six crates (about 20,300 lines of Rust, plus
about 3,300 lines of integration tests):

| Crate | Role | Depends on |
|---|---|---|
| `tabsmith-contracts` | Every record that crosses a stage boundary, with its format version; configuration; errors; file hashing | serde, serde_json, toml, thiserror, sha2 |
| `tabsmith-embedded` | The numeric kernels, `no_std` and allocation-free, shared by the runtime and by compiled models | libm |
| `tabsmith-runtime` | Loading and verifying bundles; the transform; model execution; `Predictor` | contracts, embedded, safetensors |
| `tabsmith-macros` | `model!`: a bundle compiled into a Rust module while the application builds | contracts, runtime, syn, quote |
| `tabsmith-workflow` | Data (Polars), splits, fitting, training (Burn), evaluation, bundles, reports | contracts, runtime, Burn, Polars |
| `tabsmith-cli` | The `tabsmith` binary, including `train`, `predict` and an HTTP `serve` | workflow |

A project moves through content-addressed stages:

```text
CSV/Parquet ─▶ Snapshot ─▶ Split ─▶ FittedTransform ─▶ Run (per candidate) ─▶ Evaluation ─▶ Bundle ─▶ Predictor
   s-<hash>      sp-<hash>    t-<hash>          run-NNN                 test ledger     b-<hash>
```

Each identity is a hash of everything that determines the stage. A split's id, for
example, covers the snapshot, strategy, seed, fractions, target and group column, and the
split algorithm's version. Re-running on unchanged inputs therefore reuses work, and a
change invalidates exactly its dependents. The command `tabsmith train data.csv --target
y` runs the whole chain:
1. bind and profile the data;
2. split it;
3. fit the transform on the training rows;
4. train every candidate;
5. rank the candidates on validation;
6. evaluate the chosen one once on test;
7. build and self-check its bundle.

When new data arrives, the same command retrains. It also scores the previously shipped
bundle and the new model on rows neither has seen, so the comparison between them is
fair.

A bundle is a directory. Its `manifest.json` records the format version and provenance,
and the SHA-256 of the eight other files:
- the input and output schemas;
- the fitted transform;
- the model configuration and its safetensors weights;
- the evaluation evidence;
- example inputs, with the predictions the trainer recorded for them.

The sample bundles are 18–127 KB.

## 3. Correctness by construction

### 3.1 Strict, versioned contracts

All bundle types are deserialised with `#[serde(deny_unknown_fields)]`, and every format
carries a version string, for example `tabsmith.std-tabular.v2` for the transform and
`tabsmith.trees.v1` for tree ensembles. A runtime refuses any version it does not
implement. The alternative, ignoring unknown fields, is how schema evolution usually
works, and here it would be wrong. When the transform gained missing-value tokens, an old
reader that skipped the new field would still have produced predictions, just different
ones. Tabsmith prefers a refusal to a quietly different number.

### 3.2 One preprocessing implementation

`tabsmith_runtime::transform::apply_transform` is the only function that turns raw
values into model inputs. Training, evaluation, batch prediction from the CLI, the HTTP
server and the application API all call it with the same frozen `FittedTransform`.
Values are typed (`Null`, `Bool`, `Int`, `Float`, `Text`), and coercion is strict:
- a number sent as text is rejected rather than parsed;
- `12` and `12.0` map to the same category;
- infinities are rejected.

Fitting happens in the workflow, but it calls the runtime's coercion functions, so
"what is a missing value" has exactly one definition. This makes training/serving skew
impossible by construction rather than unlikely by care. Section 7.1 measures the result
against scikit-learn.

### 3.3 Evaluation discipline

Selection uses the validation split only. Evaluations on the test split are recorded in
a *ledger* keyed by the hash of the model's weights and the data version. Scoring a
second model on the same test rows requires `--allow-test-reuse`, and the reuse is
written into the evaluation record (`test_use_index`, `test_shared_with`). With
`--group <column>`, rows of one entity never straddle splits. A property test checks this
invariant on random data (`splits_partition_rows_and_respect_groups`). Splits that cannot
support a fair evaluation are refused; a binary split without both classes in train,
validation and test is one example.

### 3.4 The task in the type system

Multiclass classification and regression were added after a binary-only first version.
The design question was how to avoid two things: task checks scattered through the code,
and structs with fields that are meaningful for one task only. The answer was to make
every task-dependent value an enum with exactly that task's data:
- `Targets`: labels or values;
- `Metrics`: binary, multiclass or regression metrics, internally tagged in JSON;
- the runtime's `Outcome`: probability and decision; class probabilities; or a value.

`Outcome` is flattened into `Prediction`, so clients still receive flat JSON. Adding the
two tasks then amounted to following the compiler's exhaustiveness errors. The binary path
stayed unchanged: its reference goldens and bitwise tests passed untouched. Regression
targets are learned standardised, and the scale travels in the output schema, so the
runtime returns values in the target's units.

### 3.5 Errors as data

Every failure is a `TabsmithError` with these parts:
- a stable code, such as `E_SCHEMA_MISSING_COLUMN` (all are listed in `docs/errors.md`);
- the phase in which it happened;
- where they apply, the offending field and the expected and actual values;
- a *next action* in plain words.

The CLI prints it for people, or as JSON with `--json`. The project's rule is that when
Tabsmith cannot do what was asked, it refuses rather than doing something nearby. A
missing GPU is an error, not a CPU fallback, and fp32 cannot be had on a TF32-only path.

## 4. Determinism

Exact reproducibility is a contract here, because three features depend on it. A crash
mid-training must be recoverable to the same model. The test ledger identifies models by
the hash of their weights. And a report is only evidence if it can be regenerated.

### 4.1 Sources of nondeterminism and their removal

| Source | Measure |
|---|---|
| Random number generator algorithms changing between library versions | Tabsmith's own SplitMix64 [4] with Lemire's unbiased bounded sampling [5]. Streams are derived from `(seed, label)`. The `rand` crate does not promise that `StdRng` stays the same across releases. |
| Minibatch order | A pure function of `(seed, epoch)`, so it can be recomputed at resume |
| Stochastic layers | None (no dropout) |
| Parallel reductions | Single-threaded CPU backend; trees sum in row order on one thread. Inference over many rows (validation, evaluation) is split by rows across cores, which changes no row's arithmetic |
| Ties | Splits are ordered by `(feature, bin)`, leaves by creation order |
| Optimizer state | Checkpoints hold model, Adam moments, loop state and best weights |

Three integration tests interrupt training and compare the result with an uninterrupted
run, bit for bit:
- a crash (the process is aborted after *n* optimizer steps);
- a clean cancel in the middle of an epoch;
- a plain stop and resume.

A further test damages the newest checkpoint and checks that resuming falls back to the
older one. The tree learner has its own bitwise resume test, and property tests check
that epoch orders are reproducible permutations.

### 4.2 GPU training: two findings

Training on GPUs uses Burn's CubeCL backends (CUDA, and Vulkan through wgpu). A probe on
an NVIDIA RTX 5090 found two independent problems for the contract above.

**Autotuning makes CUDA nondeterministic across processes.** CubeCL benchmarks candidate
kernels at run time and caches the fastest. Two processes can pick different matmul
kernels for the same shape, and those kernels accumulate in different orders, so the two
processes train different weights. Each process is consistent with itself, which is why
a single-process test does not catch it. Tabsmith builds the GPU backends with fusion
and *without* autotune. On an AMD integrated GPU, autotuning also panicked inside the
timing readback, and the device was lost. Disabling autotune removed that failure too.

**CUDA computes f32 matrix products in TF32.** Reading the source of `cubek-matmul` shows
the path. Its default strategy tries a tensor-core (`cmma`) routine first. On devices of
compute capability ≥ 8.0, that routine's `adjust_dtypes` converts f32 operands to TF32,
which has 10 mantissa bits. Weights after training differed from the CPU by up to 3e-2.
On the same GPU through Vulkan, which computes fp32, they agreed to about 1e-7. No public
setting disables this. PyTorch, JAX and TensorFlow all expose one; PyTorch has defaulted
to fp32 for matmuls since version 1.12. Tabsmith's policy:
- CUDA training requires `execution.precision = "tf32"`;
- asking for fp32 on CUDA is refused, with advice to use Vulkan for strict fp32.

Drafts of both upstream reports exist (§8).

### 4.3 Certification instead of hope

Because GPU behaviour depends on the GPU, driver and backend, a GPU path is trusted only
after it passes `tools/gpu_certify.py`. The script runs the real `tabsmith` binary and
checks four things:
1. two independent processes train bit-identical weights;
2. a run killed mid-training and resumed in a new process equals an uninterrupted run;
3. validation metrics agree with a CPU run, tightly for fp32 and more loosely for TF32;
4. the GPU-trained model bundles, and the CPU runtime's self-check passes.

Passing paths are recorded in `certified_devices.json`. An entry matches exactly on
backend, device name, OS, precision, and driver version or compute capability. With
`device = "auto-certified"`, a GPU is chosen only if it is certified and a cost model
predicts at least a 2× speed-up for the problem at hand. Section 7.3 has the results.

## 5. Models

**Linear and MLP.** These are trained with Burn 0.22 [6] on its Flex CPU backend. The
loss is class-weighted binary or softmax cross-entropy, or squared error for regression,
with L2 regularisation, and the optimizer is Adam [7]. Early stopping watches the
validation loss and keeps the weights of the best epoch. Weight initialisation uses
Tabsmith's PRNG, not Burn's, so that it is versioned with the project.

**Gradient-boosted trees.** Tree ensembles remain the strongest general method for
tabular data [8], so a third candidate was essential. The two mature libraries,
LightGBM [9] and XGBoost [10], are C++ with multi-threaded floating-point sums and their
own model formats. Binding one would have broken three properties: the pure-Rust build,
determinism, and the runtime's independence. Tabsmith implements the standard histogram
algorithm instead (in the style of LightGBM and scikit-learn's
`HistGradientBoosting` [11]):
- **Bins.** Candidate thresholds come from the training rows, up to `max_bins` per
  feature (255 by default).
- **Histograms.** Gradient and Hessian sums are taken per bin, so evaluating every split
  of a leaf costs one pass over its rows.
- **Growth.** Trees grow best-first: the leaf with the largest gain splits next. Growth
  is limited by leaves, depth and minimum rows per leaf, with an L2 penalty on leaf
  values.
- **Loss.** Gradient boosting [12] on the task's loss: class-weighted log loss; softmax
  cross-entropy with one tree per class and round; or squared error on the standardised
  target.

Three details make it deterministic and exactly portable to the runtime:
1. every sum runs in row order on one thread;
2. ties are broken by `(feature, bin)`;
3. leaf values are rounded to f32 before they update the training scores.

The trainer's validation scores are therefore exactly what the runtime computes, and the
recorded engine difference for trees is 0. One boosting round is one "epoch" of the
shared training state, so checkpoints, cancellation and bitwise resume needed no new
code.

The model format `tabsmith.trees.v1` is five flat arrays: feature, threshold, left,
right, value. Children always follow their parents, and the runtime walks the trees
iteratively. A `safetensors` file stores the arrays, with the integer arrays as I32.

## 6. The runtime

`tabsmith-runtime` depends directly on `tabsmith-contracts`, `tabsmith-embedded`, serde,
serde_json and safetensors, and nothing else: no training engine, data engine or GPU
stack. It executes dense networks and tree ensembles with its own small FP32 kernels.
They live in the `no_std` crate `tabsmith-embedded`, so that compiled models share them
(§6.1). Every run records the largest difference between Burn's outputs and the
runtime's on the validation rows ("engine Δ"): about 1e-7 for networks, and 0 for trees.
Having two implementations is the price of a small runtime, and the engine Δ is what
keeps the price safe.

Loading a bundle verifies it before anything is predicted:
- the SHA-256 of every file against the manifest;
- every format version;
- the consistency between transform, model and output schema;
- optionally, a *self-check*: the runtime replays the bundle's example records and
  compares the outputs with the predictions the trainer recorded.

The example records are synthetic, generated from the transform's statistics, so a
bundle never contains a training row. `PredictorHandle::try_update` gives applications
safe hot swaps. A new bundle replaces the active one only after it has loaded, passed its
self-check, and kept the input contract; `rollback_to` returns to an earlier bundle.

Two consequences show how small the runtime is.

**A Rust application.** The example application `examples/predict-app` builds to a
1.7 MB binary. CI checks that its dependency tree contains neither Burn nor Polars.

**A web page.** The runtime compiled to `wasm32-unknown-unknown` without a single
change. In `examples/wasm-predict`, a web page hands a bundle's files to
`Predictor::from_memory` through a 0.9 MB module (0.2 MB gzipped). It verifies the bundle
in a few milliseconds and predicts in about 10–20 µs per call, including the JSON round
trip between JavaScript and WebAssembly. The error contract survives the trip: removing a
field in the page yields `E_SCHEMA_MISSING_COLUMN` with its next action, exactly as a
Rust caller would see it.

### 6.1 Models compiled into code

Loading a bundle at run time needs a file system, an allocator and the standard library.
It also checks an application against the model's input contract only when the
application runs. `tabsmith-macros` removes all of these needs.
`model!(defects, "bundles/run-003")` runs while the application compiles. It loads the
bundle with `Predictor::load`, so it verifies everything listed above, and a broken or
edited bundle is a compile error. It then generates a module with:
- `Input`, with an enum for each categorical field and an `Option<f64>` for each
  number, so that a misspelled field or an unknown category does not compile.
  `Input::from_values` reads raw values with the runtime's rules;
- the weights as constants written with `f32::from_bits`, and the tree arrays with
  constant assertions that every split reads a real feature and points forward inside
  its own tree;
- `predict`, which allocates nothing and needs no `std`;
- `self_check()`, which replays the bundle's examples against the runtime's predictions
  recorded at compile time.

The generated code and the runtime call the same kernels in `tabsmith-embedded`: a dot
product with a fixed order of eight partial sums, dense layers, tree traversal summing
leaves in f64 in tree order, standardisation, sigmoid and softmax. Their order of
operations is part of the format. `exp` comes from `libm` [14], a pure-Rust port of
musl's math library, instead of the C runtime. Two test suites check the consequence:
- For seven bundles covering every task and model kind, the compiled model and the
  runtime return the same bits on the bundle's examples and on 3,000 random records each
  (proptest). The records include missing values, unseen categories, NaN, infinity and
  values of the wrong type. A record that one of them refuses, the other refuses too.
- The hash of 20,000 predictions per bundle equals the value recorded on Windows x86_64.
  CI runs this test on Linux x86_64 and macOS arm64 as well.

On the development machine, a compiled prediction takes 0.13–0.59 µs, against 3.5–4.2 µs
for the runtime's `predict_one`, which looks up and coerces named values first.
`examples/embedded-predict` compiles two bundles into a `no_std` library, without
`unsafe` code, that builds for a Cortex-M4F (`thumbv7em-none-eabihf`) and for
WebAssembly. With `model!(…, kani)` the input types derive `kani::Arbitrary`, and the
Kani model checker [13] proves that prediction never panics, for every input. The
proof goes stage by stage. `predict` is `decode(outputs(features(input)?))`, and each
stage is proved total for every value of its argument type, a superset of what the
stage before it returns. Rust's own checks stay on: overflow, division by zero and
indexing. Kani's additional floating-point checks are off. They are not about panics,
and CBMC treats any division by an infinity as a possible NaN, which flags `exp(-∞)` in
`libm` although IEEE 754 defines that quotient as 0. Without them, no check depends on
the network's arithmetic, so Kani slices it away. The layers then take 26 s; with
those checks on, they had not finished after 32 minutes. For the trees, `outputs` is
proved one walk at a time, through any tree; it adds the walks' leaf values, so it cannot
panic once every walk stays in range. All 14 walks in one proof ran out of memory with
14 GB. All six proofs pass with Kani 0.68 (CBMC 6.11), in 69 s together and under 2 GB
of memory each (`reports/kani/`).

## 7. Evaluation

All numbers below come from reports in the repository, which also hold the commands
that produced them.

### 7.1 Parity with scikit-learn

Every run can be exported and recomputed with scikit-learn [11]. A `ColumnTransformer`
is fitted on the same training rows, the metrics are computed on the same validation
probabilities, and, for the linear model, scikit-learn's L-BFGS solver finds the optimum
of the same weighted, L2-regularised objective. The results:
- the transforms agree to 2.3e-7 (Tabsmith computes in f32);
- the metrics agree to 3e-16, with identical confusion matrices;
- Tabsmith's Adam-trained linear model lands within 0.10 % of the solver's optimal
  objective.

CI guards these properties on a frozen fixture, with tolerances of 1e-5, 1e-12 and 1 %.

### 7.2 Against scikit-learn on the same problem

The benchmark runs the binary sample (10,000 rows) through Tabsmith and through the
equivalent pandas + scikit-learn code. Both sides get the same rows, split, features and
training settings, and each candidate is paired with its scikit-learn counterpart. The
run was on WSL2 on the same Ryzen 9 9950X3D2, with Python 3.14, scikit-learn 1.9.1 and
pandas 3.1. The full table, with p99 latency, batch throughput and artifact sizes, is in
`reports/benchmarks/2026-10-27-linux-x86_64-v2`.

| | Tabsmith linear | logistic regression | Tabsmith MLP | `MLPClassifier` | Tabsmith GBDT | `HistGradientBoosting` |
|---|---|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 | 0.7332 | 0.7295 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 | 0.8188 | 0.8184 |
| Training time | 0.11 s | 0.04 s | 0.10 s | 0.13 s | 0.67 s | 3.66 s |
| One prediction in an application (p50) | 2.4 µs | 1,993 µs | 2.6 µs | 2,007 µs | 3.0 µs | 2,844 µs |
| New process: load, predict one record | 1 ms | 618 ms | 1 ms | 612 ms | 1 ms | 625 ms |
| Peak memory of that process | 12 MB | 160 MB | 12 MB | 161 MB | 12 MB | 164 MB |

Quality is on par for the linear model and for the trees (ROC AUC 0.8188 against
0.8184). Tabsmith's small MLP is ahead of `MLPClassifier`, which was given the same
number of epochs. Tabsmith's single-threaded GBDT trained its 54 rounds in 0.67 s, while
`HistGradientBoostingClassifier`, which uses every core by default, took 3.66 s for the
same 54 rounds. That is a statement about a 7,356-row problem, where per-round overheads
dominate; on large data the parallel implementation should win.

The decisive differences are in deployment:
- a prediction from an application costs 2.4–3.0 µs instead of 2.0–2.8 ms;
- a new process answers after 1 ms instead of about 0.6 s, in 12 MB instead of 160 MB;
- the application is a 1.7 MB binary instead of a 354 MB Python environment.

### 7.3 GPU certification

Three GPU paths on one Windows 11 machine passed all four checks of §4.3. The machine
has an AMD Ryzen 9 9950X3D2, an NVIDIA RTX 5090 (driver 610.88) and an AMD Radeon
integrated GPU. Timings are for a 512×512 network with batch 2048 on 400,000 rows
(100,000 for the integrated GPU), as the marginal cost of a step: the difference between
4-epoch and 1-epoch runs, so one-off kernel compilation is excluded. The CPU reference
is the single-threaded Flex backend, at 32–35 ms per step.

| Path | Same weights in two processes | Resume after a crash | Δ vs CPU (balanced accuracy, ROC AUC) | Bundle self-check | ms/step | vs CPU |
|---|---|---|---|---|---|---|
| RTX 5090, Vulkan, fp32 | bitwise | bitwise | 0, 0 | pass, Δp = 0 | 4.8 | 6.6× |
| RTX 5090, CUDA, TF32 | bitwise | bitwise | 3.4e-3, 5.7e-4 | pass, Δp = 0 | 2.9 | 10.8× |
| Radeon integrated, Vulkan, fp32 | bitwise | bitwise | 0, 0 | pass, Δp = 0 | 44.6 | 0.8× |

The fp32 paths reproduce the CPU's balanced accuracy and ROC AUC exactly, and its log
loss to within 1e-8; TF32 moves the metrics in the third or fourth decimal. On the README's problem (a 32-16 network, batch 64), every GPU
is *slower* than the CPU end to end: 1.0–1.8 s against 0.11 s, because a step is
cheaper than one kernel launch and copy. `auto-certified` therefore keeps such problems
on the CPU. Its cost model is fitted to these measurements, and with them the
integrated GPU is never chosen.

The first certification run produced a finding of its own. Tabsmith's CUDA step was
17.6 ms, ten times slower than a plain Burn loop on the same model. The GPU was not the
cause. After every epoch, validation ran the runtime's forward pass (the arithmetic that
ships) on one CPU core, over 60,000 rows of a 512×512 network. Rows are independent, so
validation now splits them across cores. Each row's arithmetic is unchanged, so every
bitwise check still passes. The CUDA step went from 17.6 to 2.9 ms, and the CPU step
from 49 to 32 ms.

### 7.4 Robustness

The workspace has 140 tests and 6 doctests. They include property tests (proptest) that
check these invariants on random inputs:
- predictions are total and independent of batch composition;
- a compiled model and the runtime return the same bits, and refuse the same records;
- metrics do not depend on row order, and ROC AUC complements correctly;
- splits partition the rows and respect groups;
- epoch orders are reproducible permutations;
- damaged bundles are refused without a panic.

Bundles are the one input an application cannot trust, so `fuzz/` fuzzes
`Bundle::from_memory` with libFuzzer. The input is an arbitrary set of files, with hash
verification switched off for most inputs so that mutations reach the JSON, safetensors,
transform and tree parsers. A 900-second run executed 5.3 million inputs without a
crash, timeout or out-of-memory. It was seeded with network and tree bundles of all
three tasks, and coverage grew from 7,174 to 8,769 edges. An earlier run, with binary
network bundles only, executed 7.1 million inputs. The report is in `reports/fuzz/`.
All workspace crates are compiled under `unsafe_code = "forbid"`. CI denies clippy and
rustdoc warnings, including in the GPU feature builds that the CI runners cannot
execute.

## 8. Limitations and future work

- **Scale.** Data is held in memory, and CPU training is single-threaded by design (§4).
  This is the right trade at the target scale. For millions of rows, a deterministic
  parallel histogram construction would be the next step: per-thread histograms merged
  in a fixed order.
- **Trees.** There are no native categorical splits, monotone constraints or sampling
  schemes such as GOSS. Training is also slower than it needs to be: each leaf's
  histogram is built from its rows, where the usual trick derives one child's histogram
  as the parent's minus its sibling's. The learner trades breadth and speed for
  determinism and auditability, and the subtraction trick is the first speed-up to
  take.
- **GPU.** Fp32 on CUDA waits on an upstream switch. A draft issue for `cubek-matmul`
  proposes a precision policy modelled on PyTorch's
  `set_float32_matmul_precision`. A second draft, for `cubecl-wgpu`, reports the panic
  in the autotune timing path on the integrated GPU.
- **Determinism across platforms.** Inference is portable: the runtime and compiled
  models take `exp` from `libm`, and CI checks the prediction hashes of seven bundles on
  three platforms (§6.1). Training is not yet. Its guarantees are stated and tested per
  platform (OS, architecture, engine), and resuming a run on another platform is
  refused. The quick-start problem was trained on one machine under Windows 11 and
  under Linux (WSL2):
  - the gradient-boosted trees came out bit-identical (same SHA-256);
  - the two networks did not, and their test ROC AUC differs in the fifth decimal
    (0.8249 against 0.8250).

  The cause is the platform math library. Burn's CPU backend calls `exp` and `ln` from
  the C runtime. On 2 million inputs, Microsoft's runtime and glibc disagree by one ulp
  on 0.07 % of `f32` exponentials and 0.12 % of logarithms. A pure-Rust math library in
  the training engine would make the networks portable too.
- **Proofs.** The Kani proofs of §6.1 show that prediction never panics. They say
  nothing about numerical accuracy, which the equivalence and portability tests cover,
  and they cover the two models of `examples/embedded-predict`, not every bundle. Kani
  does not run on Windows; the proofs ran on Linux and are set up to run in CI.
- **Tasks.** Time-based splits, ranking and survival targets are outside 0.1. The task
  enums are where they would go.

## 9. Conclusion

Tabsmith shows that the guidance usually given to practitioners can live in the tool:
- fit preprocessing on training data only;
- keep entities together;
- use the test set once;
- make results reproducible;
- verify what you ship.

Each guideline is a type, a format or a refusal in the code, and each is checked by a
test. Rust made four of these cheap to state and enforce: ownership of the
preprocessing semantics in one crate, exhaustive handling of tasks, a runtime small
enough to run in a browser, and, through a procedural macro, a model whose input
contract the compiler checks and whose arithmetic is the same on every platform. The GPU findings show the contract's other value. Because
determinism was something the tests demanded, the first CUDA run that broke it was
caught. Both causes were then traced in the source: timing-based autotuning in CubeCL,
and TF32 inside the matmul library.

## References

1. D. Sculley et al. Hidden technical debt in machine learning systems. *NeurIPS*, 2015.
2. S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. Leakage in data mining: formulation,
   detection, and avoidance. *ACM TKDD* 6(4), 2012.
3. E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley. The ML test score: a rubric for
   ML production readiness and technical debt reduction. *IEEE Big Data*, 2017.
4. G. L. Steele Jr., D. Lea, C. H. Flood. Fast splittable pseudorandom number
   generators. *OOPSLA*, 2014.
5. D. Lemire. Fast random integer generation in an interval. *ACM TOMACS* 29(1), 2019.
6. Burn: a deep learning framework in Rust, version 0.22. https://burn.dev
7. D. P. Kingma, J. Ba. Adam: a method for stochastic optimization. *ICLR*, 2015.
8. L. Grinsztajn, E. Oyallon, G. Varoquaux. Why do tree-based models still outperform
   deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*, 2022.
9. G. Ke et al. LightGBM: a highly efficient gradient boosting decision tree.
   *NeurIPS*, 2017.
10. T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. *KDD*, 2016.
11. F. Pedregosa et al. Scikit-learn: machine learning in Python. *JMLR* 12, 2011.
12. J. H. Friedman. Greedy function approximation: a gradient boosting machine.
    *Annals of Statistics* 29(5), 2001.
13. The Kani Rust Verifier, a bit-precise model checker for Rust.
    https://github.com/model-checking/kani
14. libm: a port of musl's libm to Rust. https://crates.io/crates/libm
