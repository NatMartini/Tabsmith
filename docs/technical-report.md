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
> - single-record prediction latency 700–880× lower than an equivalent scikit-learn
>   deployment, and a 1 ms cold start instead of about 0.75 s;
> - certification of GPU paths on real hardware;
> - fuzzing of the bundle loader;
> - the unchanged runtime compiled to WebAssembly;
> - compiled models that predict the same bits as the runtime on Linux, macOS and
>   Windows, 12–57× faster, and build for a Cortex-M4F microcontroller.

> **摘要。** Tabsmith 是一个用 Rust 编写的工具：把 CSV/Parquet 文件变成可校验的“模型包”，应用程序只需依赖一个轻量的
> 运行时 crate 即可加载和预测。本文把应用机器学习中常见的失败方式视为工程缺陷，而不是靠使用者自律来避免：
> 训练与服务不一致、数据泄漏、不可复现，以及静默退化。每一种都由构造方式加以排除，并由测试守护。报告介绍了
> 体系结构（带版本的契约、唯一的预处理实现、与训练引擎无关的运行时，以及在编译期把模型包变成 `no_std` Rust 代码、
> 并沿用运行时同一套算术的过程宏）、确定性保证（逐位精确的续训，以及为何 GPU 训练必须关闭 autotune 并显式声明
> TF32），以及自研的梯度提升树。证据包括：与 scikit-learn 的数值一致性、单条预测延迟低 700–880 倍、GPU 路径在真实
> 硬件上的认证、对模型包加载器的模糊测试、运行时无需修改即可编译为 WebAssembly，以及编译进代码的模型：在 Linux、
> macOS 和 Windows 上与运行时逐位相同，快 12–57 倍，并能为 Cortex-M4F 微控制器构建。

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
   the runtime's arithmetic, so it predicts the same bits as the runtime, on every platform;
   tests check this on nine bundles and three platforms.
6. **Models imported with the library's arithmetic** (§6.4). XGBoost models become bundles
   whose margins are XGBoost's bit for bit, the C library's `logf` in the starting margin
   included; LightGBM models come within a bound the import reports, and a missing value that
   no imputation reproduces is refused.

### Scope

Release 0.4 covers supervised learning on one table: binary classification, multiclass
classification and regression. It trains three candidate families (a linear model, a
small multilayer perceptron, and gradient-boosted trees) on the CPU, and optionally the
networks on a GPU. It is aimed at datasets that fit in memory (thousands to millions of
rows), which is where most tabular problems are.

**Related tools.** scikit-learn [11], LightGBM [9] and XGBoost [10] are the reference
training libraries; Tabsmith's trees are a plain histogram GBDT in their style, without
sampling or native categorical splits, and they train between the two in speed on the
problems measured here (§7.2). In Rust, linfa and smartcore
collect algorithms in the spirit of scikit-learn; Tabsmith does not compete on the
number of algorithms but on what surrounds them: the leakage-free workflow, the
bundle, the evidence, and the model as a Rust program. Burn [5] and candle are
deep-learning frameworks; Tabsmith trains its networks with Burn and its runtime
depends on neither. PyO3 and ONNX runtimes let a Rust program call a model that lives
outside Rust's type system, which is the gap `model!` closes (Section 6.1).

## 2. System overview

The code base is a Cargo workspace of six crates (about 20,300 lines of Rust, plus
about 3,300 lines of integration tests):

| Crate | Role | Depends on |
|---|---|---|
| `tabsmith-contracts` | Every record that crosses a stage boundary, with its format version; configuration; errors; file hashing | serde, serde_json, toml, thiserror, sha2 |
| `tabsmith-embedded` | The numeric kernels, `no_std`, allocation-free and `const`, shared by the runtime and by compiled models | nothing |
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
| Parallel reductions | Single-threaded CPU backend for the networks; the trees' histograms are sums of integers, which have no order, so the result is the same on any number of threads. Inference over many rows (validation, evaluation) is split by rows across cores, which changes no row's arithmetic |
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

**Across operating systems.** Determinism within a machine is what exact resume and
the test ledger need. The trees go further: their training is plain Rust whose only
transcendental functions, `exp` in the gradients and `log` in the starting score and
the losses, come from the pure-Rust `libm` that inference uses, so the same data gives
the same trees on every operating system. Measured on Windows 11 and Linux (WSL 2) on
the same CPU, the trees' weight file has the same SHA-256 on both and all 39 epochs of
the training history are identical to every digit; the linear model's and the
network's weights, computed by Burn's CPU backend, differ, with validation losses
apart by up to 4.8 × 10⁻⁷ (`reports/reproducibility/`). The claim is about the
operating system and its C library; independence from the CPU architecture is
expected for the same reason but was not measured.

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
LightGBM [9] and XGBoost [10], are C++ with their own model formats. Binding one would have
broken three properties: the pure-Rust build, determinism across systems and partitions, and
the runtime's independence. Tabsmith implements the standard histogram algorithm instead (in the style of LightGBM and scikit-learn's
`HistGradientBoosting` [11]):
- **Bins.** Candidate thresholds come from counts of the training rows' values in cells of
  their binary form, up to `max_bins` per feature (255 by default).
- **Histograms.** Gradient and Hessian sums are taken per bin, so evaluating every split
  of a leaf costs one pass over its rows.
- **Growth.** Trees grow best-first: the leaf with the largest gain splits next. Growth
  is limited by leaves, depth and minimum rows per leaf, with an L2 penalty on leaf
  values.
- **Loss.** Gradient boosting [12] on the task's loss: class-weighted log loss; softmax
  cross-entropy with one tree per class and round; or squared error on the standardised
  target.

Exact arithmetic makes it deterministic and portable to the runtime:
1. a row's gradient and hessian become integers at a scale fixed per tree, chosen so that the
   largest is at most 2^30, and a histogram cell is a sum of integers. It has no rounding and no
   order, so it holds the same value on any number of threads and in any order of the rows, and
   one child's histogram is exactly its parent's minus its sibling's;
2. every decision is a pure function of those sums: the largest gain, first in
   `(feature, bin)` order on ties; the leaf with the largest gain, first in creation order;
3. leaf values are rounded to f32 before they update the training scores.

With the order of the sums out of the way, the rest is free to be fast
([D15](design-decisions.md#d15-the-trees-add-integers-on-workers-that-spin)). Exclusive one-hot
columns share a pass, a pass adds its rows to two tables in turn, and sections run on workers that
spin between them, which start in a microsecond or two where rayon's sleeping workers take 20 to
80 µs on Windows and 300 to 1,400 µs in WSL 2. Against LightGBM and XGBoost on one machine, the
same rows and the same settings, with quality on par or better, the whole training of Covertype
takes 8.0 s, against 6.3 s and 12.2 s (`reports/training/`).

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

A batch of 4,096 rows or more runs on up to 16 plain scoped threads, with no dependency
added: 8 times the one-thread rate on 100,000 records with the trees, 7 times with the
network. Every row is computed as on one thread and the results join in row order. If a
piece fails, the whole batch runs again on the calling thread, so the error is the one a
single thread reports. A batch below that size starts no threads (`reports/inference/`).

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
1.9 MB binary. CI checks that its dependency tree contains neither Burn nor Polars.

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
- each tree as a `const fn` of nested `if`s, up to 32,768 nodes by default;
- `predict`, which allocates nothing and needs no `std`, and is a `const fn`;
- a constant assertion that the compiler, predicting the bundle's examples itself, gets
  the runtime's bits for each;
- `self_check()`, which replays the bundle's examples against the runtime's predictions
  recorded at compile time, on the device.

The generated code and the runtime call the same kernels in `tabsmith-embedded`: a dot
product with a fixed order of eight partial sums, dense layers, tree traversal summing
leaves in f64 in tree order, standardisation, sigmoid and softmax. Their order of
operations is part of the format. `exp` is not the C runtime's: it is the algorithm of
`libm` [14] (FreeBSD's e_exp.c), written in the kernel crate operation for operation as a
`const fn`, with libm's bits for all 2^32 `f32` inputs and 3.2 billion random `f64`. A
generated tree repeats the traversal's
arithmetic in code: a row goes left on `<=`, so NaN goes right, and each output adds its
trees in order, in f64. Three test suites check the consequence:
- For nine bundles covering every task and model kind, the compiled model and the
  runtime return the same bits on the bundle's examples and on 3,000 random records each
  (proptest). The records include missing values, unseen categories, NaN, infinity and
  values of the wrong type. A record that one of them refuses, the other refuses too.
- The hash of 20,000 predictions per bundle equals the value recorded on Windows x86_64.
  CI runs this test on Linux x86_64 and macOS arm64 as well.
- For every node of every tree of the three tree fixtures, a row is built to reach it, at
  the split's exact threshold and one representable number on either side, and each tree
  is compared on its own with the arrays. Changing the generator on purpose (a threshold or
  a leaf by one representable number, `<` for `<=`, swapped children, trees added to the
  wrong output) fails this suite in all six cases.

On the development machine, a compiled prediction takes 0.06–0.33 µs, against 3.6–4.2 µs
for the runtime's `predict_one`, which looks up and coerces named values first.

Generated trees are the fast path because a branch knows its constant and its next
instruction, where a walk over arrays loads a node's feature, threshold and child from
memory that holds every other tree too. On 10,000 real rows on one thread:

| Model | Trees | Nodes | Arrays | Code | `outputs_batch` |
|---|---|---|---|---|---|
| Fixture | 14 | 854 | 0.169 µs | 0.060 µs | 0.063 µs |
| Adult | 155 | 9,151 | 1.98 µs | 0.79 µs | 0.80 µs |
| California Housing | 309 | 18,823 | 4.04 µs | 1.54 µs | 1.54 µs |
| Covertype, 400 rounds | 2,800 | 169,248 | 75.2 µs | 38.0 µs | 14.4 µs |

The program gets smaller too: on a Cortex-M4F the fixture takes 11.3 KB of flash instead of
17.4 KB. The cost is build time, 3.4 s instead of 1.2 s for a release build at 18,823 nodes
and 115 s instead of 27 s at 169,248, which is why the default stops at 32,768 nodes;
`tree_code` and `tree_arrays` override it. Above that size, `outputs_batch` and the runtime
go tree by tree over blocks of rows, which keeps one tree in the cache while the block walks
it, and which changes no bit because each output still adds its trees in order. Below it,
row by row is as fast. The numbers are in `reports/inference/`.

**The compiler runs the model.** Every kernel is a `const fn`, and so are the generated
`features`, `outputs`, `decode` and `predict`, for networks and for trees in both forms.
rustc's constant evaluator computes IEEE 754 arithmetic in software with the hardware's
rounding, so it predicts the same bits. The generated module asserts that it does: the
compiler predicts the bundle's examples while the crate compiles and compares the bits with
the runtime's, and a model it computes differently does not compile. An application can
predict in its own constants, where a refused input is a compile error. On Covertype's
169,248 nodes the check adds 0.2 s to `cargo check`. One difference had to go first:
IEEE 754 leaves the sign of a NaN from an invalid operation to the processor, and
`inf - inf` in a network's last layer gave `-NaN` on x86 but `+NaN` in the compiler and on
ARM. Every kernel that can return a NaN now returns the same quiet NaN, so the claim of
identical bits holds for NaN results too (`reports/const/`).

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
the network's arithmetic, so Kani slices it away. The layers then take 25 s; with
those checks on, they had not finished after 32 minutes. Generated trees are proved all at
once: every index is a constant the compiler has checked, so no check depends on a
comparison, and `outputs` over all 14 trees takes 4.8 s and 104 MB. Trees kept as arrays are
proved one walk at a time, through any tree, in 10 s and 1.7 GB; `outputs` adds the walks'
leaf values, so it cannot panic once every walk stays in range. All 14 walks in one proof
ran out of memory with 14 GB. The harnesses are generated by `model!` itself, with loop
bounds from the model's shape, so nothing is written by hand and every compiled model
carries the same proofs.

**From the runtime to the bounds.** Not panicking says nothing about the values, so the
trees get two more proofs each, which together link the runtime to the bounds:

1. The runtime walks the bundle's arrays with `kernels::Trees::leaf`; `model!` reads the
   same arrays through the runtime and keeps them as `TREES`. Same function, same data.
2. `<model>_tree_<t>_is_its_walk`: the code of tree `t` gives the same bits as that walk,
   for every row of features, NaN and infinities included (Kani, one proof per tree).
3. `outputs` adds, per output, the trees in tree order in `f64`, as
   `kernels::Trees::outputs` does, and rounds to `f32` at the end. This link is by
   construction, and `tests/tree_code.rs` checks it on a row for every node and on random
   rows. So the compiled model predicts what the runtime predicts for every input, not
   only for the bundle's examples.
4. `<model>_tree_<t>_stays_in_its_range`: every row of every box of feature values reaches
   a leaf of the tree inside `Trees::leaf_range` of the box (Kani, one proof per tree).
5. `Trees::bounds` adds the ranges in the order of link 3. Adding a smaller leaf to a
   smaller sum never gives a larger sum, and rounding to `f32` keeps the order: facts of
   IEEE 754 arithmetic, which tabsmith-embedded's own Kani proofs check bit for bit on the
   operations both functions use. So `output_bounds` and `OUTPUT_BOUNDS` hold for every row.

Each proof takes one tree, because with a symbolic tree index the range proof of all 14 ran
out of 14 GB, and the range is proved on the code, where every index is a constant. Through
the arrays, the range proof of one tree took 82 s and over 7 GB, against 5 s and 130 MB on
the code. Trees kept as arrays therefore get their code inside the proofs only, where link 2
connects it to the walk. Four further
harnesses must fail, each with one deliberate mistake (another tree, a threshold one unit in
the last place off, a strict comparison in the range, and the NaN rule of 0.4.0, below), so
that the generated proofs cannot pass by checking nothing. Writing link 4's assumption
down found a bug: the documentation said that a NaN bound lets a feature be NaN, and in
0.4.0 a NaN lower bound did not. A box with `lo` NaN and `hi` below a threshold reached no
leaf, while a NaN row goes right at every split. All 69 harnesses of the example (nine that
it never panics, 56 for the trees in both forms, four that must fail) pass with Kani 0.68
(CBMC 6.11) in one plain `cargo kani`, in 459 s and under 1.8 GB each; the three lemmas run
apart (`reports/kani/`).

**What else the compiler knows.** Proving the absence of panics is the floor. The
generated module also carries properties of the model's values, computed by rustc's
constant evaluator from the generated constants:

- *Output bounds.* `output_bounds(&lo, &hi)` bounds the raw outputs over a box of
  feature values; `OUTPUT_BOUNDS` is the unbounded box. For trees, each tree contributes
  the smallest and the largest leaf the box can reach, summed in the order inference
  sums; rounding is monotone, so the bound is sound with no error margin, as the Kani
  proofs above check for every box. For a
  network, interval arithmetic runs through the layers and is widened by the worst
  rounding error of the kernel (one unit roundoff per product and per addition, on the
  sum of the magnitudes). A binary tree model therefore knows `PROBABILITY_BOUNDS`
  before it runs, and a regression `VALUE_BOUNDS`; the embedded example asserts at
  compile time that its defect probability lies strictly inside (0, 1). The compiler
  computes them with the kernels `decode` calls, and the logistic function as computed
  never decreases from one `f32` logit to the next (all 2^32 checked), so no input leaves
  the probability bounds. The macro computes the same bounds at expansion time with the
  same `const fn`s and checks that rustc agrees bit for bit.
- *Monotonicity.* Trees trained under a monotone constraint (`train.gbdt.trees.monotone`)
  keep it by construction, as in XGBoost: a split on the constrained feature is taken
  only when the lesser side's weight is at most the greater side's, and from then on
  the two subtrees' leaves are bounded at the midpoint. The constraint is recorded in
  the bundle; the loader and the compiler both check the structural property (at every
  constrained split, every leaf on one side is at most every leaf on the other), so the
  claim cannot be edited in, and the sum of monotone trees is monotone because
  rounding is.
- *Cost.* `WORST_CASE` counts the comparisons, multiply-adds, additions and
  exponentials of one prediction. No loop bound depends on the data, so these are the
  worst case, and exact for a network.

### 6.2 What a bundle says about itself

Three more files of evidence travel with a model, all computed on the validation split
when the bundle is built (the test split is used once, for the evaluation, and for
nothing else):

- **A conformal calibration** (`conformal.json`): split conformal prediction with the
  finite-sample quantile of the nonconformity scores (|y − value| for a regression,
  1 − p(true label) for a classification; the ⌈(n + 1)(1 − α)⌉-th smallest). Every
  prediction then carries an interval `value ± q` or a prediction set (the labels with
  probability at least 1 − q): on rows distributed like the calibration rows, the
  truth lies inside with probability about 1 − α (α = 0.1). The calibration rows also
  selected the model, which makes the guarantee approximate rather than exact; the file
  says so, and records the coverage on its own rows. The compiled model carries the
  same calibration (`CONFORMAL`, `interval`, `prediction_set`), so a microcontroller
  can give an interval too.
- **Permutation importance** (`importance.json`): the drop of the selection metric when
  one input field's values are shuffled across the validation rows, three shuffles per
  field, largest first. It is model-agnostic and costs one prediction pass per shuffle.
  Per-prediction attributions (TreeSHAP) are not implemented: the tree format does not
  record node coverage, which exact TreeSHAP needs.
- **Drift at serving time**: `tabsmith serve` keeps, per input field, the standardised
  mean and spread of the values it has seen against the training mean and standard
  deviation in the bundle's transform, and the share of unseen categories, and lists
  the fields that moved (`GET /stats`). It needs nothing the bundle does not already
  carry.

### 6.3 Rules over regions of inputs

The bounds of §6.1 say what a model can output over a box; an application usually needs a
narrower and stronger statement about a region it cares about. `tabsmith audit`,
`Predictor::audit` and `model!(…, rule(…))` take a rule as one line of text,
`conditions => claim`: for example `age in 17..=25, capital_gain == 0 => probability <= 0.99`,
or `_ within 3 sd => value <= 600000`, where `_ within k sd` keeps every other numeric field
within `k` training standard deviations of its mean. The answer is a proof, an input that
breaks the rule, or "undecided" when the budget runs out (D18).

*Search.* The region becomes a box of feature values, and the search cuts boxes until each
one's bounds settle the claim (`Claim::holds_within`: for a binary model or a regression, the
bound on the side the claim limits decides exactly, because the logistic function and
`mean + std · z` never decrease; a multiclass label needs a margin of 10⁻⁶ between outputs
unless the box is one point). A category's slots stay one-hot throughout: a slot at 1 sets its
category's other slots to 0, and all of them at 0 is allowed only where the bundle ignores
unseen categories. A box over slots as plain numbers would hold inputs with two categories at
once; a prototype that did so reported that 87 % of Adult's labels flip within 0.05 standard
deviations, against 64 of 200 once categories were kept one-hot. For trees the search splits at
the threshold where the widest tree's range is still open, so it is exact: a box where every
tree reaches leaves of one value is a single prediction, and the search ends with a proof or a
counterexample [16, 17, 18]. Every 64 boxes a greedy dive follows each tree toward the leaves
that break the claim, which finds most counterexamples at once. For networks the boxes close
with `dense_bounds`, the interval bound propagation of §6.1 [21], and counterexamples come from
predicting each box's centre and the corner that a linearisation at the centre points to; the
split is the feature whose sensitivity times width is largest [20]. Interval bounds cannot
refute, so a network's rule can stay undecided.

*Checking the answers.* A counterexample is a record built inside the box (as close to the
training means, or to a given record, as the box allows, whole numbers where the field held
whole numbers), predicted by `predict_one`, and reported only when that prediction breaks the
claim. A proof is a certificate, the splits in the order of a walk that visits each left part
first, and `rules::Walk` with the kernels' bounds checks it again before it is reported. Kani
proves the step that makes the closed boxes cover the region: a split keeps every input of a
box in the part its own value selects, and that part is never taken for empty, for every box,
row and finite threshold of a domain with numbers and both kinds of category. A test runs the
tree search on 400 random small ensembles and compares every answer with every row of the
region, one per cell of the trees' thresholds.

*The compiler.* `model!(…, rule(…))` audits each rule while the crate compiles, with the same
budget of a million boxes. A rule that fails is a compile error that prints an input breaking
it; an undecided rule is one too. A rule that holds becomes a constant certificate and a
constant assertion that walks it with the module's `output_bounds`, so the compiler checks the
proof in its own floating point, with the per-tree ranges that the proofs of §6.1 cover.
rustc's constant evaluator takes about 3 µs per tree node per box (a proof of 3,244 boxes over
3,610 nodes took 33 s), so a proof above a million of those is checked at expansion instead,
and the module's documentation says which.

*Findings.* On Adult (155 trees), "nobody of 17 to 25 without capital gain or loss is predicted
over $50K with probability above 0.99" holds, with a proof of 338,266 boxes found and checked in
under 8 s; 0.9 fails. On California Housing (309 trees), whose training targets stop at
$500,001, the search finds a district predicted at $803,579 whose every column lies in its
training range; the combination does not occur in any census. The label stability of 200
Adult records (`--stability`): within 0.05 standard deviations of each record's numbers, 50 can
change their label, each with an input that does. Of 60 records, 15 predicted over $50K fall
to between 2 and 20 % when their capital gain and capital loss go from $0 to $1 (for 7, nothing
else moves; nobody in the data has both, and the smallest positive values are $114 and $155),
because the
trees split at "≤ 0": the threshold sits at the largest training value on its left, so a value
never seen between 0 and the smallest positive one takes the right branch, that of the small
capital gains in training (`reports/rules/`).

### 6.4 Models from XGBoost and LightGBM

Most gradient-boosted models in use were trained with XGBoost [10] or LightGBM [9].
`tabsmith import` turns their saved models into bundles, so that everything above applies to
them: the runtime, compilation with `model!`, the proofs, the browser build and rules (D19). The
aim was the library's own arithmetic, where the format can hold it, and a stated bound where it
cannot. A converter that re-implements the walk and compares within a tolerance cannot say which
of its differences are rounding and which are a missing value sent the wrong way.

*What the libraries compute.* Walking saved models by hand in several precisions and comparing
the bits with the libraries' predictions settled it: XGBoost 3.4 rounds each input to f32, sends
`x < s` left and a NaN by the split's `default_left`, and adds f32 leaves in f32, in tree order,
from a margin in f32; LightGBM 4.7 compares f64 inputs with f64 thresholds, `x <= t`, and adds
f64 leaves in f64.

*Making the bundle compute it.* The import stores `s.next_down()` as Tabsmith's threshold, which
splits every f32 input as `x < s` does. Thresholds and leaves are parsed from their decimal text
straight to f32: through f64, a decimal can round twice and land on the neighbouring f32. A tree
model may say `"sum": "f32"`, and every walk of it, the runtime row by row and tree by tree, the
compiled code in both forms, the compiler's constant evaluation, `output_bounds` and the rule
engine's search, adds its leaves in f32 (`Trees::outputs_in_f32`, `bounds_in_f32`; rounding an
f32 sum never reverses an order, which two Kani lemmas check). A multiclass model's trees are put
in Tabsmith's order, tree `t` for class `t % K`, each class keeping the order of its own sum.

*The C library in the margin.* A binary model stores its base score as a probability `p`, and
XGBoost starts each margin from `-logf(1.0f/p - 1.0f)`. C libraries round `logf` differently. Of
the 2,139,095,039 positive finite f32 inputs, the correctly rounded value differs from glibc 2.43's
`logf` on 416,909, and the Windows CRT's `logf` on 501,924, so XGBoost itself starts some binary
models from different margins on Linux and on Windows. The import carries a port of Arm's
optimized-routines `logf` [22], the code of glibc and musl, which equals glibc on all of them,
with and without fused multiply-adds, and stores the margin it gives: a bundle starts from Linux
XGBoost's bits on every platform. The inputs that matter are `1/p − 1` near the class balance,
where the difference is common: for 12 of 1,596 random base scores in (0.02, 0.98) glibc is not
correctly rounded, and models trained with those 12 would have 1,498 of 12,000 margins one ulp
off with the correctly rounded start.

*Missing values.* Each split sends a missing value one way: XGBoost by `default_left`, LightGBM
by `default_left` for its missing type NaN and as the value 0 for its missing type None. When one
value takes the missing value's way at every split on a field, the import imputes it and the
bundle stays exact. When the splits disagree, no imputation reproduces the library, and the field
refuses missing values (`MissingRule::Refuse`, `E_INVALID_VALUE`). Both libraries choose a
direction per split, so fields that had missing values in training usually refuse them; a
per-split direction in the format would remove that limit (§8).

*Evidence.* XGBoost 3.4.1 and LightGBM 4.7.0 made models on 50,000 rows of the sample data in
WSL and predicted 100,000 rows, a tenth with one or two fields set missing; Tabsmith compared on
Windows (`reports/import/`):

| Model | Trees | Rows compared | Result |
|---|---:|---:|---|
| XGBoost binary, 4 classes, regression | 300, 1,200, 300 | 270,980 | every margin bit for bit |
| XGBoost binary, 12 base scores where glibc's `logf` is not correctly rounded | 5 each | 11,901 | every margin bit for bit |
| LightGBM binary, 4 classes, regression | 300, 1,200, 300 | 270,980 | largest difference 3 % to 17 % of the reported bound |

The rows not compared hold a missing value in a refusing field, and each was refused. LightGBM's
bound adds the leaves' rounding to f32, both sums' rounding and the output's; LightGBM's thresholds
are rounded down to f32, which takes LightGBM's branch for every f32 input. Given rows that are
not f32 values, LightGBM took another branch on 31 to 91 of 90,000 rows, where an input lay
closer to a threshold than f32 resolves.

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
`reports/benchmarks/2026-11-01-wsl-linux-x86_64`.

| | Tabsmith linear | logistic regression | Tabsmith MLP | `MLPClassifier` | Tabsmith GBDT | `HistGradientBoosting` |
|---|---|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 | 0.7356 | 0.7260 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 | 0.8220 | 0.8172 |
| Training time | 0.09 s | 0.05 s | 0.13 s | 0.15 s | 0.08 s | 3.93 s |
| One prediction in an application (p50) | 2.6 µs | 2,326 µs | 2.9 µs | 2,413 µs | 3.7 µs | 3,709 µs |
| A batch of 10,000 records | 4.7 M rows/s | 0.45 M rows/s | 3.5 M rows/s | 0.21 M rows/s | 2.7 M rows/s | 0.34 M rows/s |
| New process: load, predict one record | 1 ms | 738 ms | 1 ms | 770 ms | 1 ms | 797 ms |
| Peak memory of that process | 12 MB | 162 MB | 12 MB | 162 MB | 12 MB | 165 MB |

Quality is on par for the linear model and for the trees (ROC AUC 0.8220 against
0.8172). Tabsmith's small MLP is ahead of `MLPClassifier`, which was given the same
number of epochs. Tabsmith's GBDT trained its 62 rounds in 0.08 s, while
`HistGradientBoostingClassifier` took 3.93 s for the same 62 rounds. Most of that goes to
the class weights both sides use (a separate fit took 3.3 s with them and 0.13 s
without), so this says little about the trees themselves; larger data is below.

The decisive differences are in deployment:
- a prediction from an application costs 2.6–3.7 µs instead of 2.3–3.7 ms;
- a batch of records runs 8 to 16 times faster, on up to 16 threads, and 1.3 to 3 times
  faster on one;
- a new process answers after 1 ms instead of about 0.75 s, in 12 MB instead of 160 MB;
- the application is a 1.9 MB binary instead of a 354 MB Python environment.

**Public datasets.** The same comparison on three public datasets, one per task (UCI
Adult, California Housing, UCI Covertype; `reports/datasets/`), gives the same picture
for quality: every candidate matches its scikit-learn equivalent on the same rows and
split to within the test split's resolution, and the trees agree to the third decimal
(Adult: ROC AUC 0.9301 against 0.9297; California: R² 0.8438 on both sides; Covertype:
accuracy 0.8995 against 0.8984). The trees now train faster than XGBoost and close to
LightGBM, on one machine and the same rows and settings: Covertype's 400 rounds take
8.0 s, against 6.3 s and 12.2 s, at accuracy 0.8985, 0.8903 and 0.8546; Adult's 180 rounds 0.20 s,
against 0.12 s and 0.39 s; California's 334 rounds 0.15 s, against 0.10 s and 0.36 s
(`reports/training/`). The linear model and the network train as fast or faster on one thread
than scikit-learn's.

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

The workspace has 305 tests and 15 doctests. They include property tests (proptest) that
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

- **Scale.** Data is held in memory. The networks train on one thread by design (§4);
  the trees run on up to 12 threads, with a result that does not depend on the count
  (§5). On 1,000,000 rows (a 172 MB CSV, 35 features) the whole `tabsmith train` runs in
  15 s, 140 rounds in 1.2 s (`reports/training/`). Out-of-core
  training (reading the data in row groups) is not implemented, so ten million rows
  would need about 16 GB.
- **Trees.** There are no native categorical splits or sampling schemes such as GOSS.
  (Monotone constraints exist: kept by construction, recorded in the bundle, checked by
  the loader and by the compiler.) The learner trades breadth for determinism and
  auditability. LightGBM stays 1.5 to 1.8 times as fast on the two smallest problems.
- **GPU.** Fp32 on CUDA waits on an upstream switch. A draft issue for `cubek-matmul`
  proposes a precision policy modelled on PyTorch's
  `set_float32_matmul_precision`. A second draft, for `cubecl-wgpu`, reports the panic
  in the autotune timing path on the integrated GPU.
- **Determinism across platforms.** Inference is portable: the runtime and compiled
  models compute `exp` with libm's algorithm in the kernel crate, and CI checks the prediction hashes of nine bundles on
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
- **Proofs.** The Kani proofs of §6.1 show that prediction never panics and, for trees,
  that the compiled code is the runtime's walk and that the output bounds hold. Link 3,
  the order of the additions in `outputs`, holds by construction and is tested, not
  proved. Nothing is proved about a network's values beyond its tested bounds, about
  `outputs_batch`, or about `PROBABILITY_BOUNDS`, which rounds through `exp`. The proofs
  cover the models of `examples/embedded-predict`, not every bundle, until a user runs
  `cargo kani` on their own. A walk proof grows with the whole ensemble, because the walk
  indexes the arrays of every tree: about 7 s and 1.3 GB per tree at 854 nodes and 32 s
  and 5.7 GB at 3,610, so a model of tens of thousands of nodes takes hours. Kani does not run on Windows; the
  proofs ran on Linux and run in CI, except tabsmith-embedded's rounding lemmas, which do
  not depend on the code and take about 10 minutes.
- **Rules.** A rule's "every input" is every input `predict` accepts, realistic or not:
  boxes cannot express correlations between fields, and a dataset's own one-hot columns
  (Covertype's soil types) are numbers to the model. Interval bounds make a network's
  tight rules undecided. Search time grows with the ensemble and with how close a claim is
  to the truth: a claim near Adult's true maximum stays undecided after a million boxes.
  A stability range of whole-number fields is rounded at its ends, but may hold fractions
  between them. A multiclass claim on a probability (rather than a label) is not
  supported.
- **Imports.** A field whose splits send a missing value both ways refuses missing values: the
  format has one way through a split, and a second, per-split way for missing values would touch
  every walk, bound and proof and the rule engine's boxes. LightGBM is within a bound, not bit
  for bit, since the format's leaves are f32; categorical splits, other links and random forests
  are refused. An imported bundle carries no training statistics, so rules cannot use `within k
  sd`.
- **Tasks.** Ranking and survival targets are outside 0.1; the task enums are where
  they would go. A chronological split exists (`--time`); rolling validation over
  several time windows does not.

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
15. V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World*.
    Springer, 2005.
16. A. Kantchelian, J. D. Tygar, A. D. Joseph. Evasion and hardening of tree ensemble
    classifiers. *ICML*, 2016.
17. H. Chen, H. Zhang, S. Si, Y. Li, D. Boning, C.-J. Hsieh. Robustness verification of
    tree-based models. *NeurIPS*, 2019.
18. J. Törnblom, S. Nadjm-Tehrani. Formal verification of input-output mappings of tree
    ensembles. *Science of Computer Programming* 194, 2020.
19. G. Katz, C. Barrett, D. L. Dill, K. Julian, M. J. Kochenderfer. Reluplex: an efficient
    SMT solver for verifying deep neural networks. *CAV*, 2017.
20. R. Bunel, I. Turkaslan, P. H. S. Torr, P. Kohli, M. P. Kumar. A unified view of
    piecewise linear neural network verification. *NeurIPS*, 2018.
21. S. Gowal et al. On the effectiveness of interval bound propagation for training
    verifiably robust models. arXiv:1810.12715, 2018.
22. Arm Optimized Routines: `logf` (single-precision logarithm), as in glibc and musl.
    https://github.com/ARM-software/optimized-routines
