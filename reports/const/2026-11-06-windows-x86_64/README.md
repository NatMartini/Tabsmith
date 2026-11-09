# The compiler predicts what the program predicts

[中文](#中文)

**Claim.** Every model that `model!` compiles predicts as a `const fn`: networks, and trees as
code or as arrays. While the crate compiles, the compiler predicts the bundle's examples with
its own floating point and must get the bits `tabsmith-runtime` predicted, or the crate does not
compile. For that, every kernel in `tabsmith-embedded` is a `const fn`, `exp` included: libm
0.2.16's algorithm (FreeBSD's e_exp.c), ported operation for operation. It gives libm's bits for
every input that is an `f32`, all 2^32 checked on Windows and on Linux, and for 3.2 billion
random `f64`.

**Where.** The development machine (AMD Ryzen 9 9950X3D2): Windows 11 (Rust 1.97, x86-64
MSVC) and WSL2 Ubuntu 26.04 (Rust 1.98.1). Speed and compile time on Windows, with other
programs running on the machine, so every comparison alternates the two builds.

## `exp` and the logistic function, on every `f32`

A binary model's probability is `sigmoid(z as f64)` of an `f32` logit `z`, which calls `exp(-z)`
or `exp(z)`. So every input of `exp` that a binary model can produce is an `f32`, and the check
covers all of them (`exp-probe/`, `cargo run --release -- exhaustive random 100000000`):

| | Windows | Linux |
|---|---|---|
| Inputs `x` and `-x` for every `f32` bit pattern | 4,294,967,296 | 4,294,967,296 |
| `kernels::exp` differs from `libm::exp` | 0 | 0 |
| The logistic function differs from 0.4.0's, built on libm | 0 | 0 |
| The logistic function falls from one `f32` to the next | 0 | 0 |
| Random `f64` (any bits, [-750, 750], [-30, 30], differences of two `f32`) | 3,200,000,000 | 3,200,000,000 |
| … where `exp` differs from libm | 0 | 0 |
| Time, 32 threads | 4.6 s + 2.6 s | 4.2 s + 2.4 s |

The third line makes `PROBABILITY_BOUNDS` sound: the logistic function as computed never
decreases between consecutive `f32` logits, so a logit inside `OUTPUT_BOUNDS` gives a probability
inside the bounds. Before, that held only "up to the rounding of `exp`". The kernels' own tests
repeat a sample on every CI system, macOS arm64 included: every 1,024th `f32` and 200,000 random
`f64` (`exp_gives_the_bits_of_libm`).

## The compiler against the program

`crates/tabsmith-macros/tests/const_eval.rs` compiles the seven fixture bundles, the three tree
models in both forms, and makes the compiler compute 48 rows of its own for each, with NaN and
infinities among the values, then their outputs and decisions; the test computes the same at
run time and compares every bit. The examples are predicted both ways too.

The first run found one difference, in the two linear models, on rows with infinities:

| | `multiclass-linear`, row 1, outputs 2 and 3 | `policies-linear`, row 4 |
|---|---|---|
| Program on x86-64 | `0xffc00000` (`-NaN`) | `0xffc00000` |
| Compiler | `0x7fc00000` (`+NaN`) | `0x7fc00000` |

IEEE 754 leaves the sign and payload of a NaN that an invalid operation creates, here
`inf - inf`, to the processor: x86 gives `-NaN`; ARM and rustc's constant evaluator give `+NaN`.
So the claim that the kernels give the same bits everywhere had an exception for a network's
NaN outputs. Every kernel that can return a NaN now returns one, `kernels::NAN_F32` or
`NAN_F64`, and the ten models agree bit for bit. Only inputs that overflow a network's
arithmetic reach a NaN through `predict`, which refuses non-finite features; trees never
produce one.

## Speed against 0.4.0

`speed-probe/` built twice, against 0.4.0 and against the working tree, run alternately, 8
rounds each, every measurement the median of 21 (`ab_summary.py speed-ab.txt`). Every output
hash is the same in both builds.

| Measure | 0.4.0 | Now | Change |
|---|---|---|---|
| `binary-gbdt` as code, `predict` | 64.4 ns | 40.5 ns | -37.2 % |
| `regression-gbdt` as code, `predict` | 113.3 ns | 91.3 ns | -19.4 % |
| `multiclass-gbdt` as code, `predict` | 117.1 ns | 101.2 ns | -13.6 % |
| `binary-mlp`, `predict` | 361.3 ns | 331.5 ns | -8.2 % |
| `binary-mlp`, `outputs` | 292.8 ns | 284.8 ns | -2.7 % |
| `regression-gbdt` as arrays, `outputs` of a random row | 1,534.5 ns | 1,613.5 ns | +5.2 % |
| `multiclass-gbdt` as arrays, `outputs` of a random row | 783.7 ns | 831.8 ns | +6.1 % |
| Covertype (169,248 nodes) as arrays, `outputs` | 3,718.5 ns | 3,813.7 ns | +2.6 % |

A prediction got faster, mostly in `features` and `decode`: `exp` is now in the kernel crate
and can be inlined. The walk through arrays is 3 to 6 % slower here; the same walk written with
`split_first_mut` instead of an index was no faster. Trees as code, the default up to 32,768
nodes, do not walk arrays.

## Compile time of the check on the examples

Each model in a crate of its own, built against 0.4.0 and against the working tree; three
rebuilds of the crate alone (`compile-time.ps1`, `compile-times.txt`):

| Model | Nodes | Form | `cargo build --release`, 0.4.0 / now | `cargo check`, 0.4.0 / now |
|---|---|---|---|---|
| Adult | 9,151 | code | 2.4 / 2.4 s | 1.0 / 1.1 s |
| California Housing | 18,823 | code | 5.1 / 5.0 s | 2.4 / 1.6 s |
| Covertype | 169,248 | arrays | 3.6 / 3.7 s | 1.4 / 1.6 s |

All three compile, so the compiler predicted every example of each with the runtime's bits.

## Not covered

- The comparison of `exp` with libm is a check of every `f32` and of samples of `f64`, not a
  proof. A model checker cannot carry floating-point multiplication at this size.
- The exhaustive check ran on x86-64. ARM runs the kernels' sampled test and the portability
  hashes in CI.
- The check on the examples covers the bundle's examples; `tests/const_eval.rs` adds made-up
  rows for the fixtures.

## Reproduce

```sh
cd reports/const/2026-11-06-windows-x86_64/exp-probe
cargo run --release -- exhaustive random 100000000
cargo test -p tabsmith-macros --test const_eval      # from the repository root
```

---

## 中文

**结论。** `model!` 编译的每个模型都以 `const fn` 预测：神经网络，以及代码或数组形式的树。crate 编译时，编译器用自己的浮点运算预测模型包里的示例，结果必须与 `tabsmith-runtime` 的预测逐位相同，否则 crate 不能编译。
为此，`tabsmith-embedded` 的每个内核都是 `const fn`，包括 `exp`：它是 libm 0.2.16 的算法（FreeBSD 的 e_exp.c）逐个运算的移植，对每个 f32 输入（全部 2^32 个，在 Windows 和 Linux 上都检查过）和 32 亿个随机 f64，都与 libm 逐位相同。

**logistic 函数。** 二分类模型的概率是 f32 logit 的 `sigmoid`，所以它能给 `exp` 的每个输入都是 f32，检查覆盖了全部。计算出来的 logistic 函数在相邻 f32 之间从不下降，因此 `PROBABILITY_BOUNDS` 对任何输入都成立；以前只能说“在 `exp` 的舍入范围内”成立。

**编译器对照程序。** `tests/const_eval.rs` 让编译器为 10 个模型各自生成 48 行输入（其中有 NaN 和无穷大），计算输出和决策，再在运行时算一遍逐位比较。第一次运行在两个线性模型上发现了差异：`inf - inf` 在 x86 上得到 `-NaN`（`0xffc00000`），在编译器和 ARM 上得到 `+NaN`（`0x7fc00000`）。
IEEE 754 把无效运算产生的 NaN 的符号留给处理器决定，所以“各平台逐位相同”以前对网络的 NaN 输出有例外。现在能返回 NaN 的内核都返回同一个 NaN，10 个模型逐位一致。只有让网络运算溢出的输入才会得到 NaN，树从不产生 NaN。

**速度。** 与 0.4.0 交替运行各 8 轮：生成代码形式的 `predict` 快了 14% 到 37%，网络快 8%，主要因为 `exp` 现在在内核 crate 里，可以内联；数组形式的遍历慢 3% 到 6%。所有输出哈希都相同。

**编译时间。** 编译期在示例上的检查几乎不增加编译时间：Covertype 的 16.9 万个节点，`cargo check` 多 0.2 秒。

**没有覆盖的。** `exp` 与 libm 的比较是对全部 f32 和 f64 样本的检查，不是证明；穷举检查在 x86-64 上运行，ARM 在 CI 里运行抽样测试和跨平台哈希。
