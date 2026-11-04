# tabsmith-macros

Compiles a [Tabsmith](https://github.com/NatMartini/Tabsmith) model bundle into Rust code
while your crate builds. You get a typed interface, and inference that runs without the
standard library: on a server, in WebAssembly, or on a microcontroller.

```toml
[dependencies]
tabsmith-macros = "0.4"
tabsmith-embedded = "0.4"   # the kernels the generated code calls (no_std)
```

```rust
tabsmith_macros::model!(defects, "models/defects");   // relative to Cargo.toml

let reading = defects::Input {
    line: Some(defects::Line::B),          // categories seen in training: an enum
    shift: Some(defects::Shift::Night),
    hours_in_service: Some(12_000.0),      // numbers: Option<f64>; None is imputed
    temp_mean: Some(71.4),
    ..defects::EXAMPLES[0]
};
let p = defects::predict(&reading)?;
if p.label == defects::Label::Positive {
    println!("defect risk {:.1} %", 100.0 * p.probability);
}
```

When the crate compiles, `model!`:
- **verifies the bundle** exactly as `tabsmith_runtime::Predictor::load` does: every
  file's SHA-256, the format versions, the consistency of schemas, transform and model,
  and the self-check. A broken or edited bundle is a compile error, and changing a file
  of the bundle triggers a rebuild;
- **generates a module** with the model's contract as types:
  - `Input` has one field per input the model uses. A misspelled field, a text where a
    category belongs, or a category the model never saw does not compile;
  - `Label` and `Prediction` (binary), `Class` and `Prediction` (multiclass), or
    `Prediction { value }` (regression);
  - `predict`, and its stages `features`, `outputs` and `decode`, and `outputs_batch` for
    many rows of features;
  - `Input::from_values`, which reads raw values (CSV cells, sensor registers) with
    the runtime's rules for JSON records;
- **turns the weights into constants and each tree into a function of nested `if`s**,
  so inference allocates nothing and needs no `std`. For trees, the compiler itself checks
  the structure with constant assertions: every split reads a real feature and points
  forward inside its own tree;
- **records what the runtime predicts** for the bundle's examples. `self_check()`
  replays them on the target and compares bit for bit, to catch a platform whose
  floating point differs.

## Same bits as the runtime, on every platform

A network calls the kernels of
[`tabsmith-embedded`](https://crates.io/crates/tabsmith-embedded), the same functions
`tabsmith-runtime` calls, and generated trees repeat the arithmetic of the trees' kernel.
`exp` comes from a pure-Rust math library instead of the platform's C library. So a
compiled model and the runtime return identical predictions, and so do Linux, Windows,
macOS, WebAssembly and a Cortex-M microcontroller. The tests of this crate check this on
seven bundles (every task and model kind) and on thousands of random records each,
including missing values, unseen categories, NaN, infinity and wrong types. A record that
one of them refuses, the other refuses too. A portability test compares the hash of 20,000
predictions per bundle with the value recorded on Windows x86_64; CI runs it on Linux x86_64
and macOS arm64 as well. Another test takes every node of every tree to a row built to reach
it, at the split's exact threshold, and compares each tree with its arrays.

## Trees become code

```rust
const fn tree_0(x: &[f32; N_FEATURES]) -> f32 {
    if x[24] <= 0.2705937f32 {
        if x[23] <= 1.4444073f32 { /* ... */ } else { /* ... */ }
    } else {
        if x[21] <= -0.09460988f32 { /* ... */ } else { /* ... */ }
    }
}
```

The numbers are decimal literals, so `cargo expand` shows the tree as it is. Compared with
walking the same trees in arrays, on real models and rows, on one thread:

| Nodes | Arrays, per row | Code, per row | `outputs_batch`, per row | Release build, arrays | Release build, code |
|---|---|---|---|---|---|
| 854 | 0.169 µs | 0.060 µs | 0.063 µs | 0.36 s | 0.44 s |
| 18,823 | 4.04 µs | 1.54 µs | 1.54 µs | 1.17 s | 3.44 s |
| 59,781 | 15.8 µs | 7.2 µs | 5.8 µs | 4.87 s | 18.7 s |
| 169,248 | 75.2 µs | 38.0 µs | 14.4 µs | 27.5 s | 115 s |

The program also gets smaller: a third less flash on a Cortex-M4F. And `outputs` is a
`const fn`, so the compiler can run the model:

```rust
const ALL_ZEROS: [f32; defects::N_OUTPUTS] = defects::outputs(&[0.0; defects::N_FEATURES]);
```

An ensemble of more than 32,768 nodes stays arrays unless the invocation says `tree_code`,
because of the build time. A crate that uses `outputs` in a `const` should say it anyway, so
that a larger model after retraining cannot take the `const` away. `tree_arrays` keeps the
arrays at any size, and a tree more than 64 levels deep always does. A network ignores both
options, so an invocation keeps compiling when retraining picks another model kind. The
repository's `reports/inference/` has the
measurements. `outputs_batch` runs tree by tree over blocks of rows above 32,768 nodes, where
one pass through all the code no longer fits in the cache, with 2 KB of stack per output.

## What the compiler knows about the model

Beyond the weights, the generated module tells the compiler three things, all `const`:

```rust
defects::WORST_CASE.comparisons     // the most arithmetic one prediction takes
defects::OUTPUT_BOUNDS              // [lo, hi] of the raw outputs over every input
defects::PROBABILITY_BOUNDS         // the same through the logistic function
const B: [[f32; 2]; 1] = {          // over a box: every numeric feature within
    let [lo, hi] = defects::feature_box(4.0);   // 4 standard deviations of its mean
    defects::output_bounds(&lo, &hi)
};
```

For trees the bounds are the sums of the smallest and the largest leaf each tree can
reach inside the box, summed as inference sums; rounding is monotone, so they are
sound with no error margin. For a network they come from interval arithmetic through
the layers, widened by the worst rounding error of the kernels. rustc's constant
evaluator derives them from the generated constants, and the macro checks that it
agrees, bit for bit, with the same kernels run at expansion time.

`defects::Model` implements the `tabsmith_embedded::Model` trait, as the runtime's
`Predictor` does, so an application can be written once for both.

## Proofs with Kani

`model!(name, "path", kani)` makes the module carry its own proofs for the
[Kani](https://github.com/model-checking/kani) model checker: under `cfg(kani)`, the
input types derive `kani::Arbitrary`, and `name::proofs` holds one harness per stage of
`predict` (`features`, `outputs`, `decode`), each proving that the stage never panics
for any value of its argument, with loop bounds taken from the model's shape. `cargo
kani` runs them. The crate that invokes `model!` should declare the cfg:

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ["cfg(kani)"] }
```

`examples/embedded-predict` in the repository shows the recommended Kani settings, and
`reports/kani/` the results: nine proofs in 81 s, each under 2 GB of memory. With trees as
code, all 14 trees of the example take one proof, 4.8 s and 104 MB. As arrays, that proof does
not fit in 14 GB, so `tree_arrays` proves one walk at a time.

Minimum supported Rust version: 1.88.

## 中文

`model!` 在编译期读取并校验 Tabsmith 模型包（与运行时完全相同的校验），把模型的输入输出契约生成为 Rust
类型：字段写错、类别写错、模型文件被改动都会变成编译错误。权重被生成为常量，每棵树被生成为嵌套 `if` 的 `const fn`，
推理不分配内存、不依赖 `std`，可以运行在服务器、WebAssembly 和微控制器上；网络调用和运行时同一套内核，树重复同样的算术，
预测结果与运行时逐位相同，在所有平台上也逐位相同。生成代码在真实模型上比遍历数组快 2 到 2.8 倍，在 Cortex-M4F 上少占三分之一 flash，
编译器还可以直接运行模型。超过 32,768 个节点的集成默认保留数组，因为编译时间更长；`tree_code` 和 `tree_arrays` 选项可以改变这个默认。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
