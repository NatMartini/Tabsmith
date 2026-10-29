# tabsmith-runtime

Lightweight prediction runtime for Tabsmith bundles: bundle verification, the frozen
preprocessing used in training, and FP32 execution of dense networks and
gradient-boosted tree ensembles on the CPU. It also compiles to WebAssembly unchanged.

It depends only on `tabsmith-contracts`, `tabsmith-embedded`, `serde`, `serde_json` and
`safetensors`: no training engine (Burn), no data engine (Polars), no Python. Bundles are
built with the [`tabsmith`](https://crates.io/crates/tabsmith-cli) command-line tool.
To compile a bundle into your code instead, with typed inputs and `no_std` inference
that predicts the same bits, see [`tabsmith-macros`](https://crates.io/crates/tabsmith-macros).

```rust
use tabsmith_runtime::{LoadOptions, Predictor, Record};

// A bundle built by `tabsmith train` (here from the synthetic sample data).
let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
// Every input field of the bundle is required (see its input.schema.json, or
// complete records in examples/input.json); `None` is a missing value, imputed
// as in training.
let record = Record::new()
    .with("line", "B")
    .with("temp_mean", 71.4)
    .with("hours_in_service", 1200)
    .with("calibration_offset", None::<f64>)
    /* … the other input fields … */;
let prediction = predictor.predict_one(&record)?;
match prediction.probability() {
    Some(p) => println!("{} (p = {p:.3})", prediction.label), // binary, multiclass
    None => println!("{}", prediction.label),                 // regression: the value
}
```

A prediction follows the bundle's task. `label` is always set. `probability()` is the
probability of the positive label (binary) or of the predicted class (multiclass).
`probabilities()` gives every class's probability, and `value()` the regression value
in the target's units. `outcome` holds the task's fields as an enum, and the JSON
form is flat. When the bundle carries a conformal calibration (`conformal.json`,
written by `tabsmith bundle build`), a prediction also has an `interval` (regression)
or a `prediction_set` (classification): the truth lies inside with probability about
`1 − alpha` on rows like the calibration rows. `Predictor::from_memory` loads a bundle
from file names and bytes instead of a directory, which is how `examples/wasm-predict`
runs in a browser. `Predictor` implements the `tabsmith_embedded::Model` trait, as a
model compiled with `tabsmith-macros` does, so an application can be written once for
both.

Loading is all-or-nothing: the manifest, every file's SHA-256 and size, the
bundle id, format versions and the consistency of schemas, transform and weights
are checked before a `Predictor` exists. `PredictorHandle::try_update` swaps in a
new bundle only after it loads and passes its self-check, and refuses breaking
input/output contract changes unless explicitly allowed; `rollback_to` returns to
a previous bundle.

Minimum supported Rust version: 1.88.

The bundle format, error codes and a complete example application are documented
in the [repository](https://github.com/NatMartini/Tabsmith)
(`docs/bundle-format.md`, `examples/predict-app`).

## 中文

Tabsmith 的轻量预测运行时：校验 bundle、执行与训练时相同的冻结预处理、在 CPU 上运行
FP32 的全连接网络与梯度提升树，并可原样编译为 WebAssembly。只依赖 `tabsmith-contracts`、
`tabsmith-embedded`、`serde`、`serde_json` 和 `safetensors`，不包含 Burn、Polars 或 Python。
想把模型直接编译进代码（输入是类型，推理不依赖 `std`，结果逐位相同），见 `tabsmith-macros`。预测结果随任务而定：
`probability()`（二分类、多分类）、`probabilities()`（多分类）、`value()`（回归）。模型包带有 conformal 校准（`conformal.json`）时，
预测还附带 `interval`（回归）或 `prediction_set`（分类）：在与校准行同分布的行上，真值落在其中的概率约为 `1 − alpha`。
`Predictor` 实现了 `tabsmith_embedded::Model` trait，用 `tabsmith-macros` 编译进代码的模型也实现了它，应用代码可以只写一份。
中文文档见仓库中的 `README.zh-CN.md`。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
