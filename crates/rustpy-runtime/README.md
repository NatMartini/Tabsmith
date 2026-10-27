# rustpy-runtime

Lightweight prediction runtime for RustPy bundles: bundle verification, the frozen
preprocessing used in training, and FP32 model execution on the CPU.

It depends only on `rustpy-contracts`, `serde`, `serde_json` and `safetensors`:
no training engine (Burn), no data engine (Polars), no Python. Bundles are built
with the [`rustpy`](https://crates.io/crates/rustpy-cli) command-line tool.

```rust
use rustpy_runtime::{LoadOptions, Predictor, Record};

// A bundle built by `rustpy train` (here from the synthetic sample data).
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
println!("{} (p = {:.3})", prediction.label, prediction.probability);
```

Loading is all-or-nothing: the manifest, every file's SHA-256 and size, the
bundle id, format versions and the consistency of schemas, transform and weights
are checked before a `Predictor` exists. `PredictorHandle::try_update` swaps in a
new bundle only after it loads and passes its self-check, and refuses breaking
input/output contract changes unless explicitly allowed; `rollback_to` returns to
a previous bundle.

Minimum supported Rust version: 1.88.

The bundle format, error codes and a complete example application are documented
in the [repository](https://github.com/__OWNER__/__REPO__)
(`docs/bundle-format.md`, `examples/predict-app`).

## 中文

RustPy 的轻量预测运行时：校验 bundle、执行与训练时相同的冻结预处理、在 CPU 上运行
FP32 模型。只依赖 `rustpy-contracts`、`serde`、`serde_json` 和 `safetensors`，
不包含 Burn、Polars 或 Python。中文文档见仓库中的 `README.zh-CN.md`。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
