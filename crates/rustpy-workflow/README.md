# rustpy-workflow

The RustPy workflow core behind the [`rustpy`](https://crates.io/crates/rustpy-cli)
command-line tool: data snapshots (CSV/Parquet via Polars), leakage-safe splits,
fitted preprocessing, training of a linear baseline and a small MLP on the CPU
(Burn 0.22, ndarray backend), evaluation, comparison, reports and inference
bundles.

Most users want the CLI (`cargo install rustpy-cli`) to train and
[`rustpy-runtime`](https://crates.io/crates/rustpy-runtime) to load bundles in an
application. Depend on this crate directly only to drive the workflow from Rust;
its API is not yet stable (0.1).

No GPU backend is compiled in or certified. Minimum supported Rust version: 1.95
(set by Burn 0.22.1).

Documentation and the source are in the
[repository](https://github.com/__OWNER__/__REPO__).

## 中文

`rustpy` 命令行工具背后的工作流核心：数据快照、防泄漏划分、拟合预处理、在 CPU 上
训练线性基线和小型 MLP、评估、比较、报告和推理 bundle。API 尚未稳定（0.1）。
大多数用户应使用 `rustpy-cli` 训练、用 `rustpy-runtime` 在应用中加载模型。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
