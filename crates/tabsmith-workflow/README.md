# tabsmith-workflow

The Tabsmith workflow core behind the [`tabsmith`](https://crates.io/crates/tabsmith-cli)
command-line tool: data snapshots (CSV/Parquet via Polars), leakage-safe splits,
fitted preprocessing, and training of three candidates for binary, multiclass and
regression tasks. The candidates are a linear baseline and a small MLP (Burn 0.22:
Flex on the CPU, or Vulkan/CUDA with the Cargo features `vulkan`/`cuda`) and
deterministic gradient-boosted trees. The crate also covers evaluation, comparison,
reports and inference bundles.

Most users want the CLI (`cargo install tabsmith-cli`) to train and
[`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime) to load bundles in an
application. Depend on this crate directly only to drive the workflow from Rust;
its API is not yet stable (0.1).

GPU backends are opt-in features, and a GPU is used automatically only when it is
certified (`tools/gpu_certify.py` in the repository). Minimum supported Rust version:
1.95 (set by Burn 0.22.1).

Documentation and the source are in the
[repository](https://github.com/NatMartini/Tabsmith).

## 中文

`tabsmith` 命令行工具背后的工作流核心：数据快照、防泄漏划分、拟合预处理，为二分类、多分类和
回归训练三种候选模型（线性基线、小型 MLP——CPU 上用 Flex，或通过 `vulkan`/`cuda` 特性用 GPU——以及
确定性的梯度提升树），以及评估、比较、报告和推理 bundle。API 尚未稳定（0.1）。
大多数用户应使用 `tabsmith-cli` 训练、用 `tabsmith-runtime` 在应用中加载模型。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
