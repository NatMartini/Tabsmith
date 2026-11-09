# tabsmith-contracts

Stable cross-stage contracts for Tabsmith: project configuration and its layering,
errors (`TabsmithError` with stable codes, listed in the repository's `docs/errors.md`),
input/output schemas, the fitted transform's state, the engine-neutral model description
(dense layers, or tree ensembles with how their leaves are added), hashing, and the Run,
Evaluation and Bundle manifests, each with an explicit format version. Bundle types reject
unknown fields, so a runtime that does not know a field refuses the bundle instead of
ignoring it.

This crate has no dependency on the training engine (Burn) or the data engine
(Polars). It is shared by [`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime),
which loads bundles in applications, and `tabsmith-workflow`, which builds them.
You normally use it through one of those crates.

Minimum supported Rust version: 1.88.

Documentation and the source are in the
[repository](https://github.com/NatMartini/Tabsmith).

## 中文

Tabsmith 各阶段之间的稳定契约：项目配置、错误类型（错误码见仓库的 `docs/errors.md`）、输入/输出 schema、拟合后的变换状态、与引擎无关的模型描述（全连接层或树集成），以及 Run、
Evaluation 和 Bundle 清单（均带格式版本；模型包类型拒绝未知字段）。不依赖 Burn 或 Polars，通常通过
`tabsmith-runtime` 或 `tabsmith-workflow` 间接使用。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
