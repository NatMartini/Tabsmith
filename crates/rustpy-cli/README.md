# rustpy-cli

The `rustpy` command-line tool: from a local CSV/Parquet file to a trained,
verified tabular binary classifier, packaged as a bundle that a Rust application
loads with [`rustpy-runtime`](https://crates.io/crates/rustpy-runtime).

Status: early CPU prototype (0.1). It trains a linear baseline and a small MLP on
the CPU with Burn 0.22 (ndarray backend). No GPU backend is compiled in or
certified.

## Install

Prebuilt binaries for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows
(x86_64) are attached to each
[GitHub release](https://github.com/__OWNER__/__REPO__/releases):

```sh
# Linux / macOS
curl -fsSL https://github.com/__OWNER__/__REPO__/releases/latest/download/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/__OWNER__/__REPO__/releases/latest/download/install.ps1 | iex
```

With [cargo-binstall](https://github.com/cargo-bins/cargo-binstall):
`cargo binstall rustpy-cli`.

From source (Rust 1.95 or newer; compiling Polars and Burn takes several minutes):

```sh
cargo install rustpy-cli --locked
```

Every method installs a command named `rustpy`.

## Five-minute tour

```text
rustpy init defect-model --sample-data         # project + synthetic rows, bound to target `defect`
cd defect-model
rustpy plan                                    # columns, split, preprocessing, candidates, device
rustpy run                                     # trains `linear` and `small-mlp` on the same split + transform
rustpy compare                                 # validation-only comparison and a recommendation
rustpy evaluate --run run-002 --split test     # the test split is used once, by the final candidate
rustpy bundle build --run run-002              # immutable bundle: weights + preprocessing + schemas + evidence
rustpy bundle check bundles/run-002            # verifies hashes, versions and example predictions
rustpy predict --bundle bundles/run-002 --input new.csv --output preds.csv
rustpy report                                  # offline HTML/JSON report in .rustpy/reports/
```

`rustpy --help` lists every command; `--json` gives machine-readable output.

Documentation, limitations and the source are in the
[repository](https://github.com/__OWNER__/__REPO__).

## 中文

`rustpy` 命令行工具：从本地 CSV/Parquet 文件训练、验证表格二分类模型，并打包成
Rust 应用可通过 `rustpy-runtime` 加载的 bundle。当前为 0.1 CPU 原型，GPU 路径尚未
编译或认证。安装方式见上文；中文文档见仓库中的 `README.zh-CN.md`。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
