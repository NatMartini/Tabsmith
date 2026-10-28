# tabsmith-cli

The `tabsmith` command-line tool: from a local CSV/Parquet file to a trained,
verified tabular model (binary or multiclass classifier, or regressor), packaged as a
bundle that a Rust application loads with
[`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime).

Status: early release (0.1). It trains a linear baseline and a small MLP with Burn
0.22 (Flex CPU backend), and Tabsmith's own gradient-boosted trees. The prebuilt
binaries are CPU-only. GPU training of the networks (Vulkan, CUDA) needs a build with
`--features gpu`.

## Install

Prebuilt binaries for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows
(x86_64) are attached to each
[GitHub release](https://github.com/NatMartini/Tabsmith/releases):

```sh
# Linux / macOS
curl -fsSL https://github.com/NatMartini/Tabsmith/releases/latest/download/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/NatMartini/Tabsmith/releases/latest/download/install.ps1 | iex
```

With [cargo-binstall](https://github.com/cargo-bins/cargo-binstall):
`cargo binstall tabsmith-cli`.

From source (Rust 1.95 or newer; compiling Polars and Burn takes several minutes):

```sh
cargo install tabsmith-cli --locked
cargo install tabsmith-cli --locked --features gpu   # with Vulkan and CUDA training
```

Every method installs a command named `tabsmith`.

## Quick start

```text
tabsmith sample-data machines.csv                # 10,000 synthetic rows (CC0), target `defect`
tabsmith train machines.csv --target defect --group device_id
tabsmith sample-data new.csv --rows 200 --seed 7 # some new rows
tabsmith predict --input new.csv --keep unit_id --output predictions.csv
tabsmith serve --port 8080                       # HTTP/JSON: POST /predict, GET /schema, GET /health
```

The same works for the other tasks: `tabsmith sample-data m.csv --task multiclass`, then
`tabsmith train m.csv --target failure_mode --group device_id --task multiclass`
(`--task regression` and `--target remaining_hours` for a regression).

`tabsmith train` splits the data (keeping each device's rows together), learns the
preprocessing from the training rows, trains a linear baseline, a small MLP and
gradient-boosted trees, picks one on the validation split, tests it once and builds a verified bundle.
Run it again after new data arrives: it also scores the previous bundle and the
new model on the same unseen test rows.

## Step by step

```text
tabsmith init defect-model --sample-data         # project + synthetic rows, bound to target `defect`
cd defect-model
tabsmith plan                                    # columns, split, preprocessing, candidates, device
tabsmith run                                     # trains `linear`, `small-mlp` and `gbdt` on the same split + transform
tabsmith compare                                 # validation-only comparison and a recommendation
tabsmith evaluate --run run-002 --split test     # the test split is used once, by the final candidate
tabsmith bundle build --run run-002              # immutable bundle: weights + preprocessing + schemas + evidence
tabsmith bundle check bundles/run-002            # verifies hashes, versions and example predictions
tabsmith sample-data new.csv --rows 200 --seed 7 # some new rows to predict
tabsmith predict --bundle bundles/run-002 --input new.csv --output preds.csv
tabsmith report                                  # offline HTML/JSON report in .tabsmith/reports/
```

`tabsmith --help` lists every command; `--json` gives machine-readable output.

Documentation, limitations and the source are in the
[repository](https://github.com/NatMartini/Tabsmith).

## 中文

`tabsmith` 命令行工具：从本地 CSV/Parquet 文件训练、验证表格模型（二分类、多分类或回归），
并打包成 Rust 应用可通过 `tabsmith-runtime` 加载的 bundle。训练线性基线、小型 MLP 和梯度提升树；
预编译二进制只含 CPU，GPU 训练需用 `--features gpu` 构建。快速上手：`tabsmith sample-data machines.csv`，然后
`tabsmith train machines.csv --target defect --group device_id`。安装方式见上文；中文文档见仓库中的
`README.zh-CN.md`。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
