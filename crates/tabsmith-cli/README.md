# tabsmith-cli

The `tabsmith` command-line tool: from a local CSV/Parquet file to a trained,
verified tabular model (binary or multiclass classifier, or regressor), packaged as a
bundle that a Rust application loads with
[`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime) or compiles into its
code with [`tabsmith-macros`](https://crates.io/crates/tabsmith-macros).

Release 0.4. It trains a linear baseline and a small MLP with Burn 0.22 (Flex CPU
backend) and Tabsmith's own deterministic gradient-boosted trees; checks rules over
regions of inputs (`audit`: a proof, or an input that breaks the rule); and imports models
trained by XGBoost and LightGBM (`import`: XGBoost's margins bit for bit). The prebuilt
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
tabsmith serve --port 8080                       # HTTP/JSON: POST /predict, GET /schema, GET /health, GET /stats
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
tabsmith audit --rule 'temp_mean <= 60, _ within 3 sd => probability <= 0.9'  # proved, or an input that breaks it
tabsmith audit --stability 0.05 --input new.csv   # can a small move of each row's numbers change its label?
tabsmith report                                  # offline HTML/JSON report in .tabsmith/reports/
tabsmith import model.json --out bundles/xgb     # an XGBoost (JSON) or LightGBM (text) model as a bundle; XGBoost bit for bit
```

## Commands

`tabsmith --help` and `tabsmith <command> --help` list every option. `--project DIR` names
the project (default: the nearest directory with `tabsmith.toml`); `--json` prints
machine-readable output, errors included. Exit codes follow the error's kind (2 configuration
and rules, 3 data, 4 bundles and imports, 5 devices, 6 runs, 7 evaluation discipline, 1 I/O);
`audit` ends with 8 when a rule fails and 9 when one is undecided
([docs/errors.md](https://github.com/NatMartini/Tabsmith/blob/main/docs/errors.md)).

| Command | What it does |
|---|---|
| `init DIR [--recipe tabular-binary\|tabular-multiclass\|tabular-regression] [--sample-data]` | creates a project directory with `tabsmith.toml`, and binds the synthetic sample data if asked |
| `sample-data FILE [--task binary\|multiclass\|regression] [--rows N] [--seed S] [--shifted]` | writes the synthetic equipment dataset (CC0); `--shifted` draws from a shifted distribution, for "new data" |
| `data bind FILE --target COL [--positive V] [--task T] [--split stratified\|group\|time] [--group COL] [--time COL] [--id COL] [--exclude COLS]` | points the project at a data file and records the target and split settings in `tabsmith.toml` |
| `data inspect` | column types, missing values and the label distribution of the bound data |
| `plan`, `config` | the resolved plan (data, columns, split, preprocessing, candidates, device); every effective setting and where it came from. Both take `--profile NAME` and `--set key=value` |
| `run [--candidates linear,small-mlp,gbdt] [--no-cache]` | trains the configured candidates, one run each, on the same split and transform |
| `train [FILE --target COL …] [--candidates …]` | the whole workflow in one command: bind, train, choose on validation, test once, bundle, and score the previously shipped bundle on the same unseen rows. Creates `tabsmith.toml` when there is no project yet |
| `runs`, `status RUN` | lists runs and their status; one run in detail |
| `compare [--runs RUN,…]` | compares runs on the validation split, with a recommendation (default: the latest `tabsmith run`) |
| `resume RUN [--warm-start]` | continues an interrupted or cancelled run exactly, to the same bits; `--warm-start` begins a new run from its weights |
| `evaluate --run RUN \| --bundle DIR [--split valid\|test] [--allow-test-reuse] [--allow-unverified-rows]` | evaluates a run, or a bundle on the current data, on a split; the test split is used once per data version |
| `bundle build --run RUN [--out DIR] [--allow-unevaluated]`, `bundle check DIR`, `bundle list` | builds an immutable bundle from a completed, test-evaluated run; verifies one on this machine (hashes, versions, schema, example predictions); lists the project's bundles |
| `predict --input FILE [--bundle DIR] [--output FILE] [--keep COLS]` | predicts a CSV/Parquet/JSON/JSONL file with a bundle (default: the newest evaluated bundle); CSV to stdout or `--output` (.csv or .jsonl) |
| `serve [--bundle DIR] [--host H] [--port P] [--threads N] [--allow-reload]` | serves a bundle over HTTP: `POST /predict`, `GET /schema`, `GET /health`, `GET /stats` (input drift), `POST /reload` with `--allow-reload`. No authentication or timeouts; keep `127.0.0.1` or put a proxy in front |
| `audit --rule RULE … [--rules FILE] [--boxes N] [--seconds S] [--bundle DIR]` | checks each rule `conditions => claim` for every input of its region: proved, refuted with an input, or undecided within the budget of boxes |
| `audit --stability K --input FILE [--bundle DIR]` | asks of every record whether its label survives a move of each number by up to `K` training standard deviations, and shows an input that changes it |
| `import MODEL --out DIR [--name N] [--target COL] [--labels a,b,…] [--features f1,f2,…] [--threshold P]` | turns an XGBoost (JSON) or LightGBM (text) model into a bundle; XGBoost's margins bit for bit, LightGBM's raw scores within a reported bound |
| `report` | writes a static HTML + JSON report of the project to `.tabsmith/reports/` |
| `doctor` | version, engine, compiled backends and detected devices |

Documentation, limitations and the source are in the
[repository](https://github.com/NatMartini/Tabsmith): the bundle format, the configuration
file, every error code, importing models, the design decisions and the measured reports.

## 中文

`tabsmith` 命令行工具：从本地 CSV/Parquet 文件训练、验证表格模型（二分类、多分类或回归），
并打包成 Rust 应用可通过 `tabsmith-runtime` 加载、或通过 `tabsmith-macros` 编译进代码的模型包（bundle）。
0.4 版训练线性基线、小型 MLP（Burn 0.22）和自研的确定性梯度提升树；`tabsmith audit` 对一个输入区域判定规则：
给出证明，或者给出一个违反规则、并被真实预测过的输入；`tabsmith import` 把 XGBoost（JSON）或 LightGBM（文本）模型变成模型包，
XGBoost 的 margin 逐位一致。预编译二进制只含 CPU，GPU 训练需用 `--features gpu` 构建。快速上手：`tabsmith sample-data machines.csv`，然后
`tabsmith train machines.csv --target defect --group device_id`。安装方式见上文；中文文档见仓库中的 `README.zh-CN.md`。

### 命令一览

`tabsmith --help` 和 `tabsmith <命令> --help` 列出所有选项。`--project DIR` 指定项目（默认：最近的含 `tabsmith.toml` 的目录）；`--json` 输出机器可读的结果，错误也是。退出码按错误类别划分（2 配置与规则，3 数据，4 模型包与导入，5 设备，6 Run，7 评估纪律，1 I/O）；`audit` 在有规则不成立时以 8 结束，有规则未定时以 9 结束。

| 命令 | 作用 |
|---|---|
| `init DIR [--recipe …] [--sample-data]` | 创建带 `tabsmith.toml` 的项目目录，可同时写入并绑定合成样例数据 |
| `sample-data FILE [--task …] [--rows N] [--seed S] [--shifted]` | 写出合成的设备数据集（CC0）；`--shifted` 产生分布偏移的数据，用来模拟新数据 |
| `data bind FILE --target COL [--positive V] [--task T] [--split …] [--group COL] [--time COL] [--id COL] [--exclude COLS]` | 把项目指向一个数据文件，并把目标列和切分设置记入 `tabsmith.toml` |
| `data inspect` | 列类型、缺失值和标签分布 |
| `plan`、`config` | 解析后的计划（数据、列、切分、预处理、候选模型、设备）；每个生效的设置及其来源。都接受 `--profile NAME` 和 `--set key=value` |
| `run [--candidates …] [--no-cache]` | 在同一切分和变换上训练配置的候选模型，每个一个 Run |
| `train [FILE --target COL …]` | 一条命令完成整个流程：绑定、训练、在验证集上选择、测试一次、打包，并把之前交付的模型包在同一批没见过的行上打分。没有项目时会创建 `tabsmith.toml` |
| `runs`、`status RUN` | 列出 Run 及其状态；查看一个 Run 的详情 |
| `compare [--runs …]` | 在验证集上比较 Run 并给出推荐 |
| `resume RUN [--warm-start]` | 精确续训被中断或取消的 Run（逐位相同）；`--warm-start` 用它的权重开始新的 Run |
| `evaluate --run RUN \| --bundle DIR [--split valid\|test] [--allow-test-reuse] [--allow-unverified-rows]` | 在某个切分上评估一个 Run，或在当前数据上评估一个模型包；测试集每个数据版本只用一次 |
| `bundle build --run RUN [--out DIR] [--allow-unevaluated]`、`bundle check DIR`、`bundle list` | 从已完成、已测试评估的 Run 构建不可变模型包；在本机校验模型包（哈希、版本、schema、示例预测）；列出项目里的模型包 |
| `predict --input FILE [--bundle DIR] [--output FILE] [--keep COLS]` | 用模型包预测 CSV/Parquet/JSON/JSONL 文件（默认用最新的已评估模型包） |
| `serve [--bundle DIR] [--host H] [--port P] [--threads N] [--allow-reload]` | 通过 HTTP 提供服务：`POST /predict`、`GET /schema`、`GET /health`、`GET /stats`（输入漂移），加 `--allow-reload` 后有 `POST /reload`。没有认证和超时，请保持 `127.0.0.1` 或在前面放反向代理 |
| `audit --rule RULE … [--rules FILE] [--boxes N] [--seconds S] [--bundle DIR]` | 对每条 `条件 => 结论` 规则检查其区域内的每个输入：证明、给出反例，或在盒子预算内未定 |
| `audit --stability K --input FILE [--bundle DIR]` | 问每条记录：各数值移动至多 `K` 个训练标准差后，标签是否还能保持；能改变时给出那个输入 |
| `import MODEL --out DIR [--name N] [--target COL] [--labels a,b,…] [--features …] [--threshold P]` | 把 XGBoost（JSON）或 LightGBM（文本）模型变成模型包；XGBoost 的 margin 逐位一致，LightGBM 的原始得分在报告的误差界内 |
| `report` | 把项目的静态 HTML + JSON 报告写到 `.tabsmith/reports/` |
| `doctor` | 版本、引擎、编译进来的后端和检测到的设备 |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
