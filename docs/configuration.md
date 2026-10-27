# Configuration (`rustpy.toml`)

[中文](#中文)

`rustpy.toml` is the only file you edit. Precedence, lowest to highest:

1. recipe defaults (`tabular-binary@0.1`)
2. `rustpy.toml`
3. the profile selected with `--profile <name>` (`[profiles.<name>]` tables)
4. `--set key.path=value` overrides (CLI) or override tables (SDK)

Every layer is validated on its own: unknown keys (`E_CONFIG_UNKNOWN_FIELD`) and
wrong types (`E_CONFIG_INVALID`) are errors that name the key and, for the file,
the line. `rustpy config` prints every effective value and the layer it came
from; each Run stores the fully resolved configuration (`resolved.json`) and its
hash. Environment variables never change behaviour.

```toml
schema_version = 1
recipe = "tabular-binary@0.1"

[data]
source = "data/defects.csv"   # CSV or Parquet, relative to the project
target = "defect"
positive_label = "1"          # the class to detect, compared as text
split = "group"               # "stratified" | "group" ("time" is refused in 0.1)
group_column = "device_id"    # rows of one entity never span two splits
id_column = "unit_id"         # unique row id, copied to prediction output

[profiles.quick.train.small-mlp]
epochs = 12
```

## All settings

| Key | Default | Meaning |
|---|---|---|
| `data.source` | — (required) | CSV/Parquet path |
| `data.format` | `"auto"` | `auto` (by extension), `csv`, `parquet` |
| `data.target` | — (required) | Binary target column |
| `data.positive_label` | — (required; `data bind` infers it for 0/1, false/true, no/yes) | Target value of the positive class |
| `data.id_column` | none | Unique row id; never a feature |
| `data.group_column` | none | Entity column; requires `split = "group"` |
| `data.time_column` | none | Declaring it means a time split is needed: refused in 0.1 (`E_SPLIT_UNSUPPORTED`) |
| `data.exclude` | `[]` | Columns never used as features |
| `data.columns.<name>` | inferred | `"numeric"`, `"categorical"` (read as text, keeps leading zeros) or `"ignore"` |
| `data.split` | `"stratified"` | `stratified`, `group`; `time` is refused |
| `data.seed` | `42` | Split seed |
| `data.train_fraction` / `valid_fraction` / `test_fraction` | `0.7` / `0.15` / `0.15` | Must sum to 1 |
| `data.snapshot` | `"copy"` | `copy`: the file is copied into `.rustpy/` so runs are replayable; `reference`: hash only |
| `data.csv_delimiter` | `","` | One ASCII character |
| `data.null_values` | `["", "NA", "null"]` | CSV tokens read as missing |
| `preprocess.numeric_impute` | `"mean"` | `mean`, `median` (training rows) |
| `preprocess.numeric_scale` | `"standard"` | `standard` (z-score with training mean/std), `none` |
| `preprocess.nan` | `"missing"` | `missing`: NaN is imputed like null; `error`: NaN is rejected |
| `preprocess.max_categories` | `50` | Categorical columns with more training values are rejected |
| `preprocess.unknown_category` | `"ignore"` | `ignore`: unseen category encodes as zeros; `error`: rejected |
| `preprocess.extra_columns` | `"ignore"` | `error` rejects inputs with fields the model does not use |
| `preprocess.zero_variance` | `"drop"` | `drop` (field stays accepted but unused) or `error` |
| `selection.candidates` | `["linear", "small-mlp"]` | Candidates trained by `rustpy run` |
| `selection.metric` | `"balanced_accuracy"` | `balanced_accuracy`, `f1`, `roc_auc`, `accuracy` (validation) |
| `selection.threshold` | `0.5` | Decision threshold on P(positive), fixed before evaluation |
| `selection.min_valid_metric` | `0` (off) | Quality constraint for the recommendation |
| `selection.max_p95_latency_us` | `0` (off) | Single-row latency constraint, measured on this machine |
| `selection.max_model_kib` | `0` (off) | Model size constraint |
| `train.seed` | `7` | Weight initialisation and minibatch order |
| `train.linear.*` | epochs 60, batch 256, lr 0.02, l2 1e-4, patience 10 | Logistic-regression baseline (Adam) |
| `train.small-mlp.*` | hidden [32, 16], epochs 60, batch 128, lr 0.005, l2 1e-4, patience 8 | Small MLP (ReLU, Adam) |
| `train.<c>.class_weight` | `"balanced"` | `balanced` (n / (2·n_class)) or `none` |
| `train.<c>.early_stopping_patience` | see above | Epochs without validation-loss improvement (0 = off) |
| `train.<c>.checkpoint_every_epochs` | `1` | Checkpoint at most every N epochs |
| `train.<c>.checkpoint_min_interval_seconds` | `2.0` | …and at most once per this many seconds (always on stop/cancel); does not change results |
| `execution.baseline_device` | `"cpu"` | Device for `linear` |
| `execution.candidate_device` | `"auto-certified"` | Device for `small-mlp`; only CPU is certified in this build |
| `execution.fallback` | `"error"` | The only policy: no silent fallback |
| `execution.precision` | `"fp32"` | `fp16`/`bf16` are refused (not certified) |
| `execution.max_memory_mb` | `4096` | Pre-flight memory budget (`E_RESOURCE_BUDGET_EXCEEDED`) |
| `execution.max_train_seconds` | `0` (off) | Per-candidate time budget; makes results time-dependent |

The training objective is the class-weighted mean binary cross-entropy plus
`l2 / 2 · Σ W²` over weights (biases excluded) — the same objective as
scikit-learn's `LogisticRegression(C = 1 / (n_train · l2), class_weight="balanced")`.

---

## 中文

`rustpy.toml` 是唯一需要手动编辑的文件。优先级从低到高：

1. recipe 默认值（`tabular-binary@0.1`）
2. `rustpy.toml`
3. 用 `--profile <name>` 选择的 profile（`[profiles.<name>]` 表）
4. `--set key.path=value` 覆盖（CLI）或覆盖表（SDK）

每一层都单独校验：未知字段（`E_CONFIG_UNKNOWN_FIELD`）和类型错误（`E_CONFIG_INVALID`）都会报出字段名，对文件还会给出行号。`rustpy config` 列出每个生效值及其来源层；每个 Run 保存完整解析后的配置（`resolved.json`）及其哈希。环境变量不会改变行为。

上表列出了所有设置（键名、默认值、含义）。要点：

- `data.positive_label` 是要检测的类别；`data bind` 对 0/1、false/true、no/yes 会自动推断。
- `data.group_column` 要求 `split = "group"`：同一实体的行不会跨越两个子集。
- 声明 `data.time_column` 表示需要按时间切分，0.1 版会拒绝（`E_SPLIT_UNSUPPORTED`），不会退回随机切分。
- `data.snapshot = "copy"` 会把数据文件复制进 `.rustpy/`，使 Run 可以重放；`reference` 只记录哈希。
- `preprocess.nan = "missing"` 时 NaN 与 null 一样被填补；`preprocess.unknown_category = "ignore"` 时未见过的类别编码为全零。
- `execution.candidate_device = "auto-certified"` 只会选择本版本已认证的设备（目前只有 CPU）；`execution.fallback` 只有 `error` 一种：不会静默回退。
- `execution.max_train_seconds` 会让结果依赖运行时间，因此默认关闭。

训练目标是按类别加权的平均二元交叉熵，加上对权重（不含偏置）的 `l2 / 2 · Σ W²`，与 scikit-learn 的 `LogisticRegression(C = 1 / (n_train · l2), class_weight="balanced")` 目标函数相同。
