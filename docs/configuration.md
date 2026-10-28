# Configuration (`tabsmith.toml`)

[中文](#中文)

`tabsmith.toml` is the only file you edit. Precedence, lowest to highest:

1. recipe defaults (`tabular-binary@0.1`, `tabular-multiclass@0.1` or `tabular-regression@0.1`)
2. `tabsmith.toml`
3. the profile selected with `--profile <name>` (`[profiles.<name>]` tables)
4. `--set key.path=value` overrides (CLI) or override tables (SDK)

Every layer is validated on its own: unknown keys (`E_CONFIG_UNKNOWN_FIELD`) and
wrong types (`E_CONFIG_INVALID`) are errors that name the key and, for the file,
the line. `tabsmith config` prints every effective value (unset optional keys as `(unset)`) and the layer it came
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

The recipe is the task. A multiclass or regression project differs only in the recipe
and the target (`tabsmith train data.csv --target y --task multiclass` writes both):

```toml
recipe = "tabular-regression@0.1"

[data]
source = "data/machines.csv"
target = "remaining_hours"    # a number; no positive_label
split = "group"
group_column = "device_id"
```

## All settings

| Key | Default | Meaning |
|---|---|---|
| `recipe` | `"tabular-binary@0.1"` | The task: `tabular-binary@0.1` (two labels), `tabular-multiclass@0.1` (3 to 100 labels) or `tabular-regression@0.1` (a number). `--task binary`, `multiclass` or `regression` on `train` and `data bind` sets it |
| `data.source` | — (required) | CSV/Parquet path; a relative path is resolved against the project directory. `data bind` writes a relative path when the file is inside the project or at most two levels above it, otherwise an absolute path |
| `data.format` | `"auto"` | `auto` (by extension), `csv`, `parquet`. Parquet dictionary/categorical columns are read as text, decimals as numbers, durations as seconds, and dates, times and timestamps (time-zone-aware ones in UTC) as text, which is planned as a date column, not a feature |
| `data.target` | — (required) | Target column: two labels (binary), 3 to 100 labels (multiclass, compared as text) or numbers (regression) |
| `data.positive_label` | — (binary only, required there; `data bind` infers it for 0/1, false/true, no/yes) | Target value of the positive class |
| `data.id_column` | none | Unique row id; never a feature. Rows keep their split across data versions and are copied to prediction output. Without it, the split is keyed by each row's full content (every column) plus its occurrence number among identical rows, and leakage checks recognise rows by their content (model inputs + target): an edited row counts as a new row, and adding or removing columns in the file re-splits the rows (the checks still exclude rows a model has seen). Id and group columns are always read as text and compared as keys: surrounding spaces and the leading zeros of all-digit values are ignored, so `00000001` and `1` (after a spreadsheet round trip) are the same row |
| `data.group_column` | none | Entity column; requires `split = "group"` |
| `data.time_column` | none | Declaring it means a time split is needed: refused in 0.1 (`E_SPLIT_UNSUPPORTED`) |
| `data.exclude` | `[]` | Columns never used as features |
| `data.columns.<name>` | inferred | `"numeric"`, `"categorical"` (read as text, keeps leading zeros) or `"ignore"` |
| `data.split` | `"stratified"` | `stratified` (stable hash of the id, or of the row content without one), `group` (stable hash of the group key); `time` is refused |
| `data.seed` | `42` | Split seed |
| `data.train_fraction` / `valid_fraction` / `test_fraction` | `0.7` / `0.15` / `0.15` | Must sum to 1 |
| `data.snapshot` | `"copy"` | `copy`: the file is copied into `.tabsmith/` so runs are replayable; `reference`: hash only |
| `data.csv_delimiter` | `","` | One ASCII character |
| `data.null_values` | `["", "NA", "null"]` | CSV tokens read as missing |
| `preprocess.numeric_impute` | `"mean"` | `mean`, `median` (training rows) |
| `preprocess.numeric_scale` | `"standard"` | `standard` (z-score with training mean/std), `none` |
| `preprocess.nan` | `"missing"` | `missing`: NaN is imputed like null; `error`: NaN is rejected |
| `preprocess.max_categories` | `50` | Categorical columns with more training values are rejected |
| `preprocess.unknown_category` | `"ignore"` | `ignore`: unseen category encodes as zeros; `error`: rejected |
| `preprocess.extra_columns` | `"ignore"` | `error` rejects inputs with fields the model does not use |
| `preprocess.zero_variance` | `"drop"` | `drop` (field stays accepted but unused) or `error` |
| `selection.candidates` | `["linear", "small-mlp", "gbdt"]` | Candidates trained by `tabsmith run` |
| `selection.metric` | unset: `balanced_accuracy` (classification), `rmse` (regression) | Validation metric that ranks the candidates. Classification: `balanced_accuracy`, `f1`, `roc_auc`, `accuracy` (multiclass: mean over classes, ROC AUC one-vs-rest). Regression: `rmse`, `mae` (lower is better), `r2`. A metric of the other kind is a configuration error |
| `selection.threshold` | `0.5` | Binary only: decision threshold on P(positive), fixed before evaluation |
| `selection.min_valid_metric` | `0` (off) | Quality constraint for the recommendation: at least this value, or at most for `rmse` and `mae` |
| `selection.max_p95_latency_us` | `0` (off) | Single-row latency constraint, measured on this machine |
| `selection.max_model_kib` | `0` (off) | Model size constraint |
| `train.seed` | `7` | Weight initialisation and minibatch order |
| `train.linear.*` | `epochs` 60, `batch_size` 256, `learning_rate` 0.02, `l2` 1e-4, `early_stopping_patience` 10, `hidden` [] | Logistic-regression baseline (Adam); `hidden` must stay empty |
| `train.small-mlp.*` | `hidden` [32, 16], `epochs` 60, `batch_size` 128, `learning_rate` 0.005, `l2` 1e-4, `early_stopping_patience` 8 | Small MLP (ReLU, Adam) |
| `train.gbdt.*` | `epochs` 400, `learning_rate` 0.1, `l2` 1.0, `early_stopping_patience` 25, `checkpoint_every_epochs` 50 | Gradient-boosted trees. An epoch is one boosting round: one tree, or one per class for multiclass. `learning_rate` is the shrinkage and `l2` the penalty on leaf values. `hidden` and `batch_size` do not apply |
| `train.gbdt.trees.*` | `max_leaves` 31, `max_depth` 8, `min_samples_leaf` 20, `max_bins` 255 | Tree shape. Trees grow best-first up to `max_leaves`. Split points come from the training rows, up to `max_bins` (2–256) per feature. Only `gbdt` takes a `trees` table |
| `train.<c>.class_weight` | `"balanced"` | Classification: `balanced` (n / (k·n_class) for k classes) or `none` |
| `train.<c>.early_stopping_patience` | see above | Epochs without validation-loss improvement (0 = off) |
| `train.<c>.checkpoint_every_epochs` | `1` | Checkpoint at most every N epochs |
| `train.<c>.checkpoint_min_interval_seconds` | `2.0` | …and at most once per this many seconds (always on stop/cancel); does not change results |
| `execution.baseline_device` | `"cpu"` | Device for `linear` |
| `execution.candidate_device` | `"auto-certified"` | Device for `small-mlp`: `cpu`, `vulkan` (alias `wgpu`), `cuda`, or `auto-certified`. `auto-certified` picks a GPU certified for this recipe when the cost model expects it to be at least twice as fast as the CPU, and the CPU otherwise. GPU backends exist only in builds with the Cargo features `vulkan`/`cuda` (`gpu` on the CLI). `gbdt` always trains on the CPU |
| `execution.gpu` | `0` | Which GPU of the requested backend: the index in `tabsmith doctor` (discrete GPUs first) |
| `execution.fallback` | `"error"` | The only policy: no silent fallback |
| `execution.precision` | `"fp32"` | `fp32`, or `tf32`: FP32 storage and accumulation with TF32 matrix products, which CUDA uses on compute capability 8.0 and newer. CUDA on such GPUs needs `tf32`: fp32 is refused there rather than silently lowered. Vulkan computes fp32. `fp16`/`bf16` are refused (not certified) |
| `execution.max_memory_mb` | `4096` | Pre-flight memory budget (`E_RESOURCE_BUDGET_EXCEEDED`) |
| `execution.max_train_seconds` | `0` (off) | Per-candidate time budget; makes results time-dependent |

The training objective of the networks is the class-weighted mean binary cross-entropy
plus `l2 / 2 · Σ W²` over weights (biases excluded). This is the same objective as
scikit-learn's `LogisticRegression(C = 1 / (n_train · l2), class_weight="balanced")`.
Multiclass uses the class-weighted softmax cross-entropy. Regression uses the mean
squared error of the standardised target: the model learns `(y − mean) / std` of the
training rows, and predictions are mapped back. `gbdt` boosts the same losses with
Newton steps: each leaf value is `−Σg / (Σh + l2)`, times `learning_rate`.

---

## 中文

`tabsmith.toml` 是唯一需要手动编辑的文件。优先级从低到高：

1. recipe 默认值（`tabular-binary@0.1`、`tabular-multiclass@0.1` 或 `tabular-regression@0.1`）
2. `tabsmith.toml`
3. 用 `--profile <name>` 选择的 profile（`[profiles.<name>]` 表）
4. `--set key.path=value` 覆盖（CLI）或覆盖表（SDK）

每一层都单独校验：未知字段（`E_CONFIG_UNKNOWN_FIELD`）和类型错误（`E_CONFIG_INVALID`）都会报出字段名，对文件还会给出行号。`tabsmith config` 列出每个生效值（未设置的可选键显示为 `(unset)`）及其来源层；每个 Run 保存完整解析后的配置（`resolved.json`）及其哈希。环境变量不会改变行为。

上表列出了所有设置（键名、默认值、含义）。要点：

- `recipe` 即任务：二分类、多分类（3 到 100 个类别）或回归。`train` 与 `data bind` 的 `--task` 会设置它。
- `data.positive_label` 是要检测的类别（仅二分类）；`data bind` 对 0/1、false/true、no/yes 会自动推断。
- `selection.metric` 未设置时：分类用 `balanced_accuracy`，回归用 `rmse`；回归还可选 `mae`、`r2`。
- `train.gbdt.*` 是梯度提升树：一个 epoch 即一轮提升，`learning_rate` 为收缩率，`l2` 为叶子值的惩罚；`train.gbdt.trees.*` 设定树的形状。
- `data.source` 的相对路径以项目目录为基准；`data bind` 在文件位于项目内或项目上方至多两级时写入相对路径，否则写入绝对路径。
- `data.id_column` 是唯一行 id：数据增长时各行保持原切分，并被复制到预测输出中。没有 id 时，切分以每行的完整内容（所有列）加上它在相同行中的出现序号为键；泄漏检查按内容（模型输入 + 目标列）识别行：被修改过的行视为新行；文件增删列会导致重新切分（检查仍会排除模型见过的行）。id 列和分组列总是按文本读取，并作为键比较：忽略首尾空格和纯数字值的前导零，因此 `00000001` 与（经电子表格处理后的）`1` 是同一行。
- Parquet：字典/类别列按文本读取，decimal 按数值读取，duration 按秒读取，日期、时间和时间戳（带时区的按 UTC）按文本读取，并被规划为日期列而不是特征。
- `data.group_column` 要求 `split = "group"`：同一实体的行不会跨越两个子集。
- 声明 `data.time_column` 表示需要按时间切分，0.1 版会拒绝（`E_SPLIT_UNSUPPORTED`），不会退回随机切分。
- `data.snapshot = "copy"` 会把数据文件复制进 `.tabsmith/`，使 Run 可以重放；`reference` 只记录哈希。
- `preprocess.nan = "missing"` 时 NaN 与 null 一样被填补；`preprocess.unknown_category = "ignore"` 时未见过的类别编码为全零。
- `execution.candidate_device` 可为 `cpu`、`vulkan`、`cuda` 或 `auto-certified`。`auto-certified` 只会选择经 `tools/gpu_certify.py` 认证、且成本模型预计至少快一倍的 GPU，否则用 CPU。GPU 后端只存在于启用了 `vulkan`/`cuda` 特性（CLI 为 `gpu`）的构建中；`gbdt` 始终在 CPU 上训练。
- `execution.precision = "tf32"`：CUDA（计算能力 8.0 及以上）的矩阵乘法使用 TF32，必须显式声明；要求 fp32 时会被拒绝而不是悄悄降低精度。Vulkan 按 fp32 计算。`execution.fallback` 只有 `error` 一种：不会静默回退。
- `execution.max_train_seconds` 会让结果依赖运行时间，因此默认关闭。

网络的训练目标是按类别加权的平均二元交叉熵，加上对权重（不含偏置）的 `l2 / 2 · Σ W²`，与 scikit-learn 的 `LogisticRegression(C = 1 / (n_train · l2), class_weight="balanced")` 目标函数相同。多分类使用加权 softmax 交叉熵，回归使用标准化目标的均方误差；`gbdt` 用牛顿步提升同样的损失（叶子值 `−Σg / (Σh + l2)` 乘以 `learning_rate`）。
