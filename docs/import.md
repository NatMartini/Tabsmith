# Importing models from XGBoost and LightGBM

[中文](#中文)

`tabsmith import` turns a gradient-boosted model trained by XGBoost or LightGBM into a bundle.
The bundle then works like one Tabsmith trained: `tabsmith predict` and `serve`,
`Predictor::load` in a Rust application, `model!` (compiled into the application, `no_std`,
`const`), the browser build, and `tabsmith audit`.

```sh
# Python, where the model was trained
booster.save_model("model.json")     # XGBoost: JSON (not the binary .ubj)
booster.save_model("model.txt")      # LightGBM: text

# Tabsmith
tabsmith import model.json --out bundles/xgb --target defect --labels ok,defect
#   imported XGBoost 3.4.1 (binary:logistic; 300 trees, 28240 nodes, 15 features) as bundle b-… at bundles/xgb
#   outputs: the library's margins, bit for bit, for every input (inputs are rounded to f32 as it rounds them)
#   missing values: refused in humidity, voltage_drop (their splits send one both ways); accepted in the other 13 fields
#   self-check: passed (5 examples)
tabsmith predict --bundle bundles/xgb --input rows.csv
```

| Option | Meaning |
|---|---|
| `--out DIR` | the bundle directory to create (required); an identical bundle already there is reused, a different one is an error |
| `--name NAME` | the bundle's name (default: the model file's name) |
| `--target NAME` | the name of the predicted column (default `target`) |
| `--labels a,b,…` | class labels in the model's order: negative,positive for a binary model, one per class for a multiclass one (default `0`, `1`, …) |
| `--features a,b,…` | input field names in the model's feature order (default: the names in the model; XGBoost's `f0`, `f1`, … when it has none) |
| `--threshold P` | binary: the decision threshold on the probability (default 0.5) |

`--json` prints the bundle's manifest, its self-check and what was imported.

## What it imports

| | XGBoost (JSON) | LightGBM (text) |
|---|---|---|
| Trees | `gbtree`, one tree per class and round | gradient-boosted trees (`boosting=gbdt`) |
| Objectives | `binary:logistic`, `binary:logitraw`, `multi:softprob`, `multi:softmax`, and regressions whose output is the margin: `reg:squarederror` (`reg:linear`), `reg:absoluteerror`, `reg:pseudohubererror`, `reg:quantileerror` (one quantile), `reg:squaredlogerror` | `binary` (with `sigmoid:1`), `multiclass`, and `regression`, `regression_l1`, `huber`, `fair`, `quantile`, `mape` |
| Features | numeric | numeric |

Refused, each with its reason (`E_IMPORT_UNSUPPORTED`): categorical splits and features;
outputs that pass through another link (`reg:logistic`, `count:poisson`, `reg:gamma`,
`reg:tweedie`, `survival:*`, LightGBM's `poisson`, `gamma`, `tweedie`, `cross_entropy`,
`multiclassova`, `reg_sqrt`); `binary:hinge` and ranking; XGBoost's `dart` and `gblinear`, random
forests (`num_parallel_tree` > 1, LightGBM's `rf`), trees with vector leaves and models of several
targets; LightGBM's linear trees and a `sigmoid` other than 1; a multiclass model of two
classes (train it as binary); a model that declares more than 1,048,576 features, or more
classes than it has trees. A file that is not one of the two formats is `E_IMPORT_INVALID`.

## How close the bundle is

**XGBoost: bit for bit.** The bundle's raw outputs are XGBoost's margins
(`predict(…, output_margin=True)`), to the last bit, for every input:

- XGBoost sends `x < s` left, Tabsmith `x <= t`. The import stores `t` = the f32 just below
  `s`, which splits every f32 `x` the same way, and both round the inputs to f32.
- Thresholds and leaves are read from their decimal text straight to f32 (read as f64 first, a
  number can round twice to the wrong f32).
- The leaves are added in f32, in XGBoost's tree order, from the base score rounded to f32. The
  model config says so (`"sum": "f32"`); the runtime, the compiled code and the compiler add them
  that way.
- A binary model starts from `-logf(1/p - 1)` for its `base_score` `p`. C libraries round
  `logf` differently (glibc and the Windows CRT disagree on 501,924 f32 inputs), so the import
  carries glibc's `logf`, ported from Arm's code and equal to glibc on every positive f32: the
  margin XGBoost computes on Linux, on every platform.

The probability (sigmoid, softmax) is computed from that margin in f64; XGBoost computes it in
f32, so the two can differ by a few units in the 8th digit.

**LightGBM: within a bound.** LightGBM's leaves are f64; the bundle holds the nearest f32. The
import computes a bound on |the bundle's raw output − LightGBM's raw score| (the leaves' rounding,
the sums', the output's) and reports it; on 300 trees it was about 2e-5, and the measured
difference about 7e-7. Thresholds are rounded down to f32, which takes LightGBM's branch for
every input that is an f32 value. LightGBM compares f64 inputs, so an input that is not an f32
value can take the other branch where it lies closer to a threshold than f32 resolves: 31 to 91
of 90,000 rows in [reports/import](../reports/import/2026-11-06-windows-x86_64/).

**Inputs.** Each field is used as it is (no standardisation), rounded to f32. A missing value is
null, NaN, or one of the tokens `""`, `NA`, `null` in a CSV. Infinities and numbers beyond the
f32 range are refused (`E_INVALID_VALUE`), where XGBoost would round them to infinities and
LightGBM compare them as they are.

## Missing values

Each split of a library's tree sends a missing value one way: XGBoost by the split's
`default_left`; LightGBM by `default_left` when the split's missing type is NaN, and as the value
0 when it is None. For each field, the import looks for one value that goes the missing value's
way at every split on it. When there is one, a missing value is imputed with it, and the bundle
stays exact. When there is none, because some splits send a missing value left and others right,
the field refuses missing values: `nullable` is false in `input.schema.json`, `missing` is
`refuse` in `transform.json`, and a record without it fails with `E_INVALID_VALUE` instead of
getting another prediction than the library's.

Fields that had missing values in training usually refuse them: each library chose a way per
split, and the choices differ. Fields that had none usually accept them.

## What the bundle records

- `evaluation.json` and the manifest: status `imported`, no metrics (Tabsmith did not evaluate
  the model), notes that name the library, its version, the objective, the model file's SHA-256,
  how close the outputs are, and the fields that refuse missing values. The manifest's
  `provenance.candidate` is `xgboost` or `lightgbm`, and `provenance.config_hash` the model file's
  SHA-256.
- `transform.json`: one numeric feature per field, mean 0 and std 1, `fitted_rows` 0. With no
  training statistics, `tabsmith audit` refuses `within k sd` and `--stability`: give ranges with
  `in a..=b` instead.
- `min_runtime_version` is the version that wrote it: older runtimes do not know `sum` and
  `missing`, and refuse the bundle rather than predict differently.
- `examples/`: five records built from the trees' thresholds, with the outputs Tabsmith computed.

## Check it on your own data

```python
import pandas as pd, xgboost as xgb
rows = pd.read_csv("rows.csv")
booster = xgb.Booster(model_file="model.json")
p = booster.predict(xgb.DMatrix(rows[booster.feature_names]))   # probabilities
```

```sh
tabsmith predict --bundle bundles/xgb --input rows.csv --output predictions.csv
```

The probabilities agree to about 1e-7. In Rust, `Predictor::outputs` gives the raw outputs, which
equal `booster.predict(…, output_margin=True)` bit for bit;
`crates/tabsmith-workflow/tests/import.rs` does this for the models in `fixtures/import`.

---

## 中文

`tabsmith import` 把 XGBoost 或 LightGBM 训练的梯度提升模型变成模型包。之后它和 Tabsmith 自己训练的模型包一样可用：`tabsmith predict` 和 `serve`、Rust 应用里的 `Predictor::load`、`model!`（编译进应用，支持 `no_std` 和 `const`）、浏览器版本，以及 `tabsmith audit`。

在 Python 里用 `booster.save_model("model.json")`（XGBoost，JSON 格式，不是二进制的 .ubj）或 `booster.save_model("model.txt")`（LightGBM，文本格式）保存模型，然后运行 `tabsmith import model.json --out bundles/xgb`。选项：`--out` 要创建的模型包目录（必填）；`--name` 模型包名称；`--target` 预测列的名称（默认 `target`）；`--labels` 按模型顺序给出类别标签（二分类为“负类,正类”，默认 `0`、`1`……）；`--features` 按模型的特征顺序给出输入字段名（默认用模型里的名字，XGBoost 没有名字时为 `f0`、`f1`……）；`--threshold` 二分类的判定阈值（默认 0.5）。

**能导入的模型。** XGBoost：`gbtree`；目标函数 `binary:logistic`、`binary:logitraw`、`multi:softprob`、`multi:softmax`，以及输出就是 margin 的回归（`reg:squarederror`、`reg:absoluteerror`、`reg:pseudohubererror`、单分位数的 `reg:quantileerror`、`reg:squaredlogerror`）。LightGBM：梯度提升树；目标函数 `binary`（`sigmoid:1`）、`multiclass`、`regression`、`regression_l1`、`huber`、`fair`、`quantile`、`mape`。特征必须是数值。以下情况会被拒绝并说明原因（`E_IMPORT_UNSUPPORTED`）：类别切分和类别特征；输出经过其他链接函数（如 `reg:logistic`、泊松、伽马、tweedie、`cross_entropy`、`multiclassova`、`reg_sqrt`）；`binary:hinge` 和排序；XGBoost 的 `dart`、`gblinear`、随机森林、向量叶子和多目标模型；LightGBM 的线性树和不等于 1 的 `sigmoid`；只有两个类别的多分类模型（请按二分类训练）；声明超过 1,048,576 个特征、或类别数超过树数的模型。不是这两种格式的文件报 `E_IMPORT_INVALID`。

**有多接近。** XGBoost 逐位一致：模型包的原始输出就是 XGBoost 的 margin（`predict(…, output_margin=True)`），对每个输入都精确到最后一位。XGBoost 把 `x < s` 送往左边，Tabsmith 是 `x <= t`，导入把 `t` 存为紧挨 `s` 下方的 f32，对每个 f32 输入切分结果相同，而两者都先把输入舍入到 f32。阈值和叶子从十进制文本直接解析成 f32。叶子按 XGBoost 的树序在 f32 中相加，起点是舍入到 f32 的起始值；模型配置写明这一点（`"sum": "f32"`）。二分类模型的起点是 `-logf(1/p - 1)`，不同 C 库对 `logf` 舍入不同（glibc 和 Windows CRT 在 501,924 个 f32 输入上不同），所以导入自带移植自 Arm 代码的 glibc `logf`，它在每个正 f32 上都等于 glibc：在任何平台上都得到 XGBoost 在 Linux 上的起点。概率（sigmoid、softmax）由这个 margin 在 f64 中算出，XGBoost 在 f32 中算，两者可能在第 8 位有效数字上差几个单位。LightGBM 的叶子是 f64，模型包存最接近的 f32；导入为每个模型算出模型包原始输出与 LightGBM 原始得分之差的上界并报告（300 棵树时约 2e-5，实测约 7e-7）。阈值向下舍入到 f32，对每个本身是 f32 值的输入都走 LightGBM 的分支；LightGBM 用 f64 比较输入，所以不是 f32 值的输入在离阈值比 f32 精度还近时可能走另一边（报告中 9 万行里有 31 到 91 行）。

**输入。** 每个字段按原值使用（不做标准化），舍入到 f32。缺失值是 null、NaN，或 CSV 中的 `""`、`NA`、`null`。无穷大和超出 f32 范围的数会被拒绝（`E_INVALID_VALUE`），而 XGBoost 会把它们舍入成无穷大，LightGBM 则按原值比较。

**缺失值。** 库的每个切分把缺失值送往一边：XGBoost 按该切分的 `default_left`；LightGBM 在缺失类型为 NaN 时按 `default_left`，为 None 时当作数值 0。导入为每个字段寻找一个在其所有切分上都和缺失值走同一边的值；找到就用它填补缺失值，模型包仍然精确。找不到（有的切分送往左边，有的送往右边）时，该字段拒收缺失值：`input.schema.json` 中 `nullable` 为 false，`transform.json` 中 `missing` 为 `refuse`，缺这个值的记录报 `E_INVALID_VALUE`，而不是得到和库不同的预测。训练时就有缺失值的字段通常会拒收：每个库为每个切分各选了一个方向，而这些方向并不一致。训练时没有缺失值的字段通常都接受。

**模型包里记录的内容。** `evaluation.json` 和 manifest：状态为 `imported`，没有指标（Tabsmith 没有评估这个模型），附注写明库、版本、目标函数、模型文件的 SHA-256、输出有多接近，以及拒收缺失值的字段；manifest 的 `provenance.candidate` 为 `xgboost` 或 `lightgbm`，`provenance.config_hash` 是模型文件的 SHA-256。`transform.json`：每个字段一个数值特征，均值 0、标准差 1，`fitted_rows` 为 0；由于没有训练统计量，`tabsmith audit` 拒绝 `within k sd` 和 `--stability`，请改用 `in a..=b` 给出范围。`min_runtime_version` 是写出它的版本：旧运行时不认识 `sum` 和 `missing`，会拒绝加载，而不是给出不同的预测。

**用自己的数据核对。** 用 XGBoost 在 Python 里预测同一个 CSV，再用 `tabsmith predict --bundle bundles/xgb --input rows.csv --output predictions.csv`，两边的概率相差约 1e-7。在 Rust 里，`Predictor::outputs` 给出原始输出，逐位等于 `booster.predict(…, output_margin=True)`；`crates/tabsmith-workflow/tests/import.rs` 就是这样检查 `fixtures/import` 中的模型的。
