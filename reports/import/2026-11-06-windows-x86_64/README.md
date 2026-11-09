# Models imported from XGBoost and LightGBM, against the libraries' own outputs

[中文](#中文)

**Claim.** `tabsmith import` turns an XGBoost model into a bundle that computes XGBoost's margins
bit for bit. On 100,000 rows each, for models of 300 and 1,200 trees, every row the bundle
accepts gets XGBoost's margin to the last bit: 270,980 rows, on both of the runtime's paths
(row by row, and tree by tree over blocks of rows). So do 11,901 rows of 12 binary models whose
starting margin glibc's `logf` rounds differently from the correctly rounded value. LightGBM's
leaves are f64 and the bundle holds f32 leaves, so LightGBM models come within a bound the import
computes for each model: on the same rows the largest difference was 3 % to 17 % of the bound.

**Where.** AMD Ryzen 9 9950X3D2, Windows 11, Rust 1.97, a release build of
`crates/tabsmith-workflow/tests/import.rs` (`evidence`). The models and the libraries' outputs
were made in WSL 2 (Ubuntu 26.04, glibc 2.43) on the same machine by [reference.py](reference.py),
with XGBoost 3.4.1, LightGBM 4.7.0, numpy 2.5.4 and Python 3.14.4
([summary.json](summary.json)). So the margins were computed by XGBoost on Linux and
reproduced by Tabsmith on Windows.

**Data.** `tabsmith sample-data --rows 100000 --seed 7` for the three tasks (synthetic, CC0), the
15 numeric fields. `humidity` and `voltage_drop` have missing values; for every row, with
probability 0.1, one or two random fields were set missing as well. Each model was trained on
the first 50,000 rows and predicted on all 100,000.

## XGBoost: bit for bit

| Model | Trees | Nodes | Rows compared | Same bits | Refused: a missing value in a refusing field |
|---|---:|---:|---:|---:|---:|
| binary (`binary:logistic`, depth 6) | 300 | 28,240 | 90,268 | 90,268 | 9,732 |
| 4 classes (`multi:softprob`) | 1,200 | 119,928 | 90,320 | 90,320 | 9,680 |
| regression (`reg:squarederror`) | 300 | 35,370 | 90,392 | 90,392 | 9,608 |
| 12 binary models, base score set (below) | 5 each | | 11,901 | 11,901 | 99 |

"Same bits" compares the bundle's raw outputs (`Predictor::outputs`) with
`booster.predict(…, output_margin=True)`, as f32 bits. XGBoost's `inplace_predict` gave the same
margins as its `predict` on every row. The 4-class and regression models have more than 32,768
nodes, so the runtime added their trees tree by tree over blocks of 256 rows; the binary models
row by row. Both add in f32, in XGBoost's tree order, from the base score rounded to f32
(`kernels::Trees::outputs_in_f32`).

How the bundle gets there:

- XGBoost sends `x < s` left, Tabsmith `x <= t`. With `t` the f32 just below `s`, the two agree for
  every f32 `x`, and XGBoost rounds its inputs to f32 as the bundle does.
- Thresholds and leaves are read from their decimal text straight to f32. Read as f64 first and
  then rounded, a decimal can round twice and land on the other f32 (a test shows one).
- The trees of a multiclass model are put in Tabsmith's order (tree `t` for class `t % K`), each
  class keeping the order of its own trees, which fixes the order of its sum.
- A missing value goes where XGBoost's `default_left` sends it, when one value goes that way at
  every split on its field (below).

## The starting margin and glibc's `logf`

A binary model stores `base_score` as a probability `p`, and XGBoost starts each margin from
`-logf(1.0f / p - 1.0f)`, with the C library's `logf`. C libraries round `logf` differently. Over
all 2,139,095,039 positive finite f32 inputs ([logf_probe.rs](logf_probe.rs)):

| | differs from glibc 2.43's `logf` on |
|---|---:|
| the import's port of Arm's `logf` (musl, glibc), plain or with fused multiply-adds | 0 inputs ([logf-linux.txt](logf-linux.txt)) |
| the correctly rounded value, `(x as f64).ln() as f32` | 416,909 |
| the Windows CRT's `logf` | 501,924 ([logf-windows.txt](logf-windows.txt)) |

The Windows CRT and glibc disagree on more inputs than either does with the correctly rounded
value, so XGBoost itself starts some binary models from different margins on Windows and on
Linux. The import fixes the margin XGBoost computes on Linux, with the port, so a bundle starts
from the same bits on every platform.

The inputs that matter are `1/p − 1` for base scores near the class balance, and there the
difference is common: of 1,596 random base scores in (0.02, 0.98), 12 (0.75 %) get a starting
margin from glibc that is not the correctly rounded one, most of them near 0.5. Twelve models
trained with those base scores are the last row of the table above. Started from the correctly
rounded margin instead, 1,498 of their 12,000 margins would come out one ulp off.

## LightGBM: within the bound

| Model | Trees | Nodes | Rows compared | Largest difference | Bound | Inputs as f64: rows beyond the bound, largest difference |
|---|---:|---:|---:|---:|---:|---:|
| binary (63 leaves) | 300 | 37,500 | 90,268 | 7.11e-7 | 2.31e-5 | 42, 2.01 |
| 4 classes | 1,200 | 150,000 | 90,320 | 6.20e-7 | 1.64e-5 | 91, 1.83 |
| regression | 300 | 37,500 | 90,392 | 2.55e-3 | 1.46e-2 | 31, 710 |

The difference is |the bundle's raw output − `predict(…, raw_score=True)`|. The regression's raw
scores reach 40,229 (median 19,776), so its difference is about 1e-7 of the value. The bound
adds up, per output, the largest rounding of a leaf to f32 in each tree, the rounding of both
sums, and the rounding of the output to f32; the import reports it, and the test checks every
row against it.

LightGBM compares f64 inputs with f64 thresholds; the bundle rounds each threshold down to f32,
which takes the same branch for every input that is an f32 value. The "largest difference"
column gave LightGBM the rows rounded to f32. Given the rows as they are, LightGBM took another
branch somewhere on 31 to 91 of about 90,000 rows: an input and a threshold closer than f32
can tell apart. This does not happen with XGBoost, which rounds its inputs to f32 itself.

## Missing values

Each split sends a missing value one way. XGBoost learns the way per split (`default_left`);
LightGBM sends it by `default_left` when the split's missing type is NaN, and as 0 when the
missing type is None. When one value takes the missing value's way at every split on its field,
the import imputes that value and the bundle stays exact; the rows above with missing values in
the 13 other fields are among the bit-for-bit ones. `humidity` and `voltage_drop` had missing
values in training, and both libraries sent them left at some splits and right at others, which
no single value reproduces: those fields refuse missing values (`E_INVALID_VALUE`) rather than
predict differently, and the test checks that every such row is refused.

## Reproduce

```sh
# In an environment with xgboost, lightgbm, pandas (here: a venv in WSL)
tabsmith sample-data big-binary.csv --task binary --rows 100000 --seed 7      # and multiclass, regression
python reference.py evidence                                                  # writes evidence/
TABSMITH_IMPORT_EVIDENCE=evidence cargo test -p tabsmith-workflow --test import --release -- --ignored --nocapture
rustc -O logf_probe.rs && ./logf_probe                                         # on Linux and on Windows
```

`python reference.py fixtures` made the small models in `fixtures/import/`, which
`cargo test -p tabsmith-workflow --test import` checks the same way on every run.
[evidence.md](evidence.md) is the test's output.

---

## 中文

**结论。** `tabsmith import` 把 XGBoost 模型变成一个逐位算出 XGBoost margin 的模型包。在 300 棵树和 1,200 棵树的模型上，各 10 万行里模型包接受的每一行都得到 XGBoost 的 margin，直到最后一位：共 270,980 行，运行时的两条路径（逐行，以及按块逐树）都覆盖到了。另有 12 个二分类模型，它们的起始 margin 被 glibc 的 `logf` 舍入得和正确舍入值不同，其中 11,901 行同样逐位一致。LightGBM 的叶子是 f64，模型包存的是 f32，所以 LightGBM 模型的输出落在导入时为每个模型算出的界内：在同样的行上，最大差是界的 3% 到 17%。

**环境。** AMD Ryzen 9 9950X3D2，Windows 11，Rust 1.97，release 构建的 `crates/tabsmith-workflow/tests/import.rs`（`evidence`）。模型和两个库的输出由 [reference.py](reference.py) 在同一台机器的 WSL 2（Ubuntu 26.04，glibc 2.43）里生成，用的是 XGBoost 3.4.1、LightGBM 4.7.0、numpy 2.5.4、Python 3.14.4。也就是说，margin 由 Linux 上的 XGBoost 算出，由 Windows 上的 Tabsmith 复现。

**数据。** 三种任务各用 `tabsmith sample-data --rows 100000 --seed 7`（合成数据，CC0）的 15 个数值字段。`humidity` 和 `voltage_drop` 本身有缺失值；另外每行以 0.1 的概率再随机把一两个字段设为缺失。每个模型用前 5 万行训练，在全部 10 万行上预测。

**XGBoost 逐位一致的做法。** XGBoost 把 `x < s` 送往左边，Tabsmith 是 `x <= t`；取 `t` 为紧挨 `s` 下方的 f32，两者对每个 f32 输入都一致，而 XGBoost 和模型包一样先把输入舍入到 f32。阈值和叶子从十进制文本直接解析成 f32：先解析成 f64 再舍入会舍入两次，可能落到另一个 f32 上（有测试展示了一例）。多分类模型的树按 Tabsmith 的顺序排列（第 `t` 棵树属于第 `t % K` 类），每一类保持自己的树的顺序，求和顺序因此不变。叶子在 f32 中按 XGBoost 的树序相加，起点是舍入到 f32 的起始值。

**起始 margin 与 glibc 的 `logf`。** 二分类模型把 `base_score` 存成概率 `p`，XGBoost 用 C 库的 `logf` 算 `-logf(1.0f / p - 1.0f)` 作为起点。不同 C 库对 `logf` 的舍入不同。在全部 2,139,095,039 个正有限 f32 输入上：导入用的 Arm `logf` 移植（musl、glibc 用的同一份算法）与 glibc 2.43 有 0 个不同；正确舍入值有 416,909 个不同；Windows CRT 的 `logf` 有 501,924 个不同。所以 XGBoost 自己在 Windows 和 Linux 上就会从不同的起点开始算某些二分类模型。导入固定采用 XGBoost 在 Linux 上算出的起点，模型包在任何平台上都从同样的位开始。起始值在类别比例附近时这种差别很常见：在 (0.02, 0.98) 里随机取 1,596 个 base score，有 12 个（0.75%）的 glibc 起点不是正确舍入值，大多在 0.5 附近。用这 12 个 base score 训练的模型如果从正确舍入的起点开始，12,000 个 margin 里会有 1,498 个差一个 ulp。

**LightGBM。** 二分类最大差 7.11e-7（界 2.31e-5），4 分类 6.20e-7（界 1.64e-5），回归 2.55e-3（界 1.46e-2；输出最大 40,229，相对误差约 1e-7）。界逐个输出累加每棵树中叶子舍入到 f32 的最大误差、两边求和的舍入，以及输出舍入到 f32 的误差。LightGBM 用 f64 比较输入和阈值；模型包把阈值向下舍入到 f32，对每个本身就是 f32 值的输入都走同一分支。如果把原始 f64 行直接交给 LightGBM，约 9 万行里有 31 到 91 行在某处走了另一分支：输入和阈值靠得太近，f32 分不开。XGBoost 自己先把输入舍入到 f32，所以没有这个问题。

**缺失值。** 每个切分把缺失值送往一边。XGBoost 为每个切分学习方向（`default_left`）；LightGBM 在缺失类型为 NaN 时按 `default_left`，为 None 时当作 0。如果存在一个值，在该字段的每个切分上都和缺失值走同一边，导入就填补这个值，结果仍然精确：上表中在其余 13 个字段有缺失值的行都在逐位一致的行里。`humidity` 和 `voltage_drop` 在训练时就有缺失值，两个库在一些切分上把它们送往左边、另一些送往右边，没有单个值能复现，所以这两个字段拒收缺失值（`E_INVALID_VALUE`），而不是给出不同的预测；测试检查每一个这样的行都被拒收。
