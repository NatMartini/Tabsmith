# Scaling: the gradient-boosted trees on a million rows

[中文](#中文)

**Claim.** `tabsmith train` with the `gbdt` candidate on 1,000,000 rows (172 MB CSV,
25 columns, 35 features after encoding) runs end to end in 42 s on a desktop, with a
peak working set of 1.6 GB, and trains 148 boosting rounds in 29.6 s, about 0.2 s per
round over 698,621 training rows. Quality on the 151,572 validation rows: ROC AUC 0.840,
balanced accuracy 0.761 (the synthetic defect problem of `tabsmith sample-data`, which
has the same signal at every size).

**How.** Binning, histogram building and the application of each grown tree run on
every core (32 logical CPUs here); sums run in fixed chunks of 16,384 rows merged in
order, so the trees do not depend on the core count (`docs/design-decisions.md`, D5).
The gradients, the partition of a node's rows and the per-round bookkeeping are
single-threaded, and each leaf's histogram is built from its own rows (no
parent-minus-sibling trick). The data is held in memory.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 threads), 64 GB, Windows 11 Pro
10.0.26300, the release binary of the commit that brings this report. The script is
`scale.ps1` (it generates the data, runs `tabsmith train --candidates gbdt --json`, and
samples the process's peak working set); `rows-1000000.txt` is its output, and
`run.json` the run record with the full history.

## Results

| Stage (from the run record) | Time |
|---|---|
| Read and profile the CSV | 0.88 s |
| Fit and apply the transform (train and validation rows) | 0.59 s |
| Train: 148 rounds (early stopping, patience 25; 400 planned) | 29.6 s (0.20 s per round) |
| Evaluate on the validation rows | 0.14 s |
| Whole `tabsmith train` process (generate the bundle included), wall clock | 42 s |
| Peak working set of the process | 1,602 MB |
| Generating the 1,000,000-row CSV (`tabsmith sample-data`) | 4 s, 172 MB |

| Model | |
|---|---|
| Validation ROC AUC / balanced accuracy / log loss | 0.8396 / 0.7610 / 0.4848 |
| Trees | 148 trees, `model.safetensors` 151 KB |
| Single-row prediction in the runtime (p50 / p95) | 2.9 µs / 3.6 µs |

For scale: the README benchmark's 10,000-row problem trained 54 rounds in 0.67 s on
one thread (0.012 s per round over 7,356 rows, 0.6 million row-visits per second per
round); here a round visits 3.5 million rows per second. That factor of about six comes
from the parallel histograms; the serial parts of a round (gradients, partitioning)
take the rest.

## Not measured

Ten million rows (a 1.7 GB CSV) were not run; the data path holds the table and its
feature matrix in memory, so expect about 16 GB of working set and about five minutes.
LightGBM and XGBoost, the reference trainers at this scale, were not available here
for a side-by-side number; on 10,000 rows Tabsmith's trees were faster than
scikit-learn's `HistGradientBoostingClassifier` on 32 threads, but that is a small
problem and says little about a million rows.

---

## 中文

**结论。** `gbdt` 候选在 1,000,000 行（172 MB CSV，25 列，编码后 35 个特征）上，`tabsmith train` 端到端 42 秒，
进程峰值工作集 1.6 GB；148 轮提升用时 29.6 秒，在 698,621 行训练数据上约每轮 0.2 秒。151,572 行验证集上 ROC AUC 0.840、
平衡准确率 0.761（`tabsmith sample-data` 的合成缺陷问题，各种规模下信号相同）。

**原因。** 分箱、直方图构建和每棵新树的应用都用上所有核心（这里是 32 个逻辑 CPU）；求和按 16,384 行的固定分块进行、按块顺序合并，
所以树不依赖核心数（设计决策 D5）。梯度、节点行的划分和每轮的记账是单线程的，每个叶子的直方图从自己的行重新构建（没有“父减兄弟”的技巧）。数据驻留内存。

**未测量。** 一千万行（1.7 GB CSV）没有跑：表和特征矩阵都在内存里，预计约 16 GB 工作集、约五分钟。这里没有 LightGBM 和 XGBoost 可供对照。
