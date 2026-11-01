# Scaling: the gradient-boosted trees on a million rows

[中文](#中文)

**Claim.** `tabsmith train` with the `gbdt` candidate on 1,000,000 rows (172 MB CSV,
25 columns, 35 features after encoding) runs end to end in 18 s on a desktop, with a
peak working set of 1.6 GB, and trains 148 boosting rounds in 4.5 s, 30 ms per round over
698,621 training rows. Quality on the 151,572 validation rows is unchanged from
[the report of 2026-10-29](../2026-10-29-windows-x86_64/), which took 42 s end to end and
29.6 s to train: ROC AUC 0.840, balanced accuracy 0.761 (the synthetic defect problem of
`tabsmith sample-data`, which has the same signal at every size).

**How.** A histogram is built one feature at a time on up to 12 threads, one child's
histogram is the parent's minus its sibling's, and a feature's dominant bin follows from
the node's totals (`docs/design-decisions.md`, D5). The result does not depend on the
number of threads. The data is held in memory.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 threads), 64 GB, Windows 11 Pro
10.0.26300, the release binary of the commit that brings this report. The script is
`scale.ps1` (it generates the data, runs `tabsmith train --candidates gbdt --json`, and
samples the process's peak working set); `rows-1000000.txt` is its output, and
`run.json` the run record with the full history. The same rows in WSL train in 5.2 s,
with identical weights (`../../datasets/2026-10-31-windows-x86_64/trees-linux.txt`).

## Results

| Stage (from the run record) | Time |
|---|---|
| Read and profile the CSV | 1.07 s |
| Fit and apply the transform (train and validation rows) | 0.64 s |
| Train: 148 rounds (early stopping, patience 25; 400 planned) | 4.48 s (30 ms per round) |
| Evaluate on the validation rows | 0.14 s |
| Whole `tabsmith train` process (generate the bundle included), wall clock | 18 s |
| Peak working set of the process | 1,595 MB |
| Generating the 1,000,000-row CSV (`tabsmith sample-data`) | 3 s, 172 MB |

| Model | |
|---|---|
| Validation ROC AUC / balanced accuracy / log loss | 0.8396 / 0.7610 / 0.4848 |
| Trees | 148 trees, `model.safetensors` 151 KB |
| Single-row prediction in the runtime (p50 / p95) | 3.0 µs / 3.6 µs |

## Not measured

Ten million rows (a 1.7 GB CSV) were not run; the data path holds the table and its
feature matrix in memory, so expect about 16 GB of working set. LightGBM and XGBoost, the
reference trainers at this scale, were not available here for a side-by-side number.

---

## 中文

**结论。** `gbdt` 候选在 1,000,000 行（172 MB CSV，25 列，编码后 35 个特征）上，`tabsmith train` 端到端 18 秒，
进程峰值工作集 1.6 GB；148 轮提升用时 4.5 秒，在 698,621 行训练数据上每轮 30 毫秒。151,572 行验证集上的质量与
[2026-10-29 的报告](../2026-10-29-windows-x86_64/)相同（当时端到端 42 秒，训练 29.6 秒）：ROC AUC 0.840、平衡准确率 0.761。

**原因。** 直方图按特征构建，最多用 12 个线程；一个子节点的直方图由父节点减去兄弟节点得到；占绝大多数行的桶由节点总和反推（设计决策 D5）。
结果与线程数无关。同样的行在 WSL 里训练 5.2 秒，权重完全相同。

**未测量。** 一千万行（1.7 GB CSV）没有跑：表和特征矩阵都在内存里，预计约 16 GB 工作集。这里没有 LightGBM 和 XGBoost 可供对照。
