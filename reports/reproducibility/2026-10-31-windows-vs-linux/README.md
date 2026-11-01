# The same trees on two operating systems and any number of threads

[中文](#中文)

**Claim.** Training the gradient-boosted trees (`gbdt`) on the same data with the same
settings gives bit-identical weights on Windows and on Linux, and at any number of
threads from 1 to 32. The networks (`linear`, `small-mlp`, trained with Burn) do not match
across operating systems: their weights differ, and their losses differ in the seventh
significant digit.

This repeats [the report of 2026-10-29](../2026-10-29-windows-vs-linux/) after the trees'
training was rewritten for speed, and adds the thread counts.

**Why.** The trees are plain Rust. A histogram is built one feature at a time, and each
feature sums its rows in row order on one thread, so every (feature, bin) sees the same
additions in the same order however the features are spread over threads. One child's
histogram is the parent's minus its sibling's, and which child that is depends on the row
counts alone. Their only transcendental functions, `exp` in the gradients and `log` in the
starting score and the losses, come from the pure-Rust `libm` crate, the same one
inference uses. The platform C libraries differ in the last bit of `exp` and `log`, and a
tree's split can flip on it. Burn's CPU backend computes the networks' matrix products
with platform-dependent code, so their weights are reproducible across processes on one
machine but not across operating systems.

**Where.** The development machine (AMD Ryzen 9 9950X3D2): Windows 11 Pro 10.0.26300
and WSL 2 Ubuntu 26.04 (Linux 6.6.87.2) on the same CPU. Both sides built
`tabsmith` in release mode from the same working tree (the commit that brings this
report), generated the same 3,000-row synthetic sample (`tabsmith sample-data data.csv
--rows 3000 --seed 7`; the CSV's SHA-256 is the same on both sides) and ran
`tabsmith train data.csv --target defect --group device_id --id unit_id` on the CPU.
The scripts are `windows.ps1` and `linux.sh`, the raw outputs `windows-hashes.txt` and
`linux-hashes.txt`.

## Results

| Run | Candidate | Weights (SHA-256 of `model.safetensors`) | Training history (epoch, train loss, valid loss) |
|---|---|---|---|
| run-001 | linear | **different** | 26 epochs on both; no epoch record identical; largest validation-loss difference 1.9 × 10⁻⁷ |
| run-002 | small-mlp | **different** | 12 epochs on both; no epoch record identical; largest difference 4.8 × 10⁻⁷ |
| run-003 | gbdt | **identical** (`b8c746bc…`) | 39 epochs on both; all 39 records identical to every digit |

The data file is identical on both sides (`ee2ea83c…256e`), so the comparison is about
the training code, not the input.

Larger problems, the trees only, at several thread counts on both systems
(`../../datasets/2026-10-31-windows-x86_64/trees-windows.txt` and `trees-linux.txt`):

| Problem | Threads tried on Windows | Threads tried in WSL | Weights (first 16 hex digits) |
|---|---|---|---|
| California, 334 rounds | 1, 4, 8, 16 | 1, 4, 8, 12, 16 | identical: `6cdc0dee0a7e9668` |
| Adult, 180 rounds | 1, 4, 8, 16, 32 | 1, 4, 8, 12, 16, 32 | identical: `eea00e7112c87cc1` |
| Covertype, 40 rounds | 1, 4, 6, 8, 12, 16, 24, 32 | 1, 2, 4, 6, 8, 12, 16, 24, 32 | identical: `9b86e39024e4db16` |
| Covertype, 400 rounds | 12 | 12 | identical: `ab735adb28f14eaa` |
| 1,000,000 rows, 148 rounds | 4, 8, 12, 16, 32 | 1, 4, 8, 12, 32 | identical: `46c4eea6dde5569f` |

California's weights are also the same as before the rewrite. Adult's and Covertype's are
not: histogram differences round differently from direct sums, which changes a few splits
and ties. The test metrics are the same to four decimals (`reports/datasets/`).

## What this does and does not show

- It shows independence from the operating system, its C library and the thread count, on
  one CPU model. Independence from the CPU architecture (an ARM machine, say) is expected
  for the same reason, since Rust does not fuse multiply-adds unless asked and `libm` is
  pure Rust, but it was not measured: there is no ARM machine here.
- The networks are a Burn property, not a Tabsmith one. A Tabsmith run of the networks
  is reproducible on the machine that made it, which exact resume relies on, and the
  model that ships (the bundle) predicts identically everywhere; only the training of
  the networks is tied to its machine.
- The history being identical means early stopping stops at the same round, so the
  bundle a user builds from the trees on either system is the same bundle, with the
  same bundle id.

## Reproduce

Run `windows.ps1` on Windows and `linux.sh` on Linux (they build the CLI, train, and
print the hashes; adjust the two paths at their top), then compare the two
`*-hashes.txt` files. `RAYON_NUM_THREADS` sets the thread count.

---

## 中文

**结论。** 用同样的数据和设置训练梯度提升树（`gbdt`），Windows 和 Linux 得到逐位相同的权重，从 1 到 32 的任何线程数也都相同。
用 Burn 训练的两个网络（`linear`、`small-mlp`）跨操作系统不同：权重不同，损失在第七位有效数字上有差异。

这份报告在树的训练为提速重写之后重复了 [2026-10-29 的报告](../2026-10-29-windows-vs-linux/)，并加上了线程数。

**原因。** 树是纯 Rust 实现。直方图按特征构建，每个特征在一个线程上按行序求和，所以无论特征怎样分配到线程，每个（特征，桶）看到的都是同样顺序的同样加法。
一个子节点的直方图由父节点减去兄弟节点得到，选哪个子节点只取决于行数。仅有的两个超越函数都来自纯 Rust 的 `libm`。
Burn 的 CPU 后端用与平台相关的代码算矩阵乘法，所以网络的权重只在同一台机器上可复现。

**结果。** run-003（gbdt）：权重哈希相同（`b8c746bc…`），39 轮历史逐位相同。run-001（linear）、run-002（small-mlp）：权重不同，
验证损失最大差异分别为 1.9 × 10⁻⁷ 和 4.8 × 10⁻⁷。在 California、Adult、Covertype 和 100 万行上，两个系统的多种线程数下权重也都相同。

**说明。** California 的权重与重写前相同。Adult 和 Covertype 的不同：直方图相减与直接求和的舍入不一样，个别分裂和平局因此改变，测试指标到小数点后四位不变。
这证明的是与操作系统、C 库和线程数无关，在同一型号的 CPU 上；与 CPU 架构无关（例如 ARM）按同样的理由可以预期，但没有测量。
