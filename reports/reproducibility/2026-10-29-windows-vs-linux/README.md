# The same trees on two operating systems

[中文](#中文)

**Claim.** Training the gradient-boosted trees (`gbdt`) on the same data with the same
settings gives bit-identical weights on Windows and on Linux, and an identical training
history. The networks (`linear`, `small-mlp`, trained with Burn) do not: their weights
differ, and their losses differ in the seventh significant digit.

**Why.** The trees are plain Rust: histogram sums in a fixed order, and their only
transcendental functions, `exp` in the gradients and `log` in the starting score and the
losses, come from the pure-Rust `libm` crate, the same one inference uses. The platform
C libraries (glibc, the Microsoft C runtime) differ in the last bit of `exp` and `log`,
and a tree's split can flip on it. Burn's CPU backend computes the networks' matrix
products with platform-dependent code, so their weights are reproducible across
processes on one machine (`reports/implementation-acceptance.md`, exact resume) but not
across operating systems.

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
| run-003 | gbdt | **identical** (`85bf7fd7…a6f7`) | 39 epochs on both; all 39 records identical to every digit |

The data file is identical on both sides (`ee2ea83c…256e`), so the comparison is about
the training code, not the input.

## What this does and does not show

- It shows independence from the operating system and its C library, on one CPU
  model. Both sides ran on the same x86-64 processor. Independence from the CPU
  architecture (an ARM machine, say) is expected for the same reason, since Rust
  does not fuse multiply-adds unless asked and `libm` is pure Rust, but it was not
  measured: there is no ARM machine here.
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
`*-hashes.txt` files.

---

## 中文

**结论。** 用同样的数据和设置训练梯度提升树（`gbdt`），Windows 和 Linux 得到逐位相同的权重和完全相同的训练历史；
用 Burn 训练的两个网络（`linear`、`small-mlp`）则不同：权重不同，损失在第七位有效数字上有差异。

**原因。** 树是纯 Rust 实现：直方图求和顺序固定，仅有的两个超越函数（梯度里的 `exp`、起始分和损失里的 `log`）
都来自纯 Rust 的 `libm`，和推理用的是同一个。平台 C 库（glibc、微软 C 运行时）的 `exp`、`log` 在最后一位上有差异，
一次分裂就可能因此翻转。Burn 的 CPU 后端用与平台相关的代码算矩阵乘法，所以网络的权重在同一台机器上跨进程可复现（精确续训依赖这一点），
但跨操作系统不行。

**环境。** 开发机（AMD Ryzen 9 9950X3D2）上的 Windows 11 和 WSL 2 Ubuntu 26.04，同一颗 CPU。两边都从同一份工作树构建 release 版 `tabsmith`，
生成同样的 3,000 行合成数据（CSV 的 SHA-256 两边相同），在 CPU 上运行同一条 `tabsmith train`。脚本是 `windows.ps1` 和 `linux.sh`，
原始输出是两个 `*-hashes.txt`。

**结果。** run-003（gbdt）：权重哈希相同（`85bf7fd7…a6f7`），39 轮历史逐位相同。run-001（linear）、run-002（small-mlp）：权重不同，
验证损失最大差异分别为 1.9 × 10⁻⁷ 和 4.8 × 10⁻⁷。

**说明。** 这证明的是与操作系统及其 C 库无关，在同一型号的 CPU 上；与 CPU 架构无关（例如 ARM）按同样的理由是可以预期的
（Rust 不会自动融合乘加，`libm` 是纯 Rust），但没有测量：这里没有 ARM 机器。网络的差异是 Burn 的性质，不是 Tabsmith 的；
交付的模型包在任何地方的预测都相同，只有网络的训练绑定在它的机器上。
