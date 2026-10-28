# Reports / 报告

Measured evidence behind the claims in the README. Each report says which commit
and machine it comes from and how to reproduce it; the machine-readable files
next to it are the evidence itself, with local paths replaced by `<work>` /
`<repo>`.

| Report | What it shows | Produced by |
|---|---|---|
| [implementation-acceptance.md](implementation-acceptance.md) | The whole CPU workflow through the release binary: every step's exit code and time, 12 preset failures, exact resume after a crash, determinism, the scikit-learn reference check and the training-loop overhead | [`tools/acceptance.sh`](../tools/acceptance.sh) + [`tools/acceptance_report.py`](../tools/acceptance_report.py) |
| [benchmarks/](benchmarks/) | Tabsmith vs the equivalent pandas + scikit-learn workflow on the same rows, split, features and settings: quality, training time, prediction latency, cold start, memory, deployment size. The newest run includes the gradient-boosted trees against `HistGradientBoostingClassifier` | [`tools/bench/bench.py`](../tools/bench/bench.py) |
| [gpu/](gpu/) | Certification of each GPU path on real hardware: determinism across processes, exact resume, agreement with the CPU, bundle self-check, and training time against the CPU | [`tools/gpu_certify.py`](../tools/gpu_certify.py) |
| [fuzz/](fuzz/) | Fuzzing of the bundle loader with libFuzzer: duration, executions, coverage, seeds, result | [`fuzz/`](../fuzz/) |
| [kani/](kani/) | Kani proofs that the models compiled into code never panic, for every input: each proof's checks, time and result, and a false positive in CBMC's NaN check | `cargo kani` in [`examples/embedded-predict`](../examples/embedded-predict/) |

Numbers depend on the machine. Re-run the tools on yours rather than comparing
absolute values across machines; a new report goes into a new dated directory,
and old reports are kept, favourable or not.

**Older name.** The first benchmark (`benchmarks/2026-10-27-linux-x86_64`) was made at commit `ae39055`, before the
project was renamed from RustPy to Tabsmith, and is kept as recorded:
it shows the `rustpy` command, `rustpy-*` crates, `rustpy.toml` and `.rustpy/`,
which are `tabsmith`, `tabsmith-*`, `tabsmith.toml` and `.tabsmith/` today. The
rename changed no computation: the same data gives the same splits, weights,
metrics and predictions. Only IDs that hash a format name (split, transform and
bundle IDs) differ. Later changes did change computations slightly: the move from
Burn's NdArray backend to Flex changes weights in the last bits (about 1e-7), so
numbers from before it can differ in the fourth decimal. The acceptance report, the
second benchmark and the GPU and fuzzing reports were made after these changes.

中文：README 中各项说法所依据的实测证据。每份报告都注明了来自哪个提交、哪台机器，以及如何复现；
旁边的机器可读文件就是证据本身（本地路径已替换为 `<work>` / `<repo>`）。数字取决于机器，请在自己的
机器上重新运行这些工具，而不要跨机器比较绝对值；新报告放在新的带日期目录中，旧报告无论结果好坏都会保留。

**旧名称**：第一份基准报告（`benchmarks/2026-10-27-linux-x86_64`）产生于提交 `ae39055`，当时项目还叫 RustPy，报告按原样保留：其中的 `rustpy` 命令、`rustpy-*`
crate、`rustpy.toml` 和 `.rustpy/` 如今分别是 `tabsmith`、`tabsmith-*`、`tabsmith.toml` 和 `.tabsmith/`。改名没有改变任何
计算：同样的数据得到同样的切分、权重、指标和预测，只有对格式名做哈希的 ID（切分、预处理和模型包的 ID）不同。
此后的改动确实轻微改变了计算：从 Burn 的 NdArray 后端迁移到 Flex 会让权重在最后几位上不同（约 1e-7），因此在那之前的数字可能在小数点后第四位有差异。验收报告、第二份基准报告以及 GPU 和模糊测试报告都是在这些改动之后产生的。
