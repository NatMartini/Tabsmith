# Reports / 报告

Measured evidence behind the claims in the README. Each report says which commit
and machine it comes from and how to reproduce it; the machine-readable files
next to it are the evidence itself, with local paths replaced by `<work>` /
`<repo>`. The table groups the reports by the kind of claim they support.

| Supports | Report | What it shows | Produced by |
|---|---|---|---|
| Whole workflow | [implementation-acceptance.md](implementation-acceptance.md) | The whole CPU workflow through the release binary, on version 0.4 and one binary classification problem: every step's exit code and time, 12 preset failures, exact resume after a crash, determinism, the scikit-learn reference check and the training-loop overhead | [`tools/acceptance.sh`](../tools/acceptance.sh) + [`tools/acceptance_report.py`](../tools/acceptance_report.py) |
| Quality | [datasets/](datasets/) | Three public datasets (Adult, Covertype, California Housing): every candidate against its scikit-learn equivalent on the same rows and split, test metrics side by side, including where Tabsmith loses | [`tools/datasets/compare.py`](../tools/datasets/compare.py) + [`tools/datasets/report.py`](../tools/datasets/report.py) |
| Speed and resources | [benchmarks/](benchmarks/) | Tabsmith vs the equivalent pandas + scikit-learn workflow on the same rows, split, features and settings: quality, training time, prediction latency, cold start, memory, deployment size. The newest run includes the gradient-boosted trees against `HistGradientBoostingClassifier` | [`tools/bench/bench.py`](../tools/bench/bench.py) |
| Speed and resources | [training/](training/) | The trees against LightGBM and XGBoost on one machine, the same rows and settings: time at each thread count, quality, peak memory, whether the model repeats; and what one parallel section costs under rayon and under spinning workers | the scripts in the report, [`tools/datasets/compare_gbm.py`](../tools/datasets/compare_gbm.py) |
| Speed and resources | [inference/](inference/) | Trees as generated code against trees as arrays on real models: time per row, compile time, flash on a Cortex-M4F, the tests that hold them to the same bits, and batch prediction by thread count | the probes and scripts in the report |
| Speed and resources | [scaling/](scaling/) | The trees on 1,000,000 rows: stage times, rounds per second, peak memory, quality | the script in the report |
| Determinism | [reproducibility/](reproducibility/) | The same training on Windows and on Linux: the trees' weights and history are identical to the bit, the Burn-trained networks differ in the seventh digit | the two scripts in the report |
| Determinism | [gpu/](gpu/) | Certification of each GPU path on real hardware: determinism across processes, exact resume, agreement with the CPU, bundle self-check, and training time against the CPU | [`tools/gpu_certify.py`](../tools/gpu_certify.py) |
| Guarantees | [kani/](kani/) | Kani proofs that the models compiled into code never panic, for every input: each proof's checks, time, memory and result, why generated trees can be proved all at once while arrays are proved one walk at a time, and a false positive in CBMC's NaN check | `cargo kani` in [`examples/embedded-predict`](../examples/embedded-predict/) |
| Guarantees | [fuzz/](fuzz/) | Fuzzing of the bundle loader with libFuzzer: duration, executions, coverage, seeds, result | [`fuzz/`](../fuzz/) |

Numbers depend on the machine. Re-run the tools on yours rather than comparing
absolute values across machines; a new report goes into a new dated directory,
and old reports are kept, favourable or not. Where a newer report shows that an older
one explained something wrongly, the older one carries a note saying so.

**Machine labels.** A directory name ends with the system its numbers come from.
`windows-x86_64` is native Windows 11. `wsl-linux-x86_64` is Linux under WSL 2 on the same
machine, and so are the earlier `linux-x86_64` directories, whose names stay so that
published links keep working. Speed measured in WSL 2 is not that of native Linux: waking a
sleeping thread costs far more there ([training/](training/)).

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
旁边的机器可读文件就是证据本身（本地路径已替换为 `<work>` / `<repo>`）。上面的表格按报告支持的说法分组：整体验收（Whole workflow）、质量（Quality）、速度和资源（Speed and resources）、确定性（Determinism）、保证（Guarantees）。数字取决于机器，请在自己的
机器上重新运行这些工具，而不要跨机器比较绝对值；新报告放在新的带日期目录中，旧报告无论结果好坏都会保留。如果新报告证明旧报告的某个解释是错的，旧报告里会加一条说明。

**机器标签**：目录名的结尾是数字来自的系统。`windows-x86_64` 是原生 Windows 11；`wsl-linux-x86_64` 是同一台机器上 WSL 2 里的 Linux，
更早的 `linux-x86_64` 目录也是，为了让已经发布的链接继续有效，没有改名。WSL 2 里测出的速度不等于原生 Linux：在那里唤醒一个睡眠的线程要贵得多（[training/](training/)）。

**旧名称**：第一份基准报告（`benchmarks/2026-10-27-linux-x86_64`）产生于提交 `ae39055`，当时项目还叫 RustPy，报告按原样保留：其中的 `rustpy` 命令、`rustpy-*`
crate、`rustpy.toml` 和 `.rustpy/` 如今分别是 `tabsmith`、`tabsmith-*`、`tabsmith.toml` 和 `.tabsmith/`。改名没有改变任何
计算：同样的数据得到同样的切分、权重、指标和预测，只有对格式名做哈希的 ID（切分、预处理和模型包的 ID）不同。
此后的改动确实轻微改变了计算：从 Burn 的 NdArray 后端迁移到 Flex 会让权重在最后几位上不同（约 1e-7），因此在那之前的数字可能在小数点后第四位有差异。验收报告、第二份基准报告以及 GPU 和模糊测试报告都是在这些改动之后产生的。
