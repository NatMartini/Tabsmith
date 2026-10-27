# Reports / 报告

Measured evidence behind the claims in the README. Each report says which commit
and machine it comes from and how to reproduce it; the machine-readable files
next to it are the evidence itself, with local paths replaced by `<work>` /
`<repo>`.

| Report | What it shows | Produced by |
|---|---|---|
| [implementation-acceptance.md](implementation-acceptance.md) | The whole CPU workflow through the release binary: every step's exit code and time, 12 preset failures, exact resume after a crash, determinism, the scikit-learn reference check and the training-loop overhead | [`tools/acceptance.sh`](../tools/acceptance.sh) + [`tools/acceptance_report.py`](../tools/acceptance_report.py) |
| [benchmarks/](benchmarks/) | RustPy vs the equivalent pandas + scikit-learn workflow on the same rows, split, features and settings: quality, training time, prediction latency, cold start, memory, deployment size | [`tools/bench/bench.py`](../tools/bench/bench.py) |

Numbers depend on the machine. Re-run the tools on yours rather than comparing
absolute values across machines; a new report goes into a new dated directory,
and old reports are kept, favourable or not.

中文：README 中各项说法所依据的实测证据。每份报告都注明了来自哪个提交、哪台机器，以及如何复现；
旁边的机器可读文件就是证据本身（本地路径已替换为 `<work>` / `<repo>`）。数字取决于机器，请在自己的
机器上重新运行这些工具，而不要跨机器比较绝对值；新报告放在新的带日期目录中，旧报告无论结果好坏都会保留。
