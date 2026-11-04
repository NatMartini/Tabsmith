# Benchmark 2026-11-01, Linux x86_64

[`results.md`](results.md) holds the table. The raw numbers are in
[`results.json`](results.json), and the logs of every step in [`logs/`](logs/).

Differences from [the run of 2026-10-31](../2026-10-31-linux-x86_64/):
- **The trees' training changed** ([D15](../../../docs/design-decisions.md#d15-the-trees-add-integers-on-workers-that-spin)).
  On this problem they train 62 rounds in 0.08 s, where they took 0.12 s for 54 rounds, and their
  test ROC AUC went from 0.8188 to 0.8220. scikit-learn's `HistGradientBoosting` gets the same 62
  rounds and 0.8172.
- **The machine was busier.** Other programs ran on it, so prediction and cold starts on both sides
  came out slower than on 2026-10-31, though that code did not change: a single record took 2.6 to
  3.7 µs against 2.5 to 3.0 µs, a Python cold start 738 to 797 ms against 550 to 573 ms. Read the
  ratios, not the absolute numbers.

The code is the working tree of the commit that adds this report. Setup and commands are those of
[the run of 2026-10-31](../2026-10-31-linux-x86_64/): an AMD Ryzen 9 9950X3D2 (32 logical CPUs) under
WSL 2, Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, in a virtual environment with only
what this stack needs. Training times are single runs. Latencies are medians over 20,000 predictions.

---

## 中文

表格在 [`results.md`](results.md)，原始数字在 [`results.json`](results.json)，每一步的日志在 [`logs/`](logs/)。与 2026-10-31 相比：树的训练改了（62 轮 0.08 秒，测试 ROC AUC 0.8220，scikit-learn 同样 62 轮是 0.8172）；这次机器上还有别的程序在运行，两边的预测和冷启动都比上次慢，这部分代码并没有变，所以请看比值，不要看绝对数字。
