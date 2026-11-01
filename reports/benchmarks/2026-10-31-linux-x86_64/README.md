# Benchmark 2026-10-31, Linux x86_64

[`results.md`](results.md) holds the table. The raw numbers are in
[`results.json`](results.json), and the logs of every step in [`logs/`](logs/).

Differences from [the run of 2026-10-27](../2026-10-27-linux-x86_64-v2/):
- **The trees were rewritten for speed.** They train on up to 12 threads, with a result
  that does not depend on the count (`reports/reproducibility/`). Their training time on
  this problem went from 0.67 s to 0.12 s, with the same test metrics.
- **`run.json` is rewritten at most once a second**, which speeds up every candidate on a
  problem this small. The whole `tabsmith run` went from 0.98 s to 0.51 s.
- **Batch prediction runs on several threads** from 4,096 rows on, and turning records into a
  table got faster. From records it went from 0.54–0.84 million to 3.5–4.6 million rows per
  second, and 0.7–1.3 million on one thread. This benchmark now predicts all 10,000 records
  in a batch, not 2,000, so the batch reaches the threads as a real one would.
- **The batch is measured from the same two starting points on both sides:** the records an
  application receives, and a table already in memory, a DataFrame for scikit-learn. The old
  table compared Tabsmith from records with scikit-learn from a DataFrame.
- **The example application grew** from 1.7 MB to 1.9 MB, with conformal prediction, the
  importance file and threads in the runtime.

The code is the working tree of the commit that adds this report.

Setup:
- **Machine:** an AMD Ryzen 9 9950X3D2 (32 logical CPUs) under WSL2 (Linux 6.6).
- **Python side:** Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, in a
  virtual environment with only what this stack needs, as in the earlier run. pyarrow,
  which was added to that environment later for another check, was left out: it adds
  172 MB to the environment and 40 MB to every process that imports pandas.
- **Commands:**

```sh
cargo build --release --locked -p tabsmith-cli
cargo build --release --locked --manifest-path examples/predict-app/Cargo.toml --all-targets
python3 -I tools/bench/bench.py --tabsmith target/release/tabsmith \
    --app examples/predict-app/target/release/bench \
    --python <venv>/bin/python --out reports/benchmarks/<date>-<platform> --rows 10000
```

Training times are single runs. Latencies are medians over 20,000 predictions, and cold
starts are the median of 7 fresh processes. The machine was also running other programs,
so small differences between runs mean nothing. Tabsmith's networks train on one thread
and its trees on up to 12. scikit-learn runs with its defaults, so HGB uses 16 threads.
On this 7,356-row training set most of HGB's time goes to the class weights both sides
use: a separate fit took 3.3 s with them and 0.13 s without (`hgb-small.py`,
`hgb-small.txt`), so the training-time row says little about the trees themselves.

Batch numbers are the median of 9 runs of the 10,000 records. Tabsmith uses up to 16 threads
for them and scikit-learn its defaults; the one-thread row shows Tabsmith without its threads.

中文：相比 2026-10-27 的基准，树的训练为提速重写（最多 12 个线程，结果与线程数无关），这个问题上从 0.67 秒降到 0.12 秒，测试指标不变；
`run.json` 最多每秒重写一次，所以 `tabsmith run` 整体从 0.98 秒降到 0.51 秒；批量预测在 4,096 行以上用多个线程，从记录出发从每秒 54–84 万行提高到 350–460 万行（单线程 70–130 万行），
本基准现在用全部 10,000 条记录做一批；两边都从同样的两个起点测量（应用收到的记录、内存中的表）；示例应用因为加入了 conformal 预测、线程等从 1.7 MB 增长到 1.9 MB。
Python 一侧与之前一样只装这个技术栈需要的包，没有包含后来为别的检查装的 pyarrow（它会让环境多 172 MB，让每个导入 pandas 的进程多 40 MB）。
HGB 在这个问题上的大部分时间花在两边都用的类别权重上（单独测试：有权重 3.3 秒，没有权重 0.13 秒），所以训练时间这一行并不能说明树本身的快慢。
批量数字是 10,000 条记录跑 9 次的中位数。Tabsmith 最多用 16 个线程，scikit-learn 用默认设置；单线程那一行是 Tabsmith 不用多线程的结果。
