# Benchmark 2026-10-27 (second run), Linux x86_64

[`results.md`](results.md) holds the table. The raw numbers are in
[`results.json`](results.json), and the logs of every step in [`logs/`](logs/).

Differences from the [first run](../2026-10-27-linux-x86_64/) of the same day:
- **Three candidates.** Tabsmith's `gbdt` is compared with scikit-learn's
  `HistGradientBoostingClassifier` (HGB): the same tree limits, as many boosting rounds
  as Tabsmith trained, and the same one-hot, standardised features.
- **The newer CPU engine.** Burn's Flex backend has replaced NdArray, and validation
  splits rows across cores. Single-thread training of the networks is otherwise
  unchanged.

The code is the working tree of the commit that adds this report.

Setup:
- **Machine:** an AMD Ryzen 9 9950X3D2 (32 logical CPUs) under WSL2 (Linux 6.6).
- **Python side:** Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, in a
  virtual environment whose pip came from its official wheel.
- **Commands:**

```sh
cargo build --release --locked -p tabsmith-cli
cargo build --release --locked --manifest-path examples/predict-app/Cargo.toml --all-targets
python3 -I tools/bench/bench.py --tabsmith target/release/tabsmith \
    --app examples/predict-app/target/release/bench \
    --python <venv>/bin/python --out reports/benchmarks/<date>-<platform> --rows 10000
```

The machine was otherwise idle. Training times are single runs. Latencies are medians
over 20,000 predictions, and cold starts are the median of 7 fresh processes. Tabsmith
trains on one thread. scikit-learn runs with its defaults, so HGB uses every core. On
this 7,356-row training set that made HGB slower than Tabsmith's single-threaded trees;
on large data the parallel implementation should win.

中文：同一天的第二次基准测试。与第一次相比，加入了 Tabsmith 的梯度提升树与 scikit-learn
`HistGradientBoostingClassifier` 的对比（相同的树限制、相同的提升轮数、相同的特征），CPU 引擎换成了 Flex，验证集推理按行并行
（结果不变）。在这个小数据集上，单线程的 Tabsmith 梯度提升树训练比使用全部核心的 HGB 更快；数据量大时，预计并行实现会更快。
