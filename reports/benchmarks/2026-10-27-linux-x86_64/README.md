# Benchmark 2026-10-27, Linux x86_64

> Made before the rename from RustPy to Tabsmith: read `rustpy` as `tabsmith`
> (today `--tabsmith target/release/tabsmith`, `-p tabsmith-cli`; [note](../../README.md)).
> / 本报告产生于 RustPy 改名为 Tabsmith 之前，文中的 `rustpy` 即今天的 `tabsmith`。

[`results.md`](results.md) (and the raw numbers in [`results.json`](results.json),
the logs of every step in [`logs/`](logs/)) from commit `ae39055`, on an AMD Ryzen 9
9950X3D2 (32 logical CPUs) under WSL2 (Linux 6.6), with Python 3.14.4,
scikit-learn 1.9.1, pandas 3.1.0 and numpy 2.5.4:

```sh
cargo build --release --locked -p rustpy-cli
cargo build --release --locked --manifest-path examples/predict-app/Cargo.toml --all-targets
python3 -I tools/bench/bench.py --rustpy target/release/rustpy \
    --app examples/predict-app/target/release/bench \
    --python <venv>/bin/python --out reports/benchmarks/<date>-<platform> --rows 10000
```

The machine was otherwise idle. Training times are single runs; latencies are
medians over 20,000 predictions, cold starts the median of 7 fresh processes.
