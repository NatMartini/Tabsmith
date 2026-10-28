# Fuzzing the bundle loader, 2026-10-27, Linux x86_64

The fuzz target [`bundle_from_memory`](../../../fuzz/fuzz_targets/bundle_from_memory.rs)
feeds arbitrary file sets to `Predictor::from_memory`. When loading succeeds, it also
runs the bundle's self-check (transform and model on the example records). Hash
verification is switched off for most inputs, so that mutated bytes reach the JSON,
safetensors, transform and tree-ensemble code instead of stopping at the SHA-256 check.
The property: loading may fail, but it never panics, hangs or aborts.

## Run

| | |
|---|---|
| Duration | 901 s (`-max_total_time=900`), one libFuzzer process |
| Executions | **5,339,316** (5,925/s) |
| Crashes, timeouts, out-of-memory | **none** (`fuzz/artifacts/` empty) |
| Coverage | 7,174 → 8,769 edges (`cov`), 13,469 → 18,281 features (`ft`) |
| Corpus | 1,392 → 2,366 inputs kept by libFuzzer (90 MB) |
| Peak RSS | 643 MB |
| Machine | AMD Ryzen 9 9950X3D2, WSL2 (Linux 6.6.87.2) |
| Toolchain | rustc 1.101.0-nightly (16d1ffac1 2026-10-26), cargo-fuzz 0.13.2 |
| Code | the working tree of the commit that adds this report |

Seeds were packed with [`fuzz/seed.py`](../../../fuzz/seed.py), in a verified and an
unverified variant each. They are real bundles that `tabsmith train` and
`tabsmith bundle build` built from the synthetic sample data. They cover every model
format and task:

| Seed | Task | Model |
|---|---|---|
| `binary-run-002` | binary | small-mlp (`tabsmith.dense.v1`) |
| `binary-run-003` | binary | gbdt (`tabsmith.trees.v1`) |
| `multiclass-run-001` | multiclass | linear, 4 outputs |
| `multiclass-run-003` | multiclass | gbdt, one tree per class and round |
| `regression-run-003` | regression | gbdt, standardised target |

The corpus also kept the inputs of an earlier 900-second run. That run used binary
network bundles only, made before tree models existed, and executed 7,091,390 inputs
without a crash.

The last lines of the log, including the recommended dictionary libFuzzer learned (it
found field names such as `voltage_drop` and `multiclass` on its own), are in
[`fuzz-log-tail.txt`](fuzz-log-tail.txt).

## Reproduce

```sh
cargo build --release -p tabsmith-cli
# train sample bundles (see examples/wasm-predict/build.py --tabsmith for all three tasks)
python3 fuzz/seed.py <bundle-dir> fuzz/corpus/bundle_from_memory/<name>
cd fuzz && cargo +nightly fuzz run bundle_from_memory -- -max_total_time=900 -print_final_stats=1
```

中文：对模型包加载器 `Predictor::from_memory` 做了 900 秒的模糊测试，共执行 5,339,316 次，没有崩溃、超时或内存溢出；
种子覆盖三种任务以及全连接网络和树集成两种模型格式。此前一次只用二分类网络模型包的 900 秒运行执行了 7,091,390 次，同样没有崩溃。
