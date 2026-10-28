# predict-app

A small Rust application that loads a Tabsmith bundle and predicts. It depends on
`tabsmith-runtime` only: `cargo tree` here shows everything an application pays
for (no Burn, no Polars, no Python). It is outside the workspace on purpose, with
its own `Cargo.lock`.

Build a bundle first, for example with the Quick start of the main
[README](../../README.md) (`tabsmith sample-data machines.csv`, then
`tabsmith train machines.csv --target defect --group device_id`), then:

| Binary | What it shows | Run |
|---|---|---|
| `predict-app` (`src/main.rs`) | JSON records on stdin → `label<TAB>probability` (regression: the value) | `cargo run --release -- <bundle> < records.jsonl` |
| `typed` | predicting from your own `#[derive(Serialize)]` struct | `cargo run --release --bin typed -- <bundle>` |
| `hot_swap` | `PredictorHandle::try_update`: a new bundle is loaded and self-checked before it replaces the active one, a broken one is refused, `rollback_to` goes back | `cargo run --release --bin hot_swap -- <old> <new> [<broken>]` |
| `bench` | load time, single-record latency and batch throughput as JSON (used by `tools/bench/bench.py`) | `cargo run --release --bin bench -- <bundle> <records.jsonl> [iterations]` |

`<bundle>/examples/input.json` holds complete example records; one per line
(`{"records": [...]}` → JSON lines) is the input format of `predict-app`.

中文：一个只依赖 `tabsmith-runtime` 的小型 Rust 应用，用来加载 Tabsmith 模型包并预测（没有 Burn、Polars 或
Python）。先按主 [README](../../README.zh-CN.md) 的快速上手生成模型包，再运行上表中的二进制：`predict-app` 从标准输入
读取 JSON 记录；`typed` 演示用自己的结构体预测；`hot_swap` 演示带自检的热切换与回退；`bench` 输出加载时间、单条延迟和批量吞吐。
