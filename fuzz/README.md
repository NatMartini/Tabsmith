# Fuzzing the bundle loader

A bundle is untrusted input: an application loads whatever directory it is given,
possibly downloaded. `bundle_from_memory` feeds arbitrary bytes to
`Predictor::from_memory` — first as a set of files, then, if loading succeeds,
through the bundle's self-check (transform and model on the example records).
Loading may fail; it must never panic, hang, read outside the bundle, or abort.

Most inputs switch hash verification off, so mutated bytes reach the JSON,
safetensors and transform code instead of stopping at the SHA-256 check.

## Run it (Linux or macOS, nightly)

```sh
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz
# Seed the corpus with a real bundle:
tabsmith sample-data machines.csv && tabsmith train machines.csv --target defect --group device_id
python3 fuzz/seed.py bundles/run-001 fuzz/corpus/bundle_from_memory/seed
cd fuzz && cargo +nightly fuzz run bundle_from_memory -- -max_total_time=900
```

A crash is saved under `fuzz/artifacts/bundle_from_memory/`; replay it with
`cargo +nightly fuzz run bundle_from_memory <file>`.

Seed with both model formats and every task to reach all the parsers: a dense and a
tree bundle (`tabsmith bundle build --run <gbdt-run> --allow-unevaluated`) of binary,
multiclass and regression projects (`tabsmith sample-data --task ...`).
Results of past runs are in [reports/fuzz](../reports/fuzz/).

The same property also runs on every `cargo test`, with structured damage (bit
flips, truncation, missing and extra files, unsafe paths, edited manifest fields):
`damaged_bundles_are_refused_without_panicking` in
[`crates/tabsmith-workflow/tests/properties.rs`](../crates/tabsmith-workflow/tests/properties.rs).
