# wasm-predict

`tabsmith-runtime` compiled to WebAssembly, and a page that uses it: load a bundle,
verify it, and predict, all inside the browser tab. It is the same code that serves
predictions in Rust applications and in `tabsmith serve`, not a JavaScript port: the page
hands the bundle's files to `Predictor::from_memory`, which checks the SHA-256 of every
file against the manifest, refuses formats and transform versions it does not implement,
replays the bundle's example predictions (self-check), and then predicts.

The runtime needed no change to build for `wasm32-unknown-unknown`. Its dependencies are
`tabsmith-contracts` (serde, TOML, and SHA-256 from the RustCrypto `sha2` crate),
`tabsmith-embedded` (the numeric kernels, with `exp` from `libm`), serde_json and
safetensors, all pure Rust, and it uses no threads, native libraries or
unsafe code of its own. The only difference on the web is where the files come from:
memory (`from_memory`) instead of a directory. The module is 0.9 MB, or 0.2 MB gzipped
(`opt-level = "s"`, LTO).

## Build and run

You need the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` at the version of the
`wasm-bindgen` crate in `Cargo.toml` (the two must match):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo build --release -p tabsmith-cli          # from the repository root
```

Then, from this directory:

```sh
python build.py --tabsmith ../../target/release/tabsmith
python -m http.server -d www 8000
```

Open <http://localhost:8000>. `build.py` builds the module and its JavaScript bindings into
`www/pkg/`. It also trains one sample bundle per task with the CLI: defects (binary),
failure mode (multiclass) and remaining hours (regression). These are packed into
`www/bundles/`. Pass `--bundle NAME=DIR` to pack your own bundles as well. The page can
also open any bundle folder from disk with **Open a bundle folder…**. The files are read
by the page and are never uploaded.

## What the page shows

1. **Load**: the time to verify the bundle, and what was verified.
2. **What the bundle promises**: the task and labels, the model, the preprocessing (fitted
   on training rows only), and the held-out test metrics recorded in the bundle.
3. **Predict**: a form built from `input.schema.json` and the fitted transform. Categorical
   fields offer the categories seen in training, and numeric fields show the training mean
   and standard deviation. The answer depends on the task:
   - binary: the probability of the positive label against the threshold;
   - multiclass: one probability per class;
   - regression: the value, mapped back from the standardised target.

   The request can also be edited as JSON. Remove a field or send a string where a number
   belongs, and the runtime's structured error appears with its code and next action,
   exactly as an application would get it.

The latency shown is the mean of many calls, because browsers coarsen their clock. It
includes the JSON round trip between JavaScript and WebAssembly.

## How the bundle crosses into WebAssembly

The page packs the bundle's files into one byte array: a flags byte (ignored here, since
hashes are always verified), then `[u16 LE name length][name][u32 LE data length][data]`
per file. `fuzz/seed.py` writes the same layout for the fuzzer, and `build.py` and the
folder picker in `www/index.html` both produce it. `src/lib.rs` unpacks it into a
`BTreeMap<String, Vec<u8>>` and calls `Predictor::from_memory`. The exported API is three
methods on one class:

| JavaScript | Rust | Returns |
|---|---|---|
| `new Bundle(bytes)` | `Bundle::new(&[u8])` | a verified bundle, or throws the runtime's error (`[CODE] message (next: …)`) |
| `bundle.describe()` | `Bundle::describe` | manifest, input/output schema, transform, model config, evaluation and examples, as JSON |
| `bundle.predict(json)` | `Bundle::predict(&str)` | predictions as JSON for one record or an array of records |

`unsafe_code` is not forbidden in this crate, unlike the workspace, because
`#[wasm_bindgen]` generates the unsafe ABI glue that JavaScript calls. The code in `src/`
is safe Rust.

中文：把 `tabsmith-runtime` 原样编译成 WebAssembly，在浏览器标签页里加载模型包、校验每个文件的 SHA-256、
执行自检并预测，不上传任何数据。`python build.py --tabsmith ../../target/release/tabsmith` 构建模块并为二分类、
多分类和回归各训练一个示例包；`python -m http.server -d www 8000` 后打开页面即可。页面也可以直接打开本地的
任意模型包目录。
