# tabsmith-embedded

The arithmetic of [Tabsmith](https://github.com/__OWNER__/__REPO__) models in `no_std`
Rust:
- dot products and dense layers;
- tree-ensemble traversal;
- standardisation of numeric inputs;
- the logistic function, softmax and argmax.

[`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime) and the models that
[`tabsmith-macros`](https://crates.io/crates/tabsmith-macros) compiles into Rust code
both call these kernels. That is why a compiled model predicts exactly what the runtime
predicts.

- **Same bits everywhere.** Every kernel fixes its order of operations. `exp` comes
  from [`libm`](https://crates.io/crates/libm) (pure Rust), because the platform C
  libraries disagree in the last bit: on 2 million inputs, the Microsoft C runtime and
  glibc differ for 0.07 % of `f32` exponentials and 0.12 % of logarithms.
- **No allocation, no `std`, no `unsafe`.** It builds for microcontrollers (tested
  with `thumbv7em-none-eabihf`) and WebAssembly.
- **Shared types.** `FieldValue` (a raw input value), `FieldInfo` (what a compiled model
  knows about a field), `InputError` and `SelfCheckError`.

Applications normally use it through `tabsmith-macros`, which needs this crate as a
dependency of the crate that invokes `model!`.

Minimum supported Rust version: 1.88.

## 中文

Tabsmith 模型算术的 `no_std` 实现：点积与全连接层、树集成遍历、数值标准化、logistic、softmax 与 argmax。
运行时和编译进代码的模型调用的是同一套内核，因此预测结果逐位相同。`exp` 使用纯 Rust 的 `libm`，在所有平台上
结果一致；不分配内存、不依赖 `std`、没有 `unsafe`。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
