# tabsmith-embedded

The arithmetic of [Tabsmith](https://github.com/NatMartini/Tabsmith) models in `no_std`
Rust:
- dot products and dense layers;
- tree-ensemble traversal;
- standardisation of numeric inputs;
- `exp`, the logistic function, softmax and argmax.

[`tabsmith-runtime`](https://crates.io/crates/tabsmith-runtime) and the models that
[`tabsmith-macros`](https://crates.io/crates/tabsmith-macros) compiles into Rust code
both call these kernels, and the trees `tabsmith-macros` turns into code repeat their
arithmetic. That is why a compiled model predicts exactly what the runtime predicts.

- **Same bits everywhere.** Every kernel fixes its order of operations. `exp` is not
  the platform C library's, whose last bits differ (on 2 million inputs, the Microsoft
  C runtime and glibc disagree for 0.07 % of `f32` exponentials): it is the algorithm of
  [`libm`](https://crates.io/crates/libm) 0.2.16 (FreeBSD's e_exp.c), written here
  operation for operation. It gives libm's bits for every input that is an `f32`, all
  2^32 checked, and for 3.2 billion random `f64`.
- **Every kernel is a `const fn`.** rustc evaluates IEEE 754 arithmetic in software with
  the hardware's rounding, so a model compiled with `tabsmith-macros` can predict while
  it compiles and get the same bits.
- **No allocation, no `std`, no `unsafe`, no dependencies.** It builds for
  microcontrollers (tested with `thumbv7em-none-eabihf`) and WebAssembly.
- **Bounds, at compile time.** `Trees::bounds` and `dense_bounds` are `const fn`s that
  bound a model's outputs over a box of inputs: the extreme reachable leaves of every
  tree (`Trees::leaf_range`, summed with no margin, since rounding is monotone), or
  interval arithmetic through a layer widened by the worst rounding error of the
  kernel. A compiled model bounds itself while it compiles. For trees the bounds are
  proved: the Kani proofs that `tabsmith-macros` generates check each tree's range for
  every box, and this crate's own (`cargo kani -p tabsmith-embedded`) check that adding
  a smaller leaf to a smaller sum, and rounding it to `f32`, never reverse the order.
- **XGBoost's arithmetic, for imported models.** `Trees::outputs_in_f32` adds the leaves in
  `f32`, in tree order, from the base score rounded to `f32`, as XGBoost does, and
  `bounds_in_f32` bounds it; a model imported by `tabsmith import` says so in its bundle and
  then gives XGBoost's margins bit for bit.
- **The checker of rule proofs.** `rules` holds, as `const fn`s, what a proof that a model
  keeps a rule over a region of inputs needs: `narrow`, which cuts a box of features at a
  split while keeping each category one-hot, `is_empty`, the `Walk` through a proof's splits,
  and `Claim`, which reads a box's output bounds as `decode` reads outputs. A model compiled
  with `model!(…, rule(…))` makes the compiler walk its proofs; Kani proves that a split keeps
  every input of a box in the part its own value selects.
- **Shared types.** `FieldValue` (a raw input value), `FieldInfo` (what a compiled model
  knows about a field), `InputError`, `SelfCheckError`, `Work` (the arithmetic of one
  prediction) and the `Model` trait, implemented by `tabsmith-runtime`'s `Predictor`
  and by every compiled model.

Applications normally use it through `tabsmith-macros`, which needs this crate as a
dependency of the crate that invokes `model!`.

Minimum supported Rust version: 1.88.

## 中文

Tabsmith 模型算术的 `no_std` 实现：点积与全连接层、树集成遍历、数值标准化、`exp`、logistic、softmax 与 argmax。
运行时和编译进代码的模型调用的是同一套内核，生成为代码的树重复同样的算术，因此预测结果逐位相同。`exp` 是 `libm` 0.2.16 的算法（FreeBSD 的 e_exp.c）在本 crate 里逐个运算的移植，
对所有 2^32 个 f32 输入和 32 亿个随机 f64 都与 libm 逐位相同。每个内核都是 `const fn`，编译器可以在编译时运行模型并得到相同的位；不分配内存、不依赖 `std`、没有 `unsafe`，也没有任何依赖。
`rules` 是规则证明的 `const fn` 检查器（切分盒子时保持类别独热、判断盒子是否为空、沿证明的切分走一遍、按 `decode` 的方式读取输出界），编译器用它复查 `model!(…, rule(…))` 的证明。
`Trees::outputs_in_f32` 按 XGBoost 的方式在 `f32` 中按树序相加叶子（起点是舍入到 `f32` 的起始值），`bounds_in_f32` 给出它的界；`tabsmith import` 导入的 XGBoost 模型在模型包里声明这一点，因此逐位给出 XGBoost 的 margin。

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
