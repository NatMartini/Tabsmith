# Kani proofs: compiled models never panic

[中文](#中文)

**Claim.** For the two models that `examples/embedded-predict` compiles with `model!`
(`defects`, gradient-boosted trees, and `defects_mlp`, a 35-32-16-1 network), prediction
never panics, for every possible input: any number (NaN and infinities included), any
category, any missing value.

**How.** `predict` is `decode(&outputs(&features(input)?))`. Each stage is proved total
for every value of its argument's type, which is a superset of what the stage before it
can return. The harnesses are at the end of
[`examples/embedded-predict/src/lib.rs`](../../../examples/embedded-predict/src/lib.rs).

**Where.** The development machine (AMD Ryzen 9 9950X3D2), WSL2 Ubuntu 26.04 (Linux
6.6.87.2). Kani 0.68.0 with CBMC 6.11.0 and its pinned toolchain nightly-2026-08-21.
The working tree on top of commit `31c9f44`, not yet committed (the compiled-model work).

## Settings

`examples/embedded-predict/Cargo.toml` sets `[package.metadata.kani.flags]
no-overflow-checks = true`, so a plain `cargo kani` uses it. What stays on:
- Rust's own checks, which are panics: arithmetic overflow (the crate is built with
  overflow checks), division by zero, shift overflow, and indexing;
- Kani's memory-safety checks;
- the unwinding checks, which prove that the loop bounds are large enough.

What is off: CBMC's additional checks for conversions and for floating-point NaN and
overflow. They are not about panics, and one of them is a false positive here (below).

## Results

| Harness | Settings | Checks failed / total (unreachable) | Verification time | Result |
|---|---|---|---|---|
| `tree_features_are_total` | final | 0 / 415 (8) | 11.1 s | SUCCESSFUL |
| `tree_traversal_is_total` | default (stricter) | 0 failed | 412 s (whole run) | SUCCESSFUL |
| `tree_decode_is_total` | final | 0 / 93 (17) | 0.4 s | SUCCESSFUL |
| `network_features_are_total` | default (stricter) | 0 / 421 (8) | 20.0 s | SUCCESSFUL |
| `network_layers_are_total` | final | 0 / 187 (1) | 26.1 s | SUCCESSFUL |
| `network_decode_is_total` | final | 0 / 93 (17) | 0.4 s | SUCCESSFUL |

The final settings only remove checks from the default ones. The two proofs that
passed with the default settings therefore also hold with the final ones. Their reruns
with the final settings were cut off: something outside this run shut the WSL virtual
machine down four times that afternoon.

## Findings on the way

**Proving the whole network did not finish.** An earlier single harness for the
network (`predict` on any input) was still solving after about 20 minutes. With the
default settings, `network_layers_are_total` was still running after 32 minutes when it
was cut off. With the default settings, CBMC's floating-point checks need the value of
every multiply-add in the layers. Without them, no check depends on the arithmetic,
and CBMC slices it away: 26 seconds.

**A false positive in CBMC's NaN check.** With the default settings,
`tree_decode_is_total` failed 1 of 113 checks: "NaN on division" in `libm`
0.2.16, `src/math/exp.rs` line 121. Concrete playback
(`cargo kani -Z concrete-playback --concrete-playback=print`) gives the input: a logit of
−∞. `sigmoid(−∞)` calls `exp(−∞)`. Its underflow branch evaluates
`force_eval!((-x1p_149 / x) as f32)` (−2⁻¹⁴⁹ / −∞) only to raise the underflow flag.
IEEE 754 defines a finite number divided by an infinity as 0, and `exp(−∞)` returns
0, so `sigmoid(−∞)` is 0. CBMC's NaN check treats any division by an infinity as a
possible NaN; only ∞/∞ and 0/0 are. Nothing panics on this path, and no NaN is produced.

## Reproduce

```sh
cargo install --locked kani-verifier   # Kani runs on Linux and macOS
cd examples/embedded-predict
cargo kani                              # all six; or --harness proofs::<name>
```

`bash tools/verify.sh --all` runs the same proofs together with every other check.

---

## 中文

**结论。** `examples/embedded-predict` 用 `model!` 编译了两个模型（梯度提升树 `defects` 和 35-32-16-1 的神经网络
`defects_mlp`）。对任何可能的输入（任何数值，包括 NaN 和无穷大；任何类别；任何缺失值），它们的预测都不会 panic。

**方法。** `predict` 就是 `decode(&outputs(&features(input)?))`。每一段都对其参数类型的所有取值证明是全函数，
而这个取值范围包含了前一段所有可能的输出。6 个证明全部通过：4 个用最终配置，2 个用更严格的默认配置（最终配置只是去掉了一些检查，
所以结论同样成立）。用最终配置重跑这 2 个时，WSL 虚拟机被外部关闭打断了。

**配置。** Rust 自己的检查全部保留：整数溢出、除以零、移位和数组越界都会 panic，Kani 也检查内存安全和循环展开。关掉的是 CBMC 额外的
浮点 NaN、溢出和类型转换检查：它们和 panic 无关，其中 NaN 检查在这里还是误报。CBMC 把“任何数除以无穷大”都当成可能产生 NaN，
因此在 `libm` 的 `exp(−∞)` 里报错；但按 IEEE 754，有限数除以无穷大等于 0，`exp(−∞)` 返回 0，这条路径既不会 panic，也不会产生 NaN。

**发现。** 把整个网络作为一个证明时，30 分钟内都没有解完；分段之后，没有任何检查依赖网络的浮点运算，CBMC 把它们切掉，网络层只用 26 秒。
