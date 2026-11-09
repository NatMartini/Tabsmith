# Kani proofs: the compiled trees are the runtime's trees, and their bounds hold

[中文](#中文)

**Claim.** For the 14 trees that `examples/embedded-predict` compiles with `model!`
(`defects`, 854 nodes, 35 features), Kani proves two things about each tree, for every
input:

- **Its code is the runtime's walk.** The tree's generated code gives the same bits as the
  walk through its arrays, for every row of 35 features: every `f32` value, NaN and
  infinities included. The runtime walks the same arrays with the same function.
- **Its range holds.** For every box of feature values and every row in it, the leaf the
  tree reaches lies inside the range that `Trees::leaf_range` computes for the box, the
  range that `output_bounds` adds up.

Three lemmas about rounding, proved with Kani in tabsmith-embedded, carry the second claim
from the trees to their sum. So the compiled model predicts what the runtime predicts on
every input, not only on the bundle's examples, and `output_bounds` holds for every row of
its box. All of it runs in one plain `cargo kani`, together with the nine proofs that
prediction never panics, the same trees kept as arrays, and four proofs that must fail:
69 harnesses in 459 s, compilation included, none above 1,775 MB.

## The chain, link by link

| Link | From | To | Evidence |
|---|---|---|---|
| 1 | the runtime's `Predictor` | `TREES`, the arrays in the compiled module | `model!` reads the bundle through the runtime, and both walk it with `kernels::Trees::leaf`: the same function on the same arrays |
| 2 | the walk `TREES.leaf(t, x)` | the code of tree `t` | Kani, `defects_tree_<t>_is_its_walk`, 14 proofs: the same bits for every row |
| 3 | the trees | `outputs` | each output adds its trees in tree order in `f64` and rounds to `f32` at the end, as `kernels::Trees::outputs` does. By construction; `tests/tree_code.rs` checks it on a row for every node and on 3,000 random rows, on three systems |
| 4 | each tree's leaf | `Trees::leaf_range` of the box | Kani, `defects_tree_<t>_stays_in_its_range`, 14 proofs: every box, every row in it |
| 5 | the 14 ranges | `output_bounds` | `Trees::bounds` adds the ranges in the order of link 3. Kani, in tabsmith-embedded: a larger sum stays larger when the same leaf is added, a larger leaf gives a larger sum, and rounding to `f32` keeps the order |

Link 3 is the one not proved. A proof would put the `f64` additions of both sides into one
formula, and the lemmas below show what a single such comparison costs.

## Found by writing the claim down

`output_bounds` documented that a NaN bound lets the feature be NaN. The range proofs
assume exactly that about the box, so before they could pass, the analysis had to do it. In
0.4.0 it did not for a lower bound: a box with `lo[j]` NaN and `hi[j]` below a split's
threshold reached neither child of the split, while a NaN row goes right at every split. On
the 0.4.0 code (`nan-0.4.0.rs`, output in `nan-0.4.0.txt`), with one split at 10, Kani's counterexample is `lo = NaN`,
`hi = 8`, row NaN: the row reaches the leaf 1, outside the empty range `[inf, -inf]`. With
NaN admitted only where `hi` is NaN, the same code passes, so the analysis was sound for a
rule it did not document. `leaf_range` now also follows the right child when `lo[j]` is NaN.
Only boxes with a NaN lower bound and no NaN upper bound change; `OUTPUT_BOUNDS` and every
other box give the same bits as before. `mutant_nan_rule_of_0_4_0` keeps it fixed.

## One tree per proof, and the range on the code

Every per-tree proof takes its tree by a constant index. With a symbolic tree index, the
range proof of all 14 trees ran out of memory at 13.3 GB in a trial on 2026-11-01. The range
is also proved on the tree's code, not on the walk. In the code every array index is a
constant; in the walk each index depends on the comparisons before it, and Kani must carry
every path. On the walk, the range proof of tree 0 took 82 s and over 7 GB. On the code,
the 14 take 4.9 s each (median, at most 5.1 s) and at most 128 MB, and each walk
proof 6.8 s and at most 1,264 MB.

Trees kept as arrays (`tree_arrays`) therefore get their code too, generated inside
`proofs` only, so that link 2 joins the walk they run to the range proved on the code. A
tree deeper than 64 levels, which no code is generated for, gets neither proof.
`defects_arrays` in the example is that form: its 28 tree proofs are the same proofs on the
same trees, reached through the other branch of the macro.

## Proofs that must fail

`examples/embedded-predict/src/mutants.rs` holds four harnesses marked
`#[kani::should_panic]`, each with one deliberate mistake. Kani counts them as successful
only because it finds a counterexample, so a change that made the generated proofs vacuous
would turn them into failures.

| Harness | The mistake | The check that failed |
|---|---|---|
| `mutant_another_tree` | the code of tree 0 against the walk of tree 1 | `TREE_FUNCTIONS[0](&x).to_bits() == TREES.value[TREES.leaf(1, &x)].to_bits()` |
| `mutant_one_bit_off` | the root threshold of tree 0 one unit in the last place off in the arrays | `TREE_FUNCTIONS[0](&x).to_bits() == ONE_BIT_OFF.value[ONE_BIT_OFF.leaf(0, &x)].to_bits()` |
| `mutant_strict_comparison` | `lo < threshold` where the analysis has `<=` | `min <= leaf && leaf <= max` |
| `mutant_nan_rule_of_0_4_0` | the NaN rule of 0.4.0 | `min <= leaf && leaf <= max` |

## The rounding lemmas

`Trees::bounds` and `outputs` add the same way: an `f32` leaf widened to `f64` and added to
an `f64` sum. If every leaf lies in its range, each partial sum of the smallest leaves stays
below the row's partial sum (first lemma, then second), and so does the final rounding
(third). These are facts of IEEE 754 arithmetic, since every operation rounds its exact
result and rounding is monotone, checked here bit for bit on these operations:

| Lemma (`tabsmith-embedded`, `kernels::proofs`) | Statement | Checks failed / total | Verification time | Peak CBMC memory | Result |
|---|---|---|---|---|---|
| `a_larger_sum_stays_larger` | `s <= t` gives `s + a <= t + a` (`a` a finite `f32` leaf) | 0 / 3 | 66.3 s | 4,882 MB | SUCCESSFUL |
| `a_larger_leaf_gives_a_larger_sum` | `a <= b` gives `s + a <= s + b` (`s` not NaN) | 0 / 3 | 552.0 s | 5,134 MB | SUCCESSFUL |
| `rounding_to_f32_keeps_the_order` | `s <= t` gives `s as f32 <= t as f32` | 0 / 1 | 0.6 s | 72 MB | SUCCESSFUL |

Comparing two different sums at once (`s <= t` and `a <= b` give `s + a <= t + b`)
did not finish within 30 minutes in a trial; the two one-sided lemmas give it by transitivity. The same two-sided
lemma in `f32` alone took 263 s and 10 GB. The lemmas run with `cargo kani -p
tabsmith-embedded`, which `tools/verify.sh --all` does, not CI: they do not depend on the
crate's code and took 622 s together. `lemmas.txt` is their summary.

## The other fixtures

The same generated proofs, once and not in CI, on the repository's other tree bundles, in a
scratch crate that compiles them with `kani`: 210 harnesses in 3,757 s
(63 minutes), all successful (`fixtures.txt`). The machine was also running the Linux gate for part of
this run, so its times are upper bounds.

| Bundle | Harnesses | Walk proof time (min / median / max) | Range proof time (min / median / max) | Total verification time | Peak CBMC memory |
|---|---|---|---|---|---|
| `multiclass-gbdt`: 4 outputs, 40 trees, 2,440 nodes | 83 | 14.5 / 19.5 / 25.1 s | 4.6 / 4.8 / 4.9 s | 1,351 s | 3,628 MB |
| `regression-gbdt`: 62 trees, 3,610 nodes | 127 | 19.3 / 31.5 / 49.8 s | 3.1 / 5.2 / 8.0 s | 2,371 s | 5,699 MB |

A range proof costs about the same at every size: it reads the arrays at constant indices.
A walk proof grows with the whole ensemble, not with its tree, because the walk reads the
shared arrays at indices Kani cannot know:

| Nodes in the ensemble | Walk proof, median | Walk proof, largest memory |
|---|---|---|
| 854 (`binary-gbdt`) | 6.8 s | 1,264 MB |
| 2,440 (`multiclass-gbdt`) | 19.5 s | 3,628 MB |
| 3,610 (`regression-gbdt`) | 31.5 s | 5,699 MB |

A model of tens of thousands of nodes would take hours and more memory than a CI runner
has, so it is better proved a subset at a time (`cargo kani --harness`).

## What is not proved

- Link 3, the order of the additions in `outputs`: by construction and by test.
- `outputs_batch`, and the runtime's tree-by-tree path for large ensembles. They add the
  same leaves in the same order per output; tests check them.
- `PROBABILITY_BOUNDS`, which passes the bounds through the logistic function and its `exp`.
- A network's values. Its bounds are tested with random rows of boxes, not proved.
- Accuracy, which no proof here is about.

## Where

The development machine (AMD Ryzen 9 9950X3D2), WSL2 Ubuntu 26.04 (Linux 6.6.87.2). Kani
0.68.0 with CBMC 6.11.0, whose default SAT solver here is CaDiCaL 3.0.0. One plain `cargo
kani`, the way CI runs it, on a fresh copy of the working tree, with every process limited
to 14 GB of address space (`ulimit -v`), because a GitHub-hosted Linux runner has 16 GB.
Memory was sampled once a second, so a dash means the proof ended before the first sample.

## Results

| Group | Harnesses | Time per proof (min / median / max) | Total | Peak CBMC memory |
|---|---|---|---|---|
| Prediction never panics (stage by stage) | 9 | 0.4 / 10.6 / 26.7 s | 77 s | 1,686 MB |
| `defects`: the code of each tree is its walk | 14 | 6.4 / 6.8 / 8.2 s | 98 s | 1,264 MB |
| `defects`: each tree stays in its range | 14 | 4.7 / 4.9 / 5.1 s | 68 s | 128 MB |
| `defects_arrays`: the code is the walk (code inside the proofs only) | 14 | 6.1 / 6.8 / 8.9 s | 99 s | 1,231 MB |
| `defects_arrays`: each tree stays in its range | 14 | 4.6 / 4.9 / 5.3 s | 68 s | 198 MB |
| Must fail (`mutants.rs`) | 4 | 7.1 / 7.4 / 11.6 s | 33 s | 1,775 MB |

The slowest proof is `defects_mlp_layers_are_total`, 27 s.

### Tree by tree (`defects`)

| Tree | Nodes | Depth | Walk: checks failed / total (unreachable) | Time | Memory | Range: checks failed / total (unreachable) | Time | Memory |
|---|---|---|---|---|---|---|---|---|
| 0 | 61 | 8 | 0 / 370 | 6.8 s | 1,098 MB | 0 / 509 | 4.9 s | 124 MB |
| 1 | 61 | 8 | 0 / 370 | 7.0 s | 1,224 MB | 0 / 509 | 4.7 s | 125 MB |
| 2 | 61 | 8 | 0 / 370 | 6.6 s | 1,129 MB | 0 / 509 | 4.9 s | 124 MB |
| 3 | 61 | 7 | 0 / 370 | 6.5 s | 1,006 MB | 0 / 509 | 4.8 s | 128 MB |
| 4 | 61 | 8 | 0 / 370 | 6.4 s | 1,132 MB | 0 / 509 | 4.7 s | 121 MB |
| 5 | 61 | 8 | 0 / 370 | 6.5 s | 1,134 MB | 0 / 509 | 4.9 s | 124 MB |
| 6 | 61 | 8 | 0 / 370 | 6.8 s | 1,086 MB | 0 / 509 | 4.9 s | 123 MB |
| 7 | 61 | 7 | 0 / 370 | 6.4 s | 986 MB | 0 / 509 | 5.1 s | 125 MB |
| 8 | 61 | 8 | 0 / 370 | 8.1 s | 1,123 MB | 0 / 509 | 4.7 s | 122 MB |
| 9 | 61 | 8 | 0 / 370 | 6.6 s | 1,127 MB | 0 / 509 | 4.8 s | 126 MB |
| 10 | 61 | 8 | 0 / 370 | 7.3 s | 1,264 MB | 0 / 509 | 4.9 s | 126 MB |
| 11 | 61 | 8 | 0 / 370 | 8.2 s | 1,155 MB | 0 / 509 | 5.0 s | 127 MB |
| 12 | 61 | 8 | 0 / 370 | 6.8 s | 1,063 MB | 0 / 509 | 5.0 s | 125 MB |
| 13 | 61 | 8 | 0 / 370 | 7.7 s | 1,120 MB | 0 / 509 (2) | 4.9 s | 126 MB |

### Every harness

| Harness | Checks failed / total (unreachable) | Verification time | Peak CBMC memory | Result |
|---|---|---|---|---|
| `defects_features_are_total` | 0 / 415 (8) | 11.5 s | 567 MB | SUCCESSFUL |
| `defects_outputs_are_total` | 0 / 3,390 | 4.8 s | 103 MB | SUCCESSFUL |
| `defects_decode_is_total` | 0 / 93 (17) | 0.4 s | – | SUCCESSFUL |
| `defects_mlp_features_are_total` | 0 / 415 (8) | 10.6 s | 568 MB | SUCCESSFUL |
| `defects_mlp_layers_are_total` | 0 / 187 (1) | 26.7 s | 1,470 MB | SUCCESSFUL |
| `defects_mlp_decode_is_total` | 0 / 93 (17) | 0.4 s | 58 MB | SUCCESSFUL |
| `defects_arrays_features_are_total` | 0 / 415 (8) | 10.6 s | 547 MB | SUCCESSFUL |
| `defects_arrays_tree_walk_is_total` | 0 / 130 | 11.3 s | 1,686 MB | SUCCESSFUL |
| `defects_arrays_decode_is_total` | 0 / 93 (17) | 0.4 s | – | SUCCESSFUL |
| `defects_tree_0_is_its_walk` | 0 / 370 | 6.8 s | 1,098 MB | SUCCESSFUL |
| `defects_tree_1_is_its_walk` | 0 / 370 | 7.0 s | 1,224 MB | SUCCESSFUL |
| `defects_tree_2_is_its_walk` | 0 / 370 | 6.6 s | 1,129 MB | SUCCESSFUL |
| `defects_tree_3_is_its_walk` | 0 / 370 | 6.5 s | 1,006 MB | SUCCESSFUL |
| `defects_tree_4_is_its_walk` | 0 / 370 | 6.4 s | 1,132 MB | SUCCESSFUL |
| `defects_tree_5_is_its_walk` | 0 / 370 | 6.5 s | 1,134 MB | SUCCESSFUL |
| `defects_tree_6_is_its_walk` | 0 / 370 | 6.8 s | 1,086 MB | SUCCESSFUL |
| `defects_tree_7_is_its_walk` | 0 / 370 | 6.4 s | 986 MB | SUCCESSFUL |
| `defects_tree_8_is_its_walk` | 0 / 370 | 8.1 s | 1,123 MB | SUCCESSFUL |
| `defects_tree_9_is_its_walk` | 0 / 370 | 6.6 s | 1,127 MB | SUCCESSFUL |
| `defects_tree_10_is_its_walk` | 0 / 370 | 7.3 s | 1,264 MB | SUCCESSFUL |
| `defects_tree_11_is_its_walk` | 0 / 370 | 8.2 s | 1,155 MB | SUCCESSFUL |
| `defects_tree_12_is_its_walk` | 0 / 370 | 6.8 s | 1,063 MB | SUCCESSFUL |
| `defects_tree_13_is_its_walk` | 0 / 370 | 7.7 s | 1,120 MB | SUCCESSFUL |
| `defects_tree_0_stays_in_its_range` | 0 / 509 | 4.9 s | 124 MB | SUCCESSFUL |
| `defects_tree_1_stays_in_its_range` | 0 / 509 | 4.7 s | 125 MB | SUCCESSFUL |
| `defects_tree_2_stays_in_its_range` | 0 / 509 | 4.9 s | 124 MB | SUCCESSFUL |
| `defects_tree_3_stays_in_its_range` | 0 / 509 | 4.8 s | 128 MB | SUCCESSFUL |
| `defects_tree_4_stays_in_its_range` | 0 / 509 | 4.7 s | 121 MB | SUCCESSFUL |
| `defects_tree_5_stays_in_its_range` | 0 / 509 | 4.9 s | 124 MB | SUCCESSFUL |
| `defects_tree_6_stays_in_its_range` | 0 / 509 | 4.9 s | 123 MB | SUCCESSFUL |
| `defects_tree_7_stays_in_its_range` | 0 / 509 | 5.1 s | 125 MB | SUCCESSFUL |
| `defects_tree_8_stays_in_its_range` | 0 / 509 | 4.7 s | 122 MB | SUCCESSFUL |
| `defects_tree_9_stays_in_its_range` | 0 / 509 | 4.8 s | 126 MB | SUCCESSFUL |
| `defects_tree_10_stays_in_its_range` | 0 / 509 | 4.9 s | 126 MB | SUCCESSFUL |
| `defects_tree_11_stays_in_its_range` | 0 / 509 | 5.0 s | 127 MB | SUCCESSFUL |
| `defects_tree_12_stays_in_its_range` | 0 / 509 | 5.0 s | 125 MB | SUCCESSFUL |
| `defects_tree_13_stays_in_its_range` | 0 / 509 (2) | 4.9 s | 126 MB | SUCCESSFUL |
| `defects_arrays_tree_0_is_its_walk` | 0 / 370 | 6.9 s | 1,116 MB | SUCCESSFUL |
| `defects_arrays_tree_1_is_its_walk` | 0 / 370 | 7.5 s | 1,112 MB | SUCCESSFUL |
| `defects_arrays_tree_2_is_its_walk` | 0 / 370 | 6.1 s | 1,127 MB | SUCCESSFUL |
| `defects_arrays_tree_3_is_its_walk` | 0 / 370 | 6.8 s | 1,006 MB | SUCCESSFUL |
| `defects_arrays_tree_4_is_its_walk` | 0 / 370 | 6.8 s | 1,230 MB | SUCCESSFUL |
| `defects_arrays_tree_5_is_its_walk` | 0 / 370 | 6.9 s | 1,154 MB | SUCCESSFUL |
| `defects_arrays_tree_6_is_its_walk` | 0 / 370 | 6.7 s | 1,087 MB | SUCCESSFUL |
| `defects_arrays_tree_7_is_its_walk` | 0 / 370 | 6.2 s | 991 MB | SUCCESSFUL |
| `defects_arrays_tree_8_is_its_walk` | 0 / 370 | 8.0 s | 1,143 MB | SUCCESSFUL |
| `defects_arrays_tree_9_is_its_walk` | 0 / 370 | 6.4 s | 1,128 MB | SUCCESSFUL |
| `defects_arrays_tree_10_is_its_walk` | 0 / 370 | 7.1 s | 1,103 MB | SUCCESSFUL |
| `defects_arrays_tree_11_is_its_walk` | 0 / 370 | 8.9 s | 1,231 MB | SUCCESSFUL |
| `defects_arrays_tree_12_is_its_walk` | 0 / 370 | 6.7 s | 1,059 MB | SUCCESSFUL |
| `defects_arrays_tree_13_is_its_walk` | 0 / 370 | 7.7 s | 1,122 MB | SUCCESSFUL |
| `defects_arrays_tree_0_stays_in_its_range` | 0 / 509 | 4.8 s | 123 MB | SUCCESSFUL |
| `defects_arrays_tree_1_stays_in_its_range` | 0 / 509 | 4.9 s | 125 MB | SUCCESSFUL |
| `defects_arrays_tree_2_stays_in_its_range` | 0 / 509 | 4.6 s | 125 MB | SUCCESSFUL |
| `defects_arrays_tree_3_stays_in_its_range` | 0 / 509 | 5.0 s | 126 MB | SUCCESSFUL |
| `defects_arrays_tree_4_stays_in_its_range` | 0 / 509 | 4.9 s | 122 MB | SUCCESSFUL |
| `defects_arrays_tree_5_stays_in_its_range` | 0 / 509 | 4.8 s | 121 MB | SUCCESSFUL |
| `defects_arrays_tree_6_stays_in_its_range` | 0 / 509 | 4.6 s | 123 MB | SUCCESSFUL |
| `defects_arrays_tree_7_stays_in_its_range` | 0 / 509 | 4.8 s | 125 MB | SUCCESSFUL |
| `defects_arrays_tree_8_stays_in_its_range` | 0 / 509 | 5.3 s | 122 MB | SUCCESSFUL |
| `defects_arrays_tree_9_stays_in_its_range` | 0 / 509 | 4.9 s | 122 MB | SUCCESSFUL |
| `defects_arrays_tree_10_stays_in_its_range` | 0 / 509 | 5.0 s | 124 MB | SUCCESSFUL |
| `defects_arrays_tree_11_stays_in_its_range` | 0 / 509 | 4.8 s | 198 MB | SUCCESSFUL |
| `defects_arrays_tree_12_stays_in_its_range` | 0 / 509 | 4.9 s | 125 MB | SUCCESSFUL |
| `defects_arrays_tree_13_stays_in_its_range` | 0 / 509 (2) | 5.0 s | 123 MB | SUCCESSFUL |
| `mutant_another_tree` | 1 / 3,499 (390) | 11.6 s | 1,775 MB | failed as expected |
| `mutant_one_bit_off` | 1 / 3,499 (390) | 7.2 s | 1,523 MB | failed as expected |
| `mutant_strict_comparison` | 1 / 3,695 (393) | 7.5 s | 198 MB | failed as expected |
| `mutant_nan_rule_of_0_4_0` | 1 / 3,695 (394) | 7.1 s | 190 MB | failed as expected |

`summary.txt` is the summary of this run.

## Reproduce

```sh
cargo install --locked kani-verifier   # Kani runs on Linux and macOS
cd examples/embedded-predict
cargo kani                              # all 69; or --harness <name>
cd ../..
cargo kani -p tabsmith-embedded         # the three lemmas, about 10 minutes
```

`bash tools/verify.sh --all` runs both with every other check.

---

## 中文

**结论。** `examples/embedded-predict` 用 `model!` 编译了 14 棵树（`defects`，854 个节点，35 个特征）。Kani 对每棵树、对任何输入证明了两件事：

- **生成的代码就是运行时的遍历。** 对 35 个特征的任何一行（所有 `f32` 取值，包括 NaN 和无穷大），树的生成代码给出的位与遍历它的数组完全相同。运行时用同一个函数遍历同一组数组。
- **范围成立。** 对任何特征取值的区域和其中的任何一行，树到达的叶子都在 `Trees::leaf_range` 为该区域算出的范围内，`output_bounds` 就是把这些范围加起来。

tabsmith-embedded 里另有三条关于舍入的引理，也由 Kani 证明，把第二条从单棵树推到它们的和。所以编译后的模型对每一个输入都给出与运行时相同的预测，而不只是模型包里的示例；`output_bounds` 对区域里的每一行都成立。
这些证明和原有的九个“预测不会 panic”的证明、数组形式的同一批树、四个必须失败的证明一起，在一次普通的 `cargo kani` 里运行：69 个证明，连编译共 459 秒，每个都不超过 1,775 MB。

**证明链。** 第 1 环：运行时的 `Predictor` 和编译后模块里的 `TREES` 是同一组数组、同一个遍历函数（`model!` 通过运行时读取模型包）。第 2 环：每棵树的代码与遍历逐位相同（Kani，14 个证明）。
第 3 环：每个输出按树的顺序在 `f64` 里相加，最后舍入到 `f32`，与 `kernels::Trees::outputs` 相同；这一环靠构造保证，并由 `tests/tree_code.rs` 在三个系统上测试，没有证明。
第 4 环：每棵树的叶子落在区域的 `leaf_range` 内（Kani，14 个证明）。第 5 环：`Trees::bounds` 按第 3 环的顺序把范围加起来，三条引理证明较小的和加较小的叶子不会变大、舍入到 `f32` 保持大小顺序。

**写下结论时发现的问题。** 文档说“NaN 边界允许该特征为 NaN”。范围证明正是这样假设区域的，所以分析必须照做才能通过。0.4.0 对下界没有做到：`lo[j]` 为 NaN、`hi[j]` 低于某个分裂的阈值时，分析认为两个子节点都到不了，而 NaN 的行在每个分裂都走右边。
在 0.4.0 的代码上（检查程序是 `nan-0.4.0.rs`，输出在 `nan-0.4.0.txt`），对一个阈值为 10 的分裂，Kani 给出的反例是 `lo = NaN`、`hi = 8`、输入 NaN：这一行到达叶子 1，不在空范围 `[inf, -inf]` 里。现在 `lo[j]` 为 NaN 时 `leaf_range` 也会走右子节点。只有“下界是 NaN、上界不是 NaN”的区域结果会变；`OUTPUT_BOUNDS` 和其他所有区域的结果逐位不变。`mutant_nan_rule_of_0_4_0` 防止它回退。

**为什么每个证明只取一棵树，范围在代码上证。** 树的下标是符号值时，14 棵树的范围证明在 13.3 GB 处耗尽内存。在遍历上证明范围，第 0 棵树要 82 秒、超过 7 GB；在代码上，每个范围证明中位数 4.9 秒、最多 128 MB，因为代码里每个下标都是常量。
所以数组形式的树也会在 `proofs` 里生成代码（只用于证明），由第 2 环把它运行的遍历和在代码上证明的范围连起来。

**必须失败的证明。** `mutants.rs` 里有四个带 `#[kani::should_panic]` 的证明，各有一个故意的错误：拿第 1 棵树的遍历比第 0 棵树的代码；把第 0 棵树根节点的阈值改动一个最小单位；范围分析里把 `<=` 写成 `<`；0.4.0 的 NaN 规则。Kani 都找到了反例，所以它们算作成功；如果生成的证明哪天变得什么都不检查，它们就会失败。

**舍入引理。** 三条引理逐位检查 `Trees::bounds` 和 `outputs` 用到的运算，它们是 IEEE 754 算术的性质。单边的两条分别用了表中的时间；两个和同时比较的版本在 30 分钟内没有完成，它可以由单边的两条推出。引理与 crate 的代码无关，由 `tools/verify.sh --all` 运行，不在 CI 里。

**没有证明的部分。** 第 3 环（构造保证并测试）；`outputs_batch` 和运行时大集成的逐树路径（测试）；经过 `exp` 的 `PROBABILITY_BOUNDS`；神经网络的数值（只有随机测试）；准确率不属于任何证明。

**其他模型包。** 同样的生成证明在另外两个树模型包上运行了一次（不在 CI 里）：210 个证明，3,757 秒（63 分钟），全部通过。
范围证明的耗时与模型大小无关；遍历证明随整个集成的节点数增长，中位数从 854 个节点时的 6.8 秒，到 2,440 个时的 19.5 秒、3,610 个时的 31.5 秒，内存最多 5,699 MB。几万个节点的模型要跑几个小时，适合用 `cargo kani --harness` 分批证明。
