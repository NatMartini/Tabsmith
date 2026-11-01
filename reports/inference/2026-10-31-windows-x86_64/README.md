# Inference: trees as code, and batches on several threads

[中文](#中文)

**Claim.**
- `model!` now turns each tree into a function of nested `if`s. On real models, one
  thread computes the outputs 2.0 to 2.8 times faster than walking the same trees in
  arrays, and the program gets smaller: a third less flash on a Cortex-M4F. Every result is
  the same to the bit.
- `outputs` becomes a `const fn`, so the compiler can run the model. And Kani can now prove
  all the trees at once, in 4.8 s and 104 MB, which does not fit in 14 GB with arrays:
  [the Kani report](../../kani/2026-10-31-wsl-linux-x86_64/).
- From 32,768 nodes on, a batch runs tree by tree over blocks of rows. That makes the
  runtime 1.6 to 2.8 times faster on large ensembles, with the same bits.
- `Predictor::predict` runs a batch of 4,096 rows or more on up to 16 threads. On 100,000
  records that is 8 and 7 times the one-thread rate, with the same predictions in the same
  order, and the same error when a record is wrong.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 threads), 64 GB, Windows 11 Pro 10.0.26300,
rustc 1.97.1, default release profile, one thread unless a table says otherwise. The machine
was also running other programs, so differences of a few percent mean nothing. The rows are
10,000 real rows of each dataset, so the branches are as unpredictable as real data makes
them.

## Trees: arrays or code

Time per row of `outputs`, in microseconds. Every line of the probe is checked bit for bit
against the runtime (`trees-probe.rs`, `trees.txt`, `trees.json`).

| Model | Trees | Nodes | Arrays, row by row | Code, row by row | Faster | `outputs_batch` | Runtime, one thread |
|---|---|---|---|---|---|---|---|
| Fixture binary-gbdt | 14 | 854 | 0.169 | 0.060 | 2.8× | 0.063 | 0.172 |
| Adult | 155 | 9,151 | 1.98 | 0.786 | 2.5× | 0.795 | 1.97 |
| Covertype, 40 rounds | 280 | 16,848 | 2.63 | 1.17 | 2.2× | 1.11 | 2.17 |
| California Housing | 309 | 18,823 | 4.04 | 1.54 | 2.6× | 1.54 | 3.96 |
| Covertype, 143 rounds | 1,001 | 59,781 | 15.84 | 7.22 | 2.2× | 5.85 | 9.61 |
| Covertype, 400 rounds | 2,800 | 169,248 | 75.25 | 37.98 | 2.0× | 14.44 | 27.26 |

`model!` generates code by default up to 32,768 nodes, so the last two models need the
`tree_code` option. A branch of generated code knows its own constant and its own next
instruction. A walk over arrays loads a node, its feature, its threshold and its child,
from memory that holds every other tree too. A generated tree is

```rust
const fn tree_0(x: &[f32; N_FEATURES]) -> f32 {
    if x[24] <= 0.2705937f32 {
        if x[23] <= 1.4444073f32 { /* ... */ } else { /* ... */ }
    } else {
        if x[21] <= -0.09460988f32 { /* ... */ } else { /* ... */ }
    }
}
```

## Row by row or tree by tree

Time per row in microseconds, with blocks of 256 rows. Each output still adds its trees in
order, so the order of the rows changes no bit.

| Nodes | Arrays, row by row | Arrays, tree by tree | Code, row by row | Code, tree by tree |
|---|---|---|---|---|
| 854 | 0.169 | 0.311 | 0.060 | 0.149 |
| 9,151 | 1.98 | 2.30 | 0.786 | 0.909 |
| 16,848 | 2.63 | 2.55 | 1.17 | 1.37 |
| 18,823 | 4.04 | 3.51 | 1.54 | 1.38 |
| 59,781 | 15.84 | 10.20 | 7.22 | 5.46 |
| 169,248 | 75.25 | 27.16 | 37.98 | 14.33 |

Below about 20,000 nodes it does not help, and on the smallest model it takes twice as long;
one pass through all the trees most likely fits in the cache there. At 60,000 nodes it starts
to pay, and at 170,000 it is worth 2 to 3 times. So `tabsmith-runtime` and the generated
`outputs_batch` switch at 32,768 nodes. `block-sizes.txt` holds the sweep behind the block
of 256: from 64 rows on, the arrays gain nothing more, and generated code gains about 10 %
more at 1,024 rows, for four times the stack.

This machine has an unusually large L3 cache, so the switch may pay earlier on a smaller
one. There are no numbers from other machines here.

## What generated code costs at compile time

One `model!` invocation in an otherwise empty program, built from scratch except for its
dependencies (`compile-times.py`, `compile-times.json`, `compile-times-large.json`).

| Nodes | Arrays, debug | Code, debug | Arrays, release | Code, release | Program, arrays | Program, code |
|---|---|---|---|---|---|---|
| 854 | 0.32 s | 0.35 s | 0.36 s | 0.44 s | 0.17 MB | 0.17 MB |
| 9,151 | 0.70 s | 1.00 s | 0.87 s | 1.79 s | 0.35 MB | 0.30 MB |
| 16,848 | 1.09 s | 1.71 s | 1.18 s | 3.65 s | 0.50 MB | 0.44 MB |
| 18,823 | 1.17 s | 1.83 s | 1.17 s | 3.44 s | 0.53 MB | 0.45 MB |
| 32,450 | 2.01 s | 3.16 s | 2.11 s | 7.63 s | 0.82 MB | 0.69 MB |
| 59,781 | 4.76 s | 7.54 s | 4.87 s | 18.73 s | 1.37 MB | 1.11 MB |
| 169,248 | 27.53 s | 36.22 s | 27.47 s | 114.73 s | 3.56 MB | 2.89 MB |

The program sizes are for the release build on x86-64. The default limit of 32,768 nodes keeps
a release build under 8 s here. Past it the option `tree_code` gives more speed for a longer
build, and `tree_arrays` keeps the arrays at any size.

The numbers in generated trees are decimal literals, not `f32::from_bits` as in the arrays.
That reads as the tree it is, and it is exact: Rust prints the shortest decimal that reads
back as the same bits, and rustc reads it back correctly rounded. It also builds a little
faster and keeps a debug build smaller.

| Nodes | Literals, debug | `from_bits`, debug | Literals, release | `from_bits`, release | Debug program, literals | Debug program, `from_bits` |
|---|---|---|---|---|---|---|
| 854 | 0.35 s | 0.36 s | 0.44 s | 0.45 s | 0.20 MB | 0.21 MB |
| 9,151 | 1.00 s | 1.15 s | 1.79 s | 1.87 s | 0.39 MB | 0.50 MB |
| 16,848 | 1.71 s | 1.89 s | 3.65 s | 3.79 s | 0.60 MB | 0.80 MB |
| 18,823 | 1.83 s | 2.08 s | 3.44 s | 3.64 s | 0.60 MB | 0.83 MB |

## Flash on a Cortex-M4F

`thumbv7em-none-eabihf`, `opt-level = "s"`, LTO, one exported function that calls `outputs`
(`flash-size.py`, `flash-size.json`). The sizes are the text and read-only data of that
object: the code a tree needs and its constants. Software `f64` addition from
`compiler_builtins` is left out, and both forms need it.

| Bundle | Trees | Nodes | Arrays (bytes) | Code (bytes) | Less |
|---|---|---|---|---|---|
| binary-gbdt | 14 | 854 | 17,398 | 11,308 | 35 % |
| regression-gbdt | 62 | 3,610 | 72,710 | 45,838 | 37 % |
| multiclass-gbdt | 40 | 2,440 | 49,314 | 31,844 | 35 % |

The arrays take 20 bytes for every node, leaves included. Generated code takes about 13 per
node.

## Why the bits are the same

Generated code is a second implementation of the walk, so the tests go further than for the
kernels (`crates/tabsmith-macros/tests/tree_code.rs`):
- For every node of every tree of the three tree fixtures, a row built to reach it: at the
  split's exact threshold and one representable number on either side. Each tree is compared
  on its own, so that the sum cannot round a difference away.
- 3,000 rows of any `f32`s per bundle: NaN, infinities, subnormals.
- `outputs_batch` and the tree-by-tree path at block edges, and with slices of different
  lengths.
- The compiler's own evaluation of `outputs` against the running program.

To check the tests themselves, `wrong-numbers.py` changes the generator on purpose: a
threshold one representable number up or down, a leaf one up, `<` for `<=`, children swapped,
trees added to the wrong output. All six fail the tests (`wrong-numbers.txt`).
The hash test of 20,000 predictions per bundle, recorded when the kernels were the only path,
still passes.

## Batches on several threads

Rows per second by thread count, on 100,000 records (`batch-probe.rs`, `batch-threads.txt`,
which also has 1,024, 4,096, 10,000 and 1,000,000 rows).

| Threads | binary-gbdt, from records | Against one thread | binary-gbdt, from a table | binary-mlp, from records | Against one thread | binary-mlp, from a table |
|---|---|---|---|---|---|---|
| 1 | 492,152 | 1.0× | 1,704,176 | 455,070 | 1.0× | 1,421,844 |
| 2 | 937,661 | 1.9× | 3,224,933 | 626,774 | 1.4× | 1,686,119 |
| 4 | 1,748,426 | 3.6× | 5,736,313 | 1,120,348 | 2.5× | 3,174,593 |
| 8 | 3,050,092 | 6.2× | 8,822,543 | 2,108,468 | 4.6× | 5,633,263 |
| 16 | 3,875,113 | 7.9× | 12,656,305 | 3,060,135 | 6.7× | 9,089,670 |
| 32 | 5,278,131 | 10.7× | 14,120,704 | 4,430,444 | 9.7× | 10,461,236 |

The default is the logical CPUs, up to 16: it reaches most of the gain and leaves the rest of
a shared machine alone. 24 threads give a quarter to a half more here, and
`LoadOptions::with_threads` asks for them. A batch below 4,096 rows
stays on the calling thread, so a service answering small requests starts no threads.
The runtime uses plain scoped threads and gains no dependency, and WebAssembly stays on one
thread.

A batch from records also got faster on one thread. Turning records into a table now merges
each record's sorted fields with the schema's, which took it from 1.0 to about 0.6 µs per row,
and `predict` from about 610,000 to 870,000 rows per second on the fixture. The first version
of this sorted the field names at every call, which cost a single prediction 0.4 µs (11 %);
the README benchmark showed it. The order is now computed once when a bundle loads, and
`predict_one` is back at or below where it was, against the last commit in turns
(`single-record-probe.rs`, `single-record.txt`).

If any piece of a batch fails, the whole batch runs again on the calling thread, so the error
is the one a single thread reports, with the row number of the whole batch
(`crates/tabsmith-runtime/tests/batch.rs`).

## Not measured

- Time on a Cortex-M4F. There is no board here; only the flash is measured.
- Another machine. The ratios depend on caches and branch predictors.
- macOS for this code, until CI runs it.

## Reproduce

```sh
# The bundles come from `tabsmith train` on the datasets of reports/datasets/, the records
# from their CSVs; Covertype at 40, 143 and 400 rounds with --set train.gbdt.epochs=<n>.
cargo run --release -p tabsmith-macros --example compiled_vs_runtime   # compiled-vs-runtime.txt
cargo test -p tabsmith-macros --test tree_code
cargo test -p tabsmith-runtime --test batch
```

`trees-probe.rs` with `trees-probe.toml` is the first probe, and `batch-probe.rs` needs
`tabsmith-runtime` and `serde_json`. `compile-times.py`, `flash-size.py` and
`wrong-numbers.py` drive cargo and hold the settings they used.

---

## 中文

**结论。**
- `model!` 现在把每棵树生成为一个嵌套 `if` 的函数。在真实模型上，单线程计算输出比遍历同样的树数组快 2.0 到 2.8 倍，
  程序还更小：Cortex-M4F 上少占三分之一 flash。每个结果都逐位相同。
- `outputs` 变成 `const fn`，编译器可以运行模型。Kani 现在也能把所有树放进一个证明，这在数组形式下做不到，见
  [Kani 报告](../../kani/2026-10-31-wsl-linux-x86_64/)。
- 从 32,768 个节点起，批量预测按树处理一块一块的行。大模型上运行时因此快 1.6 到 2.8 倍，结果逐位相同。
- `Predictor::predict` 在 4,096 行以上的批量上最多用 16 个线程。10 万条记录上，gbdt 和 mlp 分别达到单线程的 8 倍和 7 倍，
  预测和顺序都相同，记录出错时报告的错误也相同。

**环境。** AMD Ryzen 9 9950X3D2（16 核 32 线程），64 GB，Windows 11 Pro，rustc 1.97.1，默认 release 配置；除非表格另有说明，都是单线程。
机器上同时运行着其他程序，几个百分点的差异没有意义。数据是各数据集的 10,000 行真实数据。

**树：数组还是代码。** 表格在上面的 "Trees: arrays or code"。默认在 32,768 个节点以内生成代码，超过时用 `tree_code` 选项。
生成代码的分支知道自己的常量和下一条指令；遍历数组要从保存着所有树的内存里读取节点、特征、阈值和子节点。

**按行还是按树。** 2 万节点以下没有好处，最小的模型上反而慢一倍，多半是因为遍历一次所有树本来就放得进缓存。6 万节点开始有收益，17 万节点时值 2 到 3 倍。
所以运行时和生成的 `outputs_batch` 都在 32,768 个节点处切换。这台机器的缓存很大，缓存小的机器上可能更早就有收益，这里没有别的机器的数据。

**编译代价。** 默认上限让 release 构建在这里保持在 8 秒以内。生成的树用十进制字面量而不是数组用的 `f32::from_bits`：读起来就是树本身，而且精确：
Rust 打印能读回相同比特的最短十进制，rustc 按正确舍入读回。它编译略快，debug 程序也更小。

**Cortex-M4F 的 flash。** 数组每个节点都占 20 字节，叶子也一样；生成代码每个节点约 13 字节。

**为什么比特相同。** 生成代码是遍历的第二份实现，所以测试比内核更严格：对三个树 fixture 的每棵树每个节点，都构造一行走到它，取分裂的精确阈值和两侧相邻的可表示数，
每棵树单独比较，这样求和不会把差异舍入掉。还有每个模型 3,000 行任意 `f32`（NaN、无穷、次正规数），以及批量路径在块边界和切片长度不同时的情况。
为了检验测试本身，`wrong-numbers.py` 故意改错生成器：阈值上下移动一个可表示数、叶子值移动一个、`<` 替代 `<=`、左右子树互换、树加到错误的输出。6 种改动全部被测试抓到。
内核是唯一路径时记录的 20,000 次预测哈希测试仍然通过。

**批量多线程。** 默认用逻辑 CPU 数，最多 16：拿到大部分收益，也不占满共享机器。这里 24 线程能再多四分之一到一半，用 `LoadOptions::with_threads` 指定。4,096 行以下留在调用线程上，处理小请求的服务不会启动线程。
运行时用普通的 scoped 线程，没有新增依赖，WebAssembly 仍然单线程。单线程本身也变快了：记录转表格改为把每条记录排好序的字段和 schema 合并，从每行 1.0 µs 降到约 0.6 µs。第一版在每次调用时都排序字段名，让单次预测慢了 0.4 µs（11%），README 基准暴露了这一点；现在只在加载模型包时算一次，`predict_one` 回到了原来的水平或略好。
某一块出错时，整个批量回到调用线程重跑，所以错误和单线程报告的完全相同，行号也是整个批量的行号。

**未测量。** Cortex-M4F 上的运行时间（这里没有开发板，只测了 flash）；其他机器（比值取决于缓存和分支预测器）；这段代码在 macOS 上的结果（要等 CI 运行）。
