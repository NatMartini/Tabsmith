# Rules over regions of inputs, proved or refuted

[中文](#中文)

**Claim.** `tabsmith audit` decides rules of the form "every input of this region gets a
prediction that meets this claim" on real models. On Adult's 155 trees it proves such rules
with up to 338,266 boxes in under 8 s, the check of the proof included. Every counterexample
is an input record that `predict_one` predicts, and every proof is checked again before it is
reported. The answers show what the models do where their training data says nothing: a
district valued above every training target, and a threshold that puts $1 of capital gain with
the people who had capital gains in training.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores), Windows 11, Rust 1.97, a release build of
`probe/`, which uses `tabsmith-runtime` only and runs on one thread. The Kani proofs ran in
WSL 2 (Ubuntu 26.04) on the same machine.

**Models.** The gradient-boosted bundles that `tabsmith train` built on 2026-11-01 for the three
datasets of [reports/datasets](../../datasets/2026-10-31-windows-x86_64/), with the code
released as 0.3.0 (their manifests say 0.2.0, the version before the bump); they are not in the
repository:

| Dataset | Bundle | Task | Trees | Nodes | Inputs → features |
|---|---|---|---|---|---|
| UCI Adult | `b-711d77525a02c0f4` | binary (`>50K`) | 155 | 9,151 | 14 → 108 (102 category slots) |
| California Housing | `b-9efd299a26219c83` | regression | 309 | 18,823 | 8 → 8 |
| UCI Covertype | `b-3e1442af57b2d257` | 7 classes | 280 (40 per class) | 16,848 | 54 → 54 |

and the fixtures `binary-gbdt` (14 trees) and `binary-mlp` (a 35-32-16-1 network). 0.4.0 changed
how the trees add their histograms, not where a threshold goes (see "Thresholds" below).

## Rules

`rules/` holds the rules, one per line; `rules.txt` the output. "Visited" counts the boxes the
search took; a proof's boxes are the ones it closes. The time includes checking the proof again
with the walk of `tabsmith_embedded::rules` and the kernels' bounds (in brackets).

| Model | Rule | Result | Proof | Visited | Time |
|---|---|---|---|---|---|
| Adult | `=> probability <= 0.999` | fails: 0.99956 | | 65 | 0.00 s |
| Adult | `age in 17..=25, capital_gain == 0, capital_loss == 0 => probability <= 0.999` | holds | 24,587 boxes | 49,173 | 0.81 s (0.41) |
| Adult | the same `=> probability <= 0.99` | holds | 338,266 boxes | 676,531 | 7.91 s (4.43) |
| Adult | the same `=> probability <= 0.9` | fails: 0.9079 | | 184 | 0.00 s |
| Adult | `education_num in 1..=9, capital_gain == 0, capital_loss == 0 => probability <= 0.999` | holds | 57,938 boxes | 115,875 | 1.91 s (1.03) |
| Adult | the same `=> probability <= 0.99` | undecided | | 1,000,000 | 5.64 s |
| Adult | `hours_per_week in 1..=20, education_num in 1..=9, age in 17..=25, capital_gain == 0, capital_loss == 0 => probability <= 0.9` | holds | 47,271 boxes | 94,541 | 1.15 s (0.56) |
| Adult | the same `=> label == "<=50K"` | fails: `>50K`, 0.5701 | | 1 | 0.00 s |
| California | `=> value <= 1500000` | holds | 10 boxes | 19 | 0.00 s |
| California | `=> value <= 1000000` | holds | 17,937 boxes | 35,873 | 1.36 s (0.72) |
| California | `=> value <= 800000` | fails: $803,579 | | 818,900 | 7.79 s |
| California | `=> value <= 600000` | fails: $690,372 | | 65 | 0.00 s |
| California | `median_income in 0..=2 => value <= 800000` | holds | 5,936 boxes | 11,871 | 0.32 s (0.19) |
| California | `median_income in 0..=2 => value <= 700000` | holds | 70,517 boxes | 141,033 | 3.06 s (1.69) |
| California | `median_income in 0..=2, housing_median_age in 1..=20 => value <= 700000` | holds | 16,848 boxes | 33,695 | 0.78 s (0.41) |
| California | `_ within 3 sd => value <= 600000` | fails: $695,385 | | 65 | 0.00 s |
| Covertype | `=> label != "4"` | fails: class 4, 0.5739 | | 1 | 0.00 s |
| Covertype | `elevation >= 3500 => label != "4"` | fails: class 4, 0.7138 | | 1 | 0.00 s |
| Covertype | `elevation <= 2200 => label != "7"` | fails: class 7, 0.9996 | | 1 | 0.00 s |

The proofs and their sizes are the same as those of the prototype of 2026-11-01, which used the
same search without certificates. Counterexamples come from the greedy dives, one every 64
boxes, and most of them from the first or the second. A claim close to the true maximum takes
longer either way: California's $803,579 came after 818,900 boxes, and Adult's `<= 0.99` on low
education stayed undecided after a million.

**California.** The training targets stop at $500,001; the trees predict at least $803,579 for a
district of 14 households, 41 people and 9,602 rooms, and $695,385 for one whose every number lies
within 3 standard deviations of the training mean (499 households, 67 people). Each column is in
its training range; the combination is not. A box cannot say that rooms grow with households, so
a rule about realistic districts needs its region narrowed by hand.

**Covertype.** The class-7 counterexample has eight soil types at 1 at once (`soil_type_1`, `_4`,
`_6`, `_12`, `_18`, `_20`, `_34`, `_39`) and two wilderness areas. The dataset encodes the soil type as 40 one-hot columns, which reach
the model as 40 numbers: the bundle does not know they are exclusive, so the audit does not
either, and any combination is an input the model accepts.

## Label stability

`--stability k` asks of each record whether its label survives every move of its numbers by up
to `k` standard deviations of the field, with categories and missing values kept. A field that
held whole numbers has the ends of its range rounded inward, so at 0.05 sd Adult's age
(sd 13.7) stays put while its capital gain may move by $373. Records: every 10th or 20th of the
first 2,000 rows of each dataset (`records.py`); `stability.txt`.

| Model | Records | 0.05 sd | 0.1 sd | 0.25 sd | 0.5 sd | 1 sd | Per record |
|---|---|---|---|---|---|---|---|
| Adult | 200 | 50 can change | 62 | 71 | 152 | 198 | 0.07–0.11 ms |
| Covertype | 100 | 17 | | 70 | | 100 | 0.09–0.23 ms |
| `binary-gbdt` | 200 | 16 | | 83 | | 191 | 0.02–0.03 ms |
| `binary-mlp` | 100 | 5 (1 undecided) | | 28 (20 undecided) | | | 0.1–2.1 s |

Every "can change" comes with an input that changes the label, predicted by `predict_one`; every
other record has a proof, of at most 26 boxes for the trees. No record of the trees was
undecided. A network needs far more boxes: its largest proof at 0.05 sd has 77,604.

**What changes Adult's labels.** `adult-60.txt` lists the 17 of 60 records (every 10th of the
first 600) whose label can change within 0.05 sd. In 15 of them a person predicted to earn over
$50K drops to a probability between 0.019 and 0.20 when capital gain and capital loss both go
from $0 to $1; in 7 of those, nothing else moves. In the data, the smallest positive capital
gain is $114, the smallest capital loss $155, and no one of the 48,842 people has both.

**Thresholds.** The trees split capital gain at "≤ 0" (its lowest threshold, in dollars, is
0.0; capital loss's is -0.0). The trainer puts a threshold at the largest training value of the
bin on its left, and 91.7 % of the rows share the value 0, which therefore fills a bin of its
own: the threshold is 0 itself. A value never seen, between 0 and $114, goes right, with the
small capital gains of the training rows, whose incomes were lower. 0.4.0's trainer places a
threshold the same way: a cell refined down to one value has that value as its top
(`thresholds_from_counts` in `crates/tabsmith-workflow/src/gbdt/bins.rs`). A threshold halfway
between training values would keep $1 with $0. This is a property of the trainer that a test
on training-like rows does not show.

## What the compiler pays

`cost/` compiles the regression fixture with `rule(_ within 1 sd => value <= 30139.57)`, whose
proof has 3,244 boxes over 3,610 nodes. With `RULE_CHECK_MAX_WORK` raised so that the compiler
checks it, `cargo check` of the crate took 33.3, 33.3 and 38.5 s with incremental compilation
off (with it on, an unchanged crate reuses the result, 0.45 s), against 0.58 to 0.66 s when the
macro skips the compiler's check: about 2.8 µs per tree node per box, some 4,500 times slower
than native code. `model!` therefore leaves to the macro's own check any proof above a million
nodes times boxes, about 3 s, and says so in the module's documentation. The proofs in the
macro's tests are far below the limit: 5 boxes over 854 nodes for the defects trees, 269 boxes
over the 1,697 weights of the regression network.

## Kani

The checker's step that makes a proof's closed boxes cover its region is proved with Kani
0.68 (CBMC 6.11, CaDiCaL) in WSL 2, for every box, row and finite threshold of a domain with
two numbers, a category of three slots and one of two, each with and without values never seen
(`rules::proofs` in `crates/tabsmith-embedded/src/rules.rs`, `kani-rules.txt`):

| Proof | Checks | Result | Time | Peak memory |
|---|---|---|---|---|
| `a_split_keeps_every_input_in_its_part`: a split keeps every input of a box in the part its own value selects, and that part is not taken for empty | 669 | all pass | 5.8 s | 163 MB |
| `a_box_with_an_input_is_not_empty` | 508 | all pass | 1.6 s | 61 MB |

The assertions themselves are reached (status SUCCESS, not UNREACHABLE); the 6 unreachable
checks of each are in Kani's own models of the standard library. The feature is symbolic, so
the loops carry `#[kani::unwind(9)]`, and Kani checks that 9 iterations suffice. CI runs both.

## Not covered

- "Every input" is every input the model accepts: fractions in whole-number fields between the
  ends of a range, combinations no data has, one-hot columns of the dataset taken as numbers.
- A network's interval bounds are loose: tight rules stay undecided (`binary-mlp`: 1 to 20
  records undecided above), and a network extrapolates without bounds on its fields.
- A multiclass claim on a probability, rather than a label, is not supported.
- Times are one run each, on a machine running other programs; the box counts are exact and
  the same on every machine.

## Reproduce

```sh
# ADULT, CALIFORNIA, COVERTYPE: the three bundles; RECORDS: a folder of records.py output
reports/rules/2026-11-06-windows-x86_64/run.sh
cargo test -p tabsmith-runtime --test audit        # the fixtures, from the repository root
```

---

## 中文

**结论。** `tabsmith audit` 在真实模型上判定“这个区域里的每个输入都得到满足某个结论的预测”这类规则。在 Adult 的 155 棵树上，它能在 8 秒内（含复核）证明需要 338,266 个盒子的规则。每个反例都是用 `predict_one` 预测过的输入记录，每个证明在报告前都会复核。答案揭示了模型在训练数据没有覆盖之处的行为。

**规则。** Adult：“17 到 25 岁、没有资本收益和资本损失的人，被预测为高收入的概率不超过 0.99”成立（证明有 338,266 个盒子）；0.9 不成立。California：训练目标最高 500,001 美元，而树对一个 14 户、41 人、9,602 个房间的街区预测至少 803,579 美元；每列都在训练范围内，组合却不存在。Covertype：数据把土壤类型编码成 40 个独热列，对模型来说是 40 个数，所以审计会给出同时属于多个土壤类型的输入。

**标签稳定性。** 在每行数值 ±0.05 个标准差内（整数字段的范围两端向内取整），Adult 的 200 行中有 50 行的标签可以改变，Covertype 的 100 行中有 17 行。在 Adult 抽查的 60 行里，17 行可以改变：其中 15 个被预测年收入超过 5 万美元的人，在资本收益和资本损失都从 0 美元变成 1 美元时，概率降到 0.019 到 0.20；其中 7 个没有其他任何改动。数据中资本收益最小的正值是 114 美元，资本损失是 155 美元，48,842 人中没有人两者都有。

**阈值。** 树在资本收益“≤ 0”处切分。训练器把阈值放在左边箱子里最大的训练值上，而 91.7% 的行取值为 0，所以 0 单独成为一个箱子，阈值就是 0。训练中从未出现的 0 到 114 美元之间的值走右边，和训练中资本收益很小、收入较低的人在一起。0.4.0 的训练器放阈值的方式相同；如果阈值放在两个训练值的中点，1 美元就会和 0 在一起。这是训练器的性质，用类似训练数据的行做测试看不出来。

**编译器的代价。** 编译器复查一个 3,244 个盒子、3,610 个节点的证明用了约 33 秒，约每个节点每个盒子 2.8 微秒，比本机代码慢约 4,500 倍。所以超过一百万（约 3 秒）的证明改在宏展开时复核，模块文档会注明。

**没有覆盖的。** “每个输入”指模型接受的每个输入；网络的区间界较宽，紧的规则会停在“未定”；不支持多分类的概率结论。
