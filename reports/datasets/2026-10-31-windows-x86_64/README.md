# Three public datasets: Tabsmith against scikit-learn on the same rows

[中文](#中文)

**Claim.** On three public datasets of the three tasks, every Tabsmith candidate
reaches the quality of its scikit-learn equivalent trained on the same rows, split,
features and settings, within the noise of the test split. Its trees now train in about
the time of `HistGradientBoosting`, 11 to 36 times faster than in
[the report of 2026-10-29](../2026-10-29-windows-x86_64/), with the same test metrics to
four decimals.

**Datasets.** UCI Adult (income > 50K: binary, 48,842 rows, 14 inputs of which 8 are
text categories, "?" marks a missing value; CC BY 4.0). California Housing (median
house value: regression, 20,640 rows, 8 numeric inputs; StatLib, via scikit-learn's
mirror). UCI Covertype (forest cover type, 7 classes: multiclass, 581,012 rows, 54
numeric inputs; CC BY 4.0). `datasets-prep.py` builds a CSV with a header from each
download; nothing else is changed. The downloads are not in the repository.

**Sources.** The files used, as downloaded on 2026-10-29. scikit-learn publishes the same
SHA-256 for `cal_housing.tgz`.

| Dataset | Download | Bytes | SHA-256 | Unpack into |
|---|---|---|---|---|
| Adult | <https://archive.ics.uci.edu/static/public/2/adult.zip> | 620,237 | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` | `<folder>/adult/files/` |
| California Housing | <https://ndownloader.figshare.com/files/5976036> (`cal_housing.tgz`) | 441,963 | `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681` | `<folder>/california/files/` |
| Covertype | <https://archive.ics.uci.edu/static/public/31/covertype.zip> | 11,251,206 | `89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d` | `<folder>/covertype/files/` |

**How.** As in the earlier report. For each dataset, one `tabsmith train` on Windows
(`datasets-train.ps1`; stratified split 70/15/15 keyed by row content, every candidate,
CPU). Tabsmith chose its model on the validation split and evaluated it once on the test
split; for this comparison the two other candidates were then scored on the test split
as well (`datasets-evaluate.ps1`, `--allow-test-reuse`, recorded as reuse #2 and #3 in
their evaluation records: a benchmark, not model selection). The scikit-learn side
(`tools/datasets/compare.py`, run in WSL with scikit-learn 1.9.1) reads the project's
own split and transform and trains the equivalent of each candidate on the training
rows: the same features, the same regularisation, balanced class weights, and as many
epochs or boosting rounds as Tabsmith's run trained. It scores the same test rows with
the same metrics. `tools/datasets/report.py` makes the tables from the JSON files in `runs/`.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 threads), 64 GB. Tabsmith: the release
binary of the commit that brings this report, Windows 11, its default of 12 threads.
scikit-learn: WSL 2 Ubuntu on the same machine, its default of 16 threads. Training times
are wall-clock seconds of the fit alone, from single runs on a machine that was also
running other programs: two runs of this code took 24.9 s and 29.0 s for Covertype's trees.

## Results

### adult: binary, target `income`, 48,842 rows (34,190 train / 7,326 test), 6 numeric + 8 categorical inputs

| Model | ROC AUC | balanced accuracy | log loss | Training time |
|---|---|---|---|---|
| Tabsmith linear | 0.9067 | 0.8250 | 0.3842 | 0.2 s, 24 epochs/rounds |
| scikit-learn LogisticRegression | 0.9067 | 0.8244 | 0.3941 | 1.0 s, 69 iterations |
| Tabsmith small-mlp | 0.9157 | 0.8289 | 0.3693 | 0.8 s, 21 epochs/rounds |
| scikit-learn MLPClassifier | 0.9041 | 0.8205 | 0.3937 | 1.1 s, 21 iterations |
| Tabsmith gbdt (chosen) | 0.9301 | 0.8434 | 0.3308 | 0.5 s, 180 epochs/rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9297 | 0.8442 | 0.3309 | 0.8 s, 180 iterations |

### california: regression, target `median_house_value`, 20,640 rows (14,446 train / 3,097 test), 8 numeric + 0 categorical inputs

| Model | RMSE | MAE | R² | Training time |
|---|---|---|---|---|
| Tabsmith linear | 69,225 | 51,078 | 0.6433 | 0.1 s, 32 epochs/rounds |
| scikit-learn Ridge | 69,286 | 50,858 | 0.6427 | 0.0 s, 32 iterations (unweighted) |
| Tabsmith small-mlp | 53,308 | 36,614 | 0.7885 | 0.4 s, 40 epochs/rounds |
| scikit-learn MLPRegressor | 54,712 | 37,833 | 0.7772 | 0.6 s, 40 iterations (unweighted) |
| Tabsmith gbdt (chosen) | 45,817 | 30,664 | 0.8438 | 0.2 s, 334 epochs/rounds |
| scikit-learn HistGradientBoostingRegressor | 45,817 | 30,630 | 0.8438 | 0.5 s, 334 iterations (unweighted) |

### covertype: multiclass, target `cover_type`, 581,012 rows (406,708 train / 87,152 test), 54 numeric + 0 categorical inputs

| Model | ROC AUC (ovr) | balanced accuracy | accuracy | log loss | Training time |
|---|---|---|---|---|---|
| Tabsmith linear | 0.9206 | 0.6977 | 0.5774 | 0.9795 | 2.4 s, 17 epochs/rounds |
| scikit-learn LogisticRegression | 0.9264 | 0.7037 | 0.5969 | 0.9221 | 25.3 s, 345 iterations |
| Tabsmith small-mlp | 0.9634 | 0.8353 | 0.7474 | 0.5929 | 17.1 s, 41 epochs/rounds |
| scikit-learn MLPClassifier | 0.9627 | 0.8376 | 0.7407 | 0.6098 | 30.6 s, 41 iterations |
| Tabsmith gbdt (chosen) | 0.9924 | 0.9355 | 0.8995 | 0.2610 | 29.0 s, 400 epochs/rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9921 | 0.9361 | 0.8984 | 0.2638 | 25.5 s, 400 iterations |

## The trees, before and after

| Dataset and rounds | 2026-10-29 | Now | Faster | Test metrics |
|---|---|---|---|---|
| California, 334 rounds | 2.07 s | 0.19 s | 11 times | same to four decimals, weights identical |
| Adult, 180 rounds | 17.08 s | 0.48 s | 36 times | same to four decimals |
| Covertype, 400 rounds | 513.08 s | 29.03 s | 18 times | same to four decimals |

Retraining changes Adult's and Covertype's weights, because rounding decides some splits
and ties; the models are as good. The whole `tabsmith train` with all three candidates
went from 603 s to 71 s on Covertype and from 20 s to 3 s on Adult.

## The same operating system

The tables above compare Windows with WSL. Here each side trains only the trees, on one
operating system at a time, with the same rows per dataset (`trees-windows.txt`,
`trees-linux.txt`, `hgb-windows.txt`, `hgb-time.py`). Tabsmith uses its default of 12
threads and scikit-learn 16, its default in WSL. scikit-learn is 1.5.1 on Windows
(Anaconda) and 1.9.1 in WSL.

| Training the trees | Windows: Tabsmith | Windows: scikit-learn | WSL: Tabsmith | WSL: scikit-learn |
|---|---|---|---|---|
| California, 334 rounds | 0.19 s | 0.40 s | 0.25 s | 0.47 s |
| Adult, 180 rounds | 0.50 s | 0.91 s | 0.88 s | 0.65 s |
| Covertype, 40 rounds | 3.09 s | 4.43 s | 4.09 s | 2.89 s |

Tabsmith is 1.4 to 2.1 times faster on Windows. In WSL it is 1.9 times faster on
California and about 1.4 times slower on Adult and Covertype. So the two are about even,
and neither wins everywhere.

## What made the trees slow

Phase timers in a copy of the earlier code, on Windows:

| | California | Adult | Covertype, 40 rounds |
|---|---|---|---|
| Training time on 32 threads | 2.81 s | 17.49 s | 63.35 s |
| Share spent building histograms | 8% | 95% | 86% |
| Share spent rewriting and syncing `run.json` | about 85% | about 2% | under 1% |

- **Threads barely helped.** Every node above 16,384 rows started new threads, and every
  chunk of rows allocated, zeroed and merged a histogram of its own. The new code on one
  thread trains Covertype's 40 rounds in 8.3 s, against 63 s for the old code on 32.
- **One-hot columns did useless work.** 44 of Covertype's 54 columns and most of Adult's
  105 features are one-hot, so nearly every row updates the same bin. Skipping a bin that
  holds at least 80% of the rows, and deriving it from the node's totals, makes those
  columns several times cheaper.
- **Both children were summed from their rows.** Deriving the larger child as the parent
  minus its sibling cuts the rows visited to about a third.
- **`run.json` was rewritten and synced after every round.** Each write held the whole
  history. It now happens at most once a second.

**A correction.** The earlier report also blamed the fixed order of the sums, saying it
"rules out some of the fastest reductions". That was wrong. When a histogram is built
one feature at a time, each feature sums its rows in row order on one thread, there are
no partial sums to merge, and the weights are the same at any thread count
(`reports/reproducibility/`).

## Threads

Training time in seconds; the weights are identical in every cell of a column
(`trees-windows.txt`, `trees-linux.txt`).

| Threads | Windows: Covertype, 40 rounds | WSL: Covertype, 40 rounds | Windows: 1,000,000 rows | WSL: 1,000,000 rows |
|---|---|---|---|---|
| 1 | 8.34 | 8.13 | not run | 12.51 |
| 4 | 4.39 | 3.71 | 6.20 | 5.64 |
| 8 | 3.37 | 3.85 | 4.84 | 5.27 |
| 12 (the default) | 3.05 | 3.92 | 4.43 | 5.20 |
| 16 | 2.91 | 4.12 | 4.35 | not run |
| 32 | 2.80 | 4.86 | 4.19 | 5.86 |

Twelve threads stay within about 9% of the best count in every column. More threads gain
little on Windows and lose time in WSL. `RAYON_NUM_THREADS` overrides the default. A Linux
machine outside a virtual machine may scale differently; none was available here.

## Reading the tables

- **Quality is on par, for every candidate and task.** The largest differences are
  the networks' (Tabsmith's small MLP is a little ahead on Adult and California, 0.012
  and 0.011 of R² or AUC) and the linear model's on Covertype (scikit-learn's lbfgs
  converged further in 345 iterations than Tabsmith's 17 epochs of Adam, 0.006 of AUC);
  the trees agree to the third decimal everywhere. The test split of Adult has 7,326
  rows, so an AUC difference below about 0.01 is within its resolution.
- **Tabsmith's networks and linear model train as fast or faster** on one thread than
  scikit-learn's on every core, because the problems are small for a network: the
  epochs are the same by construction, and Tabsmith's single-threaded Burn loop has no
  Python overhead per batch. On Covertype its linear model stopped early after 17
  epochs at a slightly lower AUC than lbfgs reached.
- **The trees do not stop early on Covertype**: 400 rounds is the configured
  maximum (`train.gbdt.epochs`), and the validation loss was still falling. More
  rounds, or a larger learning rate, would improve both sides.

What this does not show: LightGBM and XGBoost, which are the reference trainers at
Covertype's size and were not installable here; Linux outside a virtual machine; and any
claim about deployment, which the README benchmark measures (`reports/benchmarks/`).

## Reproduce

1. Download the three datasets (links above), check their SHA-256, unpack each into the
   folder the table names, and run `python -I -B datasets-prep.py <folder>` to build the CSVs.
2. `datasets-train.ps1` (Windows; the same `tabsmith train` lines work in any shell),
   then `datasets-evaluate.ps1`.
3. With a Python that has scikit-learn, pandas and numpy: `datasets-compare.sh`, or
   `python -I tools/datasets/compare.py <project> <project>/sklearn.json` per project.
4. `python -I tools/datasets/report.py <runs folder> <out folder>`.
5. For the table on one operating system: `python -I hgb-time.py <folder> 40`.

---

## 中文

**结论。** 在三个任务各一个公开数据集上（UCI Adult 二分类、California Housing 回归、UCI Covertype 七分类），
Tabsmith 的每个候选都达到了在同样的行、切分、特征和设置下训练的 scikit-learn 对应模型的质量，差异在测试集的噪声以内。
树模型的训练时间现在与 `HistGradientBoosting` 差不多，比 [2026-10-29 的报告](../2026-10-29-windows-x86_64/)快 11 到 36 倍，
测试指标到小数点后四位不变。

**方法。** 与之前的报告相同。Tabsmith 在 Windows 上用默认的 12 个线程，scikit-learn 在 WSL 中用默认的 16 个线程。
训练时间来自单次运行，机器上还有其他程序在运行：同一份代码跑两次，Covertype 的树分别用了 24.9 秒和 29.0 秒。

**前后对比。** California 334 轮从 2.07 秒到 0.19 秒，Adult 180 轮从 17.08 秒到 0.48 秒，Covertype 400 轮从 513.08 秒到 29.03 秒。
重新训练会让 Adult 和 Covertype 的权重略有变化（舍入决定了个别分裂和平局），测试指标不变；California 的权重逐位相同。
三个候选完整的 `tabsmith train`，Covertype 从 603 秒降到 71 秒，Adult 从 20 秒降到 3 秒。

**同一操作系统。** 上面的表比较的是 Windows 和 WSL。只训练树、在同一系统上比较时，Tabsmith 在 Windows 上快 1.4 到 2.1 倍；
在 WSL 里 California 快 1.9 倍，Adult 和 Covertype 慢约 1.4 倍。两边大致持平，谁也不是处处领先。Windows 上的 scikit-learn 是 1.5.1 版，WSL 里是 1.9.1 版。

**之前慢在哪里。** 实测原因有四个：多线程几乎没起作用（每个大节点都新建线程，每个分块都分配并合并自己的直方图）；
独热列做了无用功（几乎所有行都落在同一个桶）；两个子节点都从行重新统计（现在用父节点减兄弟节点，访问的行数减少到约三分之一）；
每轮都重写并同步 `run.json`（现在最多每秒一次）。

**一处更正。** 之前的报告还把慢归因于固定的求和顺序，说它“排除了一些最快的合并方式”。这是错的。按特征构建直方图时，
每个特征在一个线程上按行序求和，根本没有需要合并的部分和，任何线程数下权重都相同（见 `reports/reproducibility/`）。

**线程数。** 默认 12 个线程，在 Windows 和 WSL 上都与实测最佳值相差约 9% 以内。线程再多，Windows 上收益很小，WSL 里反而变慢。
`RAYON_NUM_THREADS` 可以覆盖默认值。这里没有非虚拟机的 Linux 可测，它的扩展情况可能不同。

**没有测量的。** LightGBM 和 XGBoost（这里装不了）、非虚拟机的 Linux。
