# Three public datasets: Tabsmith against scikit-learn on the same rows

[中文](#中文)

**Claim.** On three public datasets of the three tasks, every Tabsmith candidate
reaches the quality of its scikit-learn equivalent trained on the same rows, split,
features and settings, within the noise of the test split. Where Tabsmith clearly loses
is the training time of its trees on the two larger problems: 21 times slower than
`HistGradientBoostingClassifier` on Adult and 17 times slower on Covertype. Its linear
model and network train as fast as, or faster than, their counterparts.

**Datasets.** UCI Adult (income > 50K: binary, 48,842 rows, 14 inputs of which 8 are
text categories, "?" marks a missing value; CC BY 4.0). California Housing (median
house value: regression, 20,640 rows, 8 numeric inputs; StatLib, via scikit-learn's
mirror). UCI Covertype (forest cover type, 7 classes: multiclass, 581,012 rows, 54
numeric inputs; CC BY 4.0). `datasets-prep.py` builds a CSV with a header from each
download; nothing else is changed. The downloads are not in the repository.

**How.** For each dataset, one `tabsmith train` on Windows (`datasets-train.ps1`;
stratified split 70/15/15 keyed by row content, every candidate, CPU). Tabsmith chose
its model on the validation split and evaluated it once on the test split; for this
comparison the two other candidates were then scored on the test split as well
(`datasets-evaluate.ps1`, `--allow-test-reuse`, recorded as reuse #2 and #3 in their
evaluation records: a benchmark, not model selection). The scikit-learn side
(`tools/datasets/compare.py`, run in WSL with scikit-learn 1.9.1) reads the project's
own split and transform and trains the equivalent of each candidate on the training
rows: the same features (standardised numbers, one-hot categories, mean imputation),
the same regularisation, balanced class weights, and as many epochs or boosting rounds
as Tabsmith's run trained. It scores the same test rows with the same metrics.
`tools/datasets/report.py` makes the tables below from the three JSON files per
dataset (in `runs/`).

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 threads), 64 GB. Tabsmith: the release
binary of the commit that brings this report, Windows 11. scikit-learn: WSL 2 Ubuntu on
the same machine, every core (its defaults). Training times are wall-clock seconds of
the fit alone, informational: the two sides ran on different operating systems.

## Results

### adult: binary, target `income`, 48,842 rows (34,190 train / 7,326 test), 6 numeric + 8 categorical inputs

| Model | ROC AUC | balanced accuracy | log loss | Training time |
|---|---|---|---|---|
| Tabsmith linear | 0.9067 | 0.8250 | 0.3842 | 0.3 s, 24 epochs |
| scikit-learn LogisticRegression | 0.9067 | 0.8244 | 0.3941 | 1.6 s, 69 iterations |
| Tabsmith small-mlp | 0.9157 | 0.8289 | 0.3693 | 0.9 s, 21 epochs |
| scikit-learn MLPClassifier | 0.9041 | 0.8205 | 0.3937 | 1.1 s, 21 epochs |
| Tabsmith gbdt (chosen) | 0.9301 | 0.8434 | 0.3308 | 17.1 s, 180 rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9297 | 0.8442 | 0.3309 | 0.8 s, 180 rounds |

### california: regression, target `median_house_value`, 20,640 rows (14,446 train / 3,097 test), 8 numeric inputs

| Model | RMSE | MAE | R² | Training time |
|---|---|---|---|---|
| Tabsmith linear | 69,225 | 51,078 | 0.6433 | 0.2 s, 32 epochs |
| scikit-learn Ridge | 69,286 | 50,858 | 0.6427 | 0.0 s |
| Tabsmith small-mlp | 53,308 | 36,614 | 0.7885 | 0.5 s, 40 epochs |
| scikit-learn MLPRegressor | 54,712 | 37,833 | 0.7772 | 0.6 s, 40 epochs |
| Tabsmith gbdt (chosen) | 45,817 | 30,664 | 0.8438 | 2.1 s, 334 rounds |
| scikit-learn HistGradientBoostingRegressor | 45,817 | 30,630 | 0.8438 | 0.5 s, 334 rounds |

### covertype: multiclass, target `cover_type`, 581,012 rows (406,708 train / 87,152 test), 54 numeric inputs

| Model | ROC AUC (ovr) | balanced accuracy | accuracy | log loss | Training time |
|---|---|---|---|---|---|
| Tabsmith linear | 0.9206 | 0.6977 | 0.5774 | 0.9795 | 3.0 s, 17 epochs |
| scikit-learn LogisticRegression | 0.9264 | 0.7037 | 0.5969 | 0.9221 | 26.5 s, 345 iterations |
| Tabsmith small-mlp | 0.9634 | 0.8353 | 0.7474 | 0.5929 | 16.9 s, 41 epochs |
| scikit-learn MLPClassifier | 0.9627 | 0.8376 | 0.7407 | 0.6098 | 31.2 s, 41 epochs |
| Tabsmith gbdt (chosen) | 0.9924 | 0.9355 | 0.8995 | 0.2610 | 513.1 s, 400 rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9921 | 0.9361 | 0.8984 | 0.2638 | 30.7 s, 400 rounds |

## Reading the tables

- **Quality is on par, for every candidate and task.** The largest differences are
  the networks' (Tabsmith's small MLP is a little ahead on Adult and California, 0.012
  and 0.011 of R² or AUC) and the linear model's on Covertype (scikit-learn's lbfgs
  converged further in 345 iterations than Tabsmith's 17 epochs of Adam, 0.006 of AUC);
  the trees agree to the third decimal everywhere. The test split of Adult has 7,326
  rows, so an AUC difference below about 0.01 is within its resolution.
- **Tabsmith loses on tree training time.** 17.1 s against 0.8 s on Adult (180
  rounds), 513 s against 31 s on Covertype (400 rounds of 7 trees each). Three reasons,
  all in Tabsmith's design: each leaf's histogram is built from its own rows, where
  `HistGradientBoosting` derives one child from the parent minus its sibling (half the
  work, and no re-reading of the rows); Tabsmith's parallelism is over rows in fixed
  chunks, with the gradients, the partition of each node and the leaf bookkeeping on
  one thread, while scikit-learn parallelises over features in C with OpenMP; and
  Tabsmith keeps the sums in an order that gives the same trees on any core count,
  which rules out some of the fastest reductions. Adult is the worst case because its
  8 text columns become about a hundred one-hot features; on a million rows of the
  synthetic problem (35 features) a round takes 0.2 s (`reports/scaling/`).
- **Tabsmith's networks and linear model train as fast or faster** on one thread than
  scikit-learn's on every core, because the problems are small for a network: the
  epochs are the same by construction, and Tabsmith's single-threaded Burn loop has no
  Python overhead per batch. On Covertype its linear model stopped early after 17
  epochs (3.0 s) at a slightly lower AUC than lbfgs reached in 26.5 s.
- **The trees do not stop early on Covertype**: 400 rounds is the configured
  maximum (`train.gbdt.epochs`), and the validation loss was still falling. More
  rounds, or a larger learning rate, would improve both sides.

What this does not show: LightGBM and XGBoost, which are the reference trainers at
Covertype's size and were not installable here; and any claim about deployment, which
the README benchmark measures (`reports/benchmarks/`).

## Reproduce

1. Download the three datasets (links above) and run
   `python -I -B datasets-prep.py <folder>` to build the CSVs.
2. `datasets-train.ps1` (Windows; the same `tabsmith train` lines work in any shell),
   then `datasets-evaluate.ps1`.
3. With a Python that has scikit-learn, pandas and numpy: `datasets-compare.sh`, or
   `python -I tools/datasets/compare.py <project> <project>/sklearn.json` per project.
4. `python -I tools/datasets/report.py <runs folder> <out folder>`.

---

## 中文

**结论。** 在三个任务各一个公开数据集上（UCI Adult 二分类、California Housing 回归、UCI Covertype 七分类），
Tabsmith 的每个候选都达到了在同样的行、切分、特征和设置下训练的 scikit-learn 对应模型的质量，差异在测试集的噪声以内。
Tabsmith 明显输的地方是两个较大问题上树模型的训练时间：Adult 上比 `HistGradientBoostingClassifier` 慢 21 倍，
Covertype 上慢 17 倍。线性模型和网络的训练速度与对方相当或更快。

**方法。** 每个数据集在 Windows 上跑一次 `tabsmith train`（分层切分 70/15/15，所有候选，CPU）。Tabsmith 在验证集上选模型、在测试集上只评估一次；
为了对比，另外两个候选随后也在测试集上评估（`--allow-test-reuse`，评估记录里注明是第 2、3 次使用：这是基准，不是选模型）。
scikit-learn 一侧（`tools/datasets/compare.py`，在 WSL 中用 scikit-learn 1.9.1）读取项目自己的切分和变换，用同样的特征、正则化、
平衡类别权重和同样多的 epoch/轮数训练对应模型，在同样的测试行上用同样的指标评分。

**解读。** 质量各处相当：网络的差异最大（Tabsmith 的小 MLP 在 Adult 和 California 上略领先 0.01），树在各处都一致到小数点后三位。
树的训练时间是 Tabsmith 的短板：原因是每个叶子的直方图都从自己的行重新构建（scikit-learn 用“父减兄弟”省一半工作）、
并行只在行的固定分块上（梯度、节点划分和叶子记账仍是单线程，scikit-learn 在 C 里按特征用 OpenMP 并行）、
以及为了让任何核心数都得到同样的树而固定求和顺序。Adult 是最坏情况，8 个文本列变成约一百个 one-hot 特征。
网络和线性模型单线程反而不慢：epoch 数相同，而 Tabsmith 的训练循环没有每个 batch 的 Python 开销。
Covertype 上树没有早停：400 轮是配置的上限，验证损失还在下降。没有测量 LightGBM 和 XGBoost（这里装不了）。
