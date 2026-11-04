# The trees against LightGBM and XGBoost

[中文](#中文)

**Claim.** On one machine, with the same rows, split, features and settings, Tabsmith's trees train
between LightGBM and XGBoost, at a test quality on par or better: 8 to 27% slower than LightGBM on the
largest of three public datasets, 1.5 to 1.8 times its time on the two smaller ones, and faster than
XGBoost on all three. This is safe Rust (`unsafe_code = "forbid"`), and the model is byte-identical at
every thread count and on Windows and Linux. Against version 0.3 the trees train 1.5 to 4.9 times faster.

**Where.** AMD Ryzen 9 9950X3D2 (16 cores, 32 logical CPUs), 125 GB of memory, 61 GB of it visible in WSL 2 Ubuntu 26.04 (Linux 6.6.87.2) on Windows 11. Python 3.14, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, numpy 2.5.4, Rust 1.98.1. Nothing else ran during the timings except what `run.log` shows as the load; a process on the host held most of the memory commit, idle.

**How.** `run.sh` builds `tabsmith` from a fresh copy of the working tree, and trains each dataset at
1, 4, 8, 12 and 16 threads, three times each, every fit a process of its own. Tabsmith's time is
`train_ms` of its run record, which includes binning and the run's checkpoints and excludes reading
the CSV. [`compare_gbm.py`](../../../tools/datasets/compare_gbm.py) then reads that project's own
split and features and fits each library on the same training rows. Its time is the fit alone,
binning included, with the validation rows scored every round, as Tabsmith's does for early stopping.
No early stopping: the rounds are fixed. The settings are Tabsmith's defaults on all sides: learning
rate 0.1, 31 leaves, depth 8, 20 rows per leaf, 255 bins, L2 1, balanced class weights, a standardised
regression target. [`report_training.py`](../../../tools/datasets/report_training.py) makes every table
below from `results.jsonl`; `before-0.3.jsonl` is the same script on the commit of version 0.3.0.

### Seconds at each one's best thread count

| Problem | Tabsmith 0.3 | Tabsmith | LightGBM | XGBoost |
|---|---|---|---|---|
| california, 334 rounds | 0.216 (1) | 0.148 (4) | 0.096 (8) | 0.358 (8) |
| adult, 180 rounds | 0.682 (4) | 0.202 (8) | 0.115 (8) | 0.389 (8) |
| covertype, 40 rounds | 3.714 (4) | 0.915 (12) | 0.846 (8) | 1.739 (8) |
| covertype, 400 rounds | 39.2 (16) | 8.020 (12) | 6.330 (8) | 12.2 (8) |

### Seconds by thread count (median of the runs)

**california, 334 rounds**

| Threads | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Tabsmith 0.3 | 0.216 | 0.255 | 0.255 | – | 0.252 |
| Tabsmith | 0.184 | 0.148 | 0.149 | 0.172 | 0.181 |
| LightGBM | 0.226 | 0.102 | 0.096 | 0.161 | 0.194 |
| LightGBM quantized | 0.213 | 0.110 | 0.098 | 0.211 | 0.263 |
| XGBoost | 0.471 | 0.360 | 0.358 | 0.480 | 0.520 |

**adult, 180 rounds**

| Threads | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Tabsmith 0.3 | 0.935 | 0.682 | 0.809 | – | 0.879 |
| Tabsmith | 0.406 | 0.228 | 0.202 | 0.223 | 0.249 |
| LightGBM | 0.318 | 0.129 | 0.115 | 0.160 | 0.191 |
| LightGBM quantized | 0.334 | 0.140 | 0.130 | 0.176 | 0.229 |
| XGBoost | 0.747 | 0.405 | 0.389 | 0.458 | 0.489 |

**covertype, 40 rounds**

| Threads | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Tabsmith 0.3 | 8.104 | 3.714 | 3.738 | – | 4.168 |
| Tabsmith | 3.865 | 1.311 | 0.940 | 0.915 | 0.944 |
| LightGBM | 3.576 | 1.188 | 0.846 | 0.963 | 0.996 |
| LightGBM quantized | 3.113 | 1.257 | 0.831 | 0.901 | 0.953 |
| XGBoost | 6.541 | 2.308 | 1.739 | 1.762 | 1.865 |

**covertype, 400 rounds**

| Threads | 8 | 12 |
|---|---|---|
| Tabsmith 0.3 | – | – |
| Tabsmith | 8.425 | 8.020 |
| LightGBM | 6.330 | – |
| LightGBM quantized | 5.804 | – |
| XGBoost | 12.2 | – |

### Quality on the test rows

**california, 334 rounds**

| | rmse | mae | r2 |
|---|---|---|---|
| Tabsmith | 46,008 | 30,671 | 0.8425 |
| LightGBM | 46,146 | 30,615 | 0.8415 |
| LightGBM quantized | 46,508 | 31,201 | 0.8390 |
| XGBoost | 46,112 | 30,690 | 0.8417 |

**adult, 180 rounds**

| | roc_auc | balanced_accuracy | log_loss |
|---|---|---|---|
| Tabsmith | 0.9304 | 0.8443 | 0.3300 |
| LightGBM | 0.9300 | 0.8437 | 0.3294 |
| LightGBM quantized | 0.9292 | 0.8441 | 0.3334 |
| XGBoost | 0.9298 | 0.8421 | 0.3293 |

**covertype, 40 rounds**

| | roc_auc | balanced_accuracy | accuracy | log_loss |
|---|---|---|---|---|
| Tabsmith | 0.9644 | 0.8551 | 0.7304 | 0.6081 |
| LightGBM | 0.9618 | 0.8485 | 0.7203 | 0.6360 |
| LightGBM quantized | 0.9604 | 0.8461 | 0.7119 | 0.6556 |
| XGBoost | 0.9549 | 0.8318 | 0.6853 | 0.7494 |

**covertype, 400 rounds**

| | roc_auc | balanced_accuracy | accuracy | log_loss |
|---|---|---|---|---|
| Tabsmith | 0.9923 | 0.9339 | 0.8985 | 0.2633 |
| LightGBM | 0.9912 | 0.9321 | 0.8903 | 0.2818 |
| LightGBM quantized | 0.9847 | 0.9169 | 0.8463 | 0.3805 |
| XGBoost | 0.9859 | 0.9179 | 0.8546 | 0.3607 |

### Different models, over all the runs and thread counts

A model is a hash of Tabsmith's weights, or of the libraries' raw test margins computed on one thread.

| Problem | Tabsmith | LightGBM | LightGBM quantized | XGBoost |
|---|---|---|---|---|
| california, 334 rounds | 1 | 1 | 5 | 1 |
| adult, 180 rounds | 1 | 1 | 5 | 1 |
| covertype, 40 rounds | 1 | 1 | 5 | 1 |
| covertype, 400 rounds | 1 | 2 | 2 | 1 |

### Peak memory while training, MB

Tabsmith's is the whole process's, with the data frame it read and the other splits; each library's is its process's from the fit on, with the feature matrix and Python.

| Problem | Tabsmith | LightGBM | XGBoost |
|---|---|---|---|
| california, 334 rounds | 31 | 215 | 334 |
| adult, 180 rounds | 142 | 310 | 471 |
| covertype, 40 rounds | 1,453 | 910 | 1,115 |
| covertype, 400 rounds | 1,466 | 813 | 1,115 |


## Why it got faster

Where the time goes on Covertype's 40 rounds, in seconds, with phase timers in a copy of each version
(`profile.txt`; the copy of 0.3 had integer sums, which took the time its floats took):

| Part | 0.3, 1 thread | 0.3, 8 threads | Now, 1 thread | Now, 8 threads |
|---|---|---|---|---|
| Histograms | 4.46 | 1.67 | 1.33 | 0.35 |
| Partitions | 0.37 | 0.59 | 0.47 | 0.13 |
| Gradients and probabilities | 1.6 | 0.6 | 1.16 | 0.20 |
| Applying leaves and trees | 0.43 | 0.47 | 0.32 | 0.05 |
| Losses and the metric | 1.22 | 0.28 | 0.24 | 0.09 |
| Binning | 0.20 | 0.04 | 0.21 | 0.04 |

Four changes account for it ([D15](../../../docs/design-decisions.md#d15-the-trees-add-integers-on-workers-that-spin)):
- **Sections no longer wake sleeping threads.** See below. This is the difference between stopping at
  4 threads and reaching 8 and more.
- **Exclusive one-hot columns share a pass.** Covertype takes 12 passes instead of 54, Adult 17 instead
  of 105. LightGBM calls this exclusive feature bundling.
- **A pass adds its rows to two tables in turn**, so that a row does not wait for the store of the row
  before it.
- **Probabilities are computed once per round, and the training loss on a sample.**

The sums being integers is what allows the second and the third: floats would have changed the model.
The integers cost no quality: at a scale of 2^30 with the old thresholds, the test metrics equal the
floats' to four decimals, and they stay flat down to about 8 bits (`precision-sweep.jsonl`, with a
fixed scale of 2^frac per tree and the thresholds named in `bins`).

## A million rows

The synthetic 1,000,000 rows of [`reports/scaling`](../../scaling/) (35 features, 698,621 training
rows), through the whole `tabsmith train` with early stopping, in WSL on 12 threads (`scale.txt`):

| | 0.3 | Now |
|---|---|---|
| Boosting rounds | 5.25 s, 148 rounds | 1.15 s, 140 rounds |
| The whole process | 20.1 s | 14.9 s |
| Peak memory of the process | 2.66 GB | 2.33 GB |
| Validation ROC AUC | 0.8396 | 0.8397 |

The rounds take 5.0, 1.76, 1.18, 1.15 and 1.16 s on 1, 4, 8, 12 and 16 threads, with the same hash.

## What one parallel section costs

`sections/` starts a section of work after an idle gap, 2,000 times, under rayon and under workers
that spin. The last two columns are microseconds per section beyond what the work itself would take on
that many threads.

WSL 2:

```
threads 4; a parallel section of 4 x items after an idle gap; microseconds per section
gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us
    0    10 |      9.8 |     22.7 |      2.7 |  0.43  3.63 |    20.2     0.2
    0   100 |     99.4 |     34.4 |     25.2 |  2.89  3.94 |     9.6     0.3
    0  1000 |    999.2 |    441.9 |    254.3 |  2.26  3.93 |   192.1     4.5
   20    10 |     29.8 |    318.3 |     24.5 |  0.03  2.17 |   295.8     2.1
   20   100 |    128.5 |    208.1 |     49.9 |  0.58  3.63 |   161.0     2.8
   20  1000 |   1033.0 |    507.4 |    273.3 |  2.08  4.00 |   234.1     0.1
  100    10 |    109.9 |    431.1 |    103.4 |  0.03  2.93 |   328.7     0.9
  100   100 |    198.5 |    278.7 |    126.9 |  0.55  3.66 |   154.0     2.3
  100  1000 |   1079.4 |    599.6 |    356.3 |  1.96  3.82 |   254.8    11.4
 1000    10 |   1025.5 |   1329.5 |   1005.1 |  0.08  5.01 |   323.1    -1.3
 1000   100 |   1099.7 |   1198.1 |   1027.7 |  0.50  3.60 |   173.2     2.7
 1000  1000 |   1982.6 |   1508.9 |   1260.7 |  1.93  3.77 |   263.3    15.1
threads 8; a parallel section of 8 x items after an idle gap; microseconds per section
gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us
    0    10 |      9.7 |    297.3 |      2.1 |  0.03  4.64 |   296.1     0.9
    0   100 |    101.0 |    295.0 |     13.5 |  0.34  7.49 |   282.4     0.9
    0  1000 |   1022.7 |    648.8 |    135.5 |  1.58  7.55 |   521.0     7.6
   20    10 |     30.2 |    708.8 |     22.3 |  0.01  4.43 |   687.5     1.0
   20   100 |    120.6 |    663.7 |     33.9 |  0.16  7.25 |   631.2     1.3
   20  1000 |   1026.0 |    678.8 |    152.8 |  1.53  7.57 |   533.0     7.1
  100    10 |    114.9 |    809.8 |    102.1 |  0.02  7.10 |   707.9     0.2
  100   100 |    200.8 |    758.5 |    114.8 |  0.15  6.79 |   645.9     2.2
  100  1000 |   1107.7 |    768.3 |    235.6 |  1.51  7.43 |   542.4     9.6
 1000    10 |   1020.2 |   1719.0 |   1003.8 |  0.03  5.28 |   716.5     1.3
 1000   100 |   1106.9 |   1654.3 |   1015.0 |  0.16  7.12 |   640.9     1.6
 1000  1000 |   2001.5 |   1663.4 |   1135.2 |  1.51  7.41 |   538.2    10.0
threads 16; a parallel section of 16 x items after an idle gap; microseconds per section
gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us
    0    10 |      9.3 |   1370.2 |      2.4 |  0.01  3.86 |  1369.6     1.8
    0   100 |    100.5 |   1285.0 |      8.9 |  0.08 11.30 |  1278.8     2.6
    0  1000 |    997.3 |    861.5 |     71.1 |  1.16 14.02 |   799.2     8.8
   20    10 |     30.1 |   1371.6 |     22.7 |  0.01  3.80 |  1350.9     2.0
   20   100 |    119.4 |   1401.5 |     29.0 |  0.07 11.07 |  1375.3     2.8
   20  1000 |   1041.4 |   1137.3 |     95.3 |  0.91 13.56 |  1053.4    11.5
  100    10 |    114.4 |   1497.0 |    102.7 |  0.01  5.25 |  1396.1     1.8
  100   100 |    209.7 |   1499.4 |    109.1 |  0.08 12.04 |  1392.5     2.3
  100  1000 |   1150.5 |   1256.4 |    176.9 |  0.91 13.67 |  1090.7    11.2
 1000    10 |   1013.8 |   2456.0 |   1003.0 |  0.01  4.54 |  1455.1     2.2
 1000   100 |   1102.8 |   2485.3 |   1009.5 |  0.07 10.77 |  1478.9     3.1
 1000  1000 |   2011.9 |   2294.6 |   1074.5 |  0.78 13.58 |  1231.3    11.2
```

Windows:

```
threads 8; a parallel section of 8 x items after an idle gap; microseconds per section
gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us
    0    10 |     10.6 |      3.8 |      2.0 |  2.80  5.20 |     2.5     0.7
    0   100 |    107.9 |     15.8 |     13.2 |  6.82  8.16 |     2.3    -0.3
    0  1000 |   1007.7 |    134.1 |    128.6 |  7.51  7.84 |     8.2     2.6
   20    10 |     29.7 |     36.1 |     21.8 |  0.60  5.46 |    14.9     0.6
   20   100 |    120.4 |     48.2 |     34.2 |  3.56  7.07 |    15.7     1.6
   20  1000 |   1017.7 |    177.3 |    150.9 |  6.34  7.62 |    32.5     6.2
  100    10 |    109.7 |    117.7 |    101.7 |  0.55  5.84 |    16.5     0.4
  100   100 |    200.0 |    132.2 |    113.2 |  3.11  7.60 |    19.7     0.6
  100  1000 |   1097.9 |    262.3 |    229.7 |  6.15  7.70 |    37.5     4.9
 1000    10 |   1023.1 |   1025.5 |   1007.2 |  0.91  3.22 |    22.6     4.3
 1000   100 |   1100.8 |   1037.7 |   1013.7 |  2.67  7.37 |    25.1     1.1
 1000  1000 |   2020.7 |   1173.0 |   1129.6 |  5.90  7.88 |    45.4     2.0
threads 16; a parallel section of 16 x items after an idle gap; microseconds per section
gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us
    0    10 |      9.1 |      8.3 |      2.2 |  1.09  4.12 |     7.8     1.6
    0   100 |     96.1 |     14.6 |      7.8 |  6.58 12.30 |     8.6     1.8
    0  1000 |    957.1 |     77.3 |     67.5 | 12.38 14.18 |    17.5     7.7
   20    10 |     29.2 |     63.9 |     22.4 |  0.21  3.79 |    43.3     1.9
   20   100 |    119.3 |     69.7 |     27.4 |  2.00 13.34 |    43.4     1.2
   20  1000 |    977.3 |    133.8 |     86.3 |  8.41 14.44 |    53.9     6.5
  100    10 |    109.0 |    158.9 |    102.3 |  0.15  3.88 |    58.3     1.7
  100   100 |    194.5 |    164.7 |    108.0 |  1.46 11.82 |    58.7     2.1
  100  1000 |   1047.4 |    218.7 |    167.2 |  7.98 14.10 |    59.4     8.0
 1000    10 |   1020.6 |   1076.9 |   1002.2 |  0.27  9.42 |    75.6     0.9
 1000   100 |   1095.0 |   1076.2 |   1008.7 |  1.25 10.93 |    70.2     2.8
 1000  1000 |   1956.0 |   1124.9 |   1066.7 |  7.65 14.34 |    65.2     6.9
```

A section of 100 µs of work on 8 threads takes longer under rayon in WSL than on one thread. Days, a
network simulator in Rust, makes the same choice in its worker pool (`spin_before_park`).

## Windows

The same code on native Windows, seconds of the same `train_ms`, the median of three, with the
weights' hash. The test `the_trees_have_the_same_bytes_on_every_system` pins the bytes of all three
tasks on smaller data.

| Problem | 1 threads | 4 threads | 8 threads | 12 threads | 16 threads | Weights on Windows | The same on Linux? |
|---|---|---|---|---|---|---|---|
| california, 334 rounds | 0.176 | 0.140 | 0.142 | 0.157 | 0.177 | `a45b46ddcbc328d6` | yes |
| adult, 180 rounds | 0.382 | 0.213 | 0.193 | 0.215 | 0.249 | `9eaafb8ddd7ffa1d` | yes |
| covertype, 40 rounds | 3.705 | 1.312 | 0.909 | 0.890 | 0.887 | `cf1898624256e85a` | yes |


## What this does not show

- **A virtual machine.** WSL 2 is where waking threads costs most, so the gain from spinning is largest
  here. It is smaller on bare metal, where rayon would also do better.
- **Settings.** Each library runs with its own defaults beyond the shared settings. LightGBM keeps
  `min_sum_hessian_in_leaf` at 0.001, because 0 made it fail with `left_count > 0` on Covertype; that
  floor may explain part of its lower accuracy after 400 rounds. XGBoost has no rows-per-leaf limit, so
  it runs with `min_child_weight` 0, and its `lossguide` growth differs in details. Nothing was tuned.
  LightGBM's `deterministic` flag was off, so its repeat counts are for its default.
- **Time to the same quality.** The rounds are fixed, and the libraries reach different quality at the
  same rounds.
- **Larger data.** Covertype's 406,708 training rows are the largest here.
- **Peak memory** is not like for like; see the note on its table. Tabsmith holds the data frame it read
  and all the splits, and it uses more than the libraries on Covertype.
- **Prediction.** `reports/benchmarks/` and `reports/inference/` measure it, against scikit-learn.

## Reproduce

Download the datasets ([`reports/datasets`](../../datasets/2026-10-31-windows-x86_64/)), then with a
Python that has lightgbm, xgboost, scikit-learn, pandas and numpy, in Linux, adapt the paths at the top
of `run.sh` and run it. Then `python -I -B tools/datasets/report_training.py results.jsonl before.jsonl`.
`sections/` is a Cargo project with one argument, the thread count.

---

## 中文

**结论。** 在同一台机器上，用同样的行、切分、特征和设置，Tabsmith 的树训练速度介于 LightGBM 和 XGBoost 之间，测试质量相当或更好：在三个公开数据集中最大的一个上比 LightGBM 慢 8% 到 27%，在两个较小的数据集上是它的 1.5 到 1.8 倍时间，在三个数据集上都比 XGBoost 快。这些全是安全 Rust（`unsafe_code = "forbid"`），模型在任何线程数下、在 Windows 和 Linux 上都逐字节相同。与 0.3 版相比快 1.5 到 4.9 倍。

**方法。** `run.sh` 从工作区的全新副本编译 `tabsmith`，每个数据集在 1、4、8、12、16 个线程上各训练三次，每次一个独立进程。两边的时间都包含分箱，每轮都在验证集上算损失，不做早停，轮数固定。所有表格都由 `results.jsonl` 生成，没有手抄的数字。

**变快的原因。** 并行段不再唤醒休眠的线程（这决定了能否扩展到 8 个以上线程）；互斥的独热列共用一次遍历（Covertype 从 54 次减到 12 次）；一次遍历把行交替加到两张表，避免等待上一行的写入；每轮只算一次概率，训练损失只在固定的约 4,096 行上算。后两项之所以可行，是因为直方图是整数和：换成浮点数，模型就会变。整数精度不损失质量：30 位比例配旧阈值时，测试指标与浮点版四位小数内相同，降到约 8 位质量仍然不变。

**没有说明的。** WSL 2 是唤醒线程最贵的环境，自旋的收益在这里最大，裸机上会小一些。各库除共同设置外用各自的默认值，没有调参；LightGBM 的 `min_sum_hessian_in_leaf` 保持 0.001，因为设为 0 时它在 Covertype 上会报错，这可能是它 400 轮后准确率较低的部分原因。轮数固定，所以这不是“达到同样质量所需的时间”。Covertype 的 406,708 行是这里最大的数据。峰值内存不是同口径的，Tabsmith 在 Covertype 上用得更多。
