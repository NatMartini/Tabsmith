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
