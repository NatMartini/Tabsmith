### adult: binary, target `income`, 48,842 rows (34,190 train / 7,326 test), 6 numeric + 8 categorical inputs

| Model | ROC AUC | balanced accuracy | log loss | Training time |
|---|---|---|---|---|
| Tabsmith linear | 0.9067 | 0.8250 | 0.3842 | 0.3 s, 24 epochs/rounds |
| scikit-learn LogisticRegression | 0.9067 | 0.8244 | 0.3941 | 1.6 s, 69 iterations |
| Tabsmith small-mlp | 0.9157 | 0.8289 | 0.3693 | 0.9 s, 21 epochs/rounds |
| scikit-learn MLPClassifier | 0.9041 | 0.8205 | 0.3937 | 1.1 s, 21 iterations |
| Tabsmith gbdt (chosen) | 0.9301 | 0.8434 | 0.3308 | 17.1 s, 180 epochs/rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9297 | 0.8442 | 0.3309 | 0.8 s, 180 iterations |

### california: regression, target `median_house_value`, 20,640 rows (14,446 train / 3,097 test), 8 numeric + 0 categorical inputs

| Model | RMSE | MAE | R² | Training time |
|---|---|---|---|---|
| Tabsmith linear | 69,225 | 51,078 | 0.6433 | 0.2 s, 32 epochs/rounds |
| scikit-learn Ridge | 69,286 | 50,858 | 0.6427 | 0.0 s, 32 iterations (unweighted) |
| Tabsmith small-mlp | 53,308 | 36,614 | 0.7885 | 0.5 s, 40 epochs/rounds |
| scikit-learn MLPRegressor | 54,712 | 37,833 | 0.7772 | 0.6 s, 40 iterations (unweighted) |
| Tabsmith gbdt (chosen) | 45,817 | 30,664 | 0.8438 | 2.1 s, 334 epochs/rounds |
| scikit-learn HistGradientBoostingRegressor | 45,817 | 30,630 | 0.8438 | 0.5 s, 334 iterations (unweighted) |

### covertype: multiclass, target `cover_type`, 581,012 rows (406,708 train / 87,152 test), 54 numeric + 0 categorical inputs

| Model | ROC AUC (ovr) | balanced accuracy | accuracy | log loss | Training time |
|---|---|---|---|---|---|
| Tabsmith linear | 0.9206 | 0.6977 | 0.5774 | 0.9795 | 3.0 s, 17 epochs/rounds |
| scikit-learn LogisticRegression | 0.9264 | 0.7037 | 0.5969 | 0.9221 | 26.5 s, 345 iterations |
| Tabsmith small-mlp | 0.9634 | 0.8353 | 0.7474 | 0.5929 | 16.9 s, 41 epochs/rounds |
| scikit-learn MLPClassifier | 0.9627 | 0.8376 | 0.7407 | 0.6098 | 31.2 s, 41 iterations |
| Tabsmith gbdt (chosen) | 0.9924 | 0.9355 | 0.8995 | 0.2610 | 513.1 s, 400 epochs/rounds |
| scikit-learn HistGradientBoostingClassifier | 0.9921 | 0.9361 | 0.8984 | 0.2638 | 30.7 s, 400 iterations |

