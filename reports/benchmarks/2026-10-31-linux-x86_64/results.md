Machine: AMD Ryzen 9 9950X3D2 16-Core Processor, 32 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. Data: 10000 rows (synthetic equipment-defect sample), same split, features and settings on both sides. Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4. Tabsmith's linear, small-mlp and gbdt candidates stand next to scikit-learn's LogisticRegression, MLPClassifier and HistGradientBoostingClassifier (HGB). Tabsmith's networks train on one thread and its trees on up to 12; scikit-learn runs with its defaults.

| | Tabsmith linear | scikit-learn logistic | Tabsmith small-mlp | scikit-learn MLP | Tabsmith gbdt | scikit-learn HGB |
|---|---|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 | 0.7332 | 0.7295 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 | 0.8188 | 0.8184 |
| Epochs / rounds / solver iterations | 39 | 25 | 13 | 13 | 54 | 54 |
| Training time | 0.09 s | 0.03 s | 0.11 s | 0.14 s | 0.12 s | 3.53 s |
| Single-record prediction in an app (p50) | 2.5 µs | 2021.0 µs | 2.8 µs | 1990.0 µs | 3.0 µs | 2592.0 µs |
| Single-record prediction (p99) | 3.3 µs | 3474.9 µs | 4.1 µs | 2680.0 µs | 3.9 µs | 4450.4 µs |
| Batch prediction from records | 4,608,335 rows/s | 638,634 rows/s | 4,130,750 rows/s | 377,607 rows/s | 3,544,374 rows/s | 446,299 rows/s |
| Batch prediction from a table in memory | 7,044,560 rows/s | 913,099 rows/s | 5,878,694 rows/s | 538,933 rows/s | 4,625,413 rows/s | 663,986 rows/s |
| Batch prediction from records, Tabsmith on one thread | 1,255,742 rows/s | – | 961,563 rows/s | – | 690,197 rows/s | – |
| Cold start: new process, load model, predict 1 record | 1 ms | 570 ms | 1 ms | 550 ms | 1 ms | 573 ms |
| Peak memory of that process | 12 MB | 162 MB | 12 MB | 162 MB | 12 MB | 165 MB |
| Model artifact | 0.0 MB | 0.0 MB | 0.0 MB | 0.1 MB | 0.1 MB | 0.2 MB |

| Deployment footprint | Tabsmith | Python |
|---|---|---|
| What the application ships | 1.9 MB binary (depends on tabsmith-runtime only) | 354.1 MB environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |
| Training tool | 29.9 MB `tabsmith` binary | the same environment |
| Train all three candidates from the CSV | 0.51 s for the whole `tabsmith run` process (read, profile, split, preprocess, train, validate, checkpoints), peak 49 MB | 3.71 s for read_csv + three fits (imports and process start not included) |
