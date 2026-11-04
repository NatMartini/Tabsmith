Machine: AMD Ryzen 9 9950X3D2 16-Core Processor, 32 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. Data: 10000 rows (synthetic equipment-defect sample), same split, features and settings on both sides. Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4. Tabsmith's linear, small-mlp and gbdt candidates stand next to scikit-learn's LogisticRegression, MLPClassifier and HistGradientBoostingClassifier (HGB). Tabsmith's networks train on one thread and its trees on up to 12; scikit-learn runs with its defaults.

| | Tabsmith linear | scikit-learn logistic | Tabsmith small-mlp | scikit-learn MLP | Tabsmith gbdt | scikit-learn HGB |
|---|---|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 | 0.7356 | 0.7260 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 | 0.8220 | 0.8172 |
| Epochs / rounds / solver iterations | 39 | 25 | 13 | 13 | 62 | 62 |
| Training time | 0.09 s | 0.05 s | 0.13 s | 0.15 s | 0.08 s | 3.93 s |
| Single-record prediction in an app (p50) | 2.6 µs | 2326.3 µs | 2.9 µs | 2413.1 µs | 3.7 µs | 3709.3 µs |
| Single-record prediction (p99) | 5.2 µs | 4450.8 µs | 6.0 µs | 4909.9 µs | 14.2 µs | 6217.3 µs |
| Batch prediction from records | 4,659,031 rows/s | 448,437 rows/s | 3,458,234 rows/s | 214,183 rows/s | 2,742,566 rows/s | 340,452 rows/s |
| Batch prediction from a table in memory | 6,462,791 rows/s | 713,870 rows/s | 5,048,756 rows/s | 373,519 rows/s | 3,622,087 rows/s | 603,976 rows/s |
| Batch prediction from records, Tabsmith on one thread | 921,550 rows/s | – | 662,739 rows/s | – | 441,195 rows/s | – |
| Cold start: new process, load model, predict 1 record | 1 ms | 738 ms | 1 ms | 770 ms | 1 ms | 797 ms |
| Peak memory of that process | 12 MB | 162 MB | 12 MB | 162 MB | 12 MB | 165 MB |
| Model artifact | 0.0 MB | 0.0 MB | 0.0 MB | 0.1 MB | 0.1 MB | 0.3 MB |

| Deployment footprint | Tabsmith | Python |
|---|---|---|
| What the application ships | 1.9 MB binary (depends on tabsmith-runtime only) | 354.4 MB environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |
| Training tool | 29.9 MB `tabsmith` binary | the same environment |
| Train all three candidates from the CSV | 0.49 s for the whole `tabsmith run` process (read, profile, split, preprocess, train, validate, checkpoints), peak 47 MB | 4.15 s for read_csv + three fits (imports and process start not included) |
