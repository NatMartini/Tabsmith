Machine: AMD Ryzen 9 9950X3D2 16-Core Processor, 32 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. Data: 10000 rows (synthetic equipment-defect sample), same split, features and settings on both sides. Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4. Tabsmith's linear, small-mlp and gbdt candidates stand next to scikit-learn's LogisticRegression, MLPClassifier and HistGradientBoostingClassifier (HGB). Tabsmith trains on one thread; scikit-learn runs with its defaults (HGB uses every core).

| | Tabsmith linear | scikit-learn logistic | Tabsmith small-mlp | scikit-learn MLP | Tabsmith gbdt | scikit-learn HGB |
|---|---|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 | 0.7332 | 0.7295 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 | 0.8188 | 0.8184 |
| Epochs / rounds / solver iterations | 39 | 25 | 13 | 13 | 54 | 54 |
| Training time | 0.11 s | 0.04 s | 0.10 s | 0.13 s | 0.67 s | 3.66 s |
| Single-record prediction in an app (p50) | 2.4 µs | 1993.2 µs | 2.6 µs | 2006.5 µs | 3.0 µs | 2844.1 µs |
| Single-record prediction (p99) | 5.5 µs | 2562.6 µs | 4.9 µs | 2681.7 µs | 6.7 µs | 4456.8 µs |
| Batch prediction in process | 787,675 rows/s | 736,904 rows/s | 641,489 rows/s | 400,931 rows/s | 523,179 rows/s | 731,173 rows/s |
| Cold start: new process, load model, predict 1 record | 1 ms | 618 ms | 1 ms | 612 ms | 1 ms | 625 ms |
| Peak memory of that process | 12 MB | 160 MB | 12 MB | 161 MB | 12 MB | 164 MB |
| Model artifact | 0.0 MB | 0.0 MB | 0.0 MB | 0.1 MB | 0.1 MB | 0.2 MB |

| Deployment footprint | Tabsmith | Python |
|---|---|---|
| What the application ships | 1.7 MB binary (depends on tabsmith-runtime only) | 354.1 MB environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |
| Training tool | 29.4 MB `tabsmith` binary | the same environment |
| Train all three candidates from the CSV | 0.98 s for the whole `tabsmith run` process (read, profile, split, preprocess, train, validate, checkpoints), peak 44 MB | 3.83 s for read_csv + three fits (imports and process start not included) |
