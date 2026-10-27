Machine: AMD Ryzen 9 9950X3D2 16-Core Processor, 32 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. Data: 10000 rows (synthetic equipment-defect sample), same split, features and settings on both sides. Python 3.14.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4.

| | RustPy linear | scikit-learn LogisticRegression | RustPy small-mlp | scikit-learn MLPClassifier |
|---|---|---|---|---|
| Test balanced accuracy | 0.7528 | 0.7516 | 0.7729 | 0.7314 |
| Test ROC AUC | 0.8202 | 0.8209 | 0.8250 | 0.7928 |
| Epochs / solver iterations | 39 | 25 | 13 | 13 |
| Training time | 0.15 s | 0.03 s | 0.14 s | 0.15 s |
| Single-record prediction in an app (p50) | 2.5 µs | 2004.0 µs | 2.8 µs | 2069.9 µs |
| Single-record prediction (p99) | 4.3 µs | 2587.8 µs | 5.3 µs | 3397.3 µs |
| Batch prediction in process | 825,063 rows/s | 805,501 rows/s | 685,760 rows/s | 353,438 rows/s |
| Cold start: new process, load model, predict 1 record | 1 ms | 637 ms | 1 ms | 639 ms |
| Peak memory of that process | 12 MB | 160 MB | 12 MB | 161 MB |
| Model artifact | 0.0 MB | 0.0 MB | 0.0 MB | 0.1 MB |

| Deployment footprint | RustPy | Python |
|---|---|---|
| What the application ships | 1.6 MB binary (depends on rustpy-runtime only) | 354.2 MB environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |
| Training tool | 28.7 MB `rustpy` binary | the same environment |
| Train both candidates from the CSV | 0.38 s for the whole `rustpy run` process (read, profile, split, preprocess, train, validate, checkpoints), peak 44 MB | 0.19 s for read_csv + both fits (imports and process start not included) |
