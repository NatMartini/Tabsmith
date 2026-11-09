# Models from XGBoost and LightGBM, with the libraries' own outputs

Made by [reports/import/…/reference.py](../../reports/import/2026-11-06-windows-x86_64/reference.py)
(`python reference.py fixtures`) with XGBoost 3.4.1 and LightGBM 4.7.0, from the sample data
(`tabsmith sample-data`, synthetic, CC0): each model trained on 3,000 rows, then 100 other rows,
a third of them with one or two fields set missing.

| File | What |
|---|---|
| `xgboost-{binary,multiclass,regression}.json` | `booster.save_model("….json")`: 20 rounds, depth 3 |
| `lightgbm-{binary,multiclass,regression}.txt` | `booster.save_model("….txt")`: 20 rounds, 8 leaves |
| `*.reference.json` | the rows (null for missing), and XGBoost's margins as f32 bits (`margin_bits`) or LightGBM's raw scores for the rows rounded to f32 (`raw_scores`) |

`cargo test -p tabsmith-workflow --test import` imports each model and compares: XGBoost bit for
bit, LightGBM within the bound the import reports, and a row refused exactly when it lacks a value
its field refuses to go without. `fixtures/bundles/imported-*` are two of them as bundles, which
the `model!` tests compile.
