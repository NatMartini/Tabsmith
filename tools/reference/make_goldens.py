"""Generate frozen scikit-learn reference values ("goldens") for RustPy tests.

Usage:
    python make_goldens.py <export-dir> <out-goldens.json>

<export-dir> comes from
    cargo run -p rustpy-workflow --example export_reference -- <project> <linear-run-id> <export-dir>
on a project bound to fixtures/reference/defects_1500.csv (see tools/reference/README.md).

The goldens freeze, for the fixture's split:
  * the transformed validation matrix (StandardScaler NaN-aware fit on train,
    0-imputation in standardized space, OneHotEncoder handle_unknown="ignore"),
  * metrics for a fixed probability vector,
  * the optimum of the linear objective (LogisticRegression lbfgs, balanced
    class weights, C = 1 / (n_train * l2)).
"""

import json
import math
import sys

import numpy as np
import sklearn
from sklearn import metrics as M
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sklearn_reference import load, objective  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402


def main():
    export_dir, out = sys.argv[1], sys.argv[2]
    run, num, cat, raw_tr, raw_va, x_cols, xs, preds = load(export_dir)
    ct = ColumnTransformer(
        [
            ("num", make_pipeline(StandardScaler(), SimpleImputer(strategy="constant", fill_value=0.0)), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ]
    )
    ct.fit(raw_tr)
    ref = ct.transform(raw_va)
    names = []
    for n in ct.get_feature_names_out():
        kind, rest = n.split("__", 1)
        if kind == "num":
            names.append(rest)
        else:
            col = next(c for c in cat if rest.startswith(c + "_"))
            value = rest[len(col) + 1 :]
            names.append(f"{col}=<missing>" if value == "nan" else f"{col}={value}")

    # Fixed, deterministic probabilities (with ties and extreme values) for metric parity.
    y = [int(v) for v in preds["valid"]["y"]]
    p = [round(1 / (1 + math.exp(-3 * math.sin(i * 0.7))), 6) for i in range(len(y))]
    p[0], p[1] = 1.0, 0.0
    pa, ya = np.array(p), np.array(y)
    yhat = (pa >= 0.5).astype(int)
    metric_values = {
        "accuracy": M.accuracy_score(ya, yhat),
        "balanced_accuracy": M.balanced_accuracy_score(ya, yhat),
        "precision": M.precision_score(ya, yhat, zero_division=0),
        "recall": M.recall_score(ya, yhat, zero_division=0),
        "f1": M.f1_score(ya, yhat, zero_division=0),
        "roc_auc": M.roc_auc_score(ya, pa),
        "log_loss": M.log_loss(ya, pa),
        "brier": M.brier_score_loss(ya, pa),
    }

    x_tr, y_tr = xs["train"], np.array(preds["train"]["y"])
    l2 = run["train_config"]["l2"]
    n = len(y_tr)
    n_pos = y_tr.sum()
    sw = np.where(y_tr == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))
    clf = LogisticRegression(C=1.0 / (n * l2), class_weight="balanced", solver="lbfgs", tol=1e-12, max_iter=50000)
    clf.fit(x_tr, y_tr)
    optimum = objective(clf.coef_[0], clf.intercept_[0], x_tr, y_tr, sw, l2)

    goldens = {
        "generated_by": "tools/reference/make_goldens.py",
        "reference": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "fixture": "fixtures/reference/defects_1500.csv",
        "transform": {
            "names": names,
            "valid_rows": [[float(v) for v in row] for row in ref],
            "tolerance": 1e-5,
        },
        "metrics": {"y": y, "p": p, "threshold": 0.5, "values": metric_values, "tolerance": 1e-12},
        "linear_objective": {"l2": l2, "n_train": n, "sklearn_optimum": optimum, "max_relative_gap": 0.01},
    }
    json.dump(goldens, open(out, "w"), indent=1)
    print(f"wrote {out}: {ref.shape[0]}x{ref.shape[1]} matrix, optimum {optimum:.6f}")


if __name__ == "__main__":
    main()
