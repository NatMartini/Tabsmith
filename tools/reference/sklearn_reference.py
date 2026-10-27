"""Independent reference checks of a RustPy run with scikit-learn.

Usage:
    python sklearn_reference.py <export-dir> <result.json>

<export-dir> is produced by
    cargo run -p rustpy-workflow --example export_reference -- <project> <run-id> <export-dir>

Checks:
1. Preprocessing parity: scikit-learn's StandardScaler (NaN-aware, fitted on the
   training rows) + constant-0 imputation in standardized space, and
   OneHotEncoder(handle_unknown="ignore"), fitted on the same raw training rows,
   must reproduce RustPy's transformed validation matrix column by column.
2. Metric parity: scikit-learn metrics on RustPy's validation probabilities must
   equal the metrics RustPy recorded for the run.
3. Linear objective (linear runs only): scikit-learn LogisticRegression (lbfgs,
   class_weight="balanced", C = 1 / (n_train * l2)) minimises the same objective
   as RustPy; the script reports both objective values and validation quality.
   RustPy keeps the epoch with the lowest validation loss (early stopping), so its
   objective is expected to be close to, not equal to, the optimum.

Pinned reference versions are recorded in the output.
"""

import json
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics as M
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load(export_dir):
    run = json.load(open(f"{export_dir}/rustpy_run.json"))
    features = run["fitted"]["features"]
    num = [f["name"] for f in features if f["kind"] == "numeric"]
    cat = [f["name"] for f in features if f["kind"] == "categorical"]
    dtypes = {c: str for c in cat}
    read = lambda name: pd.read_csv(
        f"{export_dir}/raw_{name}.csv", dtype=dtypes, keep_default_na=False, na_values=["", "NaN"]
    )
    x_cols = pd.read_csv(f"{export_dir}/x_valid.csv", nrows=0).columns.tolist()
    xs = {n: np.loadtxt(f"{export_dir}/x_{n}.csv", delimiter=",", skiprows=1, ndmin=2) for n in ("train", "valid")}
    preds = {n: json.load(open(f"{export_dir}/rustpy_{n}.json")) for n in ("train", "valid")}
    return run, num, cat, read("train"), read("valid"), x_cols, xs, preds


def preprocessing_parity(num, cat, raw_tr, raw_va, x_cols, x_valid):
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
    missing_in_ref = sorted(set(x_cols) - set(names))
    missing_in_ours = sorted(set(names) - set(x_cols))
    diffs = {}
    for j, n in enumerate(names):
        if n in x_cols:
            k = x_cols.index(n)
            diffs[n] = float(np.max(np.abs(ref[:, j] - x_valid[:, k])))
    return {
        "columns_rustpy": len(x_cols),
        "columns_sklearn": len(names),
        "only_in_rustpy": missing_in_ref,
        "only_in_sklearn": missing_in_ours,
        "max_abs_diff": max(diffs.values()) if diffs else None,
        "worst_column": max(diffs, key=diffs.get) if diffs else None,
        "pass": not missing_in_ref and not missing_in_ours and max(diffs.values()) < 1e-5,
        "tolerance": 1e-5,
        "note": "RustPy matrices are f32; sklearn is f64",
    }


def metric_parity(run, preds):
    y = np.array(preds["valid"]["y"])
    p = np.array(preds["valid"]["probability"])
    t = run["threshold"]
    yhat = (p >= t).astype(int)
    ref = {
        "accuracy": M.accuracy_score(y, yhat),
        "balanced_accuracy": M.balanced_accuracy_score(y, yhat),
        "precision": M.precision_score(y, yhat, zero_division=0),
        "recall": M.recall_score(y, yhat, zero_division=0),
        "f1": M.f1_score(y, yhat, zero_division=0),
        "roc_auc": M.roc_auc_score(y, p),
        "log_loss": M.log_loss(y, p),
        "brier": M.brier_score_loss(y, p),
    }
    ours = run["metrics"]["valid"]
    diffs = {k: abs(ref[k] - ours[k]) for k in ref}
    tn, fp, fn, tp = M.confusion_matrix(y, yhat).ravel()
    cm_ok = ours["confusion"] == {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    return {
        "sklearn": ref,
        "rustpy": {k: ours[k] for k in ref},
        "abs_diff": diffs,
        "confusion_matrix_equal": bool(cm_ok),
        "pass": cm_ok and max(diffs.values()) < 1e-9,
        "tolerance": 1e-9,
    }


def objective(w, b, x, y, sw, l2):
    z = x @ w + b
    loss = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(sw * loss) + 0.5 * l2 * np.sum(w * w))


def linear_objective(run, xs, preds):
    if run["candidate"] != "linear":
        return {"skipped": "not a linear run"}
    x_tr, y_tr = xs["train"], np.array(preds["train"]["y"])
    x_va, y_va = xs["valid"], np.array(preds["valid"]["y"])
    l2 = run["train_config"]["l2"]
    n = len(y_tr)
    n_pos = y_tr.sum()
    sw = np.where(y_tr == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))
    clf = LogisticRegression(C=1.0 / (n * l2), class_weight="balanced", solver="lbfgs", tol=1e-12, max_iter=50000)
    clf.fit(x_tr, y_tr)
    w_ref, b_ref = clf.coef_[0], clf.intercept_[0]
    layer = run["layers"][0]
    w_rp = np.array(layer["weight"], dtype=np.float64)
    b_rp = float(layer["bias"][0])
    j_ref = objective(w_ref, b_ref, x_tr, y_tr, sw, l2)
    j_rp = objective(w_rp, b_rp, x_tr, y_tr, sw, l2)
    p_ref = clf.predict_proba(x_va)[:, 1]
    p_rp = np.array(preds["valid"]["probability"])
    t = run["threshold"]
    return {
        "objective_sklearn_optimum": j_ref,
        "objective_rustpy": j_rp,
        "relative_gap": (j_rp - j_ref) / j_ref,
        "valid_balanced_accuracy": {
            "sklearn": M.balanced_accuracy_score(y_va, (p_ref >= t).astype(int)),
            "rustpy": M.balanced_accuracy_score(y_va, (p_rp >= t).astype(int)),
        },
        "valid_roc_auc": {"sklearn": M.roc_auc_score(y_va, p_ref), "rustpy": M.roc_auc_score(y_va, p_rp)},
        "valid_probability_correlation": float(np.corrcoef(p_ref, p_rp)[0, 1]),
        "valid_max_abs_probability_diff": float(np.max(np.abs(p_ref - p_rp))),
        "note": "RustPy minimises the same objective with minibatch Adam and keeps the best-validation epoch; sklearn is the converged lbfgs optimum",
    }


def main():
    export_dir, out = sys.argv[1], sys.argv[2]
    run, num, cat, raw_tr, raw_va, x_cols, xs, preds = load(export_dir)
    result = {
        "reference": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "run_id": run["run_id"],
        "candidate": run["candidate"],
        "preprocessing": preprocessing_parity(num, cat, raw_tr, raw_va, x_cols, xs["valid"]),
        "metrics": metric_parity(run, preds),
        "linear_objective": linear_objective(run, xs, preds),
    }
    json.dump(result, open(out, "w"), indent=2, default=float)
    print(json.dumps({k: result[k].get("pass") if isinstance(result[k], dict) else result[k] for k in result}, indent=2))


if __name__ == "__main__":
    main()
