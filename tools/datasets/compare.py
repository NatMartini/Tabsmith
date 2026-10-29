"""scikit-learn on the same rows, split and settings as a Tabsmith project, for any of
the three tasks: the Python side of reports/datasets/.

    python -I tools/datasets/compare.py <tabsmith-project> <out.json>

Reads the project's newest completed run of each candidate (linear, small-mlp, gbdt),
its fitted transform, split and snapshot, trains the scikit-learn equivalent of each
candidate on the training rows with the same features (standardised numbers, one-hot
categories, the same regularisation, as many epochs or rounds as Tabsmith's run
trained) and scores the test rows with Tabsmith's metrics, in the target's units for a
regression. Needs scikit-learn, pandas and numpy; writes one JSON file.
"""

import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics as M
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    warnings.filterwarnings("ignore")
    project, out_path = sys.argv[1:3]
    state = os.path.join(project, ".tabsmith")
    runs = {}
    for d in sorted(glob.glob(os.path.join(state, "runs", "run-*"))):
        try:
            run = load_json(os.path.join(d, "run.json"))
        except OSError:
            continue
        if run.get("status") == "completed":
            runs[run["candidate"]] = run
    if not runs:
        sys.exit("no completed runs in the project")
    ref = next(iter(runs.values()))
    fitted = load_json(os.path.join(state, "transforms", ref["transform_id"], "fitted.json"))
    split = load_json(os.path.join(state, "splits", ref["split_id"] + ".json"))
    resolved = load_json(os.path.join(state, "runs", ref["run_id"], "resolved.json"))
    null_values = resolved["config"]["data"].get("null_values", ["", "NA", "null"])
    data_csv = os.path.join(state, "snapshots", ref["snapshot_id"], "data.csv")
    out = fitted["output_schema"]
    task, target = out.get("task", "binary"), out["target"]
    num = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "numeric"]
    cat = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "categorical"]
    text_cols = {f["name"]: str for f in fitted["input_schema"]["fields"] if f["kind"] == "categorical"}
    if task != "regression":
        text_cols[target] = str

    df = pd.read_csv(data_csv, dtype=text_cols, keep_default_na=False, na_values=null_values)
    assign = np.frombuffer(split["assignment"].encode(), dtype=np.uint8) - ord("0")
    tr, te = assign == 0, assign == 2
    x = df[num + cat]
    if task == "binary":
        y = (df[target].astype(str) == out["positive_label"]).astype(int).to_numpy()
        k = 2
    elif task == "multiclass":
        index = {c: i for i, c in enumerate(out["classes"])}
        y = df[target].astype(str).map(index).to_numpy()
        k = len(out["classes"])
    else:
        y = df[target].astype(float).to_numpy()
        k = 0
    n_tr = int(tr.sum())
    if task == "regression":
        sample_weight = None
    else:
        # Tabsmith's default class_weight = "balanced": n / (k * n_class).
        counts = np.bincount(y[tr], minlength=k)
        sample_weight = n_tr / (k * counts[y[tr]])

    def preprocess():
        return ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), num),
            # Dense: HistGradientBoosting takes no sparse input, and Tabsmith's one-hot is dense too.
            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="<missing>"),
                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
        ])

    results = {"task": task, "target": target, "rows": int(len(df)), "train_rows": n_tr, "test_rows": int(te.sum()),
               "features": {"numeric": len(num), "categorical": len(cat)},
               "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
               "candidates": {}}
    for cand, run in runs.items():
        cfg = run["train_config"]
        l2, epochs = cfg["l2"], len(run["history"])
        if cand == "linear":
            if task == "regression":
                # Tabsmith: mean squared error of the standardised target + l2/2 * sum(W^2).
                model = TransformedTargetRegressor(regressor=Ridge(alpha=n_tr * l2 / 2.0), transformer=StandardScaler())
            else:
                model = LogisticRegression(C=1.0 / (n_tr * l2), solver="lbfgs", max_iter=1000)
        elif cand == "gbdt":
            t = cfg["trees"]
            kw = dict(learning_rate=cfg["learning_rate"], max_iter=epochs, max_leaf_nodes=t["max_leaves"],
                      max_depth=t["max_depth"], min_samples_leaf=t["min_samples_leaf"], max_bins=min(t["max_bins"], 255),
                      l2_regularization=l2, early_stopping=False, random_state=7)
            model = HistGradientBoostingRegressor(**kw) if task == "regression" else HistGradientBoostingClassifier(**kw)
        else:
            kw = dict(hidden_layer_sizes=tuple(cfg["hidden"]), activation="relu", solver="adam", alpha=l2,
                      batch_size=cfg["batch_size"], learning_rate_init=cfg["learning_rate"], max_iter=epochs,
                      shuffle=True, random_state=7, early_stopping=False, n_iter_no_change=epochs + 1, tol=0.0)
            if task == "regression":
                model = TransformedTargetRegressor(regressor=MLPRegressor(**kw), transformer=StandardScaler())
            else:
                model = MLPClassifier(**kw)
        pipe = make_pipeline(preprocess(), model)
        last = pipe.steps[-1][0]
        fit_kw = {}
        if sample_weight is not None:
            fit_kw = {f"{last}__sample_weight": sample_weight}
            if cand == "linear" and task == "regression":
                fit_kw = {}
        t0 = time.perf_counter()
        try:
            pipe.fit(x[tr], y[tr], **fit_kw)
        except TypeError:
            # An estimator without sample_weight support: fit unweighted and say so.
            pipe.fit(x[tr], y[tr])
            fit_kw = {}
        fit_s = time.perf_counter() - t0
        yt = y[te]
        if task == "regression":
            pred = pipe.predict(x[te])
            quality = {"rmse": float(np.sqrt(M.mean_squared_error(yt, pred))), "mae": float(M.mean_absolute_error(yt, pred)),
                       "r2": float(M.r2_score(yt, pred))}
        elif task == "binary":
            p = pipe.predict_proba(x[te])[:, 1]
            quality = {"balanced_accuracy": float(M.balanced_accuracy_score(yt, (p >= 0.5).astype(int))),
                       "roc_auc": float(M.roc_auc_score(yt, p)), "log_loss": float(M.log_loss(yt, p))}
        else:
            p = pipe.predict_proba(x[te])
            quality = {"balanced_accuracy": float(M.balanced_accuracy_score(yt, p.argmax(axis=1))),
                       "accuracy": float(M.accuracy_score(yt, p.argmax(axis=1))),
                       "roc_auc": float(M.roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=list(range(k)))),
                       "log_loss": float(M.log_loss(yt, p, labels=list(range(k))))}
        inner = model.regressor_ if hasattr(model, "regressor_") else model
        n_iter = getattr(inner, "n_iter_", None)
        results["candidates"][cand] = {
            "fit_seconds": fit_s, "weighted": bool(fit_kw), "test": quality,
            "iterations": int(np.max(n_iter)) if n_iter is not None else epochs,
            "tabsmith_epochs_or_rounds": epochs,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["candidates"], indent=1))


if __name__ == "__main__":
    main()
