"""The scikit-learn side of tools/bench/bench.py.

Trains the Python equivalent of RustPy's two candidates on exactly the same
rows, features and settings, and measures what a Python user would measure.

    python sklearn_bench.py <rustpy-project> <linear-run> <mlp-run> <records.jsonl> <out-dir>

Writes <out-dir>/sklearn.json and <out-dir>/sklearn_<candidate>.joblib.
"""

import json
import os
import resource
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics as M
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_dir(project, run_id):
    return os.path.join(project, ".rustpy", "runs", run_id)


def main():
    warnings.filterwarnings("ignore", message=".*Stochastic Optimizer: Maximum iterations.*")
    project, linear_run, mlp_run, records_path, out_dir = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)
    runs = {"linear": load_json(os.path.join(run_dir(project, linear_run), "run.json")),
            "small-mlp": load_json(os.path.join(run_dir(project, mlp_run), "run.json"))}
    ref = runs["linear"]
    state = os.path.join(project, ".rustpy")
    fitted = load_json(os.path.join(state, "transforms", ref["transform_id"], "fitted.json"))
    split = load_json(os.path.join(state, "splits", ref["split_id"] + ".json"))
    data_csv = os.path.join(state, "snapshots", ref["snapshot_id"], "data.csv")
    target = fitted["output_schema"]["target"]
    positive = fitted["output_schema"]["positive_label"]
    num = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "numeric"]
    cat = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "categorical"]
    text_cols = {f["name"]: str for f in fitted["input_schema"]["fields"] if f["kind"] == "categorical"}

    t0 = time.perf_counter()
    df = pd.read_csv(data_csv, dtype=text_cols | {target: str}, keep_default_na=False,
                     na_values=["", "NA", "null"])
    read_s = time.perf_counter() - t0
    assign = np.frombuffer(split["assignment"].encode(), dtype=np.uint8) - ord("0")
    y = (df[target].astype(str) == positive).astype(int).to_numpy()
    x = df[num + cat]
    tr, te = assign == 0, assign == 2
    n_tr = int(tr.sum())
    pos = y[tr].sum()
    sample_weight = np.where(y[tr] == 1, n_tr / (2 * pos), n_tr / (2 * (n_tr - pos)))

    def preprocess():
        return ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), num),
            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="<missing>"),
                                  OneHotEncoder(handle_unknown="ignore")), cat),
        ])

    results = {"versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
               "rows": int(len(df)), "train_rows": n_tr, "test_rows": int(te.sum()),
               "read_csv_seconds": read_s, "candidates": {}}
    records = [json.loads(line) for line in open(records_path) if line.strip()]
    one_rows = [pd.DataFrame([r]) for r in records[:2000]]
    for cand, run in runs.items():
        cfg = run["train_config"]
        l2 = cfg["l2"]
        if cand == "linear":
            # Same objective as RustPy's linear candidate: balanced class weights, C = 1 / (n * l2).
            model = LogisticRegression(C=1.0 / (n_tr * l2), solver="lbfgs", max_iter=1000)
        else:
            # Same amount of work as RustPy: as many epochs as its run actually trained
            # (it stops early on the validation split and keeps the best epoch).
            epochs = len(run["history"])
            model = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), activation="relu", solver="adam",
                                  alpha=l2, batch_size=cfg["batch_size"], learning_rate_init=cfg["learning_rate"],
                                  max_iter=epochs, shuffle=True, random_state=7, early_stopping=False,
                                  n_iter_no_change=epochs + 1, tol=0.0)
        pipe = make_pipeline(preprocess(), model)
        t0 = time.perf_counter()
        last = pipe.steps[-1][0]
        pipe.fit(x[tr], y[tr], **{f"{last}__sample_weight": sample_weight})
        fit_s = time.perf_counter() - t0
        p_test = pipe.predict_proba(x[te])[:, 1]
        yt = y[te]
        quality = {"balanced_accuracy": M.balanced_accuracy_score(yt, (p_test >= 0.5).astype(int)),
                   "roc_auc": M.roc_auc_score(yt, p_test), "log_loss": M.log_loss(yt, p_test)}
        # Batch: the whole table in one call.
        t0 = time.perf_counter()
        rounds = 5
        for _ in range(rounds):
            pipe.predict_proba(x)
        batch_s = (time.perf_counter() - t0) / rounds
        # Single record: a one-row DataFrame per call, as a request handler would do.
        for d in one_rows[:50]:
            pipe.predict_proba(d)
        lat = []
        for d in one_rows:
            t = time.perf_counter()
            pipe.predict_proba(d)
            lat.append((time.perf_counter() - t) * 1e6)
        lat.sort()
        q = lambda f: lat[round((len(lat) - 1) * f)]
        path = os.path.join(out_dir, f"sklearn_{cand}.joblib")
        joblib.dump(pipe, path)
        results["candidates"][cand] = {
            "fit_seconds": fit_s, "test": quality,
            "iterations": int(np.max(model.n_iter_)),
            "batch": {"rows": int(len(df)), "seconds": batch_s, "rows_per_second": len(df) / batch_s},
            "single_record_us": {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "iterations": len(lat)},
            "model_file_bytes": os.path.getsize(path),
        }
    results["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(os.path.join(out_dir, "sklearn.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
