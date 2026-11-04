"""One fit of LightGBM or XGBoost on the rows, split and settings of a Tabsmith project: the Python
side of reports/training/. Every fit is a process of its own, so that thread pools and peak memory
belong to this fit alone.

    python -I tools/datasets/compare_gbm.py <project> <lightgbm|lightgbm-quantized|xgboost> <threads> <rounds> <eval|noeval> <out.jsonl>

Reads the project's newest completed gbdt run, its fitted transform, split and snapshot, builds the
features as Tabsmith does (standardised numbers, one-hot categories) and trains on the training rows
with the same learning rate, leaves, depth, leaf size, bins, L2, balanced class weights and
standardised regression target. `eval` scores the validation rows every round, as Tabsmith's trainer
does for early stopping; there is no early stopping here, the rounds are fixed. Appends one JSON line:
seconds, peak memory, test metrics, and a hash of the raw test margins, computed on one thread so that
the hash speaks of the trees and not of the prediction. Needs lightgbm, xgboost, scikit-learn, pandas
and numpy, and Linux for the peak memory.
"""

import glob
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    warnings.filterwarnings("ignore")
    project, library, threads, rounds, evaluate, out_path = sys.argv[1:7]
    threads, rounds, evaluate = int(threads), int(rounds), evaluate == "eval"
    state = os.path.join(project, ".tabsmith")
    run = None
    for d in sorted(glob.glob(os.path.join(state, "runs", "run-*"))):
        if not os.path.isdir(d):
            continue
        r = load_json(os.path.join(d, "run.json"))
        if r.get("status") == "completed" and r["candidate"] == "gbdt":
            run = r
    if run is None:
        sys.exit("no completed gbdt run in the project")
    fitted = load_json(os.path.join(state, "transforms", run["transform_id"], "fitted.json"))
    split = load_json(os.path.join(state, "splits", run["split_id"] + ".json"))
    resolved = load_json(os.path.join(state, "runs", run["run_id"], "resolved.json"))
    null_values = resolved["config"]["data"].get("null_values", ["", "NA", "null"])
    out = fitted["output_schema"]
    task, target = out.get("task", "binary"), out["target"]
    num = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "numeric"]
    cat = [f["name"] for f in fitted["transform"]["features"] if f["kind"] == "categorical"]
    text_cols = {f["name"]: str for f in fitted["input_schema"]["fields"] if f["kind"] == "categorical"}
    if task != "regression":
        text_cols[target] = str
    df = pd.read_csv(os.path.join(state, "snapshots", run["snapshot_id"], "data.csv"), dtype=text_cols,
                     keep_default_na=False, na_values=null_values)
    assign = np.frombuffer(split["assignment"].encode(), dtype=np.uint8) - ord("0")
    tr, va, te = assign == 0, assign == 1, assign == 2
    if task == "binary":
        y = (df[target].astype(str) == out["positive_label"]).astype(np.int32).to_numpy()
        k = 2
    elif task == "multiclass":
        index = {c: i for i, c in enumerate(out["classes"])}
        y = df[target].astype(str).map(index).to_numpy().astype(np.int32)
        k = len(out["classes"])
    else:
        y = df[target].astype(np.float64).to_numpy()
        k = 0
    transformers = []
    if num:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), num))
    if cat:
        transformers.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="<missing>"),
                                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    pre = ColumnTransformer(transformers)
    pre.fit(df[num + cat][tr])
    x = pre.transform(df[num + cat]).astype(np.float32)
    del df
    if task == "regression":
        mean, std = y[tr].mean(), y[tr].std()
        label = (y - mean) / std
        weight = None
    else:
        mean, std = 0.0, 1.0
        label = y
        counts = np.bincount(y[tr], minlength=k)
        weight = (tr.sum() / (k * counts[y])).astype(np.float32)
    cfg = run["train_config"]
    t = cfg["trees"]
    w_tr = None if weight is None else weight[tr]
    w_va = None if weight is None else weight[va]

    # Peak memory of the fit itself, the feature matrix included: the transient float64 copy of
    # the preparation above would hide it otherwise.
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")

    t0 = time.perf_counter()
    if library.startswith("lightgbm"):
        import lightgbm as lgb

        version = lgb.__version__
        params = dict(learning_rate=cfg["learning_rate"], num_leaves=t["max_leaves"], max_depth=t["max_depth"],
                      min_data_in_leaf=t["min_samples_leaf"], max_bin=min(t["max_bins"], 255), lambda_l2=cfg["l2"],
                      num_threads=threads, seed=7, verbosity=-1)
        if task == "binary":
            params |= dict(objective="binary", metric="binary_logloss")
        elif task == "multiclass":
            params |= dict(objective="multiclass", num_class=k, metric="multi_logloss")
        else:
            params |= dict(objective="regression", metric="rmse")
        if library == "lightgbm-quantized":
            # The paper's setting: few gradient bins, leaves refit with the real gradients.
            params |= dict(use_quantized_grad=True, quant_train_renew_leaf=True)
        dtrain = lgb.Dataset(x[tr], label=label[tr], weight=w_tr, params=params, free_raw_data=True)
        valid = [lgb.Dataset(x[va], label=label[va], weight=w_va, reference=dtrain)] if evaluate else None
        booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=[])
        fit_s = time.perf_counter() - t0
        margins = booster.predict(x[te], raw_score=True, num_threads=1)
    else:
        import xgboost as xgb

        version = xgb.__version__
        params = dict(tree_method="hist", grow_policy="lossguide", eta=cfg["learning_rate"], max_leaves=t["max_leaves"],
                      max_depth=t["max_depth"], min_child_weight=0.0, max_bin=min(t["max_bins"], 255),
                      reg_lambda=cfg["l2"], nthread=threads, seed=7, verbosity=0, device="cpu")
        if task == "binary":
            params |= dict(objective="binary:logistic", eval_metric="logloss")
        elif task == "multiclass":
            params |= dict(objective="multi:softprob", num_class=k, eval_metric="mlogloss")
        else:
            params |= dict(objective="reg:squarederror", eval_metric="rmse")
        dtrain = xgb.DMatrix(x[tr], label=label[tr], weight=w_tr, nthread=threads)
        evals = [(xgb.DMatrix(x[va], label=label[va], weight=w_va, nthread=threads), "valid")] if evaluate else []
        booster = xgb.train(params, dtrain, num_boost_round=rounds, evals=evals, verbose_eval=False)
        fit_s = time.perf_counter() - t0
        booster.set_param({"nthread": 1})
        margins = booster.predict(xgb.DMatrix(x[te], nthread=1), output_margin=True)
    with open("/proc/self/status") as f:
        peak_mb = next(int(line.split()[1]) for line in f if line.startswith("VmHWM:")) / 1024.0

    margins = np.asarray(margins, dtype=np.float64)
    digest = hashlib.sha256(margins.tobytes()).hexdigest()[:16]
    yt = y[te]
    if task == "regression":
        pred = margins * std + mean
        quality = {"rmse": float(np.sqrt(M.mean_squared_error(yt, pred))), "mae": float(M.mean_absolute_error(yt, pred)),
                   "r2": float(M.r2_score(yt, pred))}
    elif task == "binary":
        p = 1.0 / (1.0 + np.exp(-margins))
        quality = {"balanced_accuracy": float(M.balanced_accuracy_score(yt, (p >= 0.5).astype(int))),
                   "roc_auc": float(M.roc_auc_score(yt, p)), "log_loss": float(M.log_loss(yt, p))}
    else:
        z = margins.reshape(-1, k)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        quality = {"balanced_accuracy": float(M.balanced_accuracy_score(yt, p.argmax(axis=1))),
                   "accuracy": float(M.accuracy_score(yt, p.argmax(axis=1))),
                   "roc_auc": float(M.roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=list(range(k)))),
                   "log_loss": float(M.log_loss(yt, p, labels=list(range(k))))}
    line = dict(library=library, version=version, project=os.path.basename(project.rstrip("/")), task=task, threads=threads,
                rounds=rounds, evaluate=evaluate, fit_seconds=fit_s, peak_mb=peak_mb, margins=digest, test=quality,
                train_rows=int(tr.sum()), features=int(x.shape[1]))
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(library, line["project"], f"t={threads} r={rounds} eval={evaluate}: {fit_s:.3f} s, {peak_mb:.0f} MB, {digest}", {a: round(b, 4) for a, b in quality.items()}, flush=True)


if __name__ == "__main__":
    main()
