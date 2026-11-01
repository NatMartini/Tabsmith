"""scikit-learn's HistGradientBoosting on the same datasets and settings, timing the fit.
Usage: hgb_time.py <datasets dir> <rounds for covertype>
Thread count comes from OMP_NUM_THREADS. Adult is one-hot encoded like Tabsmith's
features; the trees do not care about standardisation."""
import csv
import os
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

data, rounds = sys.argv[1], int(sys.argv[2])
print("sklearn", sklearn.__version__, "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "default"), flush=True)


def numeric_csv(path, target):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    arr = np.array(rows[1:], dtype=np.float64)
    t = head.index(target)
    return np.delete(arr, t, axis=1), arr[:, t]


rng = np.random.default_rng(7)
common = dict(learning_rate=0.1, max_leaf_nodes=31, max_depth=8, min_samples_leaf=20, max_bins=255, l2_regularization=1.0, early_stopping=False, random_state=7)


def fit(model, x, y, name, n_rounds):
    idx = rng.permutation(len(y))[: int(0.7 * len(y))]
    t0 = time.perf_counter()
    model.fit(x[idx], y[idx])
    s = time.perf_counter() - t0
    print(f"{name}: {len(idx)} rows, {x.shape[1]} features, {n_rounds} rounds: {s:.2f} s, {s * 1e3 / n_rounds:.1f} ms per round", flush=True)


x, y = numeric_csv(os.path.join(data, "california.csv"), "median_house_value")
fit(HistGradientBoostingRegressor(max_iter=334, **common), x, y, "california", 334)

with open(os.path.join(data, "adult.csv"), newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
head, rows = rows[0], rows[1:]
t = head.index("income")
y = np.array([r[t].strip().rstrip(".") == ">50K" for r in rows], dtype=np.int32)
cols = []
for j, name in enumerate(head):
    if j == t:
        continue
    vals = [r[j].strip() for r in rows]
    try:
        cols.append(np.array([float(v) if v not in ("", "NA", "?") else np.nan for v in vals], dtype=np.float64)[:, None])
    except ValueError:
        cats = sorted({v for v in vals if v not in ("", "NA", "?")})
        index = {c: i for i, c in enumerate(cats)}
        m = np.zeros((len(vals), len(cats)), dtype=np.float64)
        for r, v in enumerate(vals):
            if v in index:
                m[r, index[v]] = 1.0
        cols.append(m)
x = np.nan_to_num(np.hstack(cols), nan=0.0)
fit(HistGradientBoostingClassifier(max_iter=180, class_weight="balanced", **common), x, y, "adult", 180)

x, y = numeric_csv(os.path.join(data, "covertype.csv"), "cover_type")
fit(HistGradientBoostingClassifier(max_iter=rounds, class_weight="balanced", **common), x, y.astype(np.int32), "covertype", rounds)
