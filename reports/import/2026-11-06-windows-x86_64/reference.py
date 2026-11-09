"""Reference models and the libraries' own outputs for `tabsmith import`.

    reference.py fixtures   small XGBoost and LightGBM models, rows with missing values and the
                            libraries' raw outputs, for the repository's tests (fixtures/import)
    reference.py evidence   models of 300 trees on 100,000 rows, and binary XGBoost models whose
                            base score glibc's logf rounds differently from the correctly rounded
                            value (scratch/import/evidence)

XGBoost outputs are margins as f32 bits; LightGBM outputs are raw scores as f64, for the rows
rounded to f32 (the bundle's inputs) and, in the evidence, for the rows as they are.
Writes only those two folders."""
import ctypes, json, math, os, platform, sys
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "..", ".."))
NUM = ["hours_in_service", "cycles", "temp_mean", "temp_std", "temp_max", "vibration_rms", "vibration_peak",
       "pressure_mean", "pressure_std", "current_mean", "current_std", "humidity", "voltage_drop",
       "ambient_noise_db", "calibration_offset"]
TASKS = {"binary": ("defect", "binary:logistic", "binary"),
         "multiclass": ("failure_mode", "multi:softprob", "multiclass"),
         "regression": ("remaining_hours", "reg:squarederror", "regression")}
VERSIONS = {"xgboost": xgb.__version__, "lightgbm": lgb.__version__, "numpy": np.__version__,
            "glibc": " ".join(os.confstr("CS_GNU_LIBC_VERSION").split()[1:]), "python": platform.python_version()}


def data(path, task):
    df = pd.read_csv(path)
    X = df[NUM].astype("float64")
    y = df[TASKS[task][0]]
    labels = None
    if task == "multiclass":
        cat = y.astype("category")
        labels = [str(c) for c in cat.cat.categories]
        y = cat.cat.codes
    return X, y.to_numpy(), labels


def with_missing(X, rng, share):
    """Sets one or two random fields of a share of the rows to NaN."""
    X = X.copy().reset_index(drop=True)
    for r in range(len(X)):
        if rng.random() < share:
            for c in rng.choice(len(NUM), size=int(rng.integers(1, 3)), replace=False):
                X.iat[r, int(c)] = np.nan
    return X


def records(X):
    return [{k: (None if math.isnan(v) else float(v)) for k, v in row.items()} for row in X.to_dict("records")]


def to_f32(X):
    return X.astype(np.float32).astype(np.float64)


def xgb_model(X, y, task, rounds, depth, extra=None):
    params = {"objective": TASKS[task][1], "max_depth": depth, "eta": 0.3, "tree_method": "hist", "seed": 1, "nthread": 1}
    if task == "multiclass":
        params["num_class"] = int(y.max()) + 1
    params.update(extra or {})
    return xgb.train(params, xgb.DMatrix(X, label=y, missing=np.nan), num_boost_round=rounds)


def xgb_margins(b, X):
    m = np.asarray(b.predict(xgb.DMatrix(X, missing=np.nan), output_margin=True), dtype=np.float32)
    return m.reshape(len(X), -1)


def lgb_model(X, y, task, rounds, leaves):
    params = {"objective": TASKS[task][2], "num_leaves": leaves, "learning_rate": 0.3, "verbose": -1, "seed": 1,
              "deterministic": True, "num_threads": 1}
    if task == "multiclass":
        params["num_class"] = int(y.max()) + 1
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)


def lgb_raw(b, X):
    return np.asarray(b.predict(X, raw_score=True), dtype=np.float64).reshape(len(X), -1)


def dump(path, obj):
    with open(path, "w", newline="\n") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def fixtures():
    out = os.path.join(repo, "fixtures", "import")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(11)
    for task, file in [("binary", "defects.csv"), ("multiclass", "modes.csv"), ("regression", "hours.csv")]:
        X, y, labels = data(os.path.join(here, file), task)
        train, rows = X.iloc[:3000], with_missing(X.iloc[3000:3100], rng, 0.3)
        b = xgb_model(train, y[:3000], task, rounds=20, depth=3)
        name = f"xgboost-{task}"
        b.save_model(os.path.join(out, name + ".json"))
        m = xgb_margins(b, rows)
        dump(os.path.join(out, name + ".reference.json"),
             {"model": name + ".json", "library": "xgboost", "versions": VERSIONS, "task": task, "labels": labels,
              "rows": records(rows), "margin_bits": m.view(np.uint32).tolist()})
        l = lgb_model(train, y[:3000], task, rounds=20, leaves=8)
        name = f"lightgbm-{task}"
        l.save_model(os.path.join(out, name + ".txt"))
        rows32 = to_f32(rows)
        dump(os.path.join(out, name + ".reference.json"),
             {"model": name + ".txt", "library": "lightgbm", "versions": VERSIONS, "task": task, "labels": labels,
              "rows": records(rows32), "raw_scores": lgb_raw(l, rows32).tolist()})
        print(task, "fixtures written:", sorted(f for f in os.listdir(out) if task in f))


def walk_xgb(tree, row):
    i = 0
    while tree["left_children"][i] != -1:
        v = row[tree["split_indices"][i]]
        if np.isnan(v):
            i = tree["left_children"][i] if tree["default_left"][i] else tree["right_children"][i]
        elif np.float32(v) < np.float32(tree["split_conditions"][i]):
            i = tree["left_children"][i]
        else:
            i = tree["right_children"][i]
    return np.float32(tree["split_conditions"][i])


def evidence():
    out = os.path.join(here, "evidence")
    os.makedirs(out, exist_ok=True)
    summary = {"versions": VERSIONS, "models": []}
    rng = np.random.default_rng(13)
    for task in ["binary", "multiclass", "regression"]:
        X, y, labels = data(os.path.join(here, f"big-{task}.csv"), task)
        rows = with_missing(X, rng, 0.1)
        rows_file = f"rows-{task}.json"
        dump(os.path.join(out, rows_file), records(rows))
        b = xgb_model(X.iloc[:50000], y[:50000], task, rounds=300, depth=6)
        name = f"xgboost-{task}"
        b.save_model(os.path.join(out, name + ".json"))
        m = xgb_margins(b, rows)
        inplace = np.asarray(b.inplace_predict(rows.to_numpy(), predict_type="margin", missing=np.nan),
                             dtype=np.float32).reshape(len(rows), -1)
        dump(os.path.join(out, name + ".reference.json"),
             {"model": name + ".json", "library": "xgboost", "versions": VERSIONS, "task": task, "labels": labels,
              "rows": rows_file, "margin_bits": m.view(np.uint32).tolist()})
        summary["models"].append({"name": name, "trees": int(b.num_boosted_rounds()) * (int(y.max()) + 1 if task == "multiclass" else 1),
                                  "rows": len(rows), "inplace_predict_differs": int((inplace.view(np.uint32) != m.view(np.uint32)).sum())})
        l = lgb_model(X.iloc[:50000], y[:50000], task, rounds=300, leaves=63)
        name = f"lightgbm-{task}"
        l.save_model(os.path.join(out, name + ".txt"))
        dump(os.path.join(out, name + ".reference.json"),
             {"model": name + ".txt", "library": "lightgbm", "versions": VERSIONS, "task": task, "labels": labels,
              "rows": rows_file, "raw_scores": lgb_raw(l, to_f32(rows)).tolist(),
              "raw_scores_f64_inputs": lgb_raw(l, rows).tolist()})
        summary["models"].append({"name": name, "trees": l.num_trees(), "rows": len(rows)})
        print(task, "evidence written", flush=True)

    # Base scores whose starting margin -logf(1/p - 1) glibc rounds differently from the
    # correctly rounded value: XGBoost on Linux starts from glibc's.
    libm = ctypes.CDLL("libm.so.6")
    libm.logf.restype, libm.logf.argtypes = ctypes.c_float, [ctypes.c_float]
    pick = np.random.default_rng(17)
    hard, tried = [], 0
    while len(hard) < 12:
        tried += 1
        p = np.float32(pick.uniform(0.02, 0.98))
        b32 = np.float32(np.float32(1.0) / p) - np.float32(1.0)
        glibc, exact = np.float32(libm.logf(float(b32))), np.float32(math.log(float(b32)))
        if glibc != exact:
            hard.append(float(p))
    X, y, _ = data(os.path.join(here, "defects.csv"), "binary")
    rows = with_missing(X.iloc[3000:4000], rng, 0.1)
    rows_file = "rows-logf.json"
    dump(os.path.join(out, rows_file), records(rows))
    off = 0
    for k, p in enumerate(hard):
        b = xgb_model(X.iloc[:3000], y[:3000], "binary", rounds=5, depth=3, extra={"base_score": p})
        name = f"xgboost-logf-{k:02d}"
        path = os.path.join(out, name + ".json")
        b.save_model(path)
        m = xgb_margins(b, rows)
        dump(os.path.join(out, name + ".reference.json"),
             {"model": name + ".json", "library": "xgboost", "versions": VERSIONS, "task": "binary", "labels": None,
              "rows": rows_file, "margin_bits": m.view(np.uint32).tolist(), "base_score": p})
        # The same sums from the correctly rounded start: how many margins would be off.
        model = json.load(open(path))
        trees = model["learner"]["gradient_booster"]["model"]["trees"]
        p32 = np.float32(json.loads(model["learner"]["learner_model_param"]["base_score"])[0])
        start = -np.float32(math.log(float(np.float32(np.float32(1.0) / p32) - np.float32(1.0))))
        for r, row in enumerate(rows.to_numpy()):
            acc = np.float32(start)
            for tree in trees:
                acc = np.float32(acc + walk_xgb(tree, row))
            off += int(acc.view(np.uint32) != m[r, 0].view(np.uint32))
    summary["logf"] = {"base_scores": hard, "tried": tried, "rows": len(rows),
                       "margins_off_with_correct_rounding": off, "margins": len(rows) * len(hard)}
    dump(os.path.join(out, "summary.json"), summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"fixtures": fixtures, "evidence": evidence}[sys.argv[1] if len(sys.argv) > 1 else "fixtures"]()
