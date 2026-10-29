"""Turns the outputs of the dataset study into the tables of its report.

    python -I tools/datasets/report.py <runs-dir> <out-dir>

<runs-dir>/<dataset>/ holds, for each dataset: train.json (the `tabsmith train --json`
output), eval-<run>.json (one `tabsmith --json evaluate --split test` per run) and
sklearn.json (tools/datasets/compare.py). Writes <out-dir>/results.json and
<out-dir>/results.md; the report's README embeds the Markdown.
"""

import glob
import json
import os
import sys

METRICS = {
    "binary": [("roc_auc", "ROC AUC"), ("balanced_accuracy", "balanced accuracy"), ("log_loss", "log loss")],
    "multiclass": [("roc_auc", "ROC AUC (ovr)"), ("balanced_accuracy", "balanced accuracy"), ("accuracy", "accuracy"), ("log_loss", "log loss")],
    "regression": [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")],
}
PAIRS = [("linear", "linear"), ("small-mlp", "small-mlp"), ("gbdt", "gbdt")]
PYTHON_NAMES = {
    "binary": {"linear": "LogisticRegression", "small-mlp": "MLPClassifier", "gbdt": "HistGradientBoostingClassifier"},
    "multiclass": {"linear": "LogisticRegression", "small-mlp": "MLPClassifier", "gbdt": "HistGradientBoostingClassifier"},
    "regression": {"linear": "Ridge", "small-mlp": "MLPRegressor", "gbdt": "HistGradientBoostingRegressor"},
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, key):
    if v is None:
        return "—"
    if key in ("rmse", "mae"):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}"
    return f"{v:.4f}"


def main():
    runs_dir, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # the tables hold "R²"; a GBK console would choke
    except (AttributeError, ValueError):
        pass
    results, md = {}, []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if not os.path.isfile(os.path.join(d, "sklearn.json")):
            continue
        sk = load(os.path.join(d, "sklearn.json"))
        train = load(os.path.join(d, "train.json"))["result"]
        task = sk["task"]
        runs = {r["candidate"]: r for r in train["group"]["runs"]}
        evals = {}
        for p in glob.glob(os.path.join(d, "eval-*.json")):
            e = load(p)
            rec = e.get("result", e)
            subject = rec.get("subject", {})
            run_id = subject.get("run_id") if isinstance(subject, dict) else None
            cand = next((c for c, r in runs.items() if r["run_id"] == run_id), None)
            if cand:
                evals[cand] = rec
        chosen = train.get("chosen")
        entry = {"task": task, "target": sk["target"], "rows": sk["rows"], "train_rows": sk["train_rows"],
                 "test_rows": sk["test_rows"], "features": sk["features"], "chosen": chosen, "candidates": {}}
        md.append(f"### {name}: {task}, target `{sk['target']}`, {sk['rows']:,} rows "
                  f"({sk['train_rows']:,} train / {sk['test_rows']:,} test), "
                  f"{sk['features']['numeric']} numeric + {sk['features']['categorical']} categorical inputs\n")
        head = "| Model | " + " | ".join(label for _, label in METRICS[task]) + " | Training time |"
        md.append(head)
        md.append("|---|" + "---|" * (len(METRICS[task]) + 1))
        for ts_name, py_name in PAIRS:
            r, e, s = runs.get(ts_name), evals.get(ts_name), sk["candidates"].get(py_name)
            if r and e:
                m = e["metrics"]
                cells = [fmt(m.get(k), k) for k, _ in METRICS[task]]
                secs = r.get("timings", {}).get("train_ms", 0) / 1000
                rounds = len(r.get("history", []))
                star = " (chosen)" if r["run_id"] == chosen else ""
                md.append(f"| Tabsmith {ts_name}{star} | " + " | ".join(cells) + f" | {secs:.1f} s, {rounds} epochs/rounds |")
                entry["candidates"][f"tabsmith:{ts_name}"] = {"test": {k: m.get(k) for k, _ in METRICS[task]}, "train_seconds": secs, "epochs": rounds}
            if s:
                cells = [fmt(s["test"].get(k), k) for k, _ in METRICS[task]]
                md.append(f"| scikit-learn {PYTHON_NAMES[task][py_name]} | " + " | ".join(cells)
                          + f" | {s['fit_seconds']:.1f} s, {s['iterations']} iterations{'' if s.get('weighted', True) else ' (unweighted)'} |")
                entry["candidates"][f"sklearn:{py_name}"] = {"test": s["test"], "fit_seconds": s["fit_seconds"], "iterations": s["iterations"]}
        md.append("")
        results[name] = entry
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
