"""Tabsmith vs. the equivalent scikit-learn workflow, measured on the same machine.

    python3 tools/bench/bench.py --tabsmith target/release/tabsmith \
        --app examples/predict-app/target/release/bench \
        --python /path/to/venv-with-sklearn/bin/python \
        --out reports/benchmarks/<date> [--rows 10000]

Both sides use the same rows, the same split, the same features and the same
training settings (see sklearn_bench.py). Each measured process runs on its own,
and its peak memory comes from the kernel (wait4 rusage). Results go to
<out>/results.json and <out>/results.md; raw logs to <out>/logs/.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None, log=None):
    """Runs `cmd` (setup steps, not measured); returns its stdout. Fails loudly."""
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True)
    if log:
        with open(log, "wb") as f:
            f.write(b"$ " + " ".join(map(str, cmd)).encode() + b"\n" + p.stdout + b"\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr.decode()[-2000:]}")
    return p.stdout.decode()


def measured(cmd, cwd=None, stdin_path=None, log=None):
    """Runs `cmd` in a fresh Python wrapper so its peak RSS is its own."""
    wrapper = [sys.executable, "-c", (
        "import resource,subprocess,sys,time,json;"
        "f=open(sys.argv[2],'rb') if sys.argv[2] else None;"
        "t=time.perf_counter();p=subprocess.run(json.loads(sys.argv[1]),stdin=f,capture_output=True);"
        "s=time.perf_counter()-t;"
        "sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr);"
        "print(json.dumps({'__rc':p.returncode,'__s':s,'__rss':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024}),file=sys.stderr)"
    ), json.dumps([str(c) for c in cmd]), stdin_path or ""]
    p = subprocess.run(wrapper, cwd=cwd, capture_output=True)
    lines = p.stderr.decode().strip().splitlines()
    meta = json.loads(lines[-1])
    if log:
        with open(log, "wb") as f:
            f.write(b"$ " + " ".join(map(str, cmd)).encode() + b"\n" + p.stdout + b"\n--- stderr ---\n" + p.stderr)
    if meta["__rc"] != 0:
        sys.exit(f"command failed ({meta['__rc']}): {' '.join(map(str, cmd))}\n" + "\n".join(lines[-30:]))
    return meta["__s"], meta["__rss"], p.stdout.decode()


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def records_from_csv(csv_path, schema_path, out_path, limit):
    """Input records with the types the bundle expects (what an application would send)."""
    fields = json.load(open(schema_path))["fields"]
    tokens = {"", "NA", "null"}
    with open(csv_path, newline="") as f, open(out_path, "w") as out:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= limit:
                break
            rec = {}
            for fd in fields:
                v = row.get(fd["name"])
                if v is None or v in tokens:
                    rec[fd["name"]] = None
                elif fd["kind"] == "numeric":
                    x = float(v)
                    rec[fd["name"]] = None if x != x else x  # NaN is sent as a missing value
                elif fd.get("raw_dtype") == "bool":
                    rec[fd["name"]] = v.lower() == "true"
                elif fd.get("raw_dtype") == "int":
                    rec[fd["name"]] = int(v)
                else:
                    rec[fd["name"]] = v
            out.write(json.dumps(rec) + "\n")


def median_of(n, fn):
    xs = [fn() for _ in range(n)]
    return statistics.median(x[0] for x in xs), max(x[1] for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabsmith", required=True)
    ap.add_argument("--app", required=True, help="examples/predict-app/target/release/bench")
    ap.add_argument("--python", required=True, help="a Python with scikit-learn, pandas, joblib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=10000)
    a = ap.parse_args()
    tabsmith, app, py = (os.path.abspath(x) for x in (a.tabsmith, a.app, a.python))
    predict_app = os.path.join(os.path.dirname(app), "predict-app")
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    logs = os.path.join(out, "logs")
    os.makedirs(logs)
    work = os.path.join(out, "work")
    os.makedirs(work)
    R = {"machine": {}, "tabsmith": {}, "python": {}}
    R["machine"]["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    R["machine"]["logical_cpus"] = os.cpu_count()
    R["machine"]["os"] = " ".join(os.uname()[0:3:2])

    # --- Tabsmith: data, training, evaluation, bundles -----------------------------------
    run([tabsmith, "init", "bench"], cwd=work, log=f"{logs}/init.txt")
    proj = os.path.join(work, "bench")
    run([tabsmith, "sample-data", "data/defects.csv", "--rows", str(a.rows)], cwd=proj, log=f"{logs}/data.txt")
    run([tabsmith, "data", "bind", "data/defects.csv", "--target", "defect", "--group", "device_id", "--id", "unit_id"],
        cwd=proj, log=f"{logs}/bind.txt")
    secs, rss, txt = measured([tabsmith, "--json", "run", "--no-cache"], cwd=proj, log=f"{logs}/run.txt")
    result = json.loads(txt)["result"]
    runs = {r["candidate"]: r for r in result["runs"]}
    R["tabsmith"]["run_wall_seconds"] = secs
    R["tabsmith"]["run_peak_rss_mb"] = rss
    R["tabsmith"]["candidates"] = {}
    evals = {}
    for cand, r in runs.items():
        # Every candidate is scored on the test split here (a benchmark, not model selection).
        txt = run([tabsmith, "--json", "evaluate", "--run", r["run_id"], "--split", "test", "--allow-test-reuse"],
                  cwd=proj, log=f"{logs}/eval-{cand}.txt")
        evals[cand] = json.loads(txt)["result"]["metrics"]
        run([tabsmith, "bundle", "build", "--run", r["run_id"]], cwd=proj, log=f"{logs}/bundle-{cand}.txt")
    data_csv = os.path.join(proj, "data", "defects.csv")
    records = os.path.join(work, "records.jsonl")
    one = os.path.join(work, "one.jsonl")
    for cand, r in runs.items():
        bundle = os.path.join(proj, "bundles", r["run_id"])
        if not os.path.exists(records):
            records_from_csv(data_csv, os.path.join(bundle, "input.schema.json"), records, 2000)
            with open(records) as f, open(one, "w") as o:
                o.write(f.readline())
        txt = run([app, bundle, records, "20000"], log=f"{logs}/app-bench-{cand}.txt")
        app_bench = json.loads(txt)
        # Cold start: a fresh process that loads the bundle and predicts one record.
        cold, cold_rss = median_of(7, lambda: measured([predict_app, bundle], stdin_path=one))
        # Batch prediction of the whole CSV through the CLI (read + transform + predict + write).
        cli_s, cli_rss, _ = measured([tabsmith, "predict", "--bundle", bundle, "--input", data_csv, "--output",
                                      os.path.join(work, f"preds-{cand}.csv")], cwd=proj)
        m = evals[cand]
        R["tabsmith"]["candidates"][cand] = {
            "run_id": r["run_id"],
            "train_seconds": r["timings"]["train_ms"] / 1e3,
            "data_seconds": r["timings"]["data_ms"] / 1e3,
            "fit_transform_seconds": r["timings"]["fit_transform_ms"] / 1e3,
            "best_epoch": r["metrics"]["best_epoch"],
            "epochs": len(r["history"]),
            "test": {"balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"], "log_loss": m["log_loss"]},
            "app_single_record_us": app_bench["single_record_us"],
            "app_batch_rows_per_second": app_bench["batch"]["rows_per_second"],
            "app_load_ms": app_bench["load_ms"],
            "cold_start_seconds": cold,
            "cold_start_peak_rss_mb": cold_rss,
            "cli_predict_csv_seconds": cli_s,
            "cli_predict_peak_rss_mb": cli_rss,
            "bundle_bytes": du(bundle),
        }
    R["tabsmith"]["cli_binary_bytes"] = os.path.getsize(tabsmith)
    R["tabsmith"]["app_binary_bytes"] = os.path.getsize(predict_app)

    # --- scikit-learn: same rows, split, features and settings -----------------------
    sk_out = os.path.join(work, "sklearn")
    secs, rss, _ = measured([py, "-I", os.path.join(HERE, "sklearn_bench.py"), proj, runs["linear"]["run_id"],
                             runs["small-mlp"]["run_id"], runs["gbdt"]["run_id"], records, sk_out],
                            log=f"{logs}/sklearn.txt")
    sk = json.load(open(os.path.join(sk_out, "sklearn.json")))
    sk["process_wall_seconds"] = secs
    sk["process_peak_rss_mb"] = rss
    for cand in CANDIDATES:
        model = os.path.join(sk_out, f"sklearn_{cand}.joblib")
        code = ("import json,sys,joblib,pandas as pd;m=joblib.load(sys.argv[1]);"
                "r=[json.loads(l) for l in sys.stdin];print(m.predict_proba(pd.DataFrame(r))[:,1][0])")
        cold, cold_rss = median_of(7, lambda: measured([py, "-I", "-c", code, model], stdin_path=one))
        sk["candidates"][cand]["cold_start_seconds"] = cold
        sk["candidates"][cand]["cold_start_peak_rss_mb"] = cold_rss
    prefix = os.path.dirname(os.path.dirname(py))
    sp = run([py, "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"])
    site = sp.strip()
    pkgs = {}
    for name in sorted(os.listdir(site)):
        p = os.path.join(site, name)
        if os.path.isdir(p) and not name.endswith((".dist-info", "__pycache__")):
            pkgs[name] = du(p)
    R["python"] = sk
    R["python"]["environment_bytes"] = du(prefix)
    R["python"]["site_packages_bytes"] = pkgs
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(markdown(R, a.rows))
    print(open(os.path.join(out, "results.md")).read())


CANDIDATES = ("linear", "small-mlp", "gbdt")


def markdown(R, rows):
    rp, sk = R["tabsmith"], R["python"]
    mb = lambda b: f"{b / 1e6:.1f} MB"
    lines = [f"Machine: {R['machine']['cpu']}, {R['machine']['logical_cpus']} logical CPUs, {R['machine']['os']}. "
             f"Data: {rows} rows (synthetic equipment-defect sample), same split, features and settings on both sides. "
             f"Python {sk['versions']['python']}, scikit-learn {sk['versions']['scikit-learn']}, "
             f"pandas {sk['versions']['pandas']}, numpy {sk['versions']['numpy']}. "
             "Tabsmith's linear, small-mlp and gbdt candidates stand next to scikit-learn's LogisticRegression, "
             "MLPClassifier and HistGradientBoostingClassifier (HGB). Tabsmith trains on one thread; "
             "scikit-learn runs with its defaults (HGB uses every core).", ""]
    lines += ["| | Tabsmith linear | scikit-learn logistic | Tabsmith small-mlp | scikit-learn MLP | Tabsmith gbdt | scikit-learn HGB |",
              "|---|---|---|---|---|---|---|"]
    pairs = [(rp["candidates"][c], sk["candidates"][c]) for c in CANDIDATES]
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(f"{f(r, 'r')} | {f(s, 's')}" for r, s in pairs) + " |")
    row("Test balanced accuracy", lambda c, k: f"{c['test']['balanced_accuracy']:.4f}")
    row("Test ROC AUC", lambda c, k: f"{c['test']['roc_auc']:.4f}")
    row("Epochs / rounds / solver iterations", lambda c, k: str(c['epochs'] if k == 'r' else c['iterations']))
    row("Training time", lambda c, k: f"{(c['train_seconds'] if k == 'r' else c['fit_seconds']):.2f} s")
    row("Single-record prediction in an app (p50)", lambda c, k: f"{(c['app_single_record_us'] if k == 'r' else c['single_record_us'])['p50']:.1f} µs")
    row("Single-record prediction (p99)", lambda c, k: f"{(c['app_single_record_us'] if k == 'r' else c['single_record_us'])['p99']:.1f} µs")
    row("Batch prediction in process", lambda c, k: f"{(c['app_batch_rows_per_second'] if k == 'r' else c['batch']['rows_per_second']):,.0f} rows/s")
    row("Cold start: new process, load model, predict 1 record", lambda c, k: f"{c['cold_start_seconds'] * 1e3:.0f} ms")
    row("Peak memory of that process", lambda c, k: f"{c['cold_start_peak_rss_mb']:.0f} MB")
    row("Model artifact", lambda c, k: mb(c['bundle_bytes'] if k == 'r' else c['model_file_bytes']))
    lines += ["", "| Deployment footprint | Tabsmith | Python |", "|---|---|---|",
              f"| What the application ships | {mb(rp['app_binary_bytes'])} binary (depends on tabsmith-runtime only) | "
              f"{mb(sk['environment_bytes'])} environment (Python + numpy + scipy + scikit-learn + pandas + joblib) |",
              f"| Training tool | {mb(rp['cli_binary_bytes'])} `tabsmith` binary | the same environment |",
              f"| Train all three candidates from the CSV | {rp['run_wall_seconds']:.2f} s for the whole `tabsmith run` "
              f"process (read, profile, split, preprocess, train, validate, checkpoints), peak {rp['run_peak_rss_mb']:.0f} MB | "
              f"{sk['read_csv_seconds'] + sum(s['fit_seconds'] for _, s in pairs):.2f} s for read_csv + three fits "
              f"(imports and process start not included) |"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
