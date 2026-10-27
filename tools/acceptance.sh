#!/usr/bin/env bash
# End-to-end acceptance run of the RustPy CPU workflow with real measurements.
#
#   tools/acceptance.sh <rustpy-binary> <work-dir> <evidence-dir> [python-with-sklearn]
#
# Writes one JSON line per step to <evidence-dir>/steps.jsonl and the raw outputs
# to <evidence-dir>/logs/. Nothing here is simulated except where a step says so
# (the crash test uses the hidden --simulate-crash-after-steps hook).
set -u
BIN=$(realpath "$1")
WORK=$(realpath -m "$2")
EVID=$(realpath -m "$3")
PY=${4:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$WORK" "$EVID"
mkdir -p "$WORK" "$EVID/logs"
STEPS="$EVID/steps.jsonl"
FAILED=0

# step <name> <expected-exit> <dir> <cmd...>
step() {
  local name=$1 expect=$2 dir=$3
  shift 3
  local log="$EVID/logs/$name"
  local t0 t1 code
  t0=$(date +%s.%N)
  (cd "$dir" && "$@") >"$log.out" 2>"$log.err"
  code=$?
  t1=$(date +%s.%N)
  local secs
  secs=$(python3 -c "print(round($t1 - $t0, 3))")
  local ok=true
  if [ "$code" != "$expect" ]; then ok=false; FAILED=$((FAILED + 1)); fi
  python3 - "$name" "$code" "$expect" "$secs" "$ok" "$*" >>"$STEPS" <<'PY'
import json, sys
name, code, expect, secs, ok, cmd = sys.argv[1:]
print(json.dumps({"step": name, "exit": int(code), "expected_exit": int(expect), "seconds": float(secs), "ok": ok == "true", "command": cmd}))
PY
  printf '%-34s exit %-3s (expected %-3s) %7ss %s\n' "$name" "$code" "$expect" "$secs" "$([ $ok = true ] && echo ok || echo FAIL)"
}

jget() { python3 -c "import json,sys; d=json.load(open('$1')); print(eval('d'+sys.argv[1]))" "$2"; }

echo "== environment"
"$BIN" --version | tee "$EVID/logs/version.txt"
"$BIN" doctor --json >"$EVID/doctor.json"
uname -a >"$EVID/logs/uname.txt"
grep -m1 "model name" /proc/cpuinfo >>"$EVID/logs/uname.txt" || true
nproc >>"$EVID/logs/uname.txt"

echo "== first workflow (sample data, 10,000 rows)"
step 01-init 0 "$WORK" "$BIN" init defect-model --sample-data
P="$WORK/defect-model"
step 02-plan 0 "$P" "$BIN" plan
step 03-plan-json 0 "$P" "$BIN" plan --json
step 04-run 0 "$P" "$BIN" run
step 05-run-json-cached 0 "$P" "$BIN" --json run --candidates linear
step 06-compare 0 "$P" "$BIN" compare --runs run-001 run-002
"$BIN" --project "$P" compare --runs run-001 run-002 --json >"$EVID/compare_v1.json"
BEST=$(jget "$EVID/compare_v1.json" "['result']['recommended']")
echo "recommended: $BEST"
step 07-evaluate-test 0 "$P" "$BIN" evaluate --run "$BEST" --split test
step 08-bundle-build 0 "$P" "$BIN" bundle build --run "$BEST"
step 09-bundle-check 0 "$P" "$BIN" bundle check "bundles/$BEST"
"$BIN" --project "$P" bundle check "bundles/$BEST" --json >"$EVID/bundle_check_v1.json"
step 10-predict-csv 0 "$P" "$BIN" predict --bundle "bundles/$BEST" --input data/defects.csv --output preds.csv
step 11-report 0 "$P" "$BIN" report
"$BIN" --project "$P" runs --json >"$EVID/runs_v1.json"
cp -r "$P/bundles/$BEST" "$EVID/bundle_v1"

echo "== standalone application (rustpy-runtime only)"
APP="$REPO/examples/predict-app"
step 12-app-build 0 "$APP" cargo build --release --quiet
(cd "$APP" && cargo tree --edges normal) >"$EVID/logs/app_cargo_tree.txt" 2>&1
python3 -c "
import json
recs=json.load(open('$P/bundles/$BEST/examples/input.json'))['records']
print('\n'.join(json.dumps(r) for r in recs))" >"$WORK/records.jsonl"
step 13-app-predict 0 "$APP" ./target/release/predict-app "$P/bundles/$BEST"
(cd "$APP" && ./target/release/predict-app "$P/bundles/$BEST" <"$WORK/records.jsonl") >"$EVID/app_predictions.txt" 2>&1
step 14-app-typed 0 "$APP" ./target/release/typed "$P/bundles/$BEST"

echo "== second iteration: new data delivery at the same path"
step 20-new-data 0 "$P" "$BIN" sample-data data/defects.csv --rows 12000 --seed 43 --shifted
step 21-run-v2 0 "$P" "$BIN" run
"$BIN" --project "$P" runs --json >"$EVID/runs_v2.json"
LAST=$(python3 -c "
import json; runs=json.load(open('$EVID/runs_v2.json'))['result']
print(' '.join(r['run_id'] for r in runs[-2:]))")
step 22-compare-v2 0 "$P" "$BIN" compare --runs $LAST
"$BIN" --project "$P" compare --runs $LAST --json >"$EVID/compare_v2.json"
BEST2=$(jget "$EVID/compare_v2.json" "['result']['recommended']")
step 23-old-bundle-on-new-valid 0 "$P" "$BIN" evaluate --bundle "bundles/$BEST" --split valid
step 24-new-run-on-new-valid 0 "$P" "$BIN" evaluate --run "$BEST2" --split valid
step 25-cross-snapshot-compare 0 "$P" "$BIN" compare --runs "$BEST" "$BEST2"
step 26-evaluate-test-v2 0 "$P" "$BIN" evaluate --run "$BEST2" --split test
step 27-bundle-build-v2 0 "$P" "$BIN" bundle build --run "$BEST2"
mkdir -p "$WORK/broken" && cp -r "$P/bundles/$BEST2/." "$WORK/broken/"
printf 'x' >>"$WORK/broken/model.safetensors"
step 28-hot-swap-and-rollback 0 "$APP" ./target/release/hot_swap "$P/bundles/$BEST" "$P/bundles/$BEST2" "$WORK/broken"
step 29-old-bundle-still-valid 0 "$P" "$BIN" bundle check "bundles/$BEST"
step 30-report-v2 0 "$P" "$BIN" report
cp "$P/.rustpy/reports/index.html" "$EVID/report.html"
cp "$P/.rustpy/reports/report.json" "$EVID/report.json"

echo "== exact resume in an independent process (R2)"
step 40-init-a 0 "$WORK" "$BIN" init resume-a --sample-data
step 41-init-b 0 "$WORK" "$BIN" init resume-b --sample-data
step 42-uninterrupted 0 "$WORK/resume-b" "$BIN" run --candidates small-mlp
step 43-crash-after-130-steps 137 "$WORK/resume-a" "$BIN" run --candidates small-mlp --simulate-crash-after-steps 130
step 44-runs-shows-interrupted 0 "$WORK/resume-a" "$BIN" runs
step 45-resume 0 "$WORK/resume-a" "$BIN" resume run-001
"$BIN" --project "$WORK/resume-a" runs --json >"$EVID/resume_a_runs.json"
"$BIN" --project "$WORK/resume-b" runs --json >"$EVID/resume_b_runs.json"
python3 - "$EVID" >"$EVID/resume_check.json" <<'PY'
import json, sys
e = sys.argv[1]
a = json.load(open(f"{e}/resume_a_runs.json"))["result"]
b = json.load(open(f"{e}/resume_b_runs.json"))["result"]
resumed = [r for r in a if r.get("lineage", {}) and r["lineage"]["level"] == "R2"][0]
ref = b[0]
sha = lambda r: [x["sha256"] for x in r["artifacts"] if x["name"] == "model.safetensors"][0]
print(json.dumps({
    "parent": a[0]["run_id"], "parent_status": a[0]["status"], "parent_steps": a[0]["steps_done"],
    "resumed_from_step": resumed["lineage"]["from_step"], "resumed_run": resumed["run_id"],
    "resumed_model_sha256": sha(resumed), "uninterrupted_model_sha256": sha(ref),
    "bitwise_identical_weights": sha(resumed) == sha(ref),
    "identical_history": resumed["history"] == ref["history"],
}, indent=2))
PY
cat "$EVID/resume_check.json"

echo "== Ctrl-C (real SIGINT) and resume"
step 50-init-c 0 "$WORK" "$BIN" init cancel-c --sample-data
C="$WORK/cancel-c"
(cd "$C" && exec "$BIN" run --candidates small-mlp --set train.small-mlp.epochs=400 --set train.small-mlp.early_stopping_patience=0 >"$EVID/logs/51-sigint.out" 2>"$EVID/logs/51-sigint.err") &
PID=$!
sleep 3
kill -INT $PID
wait $PID
echo "{\"step\": \"51-sigint\", \"exit\": $?, \"expected_exit\": 130, \"note\": \"SIGINT after 3 s\"}" >>"$STEPS"
step 52-runs-after-sigint 0 "$C" "$BIN" runs
step 53-resume-after-sigint 0 "$C" "$BIN" resume run-001 --set train.small-mlp.epochs=400 --set train.small-mlp.early_stopping_patience=0
"$BIN" --project "$C" runs --json >"$EVID/cancel_runs.json"

echo "== determinism: two fresh projects, same data and config"
step 60-init-d1 0 "$WORK" "$BIN" init det-1 --sample-data
step 61-init-d2 0 "$WORK" "$BIN" init det-2 --sample-data
step 62-run-d1 0 "$WORK/det-1" "$BIN" run
step 63-run-d2 0 "$WORK/det-2" "$BIN" run
python3 - "$WORK" >"$EVID/determinism.json" <<'PY'
import json, subprocess, sys
w = sys.argv[1]
def shas(p):
    out = []
    for r in sorted(__import__("os").listdir(f"{w}/{p}/.rustpy/runs")):
        f = f"{w}/{p}/.rustpy/runs/{r}/run.json"
        try:
            run = json.load(open(f))
        except Exception:
            continue
        out.append({a["name"]: a["sha256"] for a in run["artifacts"]})
    return out
a, b = shas("det-1"), shas("det-2")
print(json.dumps({"det_1": a, "det_2": b, "identical": a == b}, indent=2))
PY
cat "$EVID/determinism.json"

echo "== preset failures (each must be actionable)"
F="$WORK/defect-model"
fail_step() { # name expected-exit expected-code cmd...
  local name=$1 exit=$2 code=$3
  shift 3
  step "$name" "$exit" "$F" "$@"
  python3 - "$EVID/logs/$name.out" "$code" "$name" >>"$EVID/failures.jsonl" <<'PY'
import json, sys
path, code, name = sys.argv[1:]
try:
    v = json.load(open(path))
    e = v.get("error", {})
except Exception as ex:
    e = {"parse_error": str(ex)}
print(json.dumps({"scenario": name, "expected_code": code, "code": e.get("code"), "message": e.get("message"),
                  "next_action": e.get("next_action"), "retained": e.get("retained_artifacts", []),
                  "actionable": e.get("code") == code and bool(e.get("next_action"))}))
PY
}
fail_step f01-config-typo 2 E_CONFIG_UNKNOWN_FIELD "$BIN" --json plan --set train.small-mlp.epoch=3
# A candidate bundle trained with unknown_category = "error" (not test-evaluated on purpose).
step f02a-strict-run 0 "$F" "$BIN" --json run --candidates linear --set 'preprocess.unknown_category="error"'
STRICT=$(python3 -c "import json; print(json.load(open('$EVID/logs/f02a-strict-run.out'))['result']['runs'][0]['run_id'])")
step f02b-strict-bundle 0 "$F" "$BIN" bundle build --run "$STRICT" --allow-unevaluated --out bundles/strict
python3 -c "
import json; r=json.load(open('$F/bundles/strict/examples/input.json'))['records'][0]; r['supplier']='S99'; print(json.dumps(r))" >"$F/unknown.jsonl"
fail_step f02-unknown-category 3 E_UNKNOWN_CATEGORY "$BIN" --json predict --bundle bundles/strict --input unknown.jsonl
fail_step f03-missing-column 3 E_SCHEMA_MISSING_COLUMN sh -c "echo '{\"line\": \"A\"}' > missing.jsonl && $BIN --json predict --bundle bundles/$BEST --input missing.jsonl"
fail_step f04-cardinality-budget 3 E_CARDINALITY_BUDGET "$BIN" --json plan --set preprocess.max_categories=3
fail_step f05-device-unavailable 5 E_DEVICE_UNSUPPORTED "$BIN" --json run --set 'execution.candidate_device="cuda"'
fail_step f06-memory-budget 5 E_RESOURCE_BUDGET_EXCEEDED "$BIN" --json run --set execution.max_memory_mb=1
# The other first-iteration candidate saw the same snapshot as $BEST, whose test result is already known.
OTHER=$([ "$BEST" = run-001 ] && echo run-002 || echo run-001)
fail_step f07-test-reuse 7 E_TEST_SET_REUSE "$BIN" --json evaluate --run "$OTHER" --split test
fail_step f08-corrupt-bundle 4 E_BUNDLE_CORRUPT "$BIN" --json bundle check "$WORK/broken"
rm -f "$WORK/broken/transform.json"
fail_step f09-incomplete-bundle 4 E_BUNDLE_MISSING_FILE "$BIN" --json bundle check "$WORK/broken"
fail_step f10-time-split-refused 3 E_SPLIT_UNSUPPORTED "$BIN" --json plan --set 'data.time_column="hours_in_service"'
fail_step f11-resume-completed-run 6 E_RUN_STATE_INVALID "$BIN" --json resume run-001
fail_step f12-missing-data-file 3 E_DATA_SOURCE_NOT_FOUND "$BIN" --json plan --set 'data.source="data/nope.csv"'
python3 - "$EVID/failures.jsonl" >"$EVID/failures_summary.json" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(json.dumps({"scenarios": len(rows), "actionable": sum(r["actionable"] for r in rows), "rows": rows}, indent=2))
PY
python3 -c "import json; s=json.load(open('$EVID/failures_summary.json')); print('actionable failures:', s['actionable'], '/', s['scenarios'])"

if [ -n "$PY" ]; then
  echo "== independent reference (scikit-learn)"
  for r in run-001 run-002; do
    step "70-export-$r" 0 "$REPO" cargo run --quiet -p rustpy-workflow --example export_reference -- "$P" "$r" "$WORK/ref-$r"
    step "71-sklearn-$r" 0 "$REPO" "$PY" -I tools/reference/sklearn_reference.py "$WORK/ref-$r" "$EVID/sklearn_$r.json"
  done
fi

echo "== wrapper overhead (release example; each mode in its own process, 5 interleaved repetitions)"
step 80-overhead-build 0 "$REPO" cargo build --release --quiet -p rustpy-workflow --example overhead
OH="$REPO/target/release/examples/overhead"
for i in 1 2 3 4 5; do
  for m in direct rustpy rustpy-final-checkpoint-only; do
    "$OH" "$WORK/det-1" "$m" 30 >>"$EVID/overhead.jsonl" 2>>"$EVID/logs/overhead.err"
  done
done
python3 - "$EVID/overhead.jsonl" >"$EVID/overhead_summary.json" <<'PY2'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
med = lambda m, k: statistics.median(r[k] for r in rows if r["mode"] == m)
modes = sorted({r["mode"] for r in rows})
out = {m: {"median_seconds": med(m, "train_seconds"), "median_peak_rss_mb": med(m, "peak_rss_mb"),
           "runs": sum(r["mode"] == m for r in rows)} for m in modes}
base = out["direct"]["median_seconds"]
for m in modes:
    out[m]["time_vs_direct"] = out[m]["median_seconds"] / base - 1
out["steps"] = rows[0]["steps"]; out["train_rows"] = rows[0]["train_rows"]; out["features"] = rows[0]["features"]
print(json.dumps(out, indent=2))
PY2
cat "$EVID/overhead_summary.json"

echo
echo "steps not matching the expected exit code: $FAILED"
python3 -c "
import json
rows=[json.loads(l) for l in open('$STEPS')]
print(json.dumps({'steps': len(rows), 'unexpected': [r['step'] for r in rows if r.get('exit') != r.get('expected_exit')]}, indent=2))" | tee "$EVID/summary.json"
