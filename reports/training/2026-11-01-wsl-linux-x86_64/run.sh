#!/usr/bin/env bash
# Tabsmith, LightGBM and XGBoost on one machine, one operating system, the same rows, split and
# settings. Every fit is a process of its own. Tabsmith's train_ms and each library's fit include
# binning and exclude reading the CSV. Writes only inside $HOME/tabsmith-verify and
# <repo>/scratch/compare. Deletes nothing.
set -euo pipefail
exec </dev/null
export PATH="$HOME/.cargo/bin:$PATH"

src=<repo>
data=<repo>/scratch/datasets
out=<repo>/scratch/compare
fit=<repo>/tools/datasets/compare_gbm.py
base="$HOME/tabsmith-verify"
case "$base" in
  /home/?*/tabsmith-verify) ;;
  *) echo "refusing: unexpected base folder '$base'" >&2; exit 1 ;;
esac
[ -f "$src/Cargo.toml" ] && [ -d "$src/crates/tabsmith-runtime" ] || { echo "refusing: $src is not the repository" >&2; exit 1; }
[ -f "$data/covertype.csv" ] || { echo "refusing: datasets not found" >&2; exit 1; }
py="$HOME/.cache/tabsmith-venv/bin/python"
[ -x "$py" ] || { echo "refusing: no venv at $py" >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
run="$base/gbm-$stamp"
mkdir -p "$run/repo" "$out"
rsync -a \
  --exclude '/target/' --exclude '/examples/*/target/' --exclude '/fuzz/target/' \
  --exclude '/scratch/' --exclude '/.git/' --exclude '/upstream-drafts/' --exclude '/_cloud-session/' \
  "$src/" "$run/repo/"
log="$out/gbm-$stamp.log"
jsonl="$out/gbm-$stamp.jsonl"
: > "$log"; : > "$jsonl"
say() { echo "$*" | tee -a "$log"; }
say "run folder $run"
say "kernel $(uname -r), $(nproc) cpus"
say "other load: $(cut -d' ' -f1-3 /proc/loadavg)"
say "versions: $("$py" -I -B -c "import lightgbm, xgboost, sklearn, numpy; print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'scikit-learn', sklearn.__version__, 'numpy', numpy.__version__)") $(rustc --version)"

( cd "$run/repo" && cargo build --release --locked --offline -p tabsmith-cli ) > "$run/build.log" 2>&1 || { tail -20 "$run/build.log" | tee -a "$log"; exit 1; }
t="$run/repo/target/release/tabsmith"
say "built $($t --version 2>&1 | head -n 1)"

train() { # dataset rounds threads repeat
  local name="$1" rounds="$2" threads="$3" rep="$4"
  local dir="$run/runs/$name-r$rounds-t$threads-$rep"
  mkdir -p "$dir"
  (
    cd "$dir"
    export RAYON_NUM_THREADS="$threads"
    case "$name" in
      adult)
        "$t" train "$data/adult.csv" --target income --positive '>50K' --candidates gbdt \
          --set 'data.null_values=["","NA","?"]' --set "train.gbdt.epochs=$rounds" --set 'train.gbdt.early_stopping_patience=0' --set 'execution.candidate_device="cpu"' --json >train.json 2>train.err ;;
      california)
        "$t" train "$data/california.csv" --target median_house_value --task regression --candidates gbdt \
          --set "train.gbdt.epochs=$rounds" --set 'train.gbdt.early_stopping_patience=0' --set 'execution.candidate_device="cpu"' --json >train.json 2>train.err ;;
      covertype)
        "$t" train "$data/covertype.csv" --target cover_type --task multiclass --candidates gbdt \
          --set "train.gbdt.epochs=$rounds" --set 'train.gbdt.early_stopping_patience=0' --set 'execution.candidate_device="cpu"' --json >train.json 2>train.err ;;
    esac
  )
  python3 -B - "$dir" "$name" "$threads" "$rounds" "$jsonl" <<'EOF' | tee -a "$log"
import glob, hashlib, json, sys
d, name, threads, rounds, out = sys.argv[1:6]
r = json.load(open(glob.glob(d + "/.tabsmith/runs/run-*/run.json")[0], encoding="utf-8"))
h = hashlib.sha256(open(glob.glob(d + "/bundles/*/model.safetensors")[0], "rb").read()).hexdigest()[:16]
line = dict(library="tabsmith", project=name, threads=int(threads), rounds=int(rounds), evaluate=True,
            fit_seconds=r["timings"]["train_ms"] / 1000, weights=h, done=len(r["history"]),
            peak_mb=(r.get("device_receipt") or {}).get("peak_rss_mb"), run_id=r["run_id"], dir=d)
open(out, "a", encoding="utf-8").write(json.dumps(line) + "\n")
print(f"tabsmith {name} t={threads} r={rounds}: {line['fit_seconds']:.3f} s, {len(r['history'])} rounds, weights {h}")
EOF
  ln -sfn "$dir" "$run/project-$name-r$rounds"
}

gbm() { # dataset rounds library threads eval|noeval
  local name="$1" rounds="$2" lib="$3" threads="$4" mode="$5"
  OMP_NUM_THREADS="$threads" "$py" -I -B "$fit" "$run/project-$name-r$rounds" "$lib" "$threads" "$rounds" "$mode" "$jsonl" 2>&1 | tee -a "$log"
}

metrics() { # dataset rounds: Tabsmith's test metrics, on the project the libraries also read
  local name="$1" rounds="$2"
  (
    cd "$run/project-$name-r$rounds"
    r=$(python3 -B -c "import glob,os; print(sorted(os.path.basename(d) for d in glob.glob('.tabsmith/runs/run-*') if os.path.isdir(d))[-1])")
    "$t" --json evaluate --run "$r" --split test --allow-test-reuse 2>/dev/null
  ) | python3 -B -c "
import json, sys
m = json.load(sys.stdin)['result']['metrics']
keep = {k: m[k] for k in ('roc_auc', 'balanced_accuracy', 'accuracy', 'log_loss', 'rmse', 'mae', 'r2') if k in m}
line = dict(library='tabsmith', project='$name', threads=0, rounds=$rounds, evaluate=True, fit_seconds=0, test=keep)
open('$jsonl', 'a', encoding='utf-8').write(json.dumps(line) + chr(10))
print('tabsmith $name test', {k: round(v, 4) for k, v in keep.items()})
" | tee -a "$log"
}

sweep() { # dataset rounds
  local name="$1" rounds="$2"
  for threads in 1 4 8 12 16; do
    for rep in 1 2 3; do
      train "$name" "$rounds" "$threads" "$rep"
    done
  done
  metrics "$name" "$rounds"
  for lib in lightgbm lightgbm-quantized xgboost; do
    for threads in 1 4 8 12 16; do
      for rep in 1 2 3; do
        gbm "$name" "$rounds" "$lib" "$threads" eval
      done
      gbm "$name" "$rounds" "$lib" "$threads" noeval
    done
  done
}

sweep california 334
sweep adult 180
sweep covertype 40
# The whole training at the rounds of the dataset report, for the quality table.
for rep in 1 2; do
  train covertype 400 8 "$rep"
  train covertype 400 12 "$rep"
done
metrics covertype 400
for rep in 1 2; do
  for lib in lightgbm lightgbm-quantized xgboost; do gbm covertype 400 "$lib" 8 eval; done
done
say "load at the end: $(cut -d' ' -f1-3 /proc/loadavg)"
say "done"
