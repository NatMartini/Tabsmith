#!/usr/bin/env bash
# scikit-learn on the three dataset projects (trained on Windows), with the WSL venv.
# Reads <repo>; writes only sklearn.json files inside its scratch/datasets/runs.
set -euo pipefail
exec </dev/null
py="$HOME/.cache/tabsmith-venv/bin/python"
[ -x "$py" ] || { echo "no venv python at $py" >&2; exit 1; }
script=<repo>/tools/datasets/compare.py
runs=<repo>/scratch/datasets/runs
log=<repo>/scratch/linux/datasets-compare.log
: > "$log"
for name in "${@:-adult california covertype}"; do
  echo "== $name" | tee -a "$log"
  "$py" -I -B "$script" "$runs/$name" "$runs/$name/sklearn.json" >>"$log" 2>&1 || echo "FAILED: $name" | tee -a "$log"
done
tail -n 60 "$log"
