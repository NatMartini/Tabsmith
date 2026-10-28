#!/usr/bin/env bash
# Rebuilds fixtures/bundles/: small bundles of every task and model kind. The tests of
# tabsmith-macros compile each of them into Rust code and compare it with the runtime.
#
#   tools/fixtures/make_bundles.sh <tabsmith-binary> <policies.parquet>
#
# <policies.parquet> comes from tools/fixtures/policies_data.py (needs pyarrow).
# Training is deterministic, so the same tabsmith version rebuilds the same models.
set -euo pipefail
T=$(realpath "$1")
PARQUET=$(realpath "$2")
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$REPO/fixtures/bundles
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
rm -rf "$OUT"
mkdir -p "$OUT"
CPU=(--set 'execution.candidate_device="cpu"')

# sample <task> <target> <run>=<name>...: the synthetic sample of a task, every
# candidate trained, and the listed runs bundled as fixtures/bundles/<name>.
sample() {
  local task=$1 target=$2
  shift 2
  local p=$WORK/$task
  mkdir -p "$p"
  (cd "$p" && "$T" sample-data data.csv --task "$task" --rows 3000 --seed 7 >/dev/null &&
    "$T" train data.csv --target "$target" --group device_id --id unit_id --task "$task" "${CPU[@]}" >/dev/null)
  for spec in "$@"; do
    (cd "$p" && "$T" bundle build --run "${spec%%=*}" --allow-unevaluated --out "$OUT/${spec##*=}" >/dev/null)
    echo "fixtures/bundles/${spec##*=}"
  done
}

sample binary defect run-002=binary-mlp run-003=binary-gbdt
sample multiclass failure_mode run-001=multiclass-linear run-003=multiclass-gbdt
sample regression remaining_hours run-002=regression-mlp run-003=regression-gbdt

# Error policies, missing-value markers, and categories coded by integers and floats.
p=$WORK/policies
mkdir -p "$p"
cp "$PARQUET" "$p/data.parquet"
(cd "$p" && "$T" train data.parquet --target status --positive fail "${CPU[@]}" \
  --set 'preprocess.unknown_category="error"' --set 'preprocess.nan="error"' \
  --set 'data.null_values=["", "NA", "-999"]' \
  --set 'data.columns.zone="categorical"' --set 'data.columns.grade="categorical"' >/dev/null &&
  "$T" bundle build --run run-001 --allow-unevaluated --out "$OUT/policies-linear" >/dev/null)
echo "fixtures/bundles/policies-linear"
