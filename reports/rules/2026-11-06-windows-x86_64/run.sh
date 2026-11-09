#!/usr/bin/env bash
# Audits the rules of rules/ and the label stability of 100 to 200 records per bundle, with a
# release build of probe/. Set ADULT, CALIFORNIA and COVERTYPE to bundle directories trained on
# the three datasets (reports/datasets/2026-10-31-windows-x86_64/datasets-train.ps1 trains them),
# and RECORDS to a folder of JSON-lines records made by records.py from the same CSV files.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
fixtures="$here/../../../fixtures/bundles"
# The build goes to the repository's target folder, which git ignores.
export CARGO_TARGET_DIR="$here/../../../target/rules-probe"
(cd "$here/probe" && cargo build --release)
p="$CARGO_TARGET_DIR/release/rules-probe"
{
  "$p" "$ADULT" "$here/rules/adult.rules"
  "$p" "$CALIFORNIA" "$here/rules/california.rules"
  "$p" "$COVERTYPE" "$here/rules/covertype.rules"
} > "$here/rules.txt"
{
  "$p" stability "$ADULT" "$RECORDS/adult.jsonl" 0.05,0.1,0.25,0.5,1 200
  "$p" stability "$COVERTYPE" "$RECORDS/covertype.jsonl" 0.05,0.25,1 100
  "$p" stability "$fixtures/binary-gbdt" "$RECORDS/binary-gbdt.jsonl" 0.05,0.25,1 200
  "$p" stability "$fixtures/binary-mlp" "$RECORDS/binary-mlp.jsonl" 0.05,0.25 100
} > "$here/stability.txt"
# The records behind the Adult examples of the README: every 10th of the first 600.
awk 'NR % 10 == 1' "$RECORDS/adult.jsonl" | head -60 > "$here/adult-60.jsonl"
tabsmith audit --bundle "$ADULT" --stability 0.05 --input "$here/adult-60.jsonl" > "$here/adult-60.txt" || true
