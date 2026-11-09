#!/usr/bin/env bash
# Times `cargo check` of the cost crate: one build for the dependencies, then three of the crate.
# Run from the repository root; set RULE_CHECK_MAX_WORK in crates/tabsmith-macros/src/codegen.rs
# to usize::MAX for the first measurement and back to 1_000_000 for the second.
export CARGO_TARGET_DIR="$PWD/target/rule-cost"
export CARGO_INCREMENTAL=0
cd reports/rules/2026-11-06-windows-x86_64/cost || exit 1
cargo check 2>&1 | grep -E "^error|Finished" | head -3
for i in 1 2 3; do touch src/main.rs; s=$(date +%s.%N); cargo check 2>&1 | grep -E "^error" | head -3; e=$(date +%s.%N); python -I -B -c "print('check', round($e - $s, 2), 's')"; done
