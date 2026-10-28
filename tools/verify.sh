#!/usr/bin/env bash
# Every check of .github/workflows/ci.yml, run on this machine (Linux or macOS), and the
# compiled-vs-runtime example, whose prediction hashes can be compared across machines.
#
#   bash tools/verify.sh                          the CI checks
#   bash tools/verify.sh --all                    also the minimum Rust versions, the
#                                                 GPU-feature lint (Linux), the fuzz target,
#                                                 RustSec advisories and the Kani proofs
#   bash tools/verify.sh --all 2>&1 | tee verify.log
#
# It adds the rustup targets and components it needs, and with --all the MSRV toolchains,
# cargo-audit and Kani (KANI_TIMEOUT bounds the proofs where `timeout` exists, default 4h). Otherwise
# it writes only into this repository's target folders. Exit status: the number of
# failed steps.
set -u
exec </dev/null

ALL=0
for arg in "$@"; do
  case "$arg" in
    --all) ALL=1 ;;
    -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.." || exit 2
if [ ! -f Cargo.toml ] || [ ! -d crates/tabsmith-runtime ]; then
  echo "tools/verify.sh: run it from the Tabsmith repository" >&2
  exit 2
fi
export CARGO_TERM_COLOR=never

RESULTS=()
FAILED=0

# step <name> <cmd...>: run one check and record ok or FAIL with its time.
step() {
  local name=$1 t0=$SECONDS status
  shift
  printf '\n===== %s\n> %s\n' "$name" "$*"
  if "$@"; then status=ok; else status=FAIL; FAILED=$((FAILED + 1)); fi
  RESULTS+=("$(printf '%-4s %6ds  %s' "$status" $((SECONDS - t0)) "$name")")
}

rustup_setup() {
  rustup target add thumbv7em-none-eabihf wasm32-unknown-unknown && rustup component add rustfmt clippy
}

fmt_all() {
  local manifest ok=0
  cargo fmt --all --check || ok=1
  for manifest in examples/predict-app examples/embedded-predict examples/wasm-predict fuzz; do
    cargo fmt --manifest-path "$manifest/Cargo.toml" --check || ok=1
  done
  return $ok
}

predict_app_has_no_burn_or_polars() {
  local deps
  deps=$(cargo tree --locked --manifest-path examples/predict-app/Cargo.toml -e normal --prefix none --format '{p}') || return 1
  if printf '%s\n' "$deps" | grep -E '^(burn|polars)'; then
    echo "examples/predict-app depends on Burn or Polars; tabsmith-runtime must stay free of them"
    return 1
  fi
  echo "$(printf '%s\n' "$deps" | wc -l | tr -d ' ') packages, neither Burn nor Polars"
}

rustdoc_libraries() {
  RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked -p tabsmith-contracts -p tabsmith-embedded \
    -p tabsmith-macros -p tabsmith-runtime -p tabsmith-workflow
}

msrv() {
  local workspace runtime
  workspace=$(sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)
  runtime=$(sed -n 's/^rust-version = "\(.*\)"/\1/p' crates/tabsmith-runtime/Cargo.toml)
  [ -n "$workspace" ] && [ -n "$runtime" ] || return 1
  rustup toolchain install "$workspace" "$runtime" --profile minimal --no-self-update || return 1
  cargo "+$workspace" check --workspace --all-targets --locked || return 1
  cargo "+$runtime" check -p tabsmith-contracts -p tabsmith-embedded -p tabsmith-macros -p tabsmith-runtime \
    --all-targets --locked || return 1
  cargo "+$runtime" check --manifest-path examples/predict-app/Cargo.toml --all-targets --locked
}

fuzz_target() {
  cargo check --manifest-path fuzz/Cargo.toml --locked
}

advisories() {
  local lock ok=0
  command -v cargo-audit >/dev/null 2>&1 || cargo install --locked cargo-audit || return 1
  for lock in Cargo.lock examples/predict-app/Cargo.lock examples/wasm-predict/Cargo.lock \
    examples/embedded-predict/Cargo.lock fuzz/Cargo.lock; do
    echo "== $lock"
    cargo audit --file "$lock" || ok=1
  done
  return $ok
}

kani_proofs() {
  command -v cargo-kani >/dev/null 2>&1 || cargo install --locked kani-verifier || return 1
  [ -d "$HOME/.kani" ] || cargo kani setup || return 1
  if command -v timeout >/dev/null 2>&1; then
    (cd examples/embedded-predict && timeout "${KANI_TIMEOUT:-4h}" cargo kani)
  else
    (cd examples/embedded-predict && cargo kani)
  fi
}

echo "Tabsmith verification on $(uname -srm)"
if [ "$(uname -s)" = Darwin ]; then sw_vers; fi
rustc -vV

step 'rustup targets and components' rustup_setup
step 'rustfmt (workspace and standalone crates)' fmt_all
step 'test (workspace)' cargo test --workspace --locked
step 'build (predict-app)' cargo build --locked --manifest-path examples/predict-app/Cargo.toml --all-targets
step 'predict-app has neither Burn nor Polars' predict_app_has_no_burn_or_polars
step 'clippy (workspace)' cargo clippy --workspace --all-targets --locked -- -D warnings
step 'clippy (predict-app)' cargo clippy --manifest-path examples/predict-app/Cargo.toml --all-targets --locked -- -D warnings
step 'rustdoc (library crates)' rustdoc_libraries
step 'build (embedded-predict, Cortex-M4F)' \
  cargo build --locked --release --manifest-path examples/embedded-predict/Cargo.toml --target thumbv7em-none-eabihf
step 'build (embedded-predict, wasm32)' \
  cargo build --locked --release --manifest-path examples/embedded-predict/Cargo.toml --target wasm32-unknown-unknown
step 'test (embedded-predict)' cargo test --locked --manifest-path examples/embedded-predict/Cargo.toml
step 'clippy (embedded-predict)' \
  cargo clippy --locked --manifest-path examples/embedded-predict/Cargo.toml --all-targets -- -D warnings
step 'clippy (embedded-predict, Cortex-M4F)' \
  cargo clippy --locked --manifest-path examples/embedded-predict/Cargo.toml --target thumbv7em-none-eabihf -- -D warnings
step 'build (tabsmith-runtime, wasm32)' cargo build --locked -p tabsmith-runtime --target wasm32-unknown-unknown
step 'build (wasm-predict, wasm32)' \
  cargo build --locked --release --manifest-path examples/wasm-predict/Cargo.toml --target wasm32-unknown-unknown
step 'test (wasm-predict)' cargo test --locked --manifest-path examples/wasm-predict/Cargo.toml
step 'clippy (wasm-predict, wasm32)' \
  cargo clippy --locked --manifest-path examples/wasm-predict/Cargo.toml --all-targets --target wasm32-unknown-unknown -- -D warnings
step 'compiled models vs runtime (hashes and timings)' \
  cargo run --locked --release -p tabsmith-macros --example compiled_vs_runtime

if [ "$ALL" = 1 ]; then
  if [ "$(uname -s)" = Linux ]; then
    step 'clippy (GPU features)' cargo clippy --workspace --all-targets --locked --features tabsmith-cli/gpu -- -D warnings
  fi
  step 'fuzz target compiles' fuzz_target
  step 'RustSec advisories (every Cargo.lock)' advisories
  step 'minimum Rust versions' msrv
  step 'Kani proofs (examples/embedded-predict)' kani_proofs
fi

printf '\n===== summary: %s, %s\n' "$(uname -sm)" "$(rustc -V)"
for line in ${RESULTS[@]+"${RESULTS[@]}"}; do echo "$line"; done
echo "$FAILED failed"
exit "$FAILED"
