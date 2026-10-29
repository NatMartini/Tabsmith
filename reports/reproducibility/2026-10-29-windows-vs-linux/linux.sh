#!/usr/bin/env bash
# Trains the three candidates on the same synthetic data on Linux (WSL), for the
# cross-machine reproducibility experiment (the Windows side is scratch/repro-gbdt.ps1).
# Writes only inside $HOME/tabsmith-verify and <repo>/scratch. Deletes nothing.
set -euo pipefail
exec </dev/null
export PATH="$HOME/.cargo/bin:$PATH"

src=<repo>
out=<scratch>
base="$HOME/tabsmith-verify"
case "$base" in
  /home/?*/tabsmith-verify) ;;
  *) echo "refusing: unexpected base folder '$base'" >&2; exit 1 ;;
esac
[ -f "$src/Cargo.toml" ] && [ -d "$src/crates/tabsmith-runtime" ] || { echo "refusing: $src is not the repository" >&2; exit 1; }

run="$base/repro-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run/repo"
if [ -n "$(ls -A "$run/repo")" ]; then echo "refusing: $run/repo is not empty" >&2; exit 1; fi
rsync -a \
  --exclude '/target/' --exclude '/examples/*/target/' --exclude '/fuzz/target/' \
  --exclude '/scratch/' --exclude '/.git/' --exclude '/upstream-drafts/' --exclude '/_cloud-session/' \
  "$src/" "$run/repo/"

log="$out/repro-gbdt.log"
cd "$run/repo"
cargo build --release --locked --offline -p tabsmith-cli >"$log" 2>&1
t="$run/repo/target/release/tabsmith"
work="$run/work"
mkdir -p "$work"
cd "$work"
"$t" sample-data data.csv --rows 3000 --seed 7 >>"$log" 2>&1
"$t" train data.csv --target defect --group device_id --id unit_id --set 'execution.candidate_device="cpu"' >>"$log" 2>&1
{
  echo "data.csv  $(sha256sum data.csv | cut -d ' ' -f 1)"
  find . -name model.safetensors | sort | while read -r f; do
    echo "${f#./}  $(sha256sum "$f" | cut -d ' ' -f 1)"
  done
  find . -name run.json | sort | while read -r f; do
    python3 -B - "$f" "${f#./}" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1], encoding="utf-8"))
h = " ".join(f"{e['epoch']}:{e['train_loss']}:{e['valid_loss']}" for e in j.get("history", []))
print(f"{sys.argv[2]}  candidate={j.get('candidate')} best_epoch={j.get('best_epoch')} epochs={len(j.get('history', []))}")
print(f"  history {h}")
EOF
  done
} > "$out/repro-gbdt-hashes.txt"
cat "$out/repro-gbdt-hashes.txt"
