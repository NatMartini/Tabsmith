"""How long one model! invocation takes to compile, and how large the program gets.
Usage: measure.py <out.json>. Writes only inside <repo>/scratch."""
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = "<work>"
RUNS = "../runs"
BUNDLES = [
    ("fixture binary-gbdt, 854 nodes", "<repo>/fixtures/bundles/binary-gbdt"),
    ("Adult, 9151 nodes", f"{RUNS}/final-adult/bundles/run-001"),
    ("Covertype 40 rounds, 16848 nodes", f"{RUNS}/final-covertype/bundles/run-001"),
    ("California, 18823 nodes", f"{RUNS}/final-california/bundles/run-001"),
]
if "--big" in sys.argv:
    BUNDLES = [
        ("Covertype 143 rounds, 59781 nodes", f"{RUNS}/mid-covertype/bundles/run-001"),
        ("Covertype 400 rounds, 169248 nodes", f"{RUNS}/timeline4-covertype/bundles/run-001"),
    ]
if "--limit" in sys.argv:
    BUNDLES = [("Covertype 78 rounds, 32450 nodes", f"{RUNS}/limit-covertype/bundles/run-001")]
out_path = sys.argv[1]

env = dict(os.environ)
env.update(
    CARGO_HOME="<cargo-home>",
    TEMP="<tmp>",
    TMP="<tmp>",
    VSLANG="1033",
    CARGO_TERM_COLOR="never",
    CARGO_TARGET_DIR="<target>",
    CARGO_INCREMENTAL="0",
)
MAIN = """tabsmith_macros::model!(m, "{path}", {mode});
fn main() {{
    m::self_check().unwrap();
    let x = std::hint::black_box([0.5f32; m::N_FEATURES]);
    println!("{{:?}}", m::outputs(&x));
}}
"""


def build(path, mode, profile, extra_env):
    with open(f"{HERE}/src/main.rs", "w", encoding="utf-8") as f:
        f.write(MAIN.format(path=path, mode=mode))
    args = ["cargo", "build", "--offline", "--manifest-path", f"{HERE}/Cargo.toml"]
    if profile == "release":
        args.append("--release")
    t = time.perf_counter()
    r = subprocess.run(args, env={**env, **extra_env}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - t
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"build failed: {path} {mode} {profile}")
    exe = f"<target>/{profile}/compile-probe.exe".replace("/release/", "/release/")
    return seconds, os.path.getsize(exe)


results = []
# Dependencies first, so that the timed builds compile only the invocation.
for profile in ("debug", "release"):
    build("<repo>/fixtures/bundles/binary-gbdt", "tree_arrays", profile, {})
for name, path in BUNDLES:
    for variant, mode, extra in (
        ("arrays", "tree_arrays", {}),
        ("code, decimal literals", "tree_code", {}),
    ):
        for profile in ("debug", "release"):
            seconds, size = build(path, mode, profile, extra)
            row = dict(model=name, variant=variant, profile=profile, seconds=round(seconds, 2), exe_bytes=size)
            results.append(row)
            print(f"{name:36} {variant:24} {profile:8} {seconds:8.2f} s {size / 1e6:8.2f} MB", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)
