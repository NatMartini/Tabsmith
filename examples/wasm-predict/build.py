"""Builds the WebAssembly demo into www/: the module, its JavaScript bindings,
and one packed sample bundle per task.

    python build.py --tabsmith ../../target/release/tabsmith   # trains the samples
    python build.py --bundle defects=path/to/bundles/run-002    # or packs your own
    python -m http.server -d www 8000                           # then open :8000

Needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` of the same
version as the `wasm-bindgen` crate (see README.md). Standard library only.
"""
import argparse
import json
import pathlib
import shutil
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
WWW = HERE / "www"

# (task, target column, title) of the sample bundles `--tabsmith` trains.
SAMPLES = [
    ("binary", "defect", "Will this machine be defective?"),
    ("multiclass", "failure_mode", "Which failure mode?"),
    ("regression", "remaining_hours", "How many hours until it fails?"),
]


def run(*cmd, cwd=HERE):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def pack(bundle: pathlib.Path) -> bytes:
    """The layout `Bundle::new` reads (and `fuzz/seed.py` writes): a flags byte,
    then `[u16 LE name length][name][u32 LE data length][data]` per file."""
    out = bytearray([1])
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        name = path.relative_to(bundle).as_posix().encode()
        data = path.read_bytes()
        out += struct.pack("<H", len(name)) + name + struct.pack("<I", len(data)) + data
    return bytes(out)


def train_samples(tabsmith: pathlib.Path) -> list[tuple[str, pathlib.Path, str]]:
    work = HERE / "target" / "samples"
    shutil.rmtree(work, ignore_errors=True)
    bundles = []
    for task, target, title in SAMPLES:
        project = work / task
        project.mkdir(parents=True)
        run(tabsmith, "sample-data", "machines.csv", "--task", task, cwd=project)
        run(tabsmith, "train", "machines.csv", "--target", target, "--group", "device_id", "--id", "unit_id",
            "--task", task, cwd=project)
        (bundle,) = (project / "bundles").iterdir()
        bundles.append((task, bundle, title))
    return bundles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabsmith", type=pathlib.Path, help="a tabsmith binary: train one sample bundle per task")
    ap.add_argument("--bundle", action="append", default=[], metavar="NAME=DIR", help="pack this bundle directory too")
    ap.add_argument("--skip-wasm", action="store_true", help="only (re)pack bundles")
    args = ap.parse_args()

    if not args.skip_wasm:
        run("cargo", "build", "--release", "--target", "wasm32-unknown-unknown")
        wasm = HERE / "target" / "wasm32-unknown-unknown" / "release" / "wasm_predict.wasm"
        run("wasm-bindgen", "--target", "web", "--no-typescript", "--out-dir", WWW / "pkg", wasm)

    bundles = train_samples(args.tabsmith.resolve()) if args.tabsmith else []
    for spec in args.bundle:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--bundle {spec}: expected NAME=DIR")
        bundles.append((name, pathlib.Path(path), None))

    out = WWW / "bundles"
    if bundles:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        index = []
        for name, bundle, title in bundles:
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            output = json.loads((bundle / "output.schema.json").read_text(encoding="utf-8"))
            data = pack(bundle)
            (out / f"{name}.bin").write_bytes(data)
            index.append({
                "file": f"{name}.bin",
                "title": title or manifest["name"],
                "task": output.get("task", "binary"),
                "target": output["target"],
                "bytes": len(data),
            })
            print(f"packed {bundle} -> www/bundles/{name}.bin ({len(data)} bytes)")
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
