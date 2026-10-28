"""Packs a bundle directory into the input layout of `bundle_from_memory`.

    python3 fuzz/seed.py bundles/run-002 fuzz/corpus/bundle_from_memory/run-002

Writes two seeds: `<out>-unverified` (hash checks off, so mutations reach the
parsers) and `<out>-verified`.
"""
import pathlib
import struct
import sys


def pack(bundle: pathlib.Path, verify: bool) -> bytes:
    out = bytearray([1 if verify else 0])
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        name = path.relative_to(bundle).as_posix().encode()
        data = path.read_bytes()
        out += struct.pack("<H", len(name)) + name + struct.pack("<I", len(data)) + data
    return bytes(out)


def main() -> None:
    bundle, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    for verify in (False, True):
        target = out.with_name(f"{out.name}-{'verified' if verify else 'unverified'}")
        target.write_bytes(pack(bundle, verify))
        print(target)


if __name__ == "__main__":
    main()
