"""Flash a compiled model takes on a Cortex-M4F: `outputs` as arrays or as code.
A static library with one exported function, built with LTO, so that the object holds what
that function reaches. Sizes come from the object's ELF section and symbol tables.
Writes only inside <repo>/scratch."""
import json
import os
import struct
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = "<work>"
TARGET = "<target>"
env = dict(os.environ)
env.update(
    CARGO_HOME="<cargo-home>",
    TEMP="<tmp>",
    TMP="<tmp>",
    VSLANG="1033",
    CARGO_TERM_COLOR="never",
    CARGO_TARGET_DIR=TARGET,
)
LIB = """#![no_std]
tabsmith_macros::model!(m, "{path}", {mode});

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {{
    loop {{}}
}}

#[unsafe(no_mangle)]
pub extern "C" fn score(x: &[f32; m::N_FEATURES], out: &mut [f32; m::N_OUTPUTS]) {{
    *out = m::outputs(x);
}}
"""


def members(path):
    """(name, bytes) of each member of a System V archive, with its long names resolved."""
    data = open(path, "rb").read()
    assert data[:8] == b"!<arch>\n"
    pos, names, out = 8, b"", []
    while pos + 60 <= len(data):
        name = data[pos : pos + 16].decode().rstrip()
        size = int(data[pos + 48 : pos + 58].decode().strip())
        body = data[pos + 60 : pos + 60 + size]
        pos += 60 + size + (size & 1)
        if name == "//":
            names = body
        elif name.startswith("/") and name[1:].isdigit():
            start = int(name[1:])
            out.append((names[start : names.index(b"\n", start)].decode().rstrip("/"), body))
        elif name != "/":
            out.append((name.rstrip("/"), body))
    return out


def elf_sizes(obj):
    """text, rodata and data bytes of allocated sections; and each defined symbol's size."""
    assert obj[:4] == b"\x7fELF" and obj[4] == 1 and obj[5] == 1, "ELF32, little-endian"
    shoff, = struct.unpack_from("<I", obj, 32)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", obj, 46)
    sections = []
    for i in range(shnum):
        name, typ, flags, _addr, off, size, link, _info, _align, entsize = struct.unpack_from("<10I", obj, shoff + i * shentsize)
        sections.append(dict(name=name, type=typ, flags=flags, off=off, size=size, link=link, entsize=entsize))
    strtab = obj[sections[shstrndx]["off"] : sections[shstrndx]["off"] + sections[shstrndx]["size"]]
    sizes = dict(text=0, rodata=0, data=0)
    for s in sections:
        if not s["flags"] & 2 or s["type"] == 0:  # SHF_ALLOC
            continue
        if s["flags"] & 4:  # SHF_EXECINSTR
            sizes["text"] += s["size"]
        elif s["flags"] & 1:  # SHF_WRITE
            sizes["data"] += s["size"]
        else:
            sizes["rodata"] += s["size"]
    symbols = {}
    for s in sections:
        if s["type"] != 2:  # SHT_SYMTAB
            continue
        names = obj[sections[s["link"]]["off"] : sections[s["link"]]["off"] + sections[s["link"]]["size"]]
        for i in range(s["size"] // 16):
            nm, _value, size, info, _other, shndx = struct.unpack_from("<IIIBBH", obj, s["off"] + i * 16)
            if shndx and size:
                n = names[nm : names.index(b"\0", nm)].decode()
                symbols[n] = size
    return sizes, symbols


results = []
BUNDLES = [
    ("binary-gbdt", "<repo>/fixtures/bundles/binary-gbdt"),
    ("regression-gbdt", "<repo>/fixtures/bundles/regression-gbdt"),
    ("multiclass-gbdt", "<repo>/fixtures/bundles/multiclass-gbdt"),
]
for name, path in BUNDLES:
    for opt in ("s", "3"):
        for mode in ("tree_arrays", "tree_code"):
            with open(f"{HERE}/src/lib.rs", "w", encoding="utf-8") as f:
                f.write(LIB.format(path=path, mode=mode))
            r = subprocess.run(
                ["cargo", "build", "--release", "--offline", "--manifest-path", f"{HERE}/Cargo.toml", "--target", "thumbv7em-none-eabihf"],
                env={**env, "CARGO_PROFILE_RELEASE_OPT_LEVEL": opt},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if r.returncode != 0:
                print(r.stderr[-3000:])
                raise SystemExit("build failed")
            archive = f"{TARGET}/thumbv7em-none-eabihf/release/libembedded_size.a"
            total = dict(text=0, rodata=0, data=0)
            score = None
            for member, body in members(archive):
                if not member.endswith(".o") or body[:4] != b"\x7fELF":
                    continue
                sizes, symbols = elf_sizes(body)
                if "score" in symbols:
                    total = sizes
                    score = symbols["score"]
            row = dict(model=name, opt=opt, form=mode, **total, score_symbol=score)
            results.append(row)
            print(f"{name:18} opt {opt}  {mode:12} text {total['text']:7}  rodata {total['rodata']:7}  flash {total['text'] + total['rodata'] + total['data']:7}  score {score}", flush=True)
json.dump(results, open(f"{HERE}/results.json", "w", encoding="utf-8"), indent=1)
