"""Does tests/tree_code.rs catch a wrong number? Each mutation is applied, tested and undone.
Writes only inside <repo>; restores the file from its own bytes."""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
PATH = "<repo>/crates/tabsmith-macros/src/codegen.rs"
env = dict(os.environ)
env.update(
    CARGO_HOME="<cargo-home>",
    TEMP="<tmp>",
    TMP="<tmp>",
    VSLANG="1033",
    CARGO_TERM_COLOR="never",
    RUST_TEST_THREADS="8",
)
MUTATIONS = [
    ("threshold one ulp up", "let threshold = Literal::f32_suffixed(t.threshold[i]);", "let threshold = Literal::f32_suffixed(t.threshold[i].next_up());"),
    ("threshold one ulp down", "let threshold = Literal::f32_suffixed(t.threshold[i]);", "let threshold = Literal::f32_suffixed(t.threshold[i].next_down());"),
    ("leaf one ulp up", "let value = Literal::f32_suffixed(t.value[i]);", "let value = Literal::f32_suffixed(t.value[i].next_up());"),
    ("< instead of <=", "quote!(if x[#f] <= #threshold { #left } else { #right })", "quote!(if x[#f] < #threshold { #left } else { #right })"),
    ("children swapped", "quote!(if x[#f] <= #threshold { #left } else { #right })", "quote!(if x[#f] <= #threshold { #right } else { #left })"),
    ("sum per output wrong", "let o = Literal::usize_unsuffixed(i % n_outputs);", "let o = Literal::usize_unsuffixed((i + 1) % n_outputs);"),
]
original = open(PATH, "rb").read()
text = original.decode("utf-8")
try:
    for name, old, new in MUTATIONS:
        assert text.count(old) == 1, name
        open(PATH, "wb").write(text.replace(old, new).encode("utf-8"))
        r = subprocess.run(
            ["cargo", "test", "--offline", "--locked", "-j", "4", "-p", "tabsmith-macros", "--test", "tree_code", "--no-fail-fast"],
            cwd="<repo>", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED$", r.stdout, re.M)))
        summary = re.findall(r"^test result: .*$", r.stdout, re.M)
        print(f"{name}: exit {r.returncode}, {summary[-1] if summary else 'no summary'}")
        for f in failed:
            print("   ", f)
        sys.stdout.flush()
        if r.returncode == 0:
            print("    NOT CAUGHT")
finally:
    open(PATH, "wb").write(original)
    print("restored")
