"""Medians of an A/B run of the trees probe or of compiled_vs_runtime: the lines after
`== head` belong to 0.4.0, those after `== const` to the working tree. Also checks that every
hash is the same in both."""
import re
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
values = defaultdict(list)
hashes = defaultdict(set)
side = None
for line in open(sys.argv[1], encoding="utf-8-sig"):
    line = line.strip()
    if line.startswith("== "):
        side = line[3:]
        continue
    m = re.match(r"(\w+)\s+(outputs|predict)\s+([\d.]+) ns", line)
    if m:
        values[(m.group(1) + " " + m.group(2), side)].append(float(m.group(3)))
        h = re.search(r"hash ([0-9a-f]+)", line)
        if h:
            hashes[m.group(1)].add(h.group(1))
        continue
    m = re.match(r"([\w-]+)\s+compiled ([0-9a-f]+)\s+runtime ([0-9a-f]+)\s+(\w+)\s+\|\s+runtime\s+(\d+) ns\s+compiled\(values\)\s+(\d+) ns\s+compiled\(Input\)\s+(\d+) ns", line)
    if m:
        name = m.group(1)
        hashes[name].update({m.group(2), m.group(3)})
        assert m.group(4) == "equal", line
        for k, v in (("runtime", 5), ("compiled(values)", 6), ("compiled(Input)", 7)):
            values[(f"{name} {k}", side)].append(float(m.group(v)))
for name, hs in sorted(hashes.items()):
    assert len(hs) == 1, (name, hs)
names = sorted({k for k, _ in values})
print(f"{'measure':34} {'0.4.0':>10} {'const':>10} {'change':>8}   runs")
for n in names:
    a, b = values[(n, "head")], values[(n, "const")]
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"{n:34} {ma:10.1f} {mb:10.1f} {100 * (mb / ma - 1):+7.1f} %   {len(a)}+{len(b)}")
print("hashes: every one the same in both;", ", ".join(f"{k} {next(iter(v))}" for k, v in sorted(hashes.items())))
