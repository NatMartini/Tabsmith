"""Tables of reports/training/ from the raw JSON lines of one comparison run: seconds per library
and thread count, quality, peak memory, and how many different models each library produced.

    python -I -B tools/datasets/report_training.py <results.jsonl> [<before.jsonl>] > results.md

`before.jsonl` is an earlier run of the same script; its Tabsmith becomes the column "before".
Standard library only.
"""

import json
import statistics
import sys
from collections import defaultdict

LIBRARIES = [("tabsmith", "Tabsmith"), ("lightgbm", "LightGBM"), ("lightgbm-quantized", "LightGBM quantized"), ("xgboost", "XGBoost")]
ORDER = ["california", "adult", "covertype"]


def load(path):
    times, models, quality, peak = defaultdict(list), defaultdict(set), {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            project = r["project"].removeprefix("project-").split("-r")[0]
            where = (project, r["rounds"], r["library"])
            if r["threads"] == 0:
                quality[where] = r["test"]
                continue
            if r.get("test"):
                quality[where] = r["test"]
            if not r["evaluate"]:
                continue
            times[where + (r["threads"],)].append(r["fit_seconds"])
            models[where].add((r["threads"], r.get("weights") or r["margins"]))
            if r.get("peak_mb"):
                peak[where] = max(peak.get(where, 0), r["peak_mb"])
    return times, models, quality, peak


def median(values):
    return statistics.median(values) if values else None


def seconds(v):
    if v is None:
        return "–"
    return f"{v:.3f}" if v < 10 else f"{v:.1f}"


def best(times, where):
    cells = {k[3]: median(v) for k, v in times.items() if k[:3] == where}
    if not cells:
        return None, None
    t = min(cells, key=cells.get)
    return cells[t], t


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    times, models, quality, peak = load(sys.argv[1])
    before = load(sys.argv[2])[0] if len(sys.argv) > 2 else {}
    problems = sorted({k[:2] for k in times}, key=lambda p: (ORDER.index(p[0]), p[1]))

    print("### Seconds at each one's best thread count\n")
    print("| Problem | " + ("Tabsmith 0.3 | " if before else "") + "Tabsmith | LightGBM | XGBoost |")
    print("|---|" + ("---|" if before else "") + "---|---|---|")
    for p, r in problems:
        row = [f"{p}, {r} rounds"]
        if before:
            v, t = best(before, (p, r, "tabsmith"))
            row.append(f"{seconds(v)} ({t})" if v else "–")
        for lib in ("tabsmith", "lightgbm", "xgboost"):
            v, t = best(times, (p, r, lib))
            row.append(f"{seconds(v)} ({t})" if v else "–")
        print("| " + " | ".join(row) + " |")

    print("\n### Seconds by thread count (median of the runs)\n")
    for p, r in problems:
        threads = sorted({k[3] for k in times if k[:2] == (p, r)})
        print(f"**{p}, {r} rounds**\n")
        print("| Threads | " + " | ".join(str(t) for t in threads) + " |")
        print("|---|" + "---|" * len(threads))
        if before:
            print("| Tabsmith 0.3 | " + " | ".join(seconds(median(before.get((p, r, "tabsmith", t)))) for t in threads) + " |")
        for lib, name in LIBRARIES:
            if any(k[:3] == (p, r, lib) for k in times):
                print(f"| {name} | " + " | ".join(seconds(median(times.get((p, r, lib, t)))) for t in threads) + " |")
        print()

    print("### Quality on the test rows\n")
    for p, r in problems:
        libs = [(lib, name) for lib, name in LIBRARIES if (p, r, lib) in quality]
        if not libs:
            continue
        names = [n for n in quality[(p, r, libs[0][0])] if all(n in quality[(p, r, lib)] for lib, _ in libs)]
        print(f"**{p}, {r} rounds**\n")
        print("| | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for lib, name in libs:
            q = quality[(p, r, lib)]
            print(f"| {name} | " + " | ".join(f"{q[n]:,.4f}" if abs(q[n]) < 100 else f"{q[n]:,.0f}" for n in names) + " |")
        print()

    print("### Different models, over all the runs and thread counts\n")
    print("A model is a hash of Tabsmith's weights, or of the libraries' raw test margins computed on one thread.\n")
    print("| Problem | Tabsmith | LightGBM | LightGBM quantized | XGBoost |")
    print("|---|---|---|---|---|")
    for p, r in problems:
        cells = []
        for lib, _ in LIBRARIES:
            m = models.get((p, r, lib))
            cells.append(str(len({h for _, h in m})) if m else "–")
        print(f"| {p}, {r} rounds | " + " | ".join(cells) + " |")

    print("\n### Peak memory while training, MB\n")
    print("Tabsmith's is the whole process's, with the data frame it read and the other splits; each library's is its process's from the fit on, with the feature matrix and Python.\n")
    print("| Problem | Tabsmith | LightGBM | XGBoost |")
    print("|---|---|---|---|")
    for p, r in problems:
        print(f"| {p}, {r} rounds | " + " | ".join(f"{peak.get((p, r, lib), 0):,.0f}" for lib in ("tabsmith", "lightgbm", "xgboost")) + " |")


if __name__ == "__main__":
    main()
