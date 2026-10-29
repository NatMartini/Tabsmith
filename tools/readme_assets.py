"""Draws the README's pictures into docs/assets/: the logo, the pipeline diagram and the
benchmark chart, each in a light and a dark variant for GitHub's two themes (the README
picks one with <picture>), and the source of the social preview image.

    python -B tools/readme_assets.py

The chart's numbers are read from a benchmark report's results.json (`--results`), so the
picture shows exactly what the report measured. GitHub wants the social preview as a PNG:
social-preview.png is social-preview.svg drawn at 1280x640 by a browser, uploaded under
Settings > General > Social preview. Standard library only.
"""
import argparse
import json
import pathlib
from xml.sax.saxutils import escape, quoteattr

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
RESULTS = ROOT / "reports" / "benchmarks" / "2026-10-27-linux-x86_64-v2" / "results.json"

SANS = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, "
        "'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif")
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# The text and line colours of GitHub's light and dark themes, and an ember orange for
# Tabsmith: `accent` for text, `fill` for shapes.
THEMES = {
    "light": {"fg": "#1f2328", "muted": "#59636e", "line": "#d1d9e0", "card": "#f6f8fa",
              "accent": "#c2410c", "fill": "#ea580c", "soft": "#fff1e7", "other": "#afb8c1"},
    "dark": {"fg": "#f0f6fc", "muted": "#9198a1", "line": "#3d444d", "card": "#151b23",
             "accent": "#fb923c", "fill": "#f97316", "soft": "#2b1a10", "other": "#656c76"},
}

# The words of the diagram and the chart, for README.md and README.zh-CN.md.
WORDS = {
    "en": {
        "pipeline": "From a CSV to a verified bundle, shipped four ways",
        "data": ("CSV or Parquet", "one table"),
        "steps": ["split without leakage", "preprocess on training rows", "train linear, MLP, GBDT",
                  "choose on validation", "test once, on unseen rows"],
        "bundle": ("bundle", ["weights, preprocessing,", "schema and evidence,", "SHA-256 verified"]),
        "targets": [("In your Rust application", "a small dependency, no Python"),
                    ("Compiled into your code", "no_std, no allocation"),
                    ("In a web page", "verified in the browser"),
                    ("As an HTTP service", "JSON in, JSON out")],
        "bench": "Shipping a model: Tabsmith against Python with scikit-learn",
        "legend": ("Tabsmith (small MLP)", "Python + scikit-learn (MLPClassifier)", "lower is better"),
        "rows": [("Cold start", "new process, load, predict once"),
                 ("One prediction", "single record in an app, p50"),
                 ("Memory", "peak of the cold-start process"),
                 ("Deployment", "what the application ships")],
    },
    "zh": {
        "pipeline": "从一个 CSV 到经过校验的模型包，再以四种方式交付",
        "data": ("CSV 或 Parquet", "单张表"),
        "steps": ["不泄漏地切分", "只在训练行上拟合预处理", "训练线性、MLP、GBDT",
                  "在验证集上选择", "在没见过的行上测试一次"],
        "bundle": ("模型包", ["权重、预处理、", "schema 和证据，", "SHA-256 校验"]),
        "targets": [("在 Rust 应用中", "很小的依赖，不需要 Python"),
                    ("编译进你的代码", "no_std，不分配内存"),
                    ("在网页中", "在浏览器里校验"),
                    ("作为 HTTP 服务", "JSON 进，JSON 出")],
        "bench": "交付模型：Tabsmith 与 Python + scikit-learn 对比",
        "legend": ("Tabsmith（小型 MLP）", "Python + scikit-learn（MLPClassifier）", "越低越好"),
        "rows": [("冷启动", "新进程：加载模型并预测一次"),
                 ("单次预测", "应用内的单条记录，p50"),
                 ("内存", "冷启动进程的峰值"),
                 ("交付体积", "应用需要带上的东西")],
    },
}


def svg(width: int, height: int, title: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label={quoteattr(title)}>\n'
            f"<title>{escape(title)}</title>\n{body}</svg>\n")


def text(x, y, s: str, size, fill: str, weight=400, family=SANS, anchor="start") -> str:
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>\n')


def logo(c: dict) -> str:
    """A small table on an anvil, hot as if it were being forged (128 x 128)."""
    table = "M44 58H106M64.7 47V70M85.3 47V70"
    header = "M48 36H102A4 4 0 0 1 106 40V47H44V40A4 4 0 0 1 48 36Z"
    sparks = "M101 28L109 16M91 26L92 12M111 36L123 30"
    anvil = ("M6 76C14 73 22 72 30 72H116V86H104C96 88 92 94 92 104L104 112V120H40V112L52 104"
             "C52 94 48 88 40 86H30C22 86 12 82 6 76Z")
    return (f'<g fill="none" stroke="{c["fill"]}" stroke-width="3.5" stroke-linejoin="round">'
            f'<rect x="44" y="36" width="62" height="34" rx="4"/><path d="{table}"/></g>\n'
            f'<path fill="{c["fill"]}" d="{header}"/>\n'
            f'<path fill="none" stroke="{c["fill"]}" stroke-width="5" stroke-linecap="round" d="{sparks}"/>\n'
            f'<path fill="{c["fg"]}" d="{anvil}"/>\n')


def pipeline(c: dict, w: dict) -> str:
    """CSV -> tabsmith train -> bundle -> the four ways to ship it (1000 x 384)."""
    out = [
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{c["muted"]}"/></marker></defs>\n'
    ]

    def card(x, y, w, h, stroke=None):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c["card"]}" '
                   f'stroke="{stroke or c["line"]}" stroke-width="{2 if stroke else 1}"/>\n')

    def link(d: str):
        out.append(f'<path d="{d}" fill="none" stroke="{c["muted"]}" stroke-width="1.5" '
                   'marker-end="url(#arrow)"/>\n')

    # The data: a sheet with a grid on it.
    card(0, 132, 150, 120)
    out.append(f'<g fill="none" stroke="{c["fg"]}" stroke-width="2" stroke-linejoin="round">'
               '<path d="M58 148H82L92 158V194H58Z"/><path d="M82 148V158H92"/></g>'
               f'<path d="M63 168H87M63 176H87M63 184H87M72 164V189" stroke="{c["muted"]}" '
               'stroke-width="1.5"/>\n')
    out.append(text(75, 220, w["data"][0], 15, c["fg"], 700, anchor="middle"))
    out.append(text(75, 239, w["data"][1], 13.5, c["muted"], anchor="middle"))
    link("M150 192H190")

    card(194, 52, 280, 280, stroke=c["accent"])
    out.append(text(216, 90, "tabsmith train", 17, c["accent"], 700, MONO))
    for i, step in enumerate(w["steps"]):
        y = 132 + 42 * i
        out.append(f'<circle cx="228" cy="{y - 5}" r="11" fill="{c["soft"]}" stroke="{c["accent"]}" '
                   'stroke-width="1.5"/>\n')
        out.append(text(228, y, str(i + 1), 12, c["accent"], 700, anchor="middle"))
        out.append(text(250, y, step, 14.5, c["fg"]))
    link("M474 192H514")

    # The bundle: a box, its lid glowing.
    card(518, 102, 180, 180)
    out.append(f'<path d="M608 118L632 130L608 142L584 130Z" fill="{c["fill"]}"/>'
               f'<path d="M608 118L632 130V158L608 170L584 158V130ZM584 130L608 142L632 130M608 142V170" '
               f'fill="none" stroke="{c["fg"]}" stroke-width="2" stroke-linejoin="round"/>\n')
    out.append(text(608, 198, w["bundle"][0], 16, c["fg"], 700, anchor="middle"))
    for i, line in enumerate(w["bundle"][1]):
        out.append(text(608, 220 + 18 * i, line, 13, c["muted"], anchor="middle"))

    codes = ["tabsmith-runtime", "tabsmith_macros::model!", "WebAssembly", "tabsmith serve"]
    for i, ((title, note), code) in enumerate(zip(w["targets"], codes)):
        y = 6 + 98 * i
        link(f"M698 192C720 192 720 {y + 39} 742 {y + 39}")
        card(746, y, 254, 78)
        out.append(text(764, y + 27, title, 15, c["fg"], 700))
        out.append(text(764, y + 47, code, 13, c["accent"], 600, MONO))
        out.append(text(764, y + 66, note, 13, c["muted"]))
    return "".join(out)


def ms(v: float) -> str:
    return f"{v:.0f} ms"


def us(v: float) -> str:
    return f"{v:.1f} µs" if v < 1000 else f"{v / 1000:.1f} ms"


def mb(v: float) -> str:
    return f"{v:.0f} MB" if v >= 10 else f"{v:.1f} MB"


def bench(c: dict, r: dict, w: dict) -> str:
    """The small MLP against scikit-learn's MLPClassifier where shipping differs (1000 x 336).
    Each row has its own linear scale; a bar is at least 3 px wide so that it shows."""
    ts, py = r["tabsmith"]["candidates"]["small-mlp"], r["python"]["candidates"]["small-mlp"]
    values = [
        (ts["cold_start_seconds"] * 1000, py["cold_start_seconds"] * 1000, ms),
        (ts["app_single_record_us"]["p50"], py["single_record_us"]["p50"], us),
        (ts["cold_start_peak_rss_mb"], py["cold_start_peak_rss_mb"], mb),
        (r["tabsmith"]["app_binary_bytes"] / 1e6, r["python"]["environment_bytes"] / 1e6, mb),
    ]
    ours, theirs, lower = w["legend"]
    out = [
        f'<rect x="0" y="6" width="12" height="12" rx="2" fill="{c["fill"]}"/>\n',
        text(20, 17, ours, 13.5, c["fg"]),
        f'<rect x="200" y="6" width="12" height="12" rx="2" fill="{c["other"]}"/>\n',
        text(220, 17, theirs, 13.5, c["fg"]),
        text(1000, 17, lower, 13, c["muted"], anchor="end"),
    ]
    x0, span = 300, 560
    for i, ((label, note), (t, p, fmt)) in enumerate(zip(w["rows"], values)):
        y = 44 + 72 * i
        out.append(text(0, y + 30, label, 15, c["fg"], 700))
        out.append(text(0, y + 49, note, 12.5, c["muted"]))
        for j, (v, fill, color) in enumerate(((t, c["fill"], c["accent"]), (p, c["other"], c["muted"]))):
            width = max(3.0, span * v / max(t, p))
            by = y + 14 + 26 * j
            out.append(f'<rect x="{x0}" y="{by}" width="{width:.1f}" height="18" rx="3" fill="{fill}"/>\n')
            out.append(text(f"{x0 + width + 8:.1f}", by + 14, fmt(v), 13.5, color, 700))
    return "".join(out)


def social(c: dict) -> str:
    """The repository's social preview (1280 x 640, dark only)."""
    out = [
        '<rect width="1280" height="640" fill="#0d1117"/>\n',
        f'<g transform="translate(96 176) scale(2.3)">\n{logo(c)}</g>\n',
        text(440, 300, "Tabsmith", 118, c["fg"], 700),
        text(446, 366, "Train a model from a CSV and ship it inside", 34, c["muted"]),
        text(446, 412, "your Rust application. No Python.", 34, c["muted"]),
    ]
    x = 446
    for chip in ["deterministic", "compiled into code", "no_std", "WebAssembly"]:
        w = 32 + 11 * len(chip)
        out.append(f'<rect x="{x}" y="452" width="{w}" height="44" rx="22" fill="none" '
                   f'stroke="{c["accent"]}" stroke-width="2"/>\n')
        out.append(text(x + w // 2, 481, chip, 20, c["accent"], 600, anchor="middle"))
        x += w + 14
    out.append(text(446, 576, "github.com/NatMartini/Tabsmith", 26, c["muted"], 400, MONO))
    out.append(f'<rect y="628" width="1280" height="12" fill="{c["fill"]}"/>\n')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=RESULTS,
                    help="the results.json of the benchmark report the chart shows")
    args = ap.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))

    files = {}
    for theme, c in THEMES.items():
        files[f"logo-{theme}.svg"] = svg(128, 128, "Tabsmith", logo(c))
        for lang, w in WORDS.items():
            suffix = "" if lang == "en" else f"-{lang}"
            files[f"pipeline{suffix}-{theme}.svg"] = svg(1000, 384, w["pipeline"], pipeline(c, w))
            files[f"bench{suffix}-{theme}.svg"] = svg(1000, 336, w["bench"], bench(c, results, w))
    files["social-preview.svg"] = svg(1280, 640, "Tabsmith", social(THEMES["dark"]))

    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote docs/assets/{name}")


if __name__ == "__main__":
    main()
