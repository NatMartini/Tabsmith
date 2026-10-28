"""Certifies a GPU training path for the tabular-binary recipe, end to end through
the `tabsmith` binary, and writes the evidence.

    python3 tools/gpu_certify.py --tabsmith target/release/tabsmith --device vulkan \
        --out reports/gpu/<date>-<gpu>-<os>
    python3 tools/gpu_certify.py --tabsmith ... --device cuda --precision tf32 --out ...

A path is certified when all four checks pass:
  1. determinism   two independent processes train bit-identical weights;
  2. exact resume  a run killed mid-training and resumed in a new process ends with
                   the weights and history of an uninterrupted run;
  3. CPU agreement validation metrics within tolerance of a CPU run of the same
                   configuration (tight for fp32, looser for tf32);
  4. bundle        the GPU-trained model bundles, and the CPU runtime's self-check passes.
It also times CPU and GPU training on the README's problem and on a larger one,
because a GPU only pays off once the work per step outgrows its launch overhead.

Standard library only. Exit status 0 when certified, 1 otherwise.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

TOLERANCE = {  # |GPU - CPU| on validation metrics
    "fp32": {"balanced_accuracy": 0.005, "roc_auc": 0.002, "log_loss": 0.002},
    "tf32": {"balanced_accuracy": 0.02, "roc_auc": 0.01, "log_loss": 0.01},
}


def cpu_name():
    """The CPU's marketing name where `tabsmith doctor` has none (it reads /proc/cpuinfo)."""
    commands = {
        "Windows": ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
        "Darwin": ["sysctl", "-n", "machdep.cpu.brand_string"],
    }
    cmd = commands.get(platform.system())
    if cmd is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0].strip() if out else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabsmith", required=True, help="a tabsmith binary built with --features vulkan and/or cuda")
    ap.add_argument("--device", choices=["vulkan", "cuda"], required=True)
    ap.add_argument("--gpu", type=int, default=0, help="execution.gpu (see `tabsmith doctor`)")
    ap.add_argument("--precision", choices=["fp32", "tf32"], default="fp32")
    ap.add_argument("--out", required=True, help="evidence directory, e.g. reports/gpu/<date>-<gpu>-<os>")
    ap.add_argument("--large-rows", type=int, default=400_000, help="rows of the larger timing problem")
    ap.add_argument("--work", help="keep the projects here (default: a temporary directory)")
    a = ap.parse_args()
    exe = os.path.abspath(a.tabsmith)
    work = os.path.abspath(a.work) if a.work else tempfile.mkdtemp(prefix="tabsmith-gpu-")
    os.makedirs(work, exist_ok=True)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = []

    def run(args, cwd, expect=0):
        t0 = time.perf_counter()
        p = subprocess.run([exe, *args], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        secs = time.perf_counter() - t0
        log.append({"cmd": ["tabsmith", *args], "cwd": os.path.relpath(cwd, work), "exit": p.returncode, "seconds": round(secs, 3)})
        if p.returncode != expect:
            sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
            raise SystemExit(f"`tabsmith {' '.join(args)}` exited {p.returncode}, expected {expect}")
        return p.stdout

    def runs(project):
        return json.loads(run(["--json", "runs"], project))["result"]

    def project(name, rows=None):
        path = os.path.join(work, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        if rows is None:
            run(["init", name, "--sample-data"], work)
        else:
            run(["init", name], work)
            run(["sample-data", "data/defects.csv", "--rows", str(rows)], path)
            run(["data", "bind", "data/defects.csv", "--target", "defect", "--group", "device_id", "--id", "unit_id"], path)
        return path

    gpu = [
        "--set", f'execution.candidate_device="{a.device}"',
        "--set", f"execution.gpu={a.gpu}",
        "--set", f'execution.precision="{a.precision}"',
    ]
    cpu = ["--set", 'execution.candidate_device="cpu"']
    sha = lambda r: next(x["sha256"] for x in r["artifacts"] if x["name"] == "model.safetensors")

    doctor = json.loads(run(["--json", "doctor"], work))["result"]
    device = next(
        (d for d in doctor["devices"] if d.get("backend") == a.device and d.get("index") == a.gpu),
        None,
    )
    if device is None:
        raise SystemExit(f"no {a.device} GPU with index {a.gpu}: {json.dumps(doctor['devices'], indent=1)}")
    print(f"certifying {a.device} on {device['name']} ({device.get('driver')}), precision {a.precision}")

    # 1. Determinism: two projects, two processes, same configuration.
    det = [project(f"det-{i}") for i in (1, 2)]
    for p in det:
        run(["run", "--candidates", "small-mlp", *gpu], p)
    r1, r2 = runs(det[0])[0], runs(det[1])[0]
    determinism = {
        "run_1_model_sha256": sha(r1),
        "run_2_model_sha256": sha(r2),
        "bitwise_identical_weights": sha(r1) == sha(r2),
        "identical_history": r1["history"] == r2["history"],
        "device_plan": r1["device_plan"],
        "device_receipt": r1["device_receipt"],
    }

    # 2. Exact resume: crash after 130 steps, resume in a new process.
    res = project("resume")
    run(["run", "--candidates", "small-mlp", *gpu, "--simulate-crash-after-steps", "130"], res, expect=137)
    run(["resume", "run-001", *gpu], res)
    resumed = next(r for r in runs(res) if (r.get("lineage") or {}).get("level") == "R2")
    resume = {
        "resumed_from_step": resumed["lineage"]["from_step"],
        "resumed_model_sha256": sha(resumed),
        "uninterrupted_model_sha256": sha(r1),
        "bitwise_identical_weights": sha(resumed) == sha(r1),
        "identical_history": resumed["history"] == r1["history"],
    }

    # 3. Agreement with the CPU.
    ref = project("cpu-reference")
    run(["run", "--candidates", "small-mlp", *cpu], ref)
    rc = runs(ref)[0]
    tol = TOLERANCE[a.precision]
    gm, cm = r1["metrics"]["valid"], rc["metrics"]["valid"]
    agreement = {
        k: {"gpu": gm[k], "cpu": cm[k], "difference": abs(gm[k] - cm[k]), "tolerance": tol[k], "within": abs(gm[k] - cm[k]) <= tol[k]}
        for k in tol
    }

    # 4. The GPU-trained model bundles and passes the CPU runtime's self-check.
    bundle_dir = os.path.join(det[0], "bundles", "gpu-trained")
    run(["bundle", "build", "--run", "run-001", "--allow-unevaluated", "--out", bundle_dir], det[0])
    check = json.loads(run(["--json", "bundle", "check", bundle_dir], det[0]))["result"]["check"]
    bundle = {"self_check_passed": bool(check.get("passed")), "max_abs_probability_diff": check.get("max_abs_probability_diff")}

    # Timing: the README's problem (from the runs above) and a larger one.
    def per_step(r):
        return r["timings"]["train_ms"] / max(r["steps_done"], 1)

    timing = {
        "readme_problem": {
            "model": "`tabsmith init --sample-data` problem, small-mlp hidden [32, 16], batch 64",
            "cpu_ms_per_step": round(per_step(rc), 4),
            "gpu_ms_per_step": round(per_step(r1), 4),
            "cpu_train_s": round(rc["timings"]["train_ms"] / 1e3, 2),
            "gpu_train_s": round(r1["timings"]["train_ms"] / 1e3, 2),
        }
    }
    big = [
        "--set", "train.small-mlp.hidden=[512, 512]",
        "--set", "train.small-mlp.batch_size=2048",
        "--set", "train.small-mlp.early_stopping_patience=0",
    ]
    # Each device trains the large problem for 1 and for 4 epochs. The difference is
    # the cost of 3 epochs of steps alone, without the one-off start-up (CUDA and
    # Vulkan compile their kernels during the first steps).
    large = {}
    for name, extra in (("cpu", cpu), ("gpu", gpu)):
        p = project(f"large-{name}", rows=a.large_rows)
        large[name] = {}
        for epochs in (1, 4):
            run(["run", "--candidates", "small-mlp", *big, "--set", f"train.small-mlp.epochs={epochs}", *extra], p)
            large[name][epochs] = max(runs(p), key=lambda r: r["run_id"])

    def marginal_ms(rs):
        steps = rs[4]["steps_done"] - rs[1]["steps_done"]
        return (rs[4]["timings"]["train_ms"] - rs[1]["timings"]["train_ms"]) / max(steps, 1)

    def startup_ms(rs):
        return max(rs[1]["timings"]["train_ms"] - rs[1]["steps_done"] * marginal_ms(rs), 0.0)

    cpu_ms, gpu_ms = marginal_ms(large["cpu"]), marginal_ms(large["gpu"])
    timing["large_problem"] = {
        "rows": a.large_rows,
        "model": "small-mlp, hidden [512, 512], batch 2048, 4 epochs",
        "steps": large["gpu"][4]["steps_done"],
        "cpu_ms_per_step": round(cpu_ms, 3),
        "gpu_ms_per_step": round(gpu_ms, 3),
        "gpu_startup_s": round(startup_ms(large["gpu"]) / 1e3, 2),
        "cpu_train_s": round(large["cpu"][4]["timings"]["train_ms"] / 1e3, 2),
        "gpu_train_s": round(large["gpu"][4]["timings"]["train_ms"] / 1e3, 2),
        "speedup_per_step": round(cpu_ms / max(gpu_ms, 1e-9), 1),
        "speedup": round(large["cpu"][4]["timings"]["train_ms"] / max(large["gpu"][4]["timings"]["train_ms"], 1e-9), 1),
        "valid_balanced_accuracy": {
            "cpu": large["cpu"][4]["metrics"]["valid"]["balanced_accuracy"],
            "gpu": large["gpu"][4]["metrics"]["valid"]["balanced_accuracy"],
        },
    }

    checks = {
        "determinism": determinism["bitwise_identical_weights"] and determinism["identical_history"],
        "exact_resume": resume["bitwise_identical_weights"] and resume["identical_history"],
        "cpu_agreement": all(v["within"] for v in agreement.values()),
        "bundle": bundle["self_check_passed"],
    }
    certified = all(checks.values())
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system(), platform.system().lower())
    entry = {
        "backend": a.device,
        "device": device["name"],
        "os": os_name,
        "driver": device.get("driver"),
        "precision": r1["device_plan"]["precision"],
        "certified_on": datetime.date.today().isoformat(),
        "evidence": os.path.relpath(out, os.getcwd()).replace(os.sep, "/"),
    }
    result = {
        "certified": certified,
        "checks": checks,
        "registry_entry": entry if certified else None,
        "machine": {"os": platform.platform(), "cpu": doctor["environment"].get("cpu_model") or cpu_name(), "tabsmith": doctor["environment"]["tabsmith_version"]},
        "device": device,
        "precision": a.precision,
        "determinism": determinism,
        "exact_resume": resume,
        "cpu_agreement": agreement,
        "bundle": bundle,
        "timing": timing,
        "steps": log,
    }
    # The evidence is published: the temporary projects become `<work>`.
    text = json.dumps(result, indent=2)
    text = text.replace(json.dumps(work)[1:-1], "<work>").replace(work, "<work>")
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(render(result, a))
    print(json.dumps({"certified": certified, "checks": checks, "timing": timing}, indent=2))
    if not a.work:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if certified else 1


def render(r, a) -> str:
    d, t = r["device"], r["timing"]
    yes = lambda b: "pass" if b else "**FAIL**"
    lines = [
        f"# GPU certification: {a.device} on {d['name']}",
        "",
        f"Made by [`tools/gpu_certify.py`](../../../tools/gpu_certify.py) on {r['registry_entry']['certified_on'] if r['certified'] else datetime.date.today().isoformat()}: "
        f"{r['machine']['os']}, CPU {r['machine']['cpu'] or 'unknown'}, tabsmith {r['machine']['tabsmith']}; "
        f"GPU {d['name']} ({d.get('driver')}), backend `{a.device}`, precision `{a.precision}`. "
        "Raw numbers and every command in [`results.json`](results.json).",
        "",
        f"**Result: {'certified' if r['certified'] else 'not certified'}.**",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
        f"| Determinism: two processes, same weights | {yes(r['checks']['determinism'])} | model SHA-256 `{r['determinism']['run_1_model_sha256'][:16]}…` twice |",
        f"| Exact resume after a crash at step {r['exact_resume']['resumed_from_step']} | {yes(r['checks']['exact_resume'])} | resumed and uninterrupted weights {'identical' if r['exact_resume']['bitwise_identical_weights'] else 'differ'} |",
        f"| Agreement with the CPU (validation) | {yes(r['checks']['cpu_agreement'])} | "
        + ", ".join(f"{k} Δ {v['difference']:.2e} (≤ {v['tolerance']})" for k, v in r["cpu_agreement"].items())
        + " |",
        f"| Bundle + CPU runtime self-check | {yes(r['checks']['bundle'])} | max \\|Δp\\| {r['bundle']['max_abs_probability_diff']} |",
        "",
        "## Training time",
        "",
        "| Problem | CPU train | GPU train | CPU ms/step | GPU ms/step | GPU start-up |",
        "|---|---|---|---|---|---|",
        f"| {t['readme_problem']['model']} | {t['readme_problem']['cpu_train_s']} s | {t['readme_problem']['gpu_train_s']} s | "
        f"{t['readme_problem']['cpu_ms_per_step']} | {t['readme_problem']['gpu_ms_per_step']} (incl. start-up) | – |",
        f"| {t['large_problem']['rows']:,} rows ({t['large_problem']['model']}, {t['large_problem']['steps']} steps) | "
        f"{t['large_problem']['cpu_train_s']} s | {t['large_problem']['gpu_train_s']} s | {t['large_problem']['cpu_ms_per_step']} | "
        f"{t['large_problem']['gpu_ms_per_step']} ({t['large_problem']['speedup_per_step']}× faster) | {t['large_problem']['gpu_startup_s']} s |",
        "",
        "The CPU path is single-threaded Flex. On the large problem, ms/step is the marginal",
        "cost of a step: the difference between a 4-epoch and a 1-epoch run, divided by the",
        "difference in steps, so the GPU's one-off start-up (kernel compilation in the first",
        "steps) is reported separately. On the README's problem a step is microseconds of",
        "arithmetic, less than one GPU launch and copy, so the CPU wins and `auto-certified`",
        "keeps it there; the GPU pays off once the work per step grows.",
        "",
    ]
    if r["certified"]:
        lines += [
            "Registry entry (`crates/tabsmith-workflow/src/certified_devices.json`):",
            "",
            "```json",
            json.dumps(r["registry_entry"], indent=2),
            "```",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
