"""Two-stage auto-profiling for KernelBench problems (any level).

Stage 1 — torch.profiler over one full post-warmup forward: complete op and
kernel rankings by device time (no external tool, no perf-counter
permissions). This answers "where does the time go" even for L3/L4 models
with hundreds of launches.

Stage 2 — ncu scoped with -k to only the top kernels from stage 1:
SpeedOfLight/Occupancy/LaunchStats for the kernels that matter. Skipped
gracefully (stage 1 still returned) when ncu is missing or counters are
root-restricted.

The methodology is selected automatically; the agent just calls `profile`.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

NCU_SECTIONS = ["SpeedOfLight", "LaunchStats", "Occupancy"]
KEY_METRICS = [
    ("Duration", "duration"),
    ("Compute (SM) Throughput", "compute_sol_pct"),
    ("Memory Throughput", "memory_sol_pct"),
    ("Achieved Occupancy", "achieved_occupancy_pct"),
    ("Registers Per Thread", "registers_per_thread"),
    ("Grid Size", "grid"),
    ("Block Size", "block"),
]
TOP_OPS = 12
TOP_KERNELS = 8
NCU_DEEP_KERNELS = 2

STAGE1_SCRIPT = """\
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import importlib.util
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
ref = load("_reference_problem.py", "_reference_problem")
cand = load(sys.argv[1] + "/model_new.py", "model_new")
init = [x.cuda() if hasattr(x, "cuda") else x for x in ref.get_init_inputs()]
torch.manual_seed(0)
model = cand.ModelNew(*init).cuda()
torch.manual_seed(1)
inputs = [x.cuda() if hasattr(x, "cuda") else x for x in ref.get_inputs()]
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(3):
        model(*inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model(*inputs)
        torch.cuda.synchronize()

def dev_time(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0

ops = sorted(({"name": e.key[:120], "device_us": round(dev_time(e), 1),
               "calls": e.count}
              for e in prof.key_averages() if dev_time(e) > 0),
             key=lambda r: -r["device_us"])
kernels = {}
for e in prof.events():
    for k in getattr(e, "kernels", []) or []:
        d = kernels.setdefault(k.name[:160], {"device_us": 0.0, "calls": 0})
        d["device_us"] += float(k.duration)
        d["calls"] += 1
kernel_rows = sorted(({"name": n, **v} for n, v in kernels.items()),
                     key=lambda r: -r["device_us"])
json.dump({"ops": ops, "kernels": kernel_rows},
          open("_stage1.json", "w"))
"""

STAGE2_SCRIPT = """\
import sys, torch
sys.path.insert(0, sys.argv[1])
import importlib.util
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
ref = load("_reference_problem.py", "_reference_problem")
cand = load(sys.argv[1] + "/model_new.py", "model_new")
init = [x.cuda() if hasattr(x, "cuda") else x for x in ref.get_init_inputs()]
torch.manual_seed(0)
model = cand.ModelNew(*init).cuda()
torch.manual_seed(1)
inputs = [x.cuda() if hasattr(x, "cuda") else x for x in ref.get_inputs()]
with torch.no_grad():
    for _ in range(3):
        model(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("avo_profile")
    model(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
"""


def fail(out_path: str, stage: str, detail: str, log_tail: str = "") -> None:
    Path(out_path).write_text(json.dumps(
        {"correct": False, "score": 0.0,
         "error": {"stage": stage, "detail": detail[:2000],
                   "log_tail": log_tail[-20_000:]},
         "configs": [], "meta": {}}))
    sys.exit(0)


def parse_ncu_csv(text: str) -> list[dict]:
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "Kernel Name" in l and "Metric Name" in l), None)
    if start is None:
        return []
    return [{"kernel": r.get("Kernel Name", "?"),
             "metric": r.get("Metric Name", "?"),
             "value": r.get("Metric Value", ""), "unit": r.get("Metric Unit", "")}
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))]


def render_stage1(stage1: dict) -> str:
    total = sum(r["device_us"] for r in stage1.get("kernels", [])) or \
        sum(r["device_us"] for r in stage1.get("ops", [])) or 1.0
    lines = ["# Stage 1 — torch.profiler, one full forward (complete coverage)",
             "", "Top ops by device time:"]
    for r in stage1.get("ops", [])[:TOP_OPS]:
        lines.append(f"  {r['device_us']:>10.1f} us  x{r['calls']:<4} "
                     f"({100 * r['device_us'] / total:4.1f}%)  {r['name']}")
    if stage1.get("kernels"):
        lines += ["", "Top kernels by device time:"]
        for r in stage1["kernels"][:TOP_KERNELS]:
            lines.append(f"  {r['device_us']:>10.1f} us  x{r['calls']:<4} "
                         f"({100 * r['device_us'] / total:4.1f}%)  {r['name']}")
    return "\n".join(lines)


def render_stage2(rows: list[dict]) -> str:
    per_kernel: dict[str, dict] = {}
    for row in rows:
        label = next((lab for name, lab in KEY_METRICS
                      if row["metric"].strip() == name), None)
        if label:
            per_kernel.setdefault(row["kernel"], {})[label] = \
                f"{row['value']} {row['unit']}".strip()
    lines = ["", "# Stage 2 — ncu deep metrics for the dominant kernel(s)"]
    for kernel, m in per_kernel.items():
        lines.append(f"\n## {kernel[:120]}")
        for _, label in KEY_METRICS:
            if label in m:
                lines.append(f"  {label:24} {m[label]}")
    lines.append("\nReading: optimize the top stage-1 entry first. compute_sol "
                 "vs memory_sol gives the bound; both low => latency/launch-"
                 "overhead bound (fuse kernels); many tiny launches in stage 1 "
                 "=> fusion or CUDA graphs.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(base64.b64decode(args.params_b64))
    source = params.get("problem_source")
    if not source:
        fail(args.out, "harness", "params.problem_source missing")
        return
    Path("_reference_problem.py").write_text(source)
    ws = str(Path(args.workspace).resolve())

    # -- stage 1: torch.profiler (always available) ----------------------------
    Path("_stage1_target.py").write_text(STAGE1_SCRIPT)
    p1 = subprocess.run([sys.executable, "_stage1_target.py", ws],
                        capture_output=True, text=True, timeout=900)
    if p1.returncode != 0 or not Path("_stage1.json").exists():
        fail(args.out, "harness", "torch.profiler stage failed",
             p1.stdout + "\n" + p1.stderr)
        return
    stage1 = json.loads(Path("_stage1.json").read_text())
    summary = render_stage1(stage1)

    # -- stage 2: ncu on the top kernels only -----------------------------------
    ncu = shutil.which("ncu")
    top = [r["name"] for r in stage1.get("kernels", [])[:NCU_DEEP_KERNELS]]
    if not ncu:
        summary += ("\n\n# Stage 2 skipped: ncu not on PATH "
                    "(stage 1 ranking above is still authoritative)")
    elif not top:
        summary += "\n\n# Stage 2 skipped: no kernel events recorded"
    else:
        Path("_stage2_target.py").write_text(STAGE2_SCRIPT)
        kfilter = "regex:" + "|".join(re.escape(n[:60]) for n in top)
        cmd = [ncu, "--csv", "--nvtx", "--nvtx-include", "avo_profile/",
               "-k", kfilter, "--launch-count", "8",
               "--target-processes", "all"]
        for s in NCU_SECTIONS:
            cmd += ["--section", s]
        cmd += [sys.executable, "_stage2_target.py", ws]
        try:
            p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            combined = p2.stdout + "\n" + p2.stderr
            if "ERR_NVGPUCTRPERM" in combined:
                summary += ("\n\n# Stage 2 skipped: GPU perf counters are "
                            "root-restricted (ERR_NVGPUCTRPERM); enable "
                            "NVreg_RestrictProfilingToAdminUsers=0 to unlock "
                            "deep metrics. Stage 1 ranking is still valid.")
            else:
                rows = parse_ncu_csv(p2.stdout)
                summary += (render_stage2(rows) if rows else
                            f"\n\n# Stage 2 skipped: ncu returned no metrics "
                            f"(exit {p2.returncode})")
        except subprocess.TimeoutExpired:
            summary += "\n\n# Stage 2 skipped: ncu timed out"

    Path(args.out).write_text(json.dumps(
        {"correct": True, "score": 0.0, "error": None, "configs": [],
         "meta": {"summary": summary, "stage1": stage1}}))


if __name__ == "__main__":
    main()
