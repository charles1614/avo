"""Nsight Compute profiling entry for KernelBench problems.

Profiles one post-warmup forward of the candidate ModelNew (NVTX-scoped so
warmup/JIT launches are excluded) and reports the top kernels by duration
with SpeedOfLight/Occupancy/LaunchStats metrics — enough to see which kernel
dominates and whether it is memory- or compute-bound.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

NCU_SECTIONS = ["SpeedOfLight", "LaunchStats", "Occupancy"]
KEY_METRICS = [
    ("Duration", "duration_us"),
    ("Compute (SM) Throughput", "compute_sol_pct"),
    ("Memory Throughput", "memory_sol_pct"),
    ("Achieved Occupancy", "achieved_occupancy_pct"),
    ("Registers Per Thread", "registers_per_thread"),
    ("Grid Size", "grid"),
    ("Block Size", "block"),
]
TOP_N = 10

TARGET_SCRIPT = """\
import json, sys, torch
sys.path.insert(0, sys.argv[1])            # workspace (candidate imports)
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
        model(*inputs)                      # warmup outside the NVTX range
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
    return [{"kernel": r.get("Kernel Name", "?"), "id": r.get("ID", ""),
             "metric": r.get("Metric Name", "?"),
             "value": r.get("Metric Value", ""), "unit": r.get("Metric Unit", "")}
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))]


def _num(v: str) -> float:
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return 0.0


def summarize(rows: list[dict]) -> tuple[str, list]:
    launches: dict[str, dict] = {}
    for row in rows:
        label = next((lab for name, lab in KEY_METRICS
                      if row["metric"].strip() == name), None)
        if label:
            k = launches.setdefault(f"{row['id']}|{row['kernel']}", {})
            k[label] = f"{row['value']} {row['unit']}".strip()
            k["_kernel"] = row["kernel"]
    ranked = sorted(launches.values(),
                    key=lambda m: _num(m.get("duration_us", "0").split()[0]),
                    reverse=True)[:TOP_N]
    lines = [f"ncu profile of one ModelNew forward ({len(launches)} kernel "
             f"launches; top {len(ranked)} by duration):"]
    for m in ranked:
        lines.append(f"\n## {m['_kernel'][:120]}")
        for _, label in KEY_METRICS:
            if label in m:
                lines.append(f"  {label:24} {m[label]}")
    lines.append("\nReading: the top kernel is your optimization target. "
                 "compute_sol vs memory_sol tells you the bound; both low "
                 "means latency/launch-overhead bound (consider fusion).")
    return "\n".join(lines), ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(base64.b64decode(args.params_b64))

    ncu = shutil.which("ncu")
    if not ncu:
        fail(args.out, "harness", "ncu (Nsight Compute) not on PATH")
        return
    source = params.get("problem_source")
    if not source:
        fail(args.out, "harness", "params.problem_source missing")
        return
    Path("_reference_problem.py").write_text(source)
    Path("_profile_target.py").write_text(TARGET_SCRIPT)

    cmd = [ncu, "--csv", "--nvtx", "--nvtx-include", "avo_profile/",
           "--launch-count", "60", "--target-processes", "all"]
    for s in NCU_SECTIONS:
        cmd += ["--section", s]
    cmd += [sys.executable, "_profile_target.py",
            str(Path(args.workspace).resolve())]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        fail(args.out, "harness", "ncu timed out after 900s")
        return
    combined = proc.stdout + "\n" + proc.stderr
    if "ERR_NVGPUCTRPERM" in combined:
        fail(args.out, "harness",
             "GPU perf counters restricted to root (ERR_NVGPUCTRPERM); enable "
             "NVreg_RestrictProfilingToAdminUsers=0 or run as root")
        return
    rows = parse_ncu_csv(proc.stdout)
    if not rows:
        fail(args.out, "harness",
             f"ncu produced no metrics (exit {proc.returncode})", combined)
        return
    summary, ranked = summarize(rows)
    Path(args.out).write_text(json.dumps(
        {"correct": True, "score": 0.0, "error": None, "configs": [],
         "meta": {"summary": summary, "top_kernels": ranked}}))


if __name__ == "__main__":
    main()
