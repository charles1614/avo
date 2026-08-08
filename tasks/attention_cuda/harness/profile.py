"""Nsight Compute profiling entry for the attention task.

Protocol: python harness/profile.py --workspace <dir> --params-b64 <b64> --out result.json
Profiles one benchmark-grid config's kernel launches under ncu (SpeedOfLight,
LaunchStats, Occupancy sections) and returns a curated, agent-readable
summary in meta.summary. Degrades gracefully when ncu is missing or perf
counters are root-restricted (ERR_NVGPUCTRPERM).
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

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

NCU_SECTIONS = ["SpeedOfLight", "LaunchStats", "Occupancy"]
KEY_METRICS = [  # (ncu metric display name, short label)
    ("Duration", "duration"),
    ("Compute (SM) Throughput", "compute_sol_pct"),
    ("Memory Throughput", "memory_sol_pct"),
    ("DRAM Throughput", "dram_sol_pct"),
    ("Achieved Occupancy", "achieved_occupancy_pct"),
    ("Theoretical Occupancy", "theoretical_occupancy_pct"),
    ("Registers Per Thread", "registers_per_thread"),
    ("Static Shared Memory Per Block", "static_smem_per_block"),
    ("Dynamic Shared Memory Per Block", "dynamic_smem_per_block"),
    ("Grid Size", "grid"),
    ("Block Size", "block"),
    ("Waves Per SM", "waves_per_sm"),
]

TARGET_SCRIPT = """\
import json, sys
sys.path.insert(0, "harness")
import torch
import build as builder
import common
cfg = json.loads(sys.argv[1])
mod = builder.build(sys.argv[2], json.loads(sys.argv[3]))
q, k, v = common.make_qkv(cfg, seed=0)
for _ in range(4):
    mod.attention_forward(q, k, v, cfg["causal"])
torch.cuda.synchronize()
"""


def fail(out_path: str, stage: str, detail: str, log_tail: str = "") -> None:
    Path(out_path).write_text(json.dumps(
        {"correct": False, "score": 0.0,
         "error": {"stage": stage, "detail": detail[:2000],
                   "log_tail": log_tail[-20_000:]},
         "configs": [], "meta": {}}))
    sys.exit(0)


def parse_ncu_csv(text: str) -> list[dict]:
    """ncu --csv rows -> [{kernel, metric, value, unit}]. Skips banner lines
    before the CSV header."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "Kernel Name" in l and "Metric Name" in l), None)
    if start is None:
        return []
    rows = []
    for rec in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        rows.append({"kernel": rec.get("Kernel Name", "?"),
                     "metric": rec.get("Metric Name", "?"),
                     "value": rec.get("Metric Value", ""),
                     "unit": rec.get("Metric Unit", "")})
    return rows


def summarize(rows: list[dict], cfg: dict) -> tuple[str, dict]:
    per_kernel: dict[str, dict] = {}
    for row in rows:
        wanted = next((label for name, label in KEY_METRICS
                       if row["metric"].strip() == name), None)
        if wanted:
            k = per_kernel.setdefault(row["kernel"], {})
            k[wanted] = f"{row['value']} {row['unit']}".strip()
    lines = [f"ncu profile of config {cfg} "
             f"(one launch, after warmup):"]
    for kernel, m in per_kernel.items():
        lines.append(f"\n## {kernel[:120]}")
        for _, label in KEY_METRICS:
            if label in m:
                lines.append(f"  {label:26} {m[label]}")
    lines.append(
        "\nReading: compute_sol vs memory_sol tells you the bound "
        "(both low => latency/occupancy bound). Low achieved vs theoretical "
        "occupancy => launch config or register pressure. High "
        "registers_per_thread with spills => rebalance or reduce live state.")
    return "\n".join(lines), per_kernel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(base64.b64decode(args.params_b64))

    ncu = shutil.which("ncu")
    if not ncu:
        fail(args.out, "harness",
             "ncu (Nsight Compute) not on PATH — install nsight-compute or "
             "add /usr/local/cuda/bin to PATH")
        return

    import common
    configs = common.config_grid(params)
    idx = int(params.get("profile_config_index", 0)) % len(configs)
    cfg = configs[idx]

    Path("_profile_target.py").write_text(TARGET_SCRIPT)
    cmd = [ncu, "--csv", "--launch-skip", "2", "--launch-count", "1",
           "--target-processes", "all"]
    for s in NCU_SECTIONS:
        cmd += ["--section", s]
    cmd += [sys.executable, "_profile_target.py", json.dumps(cfg),
            args.workspace, json.dumps(params.get("arch_flags", []))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        fail(args.out, "harness", "ncu timed out after 900s")
        return

    combined = proc.stdout + "\n" + proc.stderr
    if "ERR_NVGPUCTRPERM" in combined:
        fail(args.out, "harness",
             "GPU performance counters are restricted to root on this host "
             "(ERR_NVGPUCTRPERM). Enable with: "
             "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
             "sudo tee /etc/modprobe.d/nvidia-profiler.conf && reboot "
             "(or run ncu as root). Profiling unavailable until then.")
        return
    rows = parse_ncu_csv(proc.stdout)
    if not rows:
        fail(args.out, "harness",
             f"ncu produced no metrics (exit {proc.returncode})", combined)
        return

    summary, metrics = summarize(rows, cfg)
    Path(args.out).write_text(json.dumps(
        {"correct": True, "score": 0.0, "error": None, "configs": [],
         "meta": {"summary": summary, "metrics": metrics,
                  "profiled_config": cfg}}))


if __name__ == "__main__":
    main()
