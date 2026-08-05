"""Shared harness utilities: config grid, FP32 reference (chunked), BF16 error
floor, CUDA-event timing, TFLOPS accounting, GPU-busy detection.

All scoring math lives here, outside the agent-writable workspace.
"""
from __future__ import annotations

import math
import os
import statistics
import subprocess
import time

import torch

DEFAULT_GRID = {"heads": 16, "head_dim": 128, "total_tokens": 8192,
                "seqlens": [1024, 2048, 4096, 8192], "causal": [False, True],
                "warmup": 10, "repeats": 30}
REF_CHUNK = 1024


def config_grid(params: dict) -> list[dict]:
    g = {**DEFAULT_GRID, **params}
    configs = []
    for causal in g["causal"]:
        for s in g["seqlens"]:
            batch = max(1, g["total_tokens"] // s)
            configs.append({"batch": batch, "heads": g["heads"], "seqlen": s,
                            "head_dim": g["head_dim"], "causal": bool(causal)})
    return configs


def make_qkv(cfg: dict, seed: int, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (cfg["batch"], cfg["heads"], cfg["seqlen"], cfg["head_dim"])
    q = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)
    return q, k, v


def _chunked_attention(qh, kh, vh, causal: bool, out_dtype=torch.float32,
                       p_dtype=None):
    """Attention for one [S, D] head; q/k/v may be fp32 or bf16.
    p_dtype: cast softmax probabilities before the PV matmul (bf16 emulation)."""
    s_len = qh.shape[0]
    scale = 1.0 / math.sqrt(qh.shape[1])
    out = torch.empty(qh.shape, device=qh.device, dtype=out_dtype)
    for i0 in range(0, s_len, REF_CHUNK):
        i1 = min(i0 + REF_CHUNK, s_len)
        s = (qh[i0:i1] @ kh.T).float() * scale
        if causal:
            cols = torch.arange(s_len, device=s.device)
            rows = torch.arange(i0, i1, device=s.device)
            s.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        if p_dtype is not None:
            p = p.to(p_dtype)
        out[i0:i1] = (p @ vh).to(out_dtype)
    return out


def reference_fp32(q, k, v, causal: bool) -> torch.Tensor:
    """FP32 reference on the exact (bf16-representable) input values."""
    B, H = q.shape[0], q.shape[1]
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            out[b, h] = _chunked_attention(q[b, h].float(), k[b, h].float(),
                                           v[b, h].float(), causal)
    return out


def reference_bf16(q, k, v, causal: bool) -> torch.Tensor:
    """An honest BF16 implementation (bf16 matmuls, fp32 softmax, bf16 PV):
    its deviation from the FP32 reference is the intrinsic error floor."""
    B, H = q.shape[0], q.shape[1]
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            out[b, h] = _chunked_attention(q[b, h], k[b, h], v[b, h], causal,
                                           p_dtype=torch.bfloat16)
    return out


def check_config(kernel_fn, cfg: dict, seed: int) -> dict:
    q, k, v = make_qkv(cfg, seed)
    out = kernel_fn(q, k, v, cfg["causal"]).float()
    ref = reference_fp32(q, k, v, cfg["causal"])
    floor = (reference_bf16(q, k, v, cfg["causal"]) - ref).abs().max().item()
    err = (out - ref).abs().max().item()
    threshold = 2.0 * floor + 1e-3
    return {"max_abs_err": err, "err_threshold": threshold,
            "ok": bool(err <= threshold) and math.isfinite(err)}


def bench_ms(fn, warmup: int, repeats: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def attention_tflops(cfg: dict, ms: float) -> float:
    flops = 4.0 * cfg["batch"] * cfg["heads"] * cfg["seqlen"] ** 2 * cfg["head_dim"]
    if cfg["causal"]:
        flops /= 2.0
    return flops / (ms * 1e-3) / 1e12


def geomean(xs: list[float]) -> float:
    if not xs or any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


BUSY_MEMORY_MIB = 1024  # idle desktop/monitoring daemons sit well below this
BUSY_UTIL_PCT = 5


def _smi(query: str, fields: str) -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [[c.strip() for c in line.split(",")]
            for line in out.splitlines() if line.strip()]


def gpu_busy_reason() -> str | None:
    """Refuse to bench only when the GPU is actually doing work: another
    process holding real memory, or sustained nonzero utilization. Persistent
    desktop/monitoring daemons with small footprints are fine."""
    me = str(os.getpid())
    for row in _smi("compute-apps", "pid,process_name,used_memory"):
        if len(row) >= 3 and row[0] != me:
            try:
                mem = int(row[2])
            except ValueError:
                continue
            if mem >= BUSY_MEMORY_MIB:
                return (f"process {row[0]} ({row[1]}) holds {mem} MiB on the GPU")
    utils = []
    for _ in range(2):
        for row in _smi("gpu", "utilization.gpu"):
            try:
                utils.append(int(row[0]))
            except ValueError:
                pass
        time.sleep(0.25)
    if utils and max(utils) > BUSY_UTIL_PCT:
        return f"GPU utilization at {max(utils)}% before benching"
    return None


def gpu_meta() -> dict:
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda}
