"""Attention-specific harness utilities: benchmark grid, FP32/BF16 references,
the BF16 error floor, and TFLOPS accounting.

Generic guards/timing/aggregation come from the shared integrity library and
are re-exported here for callers (bench_baselines). All scoring math lives
outside the agent-writable workspace.
"""
from __future__ import annotations

import math

import torch

# one implementation for every task, staged beside this file
from avo_harness import (bench_ms, geomean,  # noqa: F401
                         gpu_busy_reason, gpu_meta, max_abs_err)

DEFAULT_GRID = {"heads": 16, "head_dim": 128, "total_tokens": 8192,
                "seqlens": [1024, 2048, 4096, 8192], "causal": [False, True],
                "warmup": 10, "repeats": 30}
REF_CHUNK = 1024


def config_grid(params: dict) -> list[dict]:
    g = {**DEFAULT_GRID, **params}
    kv_heads = int(g.get("kv_heads", g["heads"]))  # GQA when < heads
    configs = []
    for causal in g["causal"]:
        for s in g["seqlens"]:
            batch = max(1, g["total_tokens"] // s)
            configs.append({"batch": batch, "heads": g["heads"],
                            "kv_heads": kv_heads, "seqlen": s,
                            "head_dim": g["head_dim"], "causal": bool(causal)})
    return configs


def make_qkv(cfg: dict, seed: int, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    B, H, S, D = cfg["batch"], cfg["heads"], cfg["seqlen"], cfg["head_dim"]
    Hkv = cfg.get("kv_heads", H)
    q = torch.randn((B, H, S, D), generator=gen, device=device, dtype=torch.bfloat16)
    k = torch.randn((B, Hkv, S, D), generator=gen, device=device, dtype=torch.bfloat16)
    v = torch.randn((B, Hkv, S, D), generator=gen, device=device, dtype=torch.bfloat16)
    return q, k, v


def expand_kv(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[B, Hkv, S, D] -> [B, H, S, D] by repeating each KV head over its group."""
    if x.shape[1] == heads:
        return x
    return x.repeat_interleave(heads // x.shape[1], dim=1)


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
    """FP32 reference on the exact (bf16-representable) input values.
    Supports GQA: k/v with fewer heads are expanded over their groups."""
    B, H = q.shape[0], q.shape[1]
    k, v = expand_kv(k, H), expand_kv(v, H)
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
    k, v = expand_kv(k, H), expand_kv(v, H)
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


def attention_tflops(cfg: dict, ms: float) -> float:
    flops = 4.0 * cfg["batch"] * cfg["heads"] * cfg["seqlen"] ** 2 * cfg["head_dim"]
    if cfg["causal"]:
        flops /= 2.0
    return flops / (ms * 1e-3) / 1e12




