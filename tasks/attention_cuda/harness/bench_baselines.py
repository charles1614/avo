"""Benchmark baselines on the same grid with the same timing code as candidate
scoring: PyTorch SDPA (forced flash / cudnn / efficient / math backends) and
the flash-attn package (FA2; FA3 via flash_attn_interface on Hopper).

Protocol matches score.py; result.json carries the numbers in meta.baselines.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

MATH_BACKEND_MAX_SEQLEN = 4096  # math backend materializes S x S; cap it


def cfg_key(cfg: dict) -> str:
    return f"s{cfg['seqlen']}_b{cfg['batch']}_{'causal' if cfg['causal'] else 'full'}"


def sdpa_fn(backend_name: str):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {"sdpa_flash": SDPBackend.FLASH_ATTENTION,
               "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
               "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
               "sdpa_math": SDPBackend.MATH}[backend_name]

    def run(q, k, v, causal):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return run


def flash_attn_fn(fa3: bool):
    if fa3:
        from flash_attn_interface import flash_attn_func  # FA3 (Hopper)
    else:
        from flash_attn import flash_attn_func  # FA2

    def run(q, k, v, causal):
        # flash-attn expects [B, S, H, D]
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2),
                              v.transpose(1, 2), causal=causal)
        if isinstance(out, tuple):  # FA3 may return (out, lse)
            out = out[0]
        return out.transpose(1, 2)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)  # unused; protocol compat
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    params = json.loads(base64.b64decode(args.params_b64))

    import torch
    import common

    if not torch.cuda.is_available():
        Path(args.out).write_text(json.dumps(
            {"correct": False, "score": 0.0,
             "error": {"stage": "harness", "detail": "CUDA not available",
                       "log_tail": ""},
             "configs": [], "meta": {}}))
        return

    configs = common.config_grid(params)
    warmup = int(params.get("warmup", common.DEFAULT_GRID["warmup"]))
    repeats = int(params.get("repeats", common.DEFAULT_GRID["repeats"]))

    candidates: dict = {n: sdpa_fn(n) for n in
                        ["sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "sdpa_math"]}
    try:
        candidates["flash_attn2"] = flash_attn_fn(fa3=False)
    except ImportError:
        pass
    try:
        candidates["flash_attn3"] = flash_attn_fn(fa3=True)
    except ImportError:
        pass

    per_config: dict = {}
    geomeans: dict = {}
    failures: dict = {}
    for name, fn in candidates.items():
        rows = {}
        for cfg in configs:
            if name == "sdpa_math" and cfg["seqlen"] > MATH_BACKEND_MAX_SEQLEN:
                continue
            q, k, v = common.make_qkv(cfg, seed=0)
            causal = cfg["causal"]
            try:
                fn(q, k, v, causal)  # probe support for this config
                ms = common.bench_ms(lambda: fn(q, k, v, causal), warmup, repeats)
                rows[cfg_key(cfg)] = round(common.attention_tflops(cfg, ms), 3)
            except Exception as e:
                failures.setdefault(name, {})[cfg_key(cfg)] = f"{type(e).__name__}: {e}"
            finally:
                del q, k, v
                torch.cuda.empty_cache()
        if rows:
            per_config[name] = rows
            full_grid = [r for kk, r in rows.items()]
            geomeans[name] = round(common.geomean(full_grid), 3)

    baselines = {"geomeans": geomeans, "per_config": per_config,
                 "failures": failures, **common.gpu_meta(),
                 "warmup": warmup, "repeats": repeats}
    Path(args.out).write_text(json.dumps(
        {"correct": True, "score": 0.0, "error": None, "configs": [],
         "meta": {"baselines": baselines}}, indent=1))


if __name__ == "__main__":
    main()
