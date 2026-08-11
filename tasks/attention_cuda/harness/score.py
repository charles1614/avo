"""Scoring for the attention_cuda task.

Everything that protects the experiment — GPU busy guard, banned-API scan,
correctness-before-timing, the post-benchmark anti-memoization recheck, result
tokens, structured failures, CUDA-event timing — lives in the shared
`avo_harness` library and is applied by `run_scoring`. This file supplies only
what is specific to attention: how to build the kernel, the benchmark grid,
the correctness reference, and the TFLOPS metric.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import avo_harness as ah  # noqa: E402  (staged beside this file)
import build as builder  # noqa: E402
import common  # noqa: E402

# attention-specific default bans: hand-write the kernel, don't call a fused
# vendor implementation. GEMM primitives (cuBLAS/CUTLASS) remain allowed.
DEFAULT_BANNED_APIS = [
    r"scaled_dot_product_attention",
    r"cudnnMultiHeadAttn", r"cudnn_attention", r"cudnnAttn", r"cudnnFusedAttn",
    r"_flash_attention", r"flash_attn_", r"mem_efficient_attention",
    r"at::native::[A-Za-z_]*attention", r"_scaled_dot_product", r"xformers",
]


def load(args: ah.HarnessArgs):
    module = builder.build(args.workspace, args.params.get("arch_flags", []))
    return module.attention_forward


def configs(kernel_fn, args: ah.HarnessArgs) -> list:
    return common.config_grid(args.params)


def check(kernel_fn, cfg: dict, seed: int, args: ah.HarnessArgs) -> dict:
    res = common.check_config(kernel_fn, cfg, seed)
    return {"ok": res["ok"],
            "detail": (f"max_abs_err={res['max_abs_err']:.5f} > "
                       f"threshold={res['err_threshold']:.5f}"),
            "max_abs_err": res["max_abs_err"],
            "err_threshold": res["err_threshold"]}


def measure(kernel_fn, cfg: dict, args: ah.HarnessArgs) -> dict:
    import torch
    p = args.params
    warmup = int(p.get("warmup", common.DEFAULT_GRID["warmup"]))
    repeats = int(p.get("repeats", common.DEFAULT_GRID["repeats"]))
    q, k, v = common.make_qkv(cfg, seed=int(p.get("rng_seed", 0)) % (2**31))
    causal = cfg["causal"]
    try:
        ms = ah.bench_ms(lambda: kernel_fn(q, k, v, causal), warmup, repeats)
        tflops = common.attention_tflops(cfg, ms)
    finally:
        del q, k, v
        torch.cuda.empty_cache()
    return {"median_ms": ms, "tflops": tflops, "metric_value": tflops}


def main() -> None:
    args = ah.parse_args()
    args.params.setdefault("banned_apis", DEFAULT_BANNED_APIS)
    ah.run_scoring(args, ah.ScoringHooks(
        load=load, configs=configs, check=check, measure=measure,
        meta=lambda: {"warmup": args.params.get("warmup",
                                                common.DEFAULT_GRID["warmup"]),
                      "repeats": args.params.get("repeats",
                                                 common.DEFAULT_GRID["repeats"])},
        correctness_trials=3))


if __name__ == "__main__":
    main()
