"""Scoring harness for the attention_cuda task.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json
Stages: compile -> correctness on every grid config (3 randomized trials each,
zero score on any failure) -> CUDA-event benchmark -> geomean TFLOPS.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import traceback
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

CORRECTNESS_TRIALS = 3


RESULT_TOKEN = ""


def write(out_path: str, payload: dict) -> None:
    payload.setdefault("meta", {})["result_token"] = RESULT_TOKEN
    Path(out_path).write_text(json.dumps(payload, indent=1))


def fail(out_path: str, stage: str, detail: str, log_tail: str = "") -> None:
    write(out_path, {"correct": False, "score": 0.0,
                     "error": {"stage": stage, "detail": detail[:2000],
                               "log_tail": log_tail[-20_000:]},
                     "configs": [], "meta": {}})
    sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-token", default="")
    args = ap.parse_args()
    global RESULT_TOKEN
    RESULT_TOKEN = args.result_token
    params = json.loads(base64.b64decode(args.params_b64))

    try:
        import torch
    except ImportError as e:
        fail(args.out, "harness", f"torch not importable: {e}")
        return
    if not torch.cuda.is_available():
        fail(args.out, "harness", "CUDA not available on this host")
        return

    import common
    import build as builder

    if not params.get("allow_busy", False):
        busy = common.gpu_busy_reason()
        if busy:
            # stage "harness": a transient machine condition, never cached
            fail(args.out, "harness",
                 f"GPU busy ({busy}); refusing to produce noisy numbers")
            return

    # -- forbid delegating to a pre-built fused attention op -------------------
    # the task is to WRITE the kernel; calling cuDNN/SDPA/flash-attn measures
    # the library, not the agent, and makes cross-model comparison meaningless
    import checks
    banned = checks.scan_banned_apis(Path(args.workspace))
    if banned:
        fail(args.out, "compile",
             f"workspace calls a forbidden pre-built attention op ({banned}). "
             "You must implement the attention kernel yourself; delegating to "
             "a fused library API is not a valid solution.")
        return

    # -- compile ---------------------------------------------------------------
    try:
        module = builder.build(Path(args.workspace),
                               params.get("arch_flags", []))
    except Exception as e:
        fail(args.out, "compile", f"{type(e).__name__}: {e}",
             traceback.format_exc())
        return
    kernel_fn = module.attention_forward

    configs = common.config_grid(params)
    rng_seed = int(params.get("rng_seed", 0))

    # -- correctness (all configs before any benching) ---------------------------
    checked = []
    for i, cfg in enumerate(configs):
        for trial in range(CORRECTNESS_TRIALS):
            seed = (rng_seed * 1_000_003 + i * 101 + trial) % (2**31)
            try:
                res = common.check_config(kernel_fn, cfg, seed)
            except Exception as e:
                fail(args.out, "correctness",
                     f"kernel raised on config {cfg}: {type(e).__name__}: {e}",
                     traceback.format_exc())
                return
            if not res["ok"]:
                fail(args.out, "correctness",
                     f"config {cfg} trial {trial}: max_abs_err={res['max_abs_err']:.5f} "
                     f"> threshold={res['err_threshold']:.5f}")
                return
        checked.append(res)

    # -- benchmark ----------------------------------------------------------------
    warmup = int(params.get("warmup", common.DEFAULT_GRID["warmup"]))
    repeats = int(params.get("repeats", common.DEFAULT_GRID["repeats"]))
    results = []
    try:
        for cfg, chk in zip(configs, checked):
            q, k, v = common.make_qkv(cfg, seed=rng_seed % (2**31))
            causal = cfg["causal"]
            ms = common.bench_ms(lambda: kernel_fn(q, k, v, causal),
                                 warmup, repeats)
            results.append({**cfg, "max_abs_err": chk["max_abs_err"],
                            "err_threshold": chk["err_threshold"],
                            "median_ms": ms,
                            "tflops": common.attention_tflops(cfg, ms)})
            del q, k, v
            torch.cuda.empty_cache()
    except Exception as e:
        fail(args.out, "bench", f"{type(e).__name__}: {e}", traceback.format_exc())
        return

    # -- post-benchmark re-verification (anti-memoization) ---------------------
    # a kernel that caches its output would pass the pre-bench checks and then
    # time near-zero on the repeated identical bench calls; re-check with a
    # FRESH seed after timing to catch stale/cached results
    for i, cfg in enumerate(configs):
        seed = (rng_seed * 7_919 + i * 104_729 + 31) % (2**31)
        try:
            res = common.check_config(kernel_fn, cfg, seed)
        except Exception as e:
            fail(args.out, "correctness",
                 f"post-bench recheck raised on {cfg}: {type(e).__name__}: {e}",
                 traceback.format_exc())
            return
        if not res["ok"]:
            fail(args.out, "correctness",
                 f"post-benchmark recheck FAILED on {cfg} "
                 f"(max_abs_err={res['max_abs_err']:.5f}) — the kernel returns "
                 "stale/cached output for new inputs")
            return

    score = common.geomean([r["tflops"] for r in results])
    write(args.out, {"correct": True, "score": score, "error": None,
                     "configs": results,
                     "meta": {**common.gpu_meta(),
                              "warmup": warmup, "repeats": repeats}})


if __name__ == "__main__":
    main()
