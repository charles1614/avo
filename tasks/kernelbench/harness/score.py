"""Scoring harness for KernelBench problems under AVO's discipline.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json

The AUTHORITATIVE reference comes from params["problem_source"] (immutable,
part of the eval-cache key) — the agent-visible workspace copy of problem.py
only feeds the candidate's own imports. Correctness runs on multiple
content-hash-seeded random trials (stricter than stock KernelBench's single
trial); score = median reference time / median candidate time (speedup vs
PyTorch eager, comparable to fast_p).
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

# fragmentation, not capacity, kills borderline multi-GiB problems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULTS = {"num_correct_trials": 3, "warmup": 5, "repeats": 20,
            "tolerance": 1e-2, "allow_busy": False}
BUSY_MEMORY_MIB = 1024
BUSY_UTIL_PCT = 5


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


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gpu_busy_reason() -> str | None:
    def smi(query, fields):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-{query}={fields}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        return [[c.strip() for c in l.split(",")] for l in out.splitlines() if l.strip()]
    # memory-based, not PID-based: in containers nvidia-smi reports host pids
    # while os.getpid() is the namespace pid, so PID comparison would flag our
    # own process as foreign and fail every eval
    for row in smi("gpu", "memory.used"):
        try:
            if int(row[0]) >= BUSY_MEMORY_MIB:
                return f"{row[0]} MiB already in use on this GPU"
        except (ValueError, IndexError):
            continue
    utils = []
    for _ in range(2):
        for row in smi("gpu", "utilization.gpu"):
            try:
                utils.append(int(row[0]))
            except ValueError:
                pass
        time.sleep(0.25)
    if utils and max(utils) > BUSY_UTIL_PCT:
        return f"GPU utilization at {max(utils)}%"
    return None


def to_cuda(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x


CHUNK_ELEMS = 32_000_000  # compare in ~128 MB fp32 slices, not whole tensors


def compare_chunked(r, n, tol: float) -> tuple[bool, float]:
    """Memory-frugal allclose + max-abs-err: KernelBench v0.1 outputs reach
    multiple GiB, and a whole-tensor diff OOMs alongside inputs/outputs."""
    rf, nf = r.reshape(-1), n.reshape(-1)
    import torch
    ok, max_err = True, 0.0
    for i in range(0, rf.numel(), CHUNK_ELEMS):
        rc = rf[i:i + CHUNK_ELEMS].float()
        nc = nf[i:i + CHUNK_ELEMS].float()
        err = (rc - nc).abs().max().item()
        max_err = max(max_err, err)
        if ok and not torch.allclose(rc, nc, atol=tol, rtol=tol):
            ok = False
        del rc, nc
    return ok, max_err


def bench_ms(fn, warmup: int, repeats: int) -> float:
    import torch
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-token", default="")
    args = ap.parse_args()
    global RESULT_TOKEN
    RESULT_TOKEN = args.result_token
    params = {**DEFAULTS, **json.loads(base64.b64decode(args.params_b64))}

    try:
        import torch
    except ImportError as e:
        fail(args.out, "harness", f"torch not importable: {e}")
        return
    if not torch.cuda.is_available():
        fail(args.out, "harness", "CUDA not available on this host")
        return
    if not params.get("allow_busy") and (busy := gpu_busy_reason()):
        fail(args.out, "harness", f"GPU busy ({busy}); refusing noisy numbers")
        return

    source = params.get("problem_source")
    if not source:
        fail(args.out, "harness", "params.problem_source missing "
             "(runs must be launched via scripts/run_kernelbench.py)")
        return

    # authoritative reference: reconstructed from params, never from workspace
    ref_path = Path("_reference_problem.py")
    ref_path.write_text(source)
    ref_mod = import_from(ref_path, "_reference_problem")

    workspace = Path(args.workspace).resolve()
    # APPEND, never insert(0): a workspace file named torch.py/numpy.py would
    # otherwise shadow the real package for anything imported after this point
    sys.path.append(str(workspace))
    try:
        cand_mod = import_from(workspace / "model_new.py", "model_new")
        ModelNew = cand_mod.ModelNew
    except Exception as e:
        fail(args.out, "compile", f"loading ModelNew failed: {type(e).__name__}: {e}",
             traceback.format_exc())
        return

    rng_seed = int(params.get("rng_seed", 0))
    tol = float(params["tolerance"])
    try:
        init_inputs = [to_cuda(x) for x in ref_mod.get_init_inputs()]
        torch.manual_seed(rng_seed)
        ref_model = ref_mod.Model(*init_inputs).cuda()
        torch.manual_seed(rng_seed)  # identical init for stateful models
        new_model = ModelNew(*init_inputs).cuda()
        # eval mode: 18 KernelBench problems use Dropout (random per call =>
        # false correctness failures) and 30 use BatchNorm (train mode uses
        # batch stats and mutates running stats). Inference is what we score.
        ref_model.eval()
        new_model.eval()
        # seeded construction only matches while the candidate creates
        # parameters in the reference's order; a restructured/fused model
        # would otherwise be judged against different weights. Transfer the
        # reference's weights whenever the parameter set is identical.
        weights_transferred = False
        ref_sd = ref_model.state_dict()
        if set(new_model.state_dict()) == set(ref_sd):
            try:
                new_model.load_state_dict(ref_sd)
                weights_transferred = True
            except RuntimeError:
                pass  # shape mismatch: fall back to seeded init
    except Exception as e:
        fail(args.out, "compile", f"model construction failed: {type(e).__name__}: {e}",
             traceback.format_exc())
        return

    # -- correctness: multiple seeded random trials ---------------------------
    max_err = 0.0
    for trial in range(int(params["num_correct_trials"])):
        torch.cuda.empty_cache()
        seed = (rng_seed * 1_000_003 + trial * 7919) % (2**31)
        torch.manual_seed(seed)
        inputs = [to_cuda(x) for x in ref_mod.get_inputs()]
        try:
            with torch.no_grad():
                ref_out = ref_model(*inputs)
                new_out = new_model(*inputs)
        except Exception as e:
            fail(args.out, "correctness",
                 f"candidate raised on trial {trial}: {type(e).__name__}: {e}",
                 traceback.format_exc())
            return
        refs = ref_out if isinstance(ref_out, (tuple, list)) else [ref_out]
        news = new_out if isinstance(new_out, (tuple, list)) else [new_out]
        if len(refs) != len(news):
            fail(args.out, "correctness", "output arity mismatch")
            return
        for r, n in zip(refs, news):
            if r.shape != n.shape:
                fail(args.out, "correctness",
                     f"shape mismatch: {tuple(n.shape)} vs {tuple(r.shape)}")
                return
            ok, err = compare_chunked(r, n, tol)
            max_err = max(max_err, err)
            if not ok:
                fail(args.out, "correctness",
                     f"trial {trial}: max_abs_err={err:.6f} beyond "
                     f"atol=rtol={tol}")
                return
        del inputs, ref_out, new_out, refs, news
        torch.cuda.empty_cache()

    # -- timing: same inputs, median CUDA-event time for both -----------------
    torch.manual_seed(rng_seed)
    inputs = [to_cuda(x) for x in ref_mod.get_inputs()]
    try:
        with torch.no_grad():
            ref_ms = bench_ms(lambda: ref_model(*inputs),
                              int(params["warmup"]), int(params["repeats"]))
            new_ms = bench_ms(lambda: new_model(*inputs),
                              int(params["warmup"]), int(params["repeats"]))
    except Exception as e:
        fail(args.out, "bench", f"{type(e).__name__}: {e}", traceback.format_exc())
        return

    # post-benchmark re-verification with a FRESH input: a model that memoizes
    # its output would pass the pre-bench trials and then "run" in ~0 ms on the
    # repeated identical bench calls
    torch.manual_seed((rng_seed * 7_919 + 31) % (2**31))
    fresh = [to_cuda(x) for x in ref_mod.get_inputs()]
    try:
        with torch.no_grad():
            r_out, n_out = ref_model(*fresh), new_model(*fresh)
        rr = r_out if isinstance(r_out, (tuple, list)) else [r_out]
        nn = n_out if isinstance(n_out, (tuple, list)) else [n_out]
        for r, n in zip(rr, nn):
            ok, err = compare_chunked(r, n, tol)
            if not ok:
                fail(args.out, "correctness",
                     f"post-benchmark recheck FAILED (max_abs_err={err:.6f}) — "
                     "the model returns stale/cached output for new inputs")
                return
    except Exception as e:
        fail(args.out, "correctness",
             f"post-bench recheck raised: {type(e).__name__}: {e}",
             traceback.format_exc())
        return

    speedup = ref_ms / new_ms if new_ms > 0 else 0.0
    write(args.out, {
        "correct": True, "score": speedup, "error": None,
        "configs": [{"problem": params.get("problem_name", "?"),
                     "ref_ms": ref_ms, "new_ms": new_ms,
                     "speedup": speedup, "max_abs_err": max_err,
                     "throughput": speedup}],
        "meta": {"weights_transferred": weights_transferred,
                 "gpu": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "warmup": params["warmup"], "repeats": params["repeats"],
                 "tolerance": tol}})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # any escape still yields structured output
        import sys as _sys
        out = None
        for i, a in enumerate(_sys.argv):
            if a == "--out" and i + 1 < len(_sys.argv):
                out = _sys.argv[i + 1]
        if out:
            fail(out, "harness",
                 f"unhandled {type(e).__name__}: {e}", traceback.format_exc())
        raise
