"""Shared harness library — the integrity layer every AVO task inherits.

Staged automatically next to each task's harness by the runners, so a task
harness just does `import avo_harness as ah`. Task code supplies ONLY what is
task-specific (how to load the candidate, the reference, the config grid, the
metric); everything that protects the experiment lives here, once:

  * result-token emission            (forged-result rejection)
  * structured failure reporting     (stage-tagged, cacheability semantics)
  * container-safe GPU busy guard    (memory-based, not PID-based)
  * banned-API source scan           (no delegating to the library under test)
  * CUDA-event timing + geomean      (identical protocol across tasks)
  * run_scoring(): the ENFORCED sequence
        ban scan -> load -> correctness -> benchmark -> POST-BENCH RECHECK
    so a new task cannot silently omit the anti-memoization recheck.

Deliberately dependency-free (stdlib + torch only): it runs inside the eval
sandbox, on remote hosts, without the avo framework installed.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# fragmentation, not capacity, kills borderline multi-GiB problems
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BUSY_MEMORY_MIB = 1024
BUSY_UTIL_PCT = 5
CHUNK_ELEMS = 32_000_000
DEFAULT_BANNED: list[str] = []


# ---------------------------------------------------------------------------
# protocol: arguments, results, failures
# ---------------------------------------------------------------------------

@dataclass
class HarnessArgs:
    workspace: Path
    params: dict
    out: str
    token: str


def parse_args(defaults: dict | None = None) -> HarnessArgs:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-token", default="")
    a = ap.parse_args()
    params = {**(defaults or {}), **json.loads(base64.b64decode(a.params_b64))}
    return HarnessArgs(Path(a.workspace), params, a.out, a.result_token)


def write_result(args: HarnessArgs, *, correct: bool, score: float,
                 configs: list | None = None, meta: dict | None = None,
                 error: dict | None = None) -> None:
    """Single writer for every task: always stamps the per-eval token, without
    which the runner rejects a positive score as forged."""
    payload = {"correct": correct, "score": score, "error": error,
               "configs": configs or [],
               "meta": {**(meta or {}), "result_token": args.token}}
    Path(args.out).write_text(json.dumps(payload, indent=1))


def fail(args: HarnessArgs, stage: str, detail: str, log_tail: str = "") -> None:
    """Structured failure. Stage drives cacheability upstream:
    compile/correctness/bench are code properties (cached); harness is a
    transient machine condition (never cached)."""
    write_result(args, correct=False, score=0.0,
                 error={"stage": stage, "detail": str(detail)[:2000],
                        "log_tail": log_tail[-20_000:]})
    sys.exit(0)


def install_crash_handler(args: HarnessArgs) -> None:
    """Any escape still produces a parseable result instead of a bare crash."""
    hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            write_result(args, correct=False, score=0.0,
                         error={"stage": "harness",
                                "detail": f"unhandled {exc_type.__name__}: {exc}",
                                "log_tail": "".join(
                                    traceback.format_exception(exc_type, exc, tb))[-20_000:]})
        finally:
            hook(exc_type, exc, tb)
    sys.excepthook = _hook


# ---------------------------------------------------------------------------
# environment guards
# ---------------------------------------------------------------------------

def _smi(query: str, fields: str) -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [[c.strip() for c in line.split(",")]
            for line in out.splitlines() if line.strip()]


def gpu_busy_reason() -> str | None:
    """Refuse to benchmark when the GPU is already working.

    Memory-based, NOT PID-based: inside a container nvidia-smi reports host
    pids while os.getpid() is the namespace pid, so comparing pids marks our
    own process foreign and fails every eval. This runs before we allocate,
    so any sizeable usage belongs to someone else.
    """
    for row in _smi("gpu", "memory.used"):
        try:
            used = int(row[0])
        except (ValueError, IndexError):
            continue
        if used >= BUSY_MEMORY_MIB:
            return f"{used} MiB already in use on this GPU"
    utils: list[int] = []
    for _ in range(2):
        for row in _smi("gpu", "utilization.gpu"):
            try:
                utils.append(int(row[0]))
            except (ValueError, IndexError):
                pass
        time.sleep(0.25)
    if utils and max(utils) > BUSY_UTIL_PCT:
        return f"GPU utilization at {max(utils)}%"
    return None


def scan_banned_apis(workspace, patterns: list[str] | None = None) -> str | None:
    """First banned symbol in workspace source, or None. Comments stripped so
    a mention in a comment doesn't false-trip. Tasks declare their own list
    (`task_params.banned_apis`) — an attention task bans SDPA/cuDNN-attention,
    a GEMM task bans cuBLAS, etc. Without it the score measures the vendor
    library, not the agent."""
    import re
    pats = patterns if patterns is not None else DEFAULT_BANNED
    if not pats:
        return None
    rx = re.compile("|".join(pats))
    for p in sorted(Path(workspace).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (
                ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py"):
            continue
        text = p.read_text(errors="replace")
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"(?m)#.*$", "", text)
        m = rx.search(text)
        if m:
            return f"{m.group(0)} in {p.name}"
    return None


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def bench_ms(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    """Median CUDA-event time. One timing protocol for every task."""
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


def max_abs_err(ref, got) -> float:
    """Chunked max |ref-got|: task outputs reach multiple GiB and a whole
    tensor diff OOMs next to the inputs."""
    import torch
    rf, gf = ref.reshape(-1), got.reshape(-1)
    worst = 0.0
    for i in range(0, rf.numel(), CHUNK_ELEMS):
        a = rf[i:i + CHUNK_ELEMS].float()
        b = gf[i:i + CHUNK_ELEMS].float()
        worst = max(worst, (a - b).abs().max().item())
        del a, b
    return worst


def geomean(xs: list[float]) -> float:
    if not xs or any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def gpu_meta() -> dict:
    import torch
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda}


# ---------------------------------------------------------------------------
# the enforced scoring sequence
# ---------------------------------------------------------------------------

@dataclass
class ScoringHooks:
    """Task-specific pieces. Everything else is fixed by run_scoring()."""
    load: Callable[[HarnessArgs], object]
    # (candidate, args) -> list of config dicts to score
    configs: Callable[[object, HarnessArgs], list]
    # (candidate, cfg, seed, args) -> {"ok": bool, "detail": str, **metrics}
    check: Callable[[object, dict, int, HarnessArgs], dict]
    # (candidate, cfg, args) -> {"metric_value": float, **row}
    measure: Callable[[object, dict, HarnessArgs], dict]
    # rows -> final score (default: geomean of "metric_value")
    aggregate: Callable[[list], float] | None = None
    meta: Callable[[], dict] = field(default=lambda: {})
    correctness_trials: int = 3


def run_scoring(args: HarnessArgs, hooks: ScoringHooks) -> None:
    """The invariant scoring sequence, identical for every task:

      1. GPU busy guard          (unless params.allow_busy)
      2. banned-API source scan  (params.banned_apis)
      3. load candidate          (task hook; failures => stage "compile")
      4. correctness             N seeded trials per config, BEFORE any timing
      5. benchmark               (task hook)
      6. POST-BENCH RECHECK      fresh seeds — catches a candidate that
                                 memoizes its output and then "runs" in ~0 ms
      7. write result            with the per-eval token

    A task cannot skip 1, 2, 6 or 7 by forgetting them: they are not its code.
    """
    install_crash_handler(args)
    try:
        import torch
    except ImportError as e:
        fail(args, "harness", f"torch not importable: {e}")
        return
    if not torch.cuda.is_available():
        fail(args, "harness", "CUDA not available on this host")
        return
    if not args.params.get("allow_busy") and (busy := gpu_busy_reason()):
        fail(args, "harness", f"GPU busy ({busy}); refusing noisy numbers")
        return

    banned = scan_banned_apis(args.workspace, args.params.get("banned_apis"))
    if banned:
        fail(args, "compile",
             f"workspace calls a forbidden pre-built implementation ({banned}). "
             "Implement it yourself; delegating to the library under test is "
             "not a valid solution.")
        return

    try:
        candidate = hooks.load(args)
    except Exception as e:
        fail(args, "compile", f"{type(e).__name__}: {e}", traceback.format_exc())
        return

    rng_seed = int(args.params.get("rng_seed", 0))
    configs = hooks.configs(candidate, args)

    checked = []
    for i, cfg in enumerate(configs):
        last = None
        for trial in range(hooks.correctness_trials):
            seed = (rng_seed * 1_000_003 + i * 101 + trial) % (2**31)
            try:
                last = hooks.check(candidate, cfg, seed, args)
            except Exception as e:
                fail(args, "correctness",
                     f"candidate raised on {cfg}: {type(e).__name__}: {e}",
                     traceback.format_exc())
                return
            if not last.get("ok"):
                fail(args, "correctness",
                     f"{cfg} trial {trial}: {last.get('detail', 'mismatch')}")
                return
        checked.append(last or {})

    rows = []
    try:
        for cfg, chk in zip(configs, checked):
            row = hooks.measure(candidate, cfg, args)
            rows.append({**cfg, **{k: v for k, v in chk.items()
                                   if k not in ("ok", "detail")}, **row})
    except Exception as e:
        fail(args, "bench", f"{type(e).__name__}: {e}", traceback.format_exc())
        return

    # anti-memoization: fresh seeds AFTER timing
    for i, cfg in enumerate(configs):
        seed = (rng_seed * 7_919 + i * 104_729 + 31) % (2**31)
        try:
            res = hooks.check(candidate, cfg, seed, args)
        except Exception as e:
            fail(args, "correctness",
                 f"post-bench recheck raised on {cfg}: {type(e).__name__}: {e}",
                 traceback.format_exc())
            return
        if not res.get("ok"):
            fail(args, "correctness",
                 f"post-benchmark recheck FAILED on {cfg}: "
                 f"{res.get('detail', 'mismatch')} — the candidate returns "
                 "stale/cached output for new inputs")
            return

    agg = hooks.aggregate or (lambda rs: geomean([r["metric_value"] for r in rs]))
    write_result(args, correct=True, score=agg(rows), configs=rows,
                 meta={**gpu_meta(), **hooks.meta()})
