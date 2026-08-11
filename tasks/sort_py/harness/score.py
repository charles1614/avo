"""Scoring for the sort_py toy task (CPU-only).

Uses the same shared `avo_harness` sequence as the GPU tasks — result tokens,
correctness before timing, and the post-benchmark recheck all apply — with
requires_cuda=False so no GPU guard runs. Task specifics: the banned-construct
AST check (you must implement the sort, not call the builtin) and elements/sec.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json
"""
from __future__ import annotations

import ast
import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import avo_harness as ah  # noqa: E402  (staged beside this file)

DEFAULTS = {"sizes": [500, 2000, 8000]}
TIMING_REPEATS = 5
SLOW_SINGLE_RUN_S = 5.0
BANNED_IMPORT_ROOTS = ("subprocess", "ctypes", "os")


def banned_constructs(source: str) -> str | None:
    """The builtin sort is the thing being optimized: calling it is delegation
    (this task's analogue of banned_apis)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "sorted":
                return f"call to builtin sorted() at line {node.lineno}"
            if isinstance(fn, ast.Attribute) and fn.attr == "sort":
                return f".sort() call at line {node.lineno}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.split(".")[0] in BANNED_IMPORT_ROOTS:
                    return f"banned import '{name}' at line {node.lineno}"
    return None


def load(args: ah.HarnessArgs):
    path = args.workspace / "solution.py"
    if not path.exists():
        raise FileNotFoundError("solution.py missing from workspace")
    banned = banned_constructs(path.read_text())
    if banned:
        raise ValueError(f"banned construct: {banned}")
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "sort_list"):
        raise AttributeError("solution.py must define sort_list(arr)")
    return mod.sort_list


def configs(sort_fn, args: ah.HarnessArgs) -> list:
    return [{"size": n} for n in args.params["sizes"]]


def check(sort_fn, cfg: dict, seed: int, args: ah.HarnessArgs) -> dict:
    rng = random.Random(seed)
    cases = [[], [1], [2, 1], [5, 5, 5],
             [rng.randint(-10**6, 10**6) for _ in range(min(cfg["size"], 2000))],
             [rng.randint(-50, 50) for _ in range(200)]]  # many duplicates
    for case in cases:
        original = list(case)
        got = sort_fn(case)
        if case != original:
            return {"ok": False, "detail": "input list was mutated"}
        if got != sorted(original):
            return {"ok": False,
                    "detail": f"wrong result for n={len(original)}: got[:8]="
                              f"{got[:8] if isinstance(got, list) else type(got)}"}
    return {"ok": True, "detail": ""}


def measure(sort_fn, cfg: dict, args: ah.HarnessArgs) -> dict:
    rng = random.Random(cfg["size"])
    arr = [rng.randint(-10**6, 10**6) for _ in range(cfg["size"])]
    times = []
    for _ in range(TIMING_REPEATS):
        data = list(arr)
        t0 = time.perf_counter()
        sort_fn(data)
        dt = time.perf_counter() - t0
        times.append(dt)
        if dt > SLOW_SINGLE_RUN_S:
            break
    median_s = statistics.median(times)
    kelems = cfg["size"] / median_s / 1e3
    return {"median_ms": median_s * 1e3, "throughput": kelems,
            "metric_value": kelems}


def main() -> None:
    args = ah.parse_args(DEFAULTS)
    ah.run_scoring(args, ah.ScoringHooks(
        load=load, configs=configs, check=check, measure=measure,
        meta=lambda: {"python": sys.version.split()[0],
                      "sizes": args.params["sizes"]},
        correctness_trials=3, requires_cuda=False))


if __name__ == "__main__":
    main()
