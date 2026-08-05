"""Scoring harness for the sort_py task.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json
Never trusts the workspace: bans sorted()/.sort() via AST, checks correctness on
randomized inputs, then times throughput.
"""
from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

DEFAULT_SIZES = [500, 2000, 8000]
CORRECTNESS_TRIALS = 10
TIMING_REPEATS = 5
SLOW_SINGLE_RUN_S = 5.0  # if one run is slower than this, skip repeats


def fail(stage: str, detail: str, out_path: str) -> None:
    result = {"correct": False, "score": 0.0,
              "error": {"stage": stage, "detail": detail, "log_tail": ""},
              "configs": [], "meta": {}}
    Path(out_path).write_text(json.dumps(result, indent=1))
    sys.exit(0)


def banned_constructs(source: str) -> str | None:
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
                root = name.split(".")[0]
                if root in ("subprocess", "ctypes", "os"):
                    return f"banned import '{root}' at line {node.lineno}"
    return None


def load_solution(workspace: Path):
    path = workspace / "solution.py"
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


def check_correctness(sort_fn, rng: random.Random) -> None:
    cases = [[], [1], [2, 1], [5, 5, 5]]
    for _ in range(CORRECTNESS_TRIALS):
        n = rng.randint(2, 2000)
        cases.append([rng.randint(-10**6, 10**6) for _ in range(n)])
    for case in cases:
        original = list(case)
        got = sort_fn(case)
        expected = sorted(original)
        if case != original:
            raise ValueError("input list was mutated")
        if got != expected:
            raise ValueError(
                f"wrong result for n={len(original)}: "
                f"got[:8]={got[:8] if isinstance(got, list) else type(got)}, "
                f"expected[:8]={expected[:8]}")


def bench(sort_fn, sizes: list[int], rng: random.Random) -> list[dict]:
    configs = []
    for n in sizes:
        arr = [rng.randint(-10**6, 10**6) for _ in range(n)]
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
        configs.append({"size": n, "median_ms": median_s * 1e3,
                        "throughput": n / median_s / 1e3})  # kElem/s
    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = json.loads(base64.b64decode(args.params_b64))
    sizes = params.get("sizes", DEFAULT_SIZES)
    rng = random.Random(params.get("rng_seed", random.SystemRandom().randint(0, 2**31)))

    try:
        sort_fn = load_solution(Path(args.workspace))
    except (SyntaxError, ValueError, FileNotFoundError, AttributeError, Exception) as e:
        fail("compile" if isinstance(e, SyntaxError) else "correctness",
             f"{type(e).__name__}: {e}", args.out)
        return

    try:
        check_correctness(sort_fn, rng)
    except Exception as e:
        fail("correctness", f"{type(e).__name__}: {e}", args.out)
        return

    try:
        configs = bench(sort_fn, sizes, rng)
        score = math.exp(sum(math.log(c["throughput"]) for c in configs) / len(configs))
    except Exception as e:
        fail("bench", f"{type(e).__name__}: {e}", args.out)
        return

    result = {"correct": True, "score": score, "error": None, "configs": configs,
              "meta": {"python": sys.version.split()[0], "sizes": sizes}}
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
