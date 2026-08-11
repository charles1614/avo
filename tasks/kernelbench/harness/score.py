"""Scoring for KernelBench problems.

Integrity (result tokens, busy guard, correctness-before-timing, the
post-benchmark anti-memoization recheck, structured failures, timing protocol)
comes from the shared `avo_harness` library via `run_scoring`. This file holds
only the KernelBench specifics: the authoritative reference comes from
params["problem_source"] (immutable, part of the eval-cache key — the
agent-visible workspace copy of problem.py only feeds the candidate's own
imports), and the metric is speedup vs PyTorch eager (comparable to fast_p).

Note: no banned-API list. Unlike the hand-written-kernel tasks, reaching for a
better library call IS a legitimate optimization in KernelBench.

Protocol: python harness/score.py --workspace <dir> --params-b64 <b64> --out result.json
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import avo_harness as ah  # noqa: E402  (staged beside this file)

DEFAULTS = {"num_correct_trials": 3, "warmup": 5, "repeats": 20,
            "tolerance": 1e-2}


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_cuda(x):
    import torch
    return x.cuda() if isinstance(x, torch.Tensor) else x


@dataclass
class Candidate:
    ref_mod: object
    ref_model: object
    new_model: object
    weights_transferred: bool


def load(args: ah.HarnessArgs) -> Candidate:
    import torch
    source = args.params.get("problem_source")
    if not source:
        raise ValueError("params.problem_source missing "
                         "(launch via scripts/run_kernelbench.py)")
    # authoritative reference: reconstructed from params, never from workspace
    Path("_reference_problem.py").write_text(source)
    ref_mod = import_from(Path("_reference_problem.py"), "_reference_problem")

    # APPEND, never insert(0): a workspace torch.py would otherwise shadow the
    # real package for anything imported after this point
    sys.path.append(str(args.workspace.resolve()))
    cand_mod = import_from(args.workspace.resolve() / "model_new.py", "model_new")

    seed = int(args.params.get("rng_seed", 0))
    init_inputs = [to_cuda(x) for x in ref_mod.get_init_inputs()]
    torch.manual_seed(seed)
    ref_model = ref_mod.Model(*init_inputs).cuda()
    torch.manual_seed(seed)  # identical init when param order matches
    new_model = cand_mod.ModelNew(*init_inputs).cuda()
    # eval mode: 18 problems use Dropout (random per call => false correctness
    # failures) and 30 use BatchNorm (train mode uses batch stats and mutates
    # running stats). Inference is what we score.
    ref_model.eval()
    new_model.eval()
    # seeded construction only matches while the candidate creates parameters
    # in the reference's order; a restructured/fused model would otherwise be
    # judged against different weights.
    transferred = False
    ref_sd = ref_model.state_dict()
    if set(new_model.state_dict()) == set(ref_sd):
        try:
            new_model.load_state_dict(ref_sd)
            transferred = True
        except RuntimeError:
            pass  # shape mismatch: fall back to seeded init
    return Candidate(ref_mod, ref_model, new_model, transferred)


def configs(cand: Candidate, args: ah.HarnessArgs) -> list:
    return [{"problem": args.params.get("problem_name", "?")}]


def _inputs(cand: Candidate, seed: int):
    import torch
    torch.manual_seed(seed)
    return [to_cuda(x) for x in cand.ref_mod.get_inputs()]


def check(cand: Candidate, cfg: dict, seed: int, args: ah.HarnessArgs) -> dict:
    import torch
    tol = float(args.params["tolerance"])
    torch.cuda.empty_cache()
    inputs = _inputs(cand, seed)
    with torch.no_grad():
        ref_out = cand.ref_model(*inputs)
        new_out = cand.new_model(*inputs)
    refs = ref_out if isinstance(ref_out, (tuple, list)) else [ref_out]
    news = new_out if isinstance(new_out, (tuple, list)) else [new_out]
    if len(refs) != len(news):
        return {"ok": False, "detail": "output arity mismatch"}
    worst = 0.0
    for r, n in zip(refs, news):
        if r.shape != n.shape:
            return {"ok": False,
                    "detail": f"shape mismatch {tuple(n.shape)} vs {tuple(r.shape)}"}
        err = ah.max_abs_err(r, n)
        worst = max(worst, err)
        if not torch.allclose(r.float(), n.float(), atol=tol, rtol=tol):
            return {"ok": False,
                    "detail": f"max_abs_err={err:.6f} beyond atol=rtol={tol}"}
    del inputs, ref_out, new_out
    torch.cuda.empty_cache()
    return {"ok": True, "detail": "", "max_abs_err": worst}


def measure(cand: Candidate, cfg: dict, args: ah.HarnessArgs) -> dict:
    import torch
    p = args.params
    warmup, repeats = int(p["warmup"]), int(p["repeats"])
    inputs = _inputs(cand, int(p.get("rng_seed", 0)))
    with torch.no_grad():
        ref_ms = ah.bench_ms(lambda: cand.ref_model(*inputs), warmup, repeats)
        new_ms = ah.bench_ms(lambda: cand.new_model(*inputs), warmup, repeats)
    speedup = ref_ms / new_ms if new_ms > 0 else 0.0
    return {"ref_ms": ref_ms, "new_ms": new_ms, "speedup": speedup,
            "metric_value": speedup, "throughput": speedup}


def main() -> None:
    args = ah.parse_args(DEFAULTS)
    cand_box: dict = {}

    def load_and_keep(a):
        cand_box["c"] = load(a)
        return cand_box["c"]

    ah.run_scoring(args, ah.ScoringHooks(
        load=load_and_keep, configs=configs, check=check, measure=measure,
        aggregate=lambda rows: rows[0]["metric_value"] if rows else 0.0,
        meta=lambda: {"warmup": args.params["warmup"],
                      "repeats": args.params["repeats"],
                      "tolerance": args.params["tolerance"],
                      "weights_transferred":
                          cand_box["c"].weights_transferred if cand_box else False},
        correctness_trials=int(args.params["num_correct_trials"])))


if __name__ == "__main__":
    main()
