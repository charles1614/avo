#!/usr/bin/env python3
"""KernelBench campaign runner: unattended AVO evolution across a problem set.

For each selected problem it materializes a seed workspace (reference wrapped
as ModelNew), embeds the authoritative problem source into task_params, and
runs one bounded evolution. Crash-safe free running: finished problems are
skipped on restart, so `nohup python scripts/run_kernelbench.py ... &` can be
killed and relaunched at any time. Ends with a fast_p report.

Usage (on the GPU host, after fetch_kb.py has pulled kernelbench):
  python scripts/run_kernelbench.py --config configs/kernelbench_h100.yaml \
      --problems level1 --limit 10 --confirm-spend
  python scripts/run_kernelbench.py --config ... --report-only
  python scripts/run_kernelbench.py --config ... --problems level1 --dry   # seed eval only, no LLM
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from avo.config import RunConfig
from avo.eval.ssh_runner import make_runner

PROBLEMS_ROOT = Path("knowledge_base/external/kernelbench/KernelBench")
SEED_TEMPLATE = Path("tasks/kernelbench/seed_template")
FAST_THRESHOLDS = (1.0, 1.5, 2.0)


def slug(problem_rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", problem_rel.replace(".py", "")).strip("-").lower()


def discover(selector: str) -> list[Path]:
    """selector: 'level1' | 'level1,level2' | a file of relative paths |
    a comma list of problem paths like level1/97_ScaledDotProductAttention.py"""
    if Path(selector).is_file() and not selector.endswith(".py"):
        rels = [l.strip() for l in Path(selector).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        return [PROBLEMS_ROOT / r for r in rels]
    out: list[Path] = []
    for part in selector.split(","):
        part = part.strip()
        if part.endswith(".py"):
            out.append(PROBLEMS_ROOT / part)
        else:
            out.extend(sorted((PROBLEMS_ROOT / part).glob("*.py")))
    return out


def materialize_seed(problem_file: Path, seeds_root: Path) -> Path:
    seed_dir = seeds_root / slug(str(problem_file.relative_to(PROBLEMS_ROOT)))
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    shutil.copytree(SEED_TEMPLATE, seed_dir)
    shutil.copy2(problem_file, seed_dir / "problem.py")
    return seed_dir


def problem_config(base: dict, problem_file: Path, seed_dir: Path) -> RunConfig:
    cfg = json.loads(json.dumps(base))  # deep copy
    rel = str(problem_file.relative_to(PROBLEMS_ROOT))
    cfg["run_name"] = f"kb-{slug(rel)}"
    cfg["seed_dir"] = str(seed_dir)
    cfg.setdefault("task_params", {})
    cfg["task_params"]["problem_name"] = rel
    cfg["task_params"]["problem_source"] = problem_file.read_text()
    return RunConfig.model_validate(cfg)


def finished_run(runs_dir: Path, run_name: str) -> Path | None:
    candidates = sorted(runs_dir.glob(f"{run_name}-*"),
                        key=lambda p: p.stat().st_mtime)
    for c in reversed(candidates):
        if (c / "summary.json").exists():
            return c
    return None


def aggregate(runs_dir: Path, run_names: dict[str, str], out_path: Path) -> str:
    rows, speedups = [], []
    for rel, run_name in sorted(run_names.items()):
        run = finished_run(runs_dir, run_name)
        if run is None:
            rows.append((rel, None, None, "pending/failed"))
            continue
        s = json.loads((run / "summary.json").read_text())
        best = s.get("best_score") or 0.0
        speedups.append(best)
        rows.append((rel, best, s.get("versions"), f"${s.get('usd', 0):.2f}"))
    done = [r for r in rows if r[1] is not None]
    lines = ["# KernelBench campaign report", "",
             f"problems attempted: {len(rows)}, finished: {len(done)}", ""]
    for p in FAST_THRESHOLDS:
        frac = (sum(1 for v in speedups if v > p) / len(speedups)) if speedups else 0.0
        lines.append(f"- fast_{p:g} (best speedup > {p:g}x): "
                     f"{frac:.2%} ({sum(1 for v in speedups if v > p)}/{len(speedups)})")
    lines += ["", "| problem | best speedup | versions | cost |", "|---|---|---|---|"]
    for rel, best, versions, cost in rows:
        b = f"{best:.3f}x" if best is not None else "—"
        lines.append(f"| {rel} | {b} | {versions if versions is not None else '—'} | {cost} |")
    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base RunConfig YAML "
                    "(task: tasks/kernelbench); campaign caps under `campaign:`")
    ap.add_argument("--problems", default="level1",
                    help="level name(s), problem path(s), or a list file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="evaluate each problem's seed only (no LLM)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    campaign = raw.pop("campaign", {}) or {}
    max_campaign_usd = float(campaign.get("max_total_usd", 100.0))
    root = Path(".")
    runs_dir = root / raw.get("runs_dir", "runs")
    report_path = root / "results" / "kernelbench_report.md"

    if not PROBLEMS_ROOT.is_dir():
        print(f"{PROBLEMS_ROOT} missing — run: python scripts/fetch_kb.py --only kernelbench")
        return 1
    problems = discover(args.problems)[: args.limit]
    run_names = {str(p.relative_to(PROBLEMS_ROOT)): f"kb-{slug(str(p.relative_to(PROBLEMS_ROOT)))}"
                 for p in problems}

    if args.report_only:
        print(aggregate(runs_dir, run_names, report_path))
        return 0

    if not args.dry and not args.confirm_spend:
        print(f"Would run AVO evolution on {len(problems)} problems with LLM "
              f"'{raw['llm']['model']}' (campaign cap ${max_campaign_usd}).\n"
              "Re-run with --confirm-spend (or use --dry for seed-only evals).")
        return 1

    seeds_root = runs_dir / "kb_seeds"
    spent = 0.0
    for i, problem in enumerate(problems, 1):
        rel = str(problem.relative_to(PROBLEMS_ROOT))
        run_name = run_names[rel]
        if (done := finished_run(runs_dir, run_name)) is not None:
            print(f"[{i}/{len(problems)}] {rel}: already finished ({done.name}); skipping")
            continue
        if spent >= max_campaign_usd:
            print(f"campaign cap ${max_campaign_usd} reached; stopping")
            break
        seed_dir = materialize_seed(problem, seeds_root)
        config = problem_config(raw, problem, seed_dir)

        if args.dry:
            from avo.config import load_task_spec
            task_dir = root / config.task
            task = load_task_spec(task_dir)
            runner = make_runner(config.runner, f"kb-dry-{slug(rel)}")
            params = {**config.task_params, "rng_seed": 12345}
            r = runner.score(seed_dir, task_dir / task.harness_dir,
                             task.score_entry, params)
            print(f"[{i}/{len(problems)}] {rel}: correct={r.correct} "
                  f"speedup={r.score:.3f} "
                  f"{(r.error or {}).get('detail', '')[:120]}")
            continue

        print(f"[{i}/{len(problems)}] {rel}: evolving ...")
        from avo.evolution.controller import Controller
        from avo.llm.base import make_client
        llm = make_client(config.llm)
        try:
            summary = Controller(config, llm, project_root=root).run()
            spent += summary.get("usd", 0.0)
            print(f"  -> best {summary.get('best_score', 0):.3f}x, "
                  f"${summary.get('usd', 0):.2f} (campaign ${spent:.2f})")
        except Exception as e:
            print(f"  -> FAILED: {type(e).__name__}: {e}")

    print()
    print(aggregate(runs_dir, run_names, report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
