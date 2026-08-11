"""CLI entry point.

Cost-safety invariant: only `run`, `resume`, and scripts/llm_smoke.py can call
an LLM, and all of them require --confirm-spend. `eval-once`, `baselines`,
`report`, and `rebench` never construct an LLM client.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avo.config import RunConfig, load_run_config, load_task_spec
from avo.eval.cache import EvalCache, eval_key
from avo.eval.ssh_runner import make_runner


def _spend_banner(config: RunConfig) -> str:
    llm, b = config.llm, config.budgets
    lines = [
        "This command makes LLM API calls that cost money.",
        f"  provider/model : {llm.provider} / {llm.model}"
        + (f" @ {llm.base_url}" if llm.base_url else ""),
        f"  prices         : ${llm.price_input_per_mtok}/M in, "
        f"${llm.price_output_per_mtok}/M out",
        f"  budgets        : max_usd=${b.max_usd}, max_total_tokens={b.max_total_tokens}, "
        f"max_versions={b.max_versions}, max_steps={b.max_steps}",
        f"  key env var    : {llm.key_env_name()} "
        f"({'set' if llm.resolve_api_key() else 'NOT SET'})",
        "",
        "Re-run with --confirm-spend to proceed.",
    ]
    return "\n".join(lines)


def _require_spend_confirmation(args, config: RunConfig) -> None:
    if not args.confirm_spend:
        print(_spend_banner(config))
        sys.exit(1)
    if (config.llm.price_input_per_mtok <= 0
            or config.llm.price_output_per_mtok <= 0):
        print("[avo] WARNING: token prices are 0 in the config -> the max_usd "
              "cap cannot trigger. max_total_tokens "
              f"({config.budgets.max_total_tokens}) is the only spend backstop.")


def cmd_run(args) -> int:
    config = load_run_config(args.config)
    _require_spend_confirmation(args, config)
    from avo.evolution.controller import Controller
    from avo.llm.base import make_client
    llm = make_client(config.llm)
    controller = Controller(config, llm, project_root=Path(args.root))
    controller.run()
    return 0


def cmd_resume(args) -> int:
    run_dir = Path(args.run)
    config = load_run_config(run_dir / "config.yaml")  # JSON is valid YAML
    _require_spend_confirmation(args, config)
    from avo.evolution.controller import Controller
    from avo.llm.base import make_client
    llm = make_client(config.llm)
    controller = Controller(config, llm, project_root=Path(args.root),
                            run_dir=run_dir)
    controller.run()
    return 0


def cmd_eval_once(args) -> int:
    config = load_run_config(args.config)
    root = Path(args.root)
    task_dir = root / config.task
    task = load_task_spec(task_dir)
    workspace = Path(args.workspace) if args.workspace else task_dir / task.seed_dir
    harness = task_dir / task.harness_dir
    runner = make_runner(config.runner, "eval-once")

    from avo.eval.scoring import cacheable
    cache = EvalCache(root / config.runs_dir / "eval_once_cache")
    key = eval_key(workspace, harness, config.task_params, runner.identity())
    result = None if args.fresh else cache.get(key)
    if result is None:
        params = {**config.task_params, "rng_seed": int(key[:12], 16) % (2**31)}
        result = runner.score(workspace, harness, task.score_entry, params)
        if cacheable(result):
            cache.put(key, result)
    print(result.to_json())
    return 0 if result.correct else 2


def cmd_baselines(args) -> int:
    config = load_run_config(args.config)
    root = Path(args.root)
    task_dir = root / config.task
    task = load_task_spec(task_dir)
    bench_entry = "bench_baselines.py"
    if not (task_dir / task.harness_dir / bench_entry).exists():
        print(f"task {task.name} has no {bench_entry}")
        return 1
    runner = make_runner(config.runner, "baselines")
    result = runner.score(task_dir / task.seed_dir, task_dir / task.harness_dir,
                          bench_entry, config.task_params)
    out_dir = root / config.runs_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.name}_{config.runner.host or 'local'}.json"
    out.write_text(json.dumps(result.meta.get("baselines", result.meta), indent=1))
    print(f"[avo] baselines written to {out}")
    print(result.to_json())
    return 0


def _latest_run(root: Path) -> Path | None:
    runs = [p for p in (root / "runs").glob("*") if (p / "lineage.jsonl").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def cmd_dashboard(args) -> int:
    from avo.report.dashboard import build, watch
    run_dir = Path(args.run) if args.run else _latest_run(Path(args.root))
    if run_dir is None:
        print("no runs found under runs/ — pass --run explicitly")
        return 1
    baselines = Path(args.baselines) if args.baselines else None
    out = build(run_dir, baselines, refresh_s=args.watch)
    print(f"[avo] dashboard: {out}")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    if args.watch:
        try:
            watch(run_dir, baselines, interval_s=args.watch)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_export(args) -> int:
    """Package a run into a committable results/<run_id>/ directory: lineage,
    scores, evolved-kernel git history (bundle), final source, baselines,
    dashboard. No LLM."""
    import shutil
    import subprocess
    run_dir = Path(args.run)
    dest = Path(args.dest) / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    for name in ("lineage.jsonl", "summary.json", "state.json", "config.yaml"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)
    if (run_dir / "evals").is_dir():
        shutil.copytree(run_dir / "evals", dest / "evals", dirs_exist_ok=True)
    (dest / "logs").mkdir(exist_ok=True)
    sup = run_dir / "logs" / "supervisor.jsonl"
    if sup.exists():
        shutil.copy2(sup, dest / "logs" / "supervisor.jsonl")
    if args.with_transcripts:
        for f in (run_dir / "logs").glob("*"):
            shutil.copy2(f, dest / "logs" / f.name)

    workspace = run_dir / "workspace"
    subprocess.run(["git", "-C", str(workspace), "bundle", "create",
                    str((dest / "workspace.bundle").resolve()), "--all"],
                   check=True, capture_output=True)
    kernel_dir = dest / "final_solution"
    shutil.rmtree(kernel_dir, ignore_errors=True)
    shutil.copytree(workspace, kernel_dir,
                    ignore=shutil.ignore_patterns(".git", "__pycache__"))

    baselines_dir = run_dir.parent / "baselines"
    if baselines_dir.is_dir():
        shutil.copytree(baselines_dir, dest / "baselines", dirs_exist_ok=True)
    from avo.report.dashboard import build as build_dash
    shutil.copy2(build_dash(run_dir), dest / "dashboard.html")
    for name in ("report.md", "rebench.md"):
        f = run_dir / "report" / name
        if f.exists():
            shutil.copy2(f, dest / name)

    (dest / "README.md").write_text(
        f"# AVO run export: {run_dir.name}\n\n"
        "- `lineage.jsonl` / `evals/` — committed versions and their full score records\n"
        "- `workspace.bundle` — complete git history of the evolution "
        "(`git clone workspace.bundle kernel-history` to inspect every version)\n"
        "- `final_solution/` — the best committed solution's source tree\n"
        "- `baselines/`, `dashboard.html` — reference numbers and visualization\n\n"
        "Re-verify the headline score on matching hardware:\n"
        "```\navo eval-once --config <the config in config.yaml> "
        "--workspace final_solution --fresh\n```\n"
        "Note: the agentic evolution itself is not bit-reproducible (LLM "
        "sampling); what is reproducible is verification of every committed "
        "artifact and re-running the method.\n")
    print(f"[avo] exported to {dest}")
    return 0


def cmd_audit(args) -> int:
    """Contamination audit of a finished (or running) route — no LLM."""
    from avo.evolution.integrity import audit_run, write_report
    report = audit_run(Path(args.run), isolation=args.isolation)
    out = write_report(Path(args.run), report)
    print(json.dumps(report.summary(), indent=1))
    print(f"[avo] full report: {out}")
    return 2 if report.contaminated else 0


def cmd_report(args) -> int:
    from avo.report.report import write_report
    path = write_report(Path(args.run),
                        Path(args.baselines) if args.baselines else None)
    print(f"[avo] report written to {path}")
    print(path.read_text())
    return 0


def cmd_rebench(args) -> int:
    config = load_run_config(args.config)
    root = Path(args.root)
    task_dir = root / config.task
    task = load_task_spec(task_dir)
    runner = make_runner(config.runner, "rebench")
    from avo.report.report import rebench
    out = rebench(Path(args.run), runner, task_dir, task.score_entry,
                  config.task_params, rounds=args.rounds)
    print(f"[avo] rebench written to {out}")
    print(out.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="avo",
        description="AVO: agentic variation operators (arXiv:2603.24517 reproduction)")
    parser.add_argument("--root", default=".",
                        help="project root (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="start an evolution run (spends LLM tokens)")
    p.add_argument("--config", required=True)
    p.add_argument("--confirm-spend", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("resume", help="resume a run (spends LLM tokens)")
    p.add_argument("--run", required=True, help="runs/<id> directory")
    p.add_argument("--confirm-spend", action="store_true")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("eval-once", help="score a workspace once (no LLM)")
    p.add_argument("--config", required=True)
    p.add_argument("--workspace", help="default: the task's seed dir")
    p.add_argument("--fresh", action="store_true", help="bypass the eval cache")
    p.set_defaults(fn=cmd_eval_once)

    p = sub.add_parser("baselines", help="benchmark baselines (no LLM)")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=cmd_baselines)

    p = sub.add_parser("dashboard",
                       help="self-contained HTML dashboard for a run (no LLM)")
    p.add_argument("--run", help="run directory (default: latest under runs/)")
    p.add_argument("--baselines", help="baseline JSON path (default: latest)")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="regenerate every N seconds; page auto-reloads")
    p.add_argument("--open", action="store_true",
                   help="open the dashboard in the default browser")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("export",
                       help="package a run into committable results/ (no LLM)")
    p.add_argument("--run", required=True)
    p.add_argument("--dest", default="results")
    p.add_argument("--with-transcripts", action="store_true",
                   help="include full per-step agent transcripts (larger)")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("audit",
                       help="detect cross-route / shared-tmp contamination (no LLM)")
    p.add_argument("--run", required=True)
    p.add_argument("--isolation", default="unknown",
                   help="isolation mode in force during the run (for the record)")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("report", help="render lineage table + plot (no LLM)")
    p.add_argument("--run", required=True)
    p.add_argument("--baselines", help="baseline JSON path (default: latest)")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("rebench",
                       help="re-score all committed versions, mean±std (no LLM)")
    p.add_argument("--run", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--rounds", type=int, default=10)
    p.set_defaults(fn=cmd_rebench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
