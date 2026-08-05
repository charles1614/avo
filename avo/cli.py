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

    cache = EvalCache(root / config.runs_dir / "eval_once_cache")
    key = eval_key(workspace, harness, config.task_params, runner.identity())
    result = None if args.fresh else cache.get(key)
    if result is None:
        result = runner.score(workspace, harness, task.score_entry,
                              config.task_params)
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
