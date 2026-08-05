"""Run reporting: lineage table, score-per-version plot vs baselines, rebench."""
from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from avo.evolution.lineage import Lineage
from avo.types import LineageEntry


def load_baselines(path: Path | None, search_dir: Path | None = None) -> dict:
    """Baseline JSON: {"geomeans": {"sdpa_flash": 55.2, ...}, ...}."""
    if path is None and search_dir and search_dir.is_dir():
        candidates = sorted(search_dir.glob("*.json"),
                            key=lambda p: p.stat().st_mtime)
        path = candidates[-1] if candidates else None
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def lineage_table_text(entries: list[LineageEntry], baselines: dict) -> str:
    lines = ["| version | step | score | change |", "|---|---|---|---|"]
    for e in entries:
        lines.append(f"| {e.version} | {e.step} | {e.score:.4f} | "
                     f"{e.message.splitlines()[0][:80]} |")
    if entries:
        seed, best = entries[0].score, max(e.score for e in entries)
        if seed > 0:
            lines.append(f"\nSeed -> best: {seed:.4f} -> {best:.4f} "
                         f"({(best / seed - 1) * 100:+.1f}%)")
    for name, val in (baselines.get("geomeans") or {}).items():
        lines.append(f"- baseline {name}: {val:.4f}")
    return "\n".join(lines)


def write_report(run_dir: Path, baselines_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    lineage = Lineage(run_dir / "workspace", run_dir / "lineage.jsonl")
    entries = lineage.entries()
    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)
    baselines = load_baselines(baselines_path,
                               run_dir.parent / "baselines")

    md = ["# AVO run report", f"\nRun: `{run_dir.name}`\n",
          lineage_table_text(entries, baselines)]
    if (run_dir / "summary.json").exists():
        md.append("\n## Summary\n```json\n"
                  + (run_dir / "summary.json").read_text() + "\n```")
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(md))

    try:
        _plot(entries, baselines, out_dir / "scores.png")
    except ImportError:
        print("[avo] matplotlib not installed; skipping plot "
              "(pip install 'avo[report]')")
    return report_md


def _plot(entries: list[LineageEntry], baselines: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(entries)))
    ys = [e.score for e in entries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", label="AVO committed versions")
    for name, val in (baselines.get("geomeans") or {}).items():
        ax.axhline(val, linestyle="--", alpha=0.6, label=f"baseline: {name}")
    ax.set_xlabel("committed version")
    ax.set_ylabel("score (geomean)")
    ax.set_xticks(xs)
    ax.set_xticklabels([e.version for e in entries], rotation=45, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def rebench(run_dir: Path, runner, task_dir: Path, score_entry: str,
            params: dict, rounds: int = 10, log=print) -> Path:
    """Re-score every committed version with `rounds` independent runs and
    report mean±std — the paper-style final measurement (no cache, no LLM)."""
    run_dir = Path(run_dir)
    lineage = Lineage(run_dir / "workspace", run_dir / "lineage.jsonl")
    harness = task_dir / "harness"
    rows = ["| version | mean | std | change |", "|---|---|---|---|"]
    for e in lineage.entries():
        scores = []
        with tempfile.TemporaryDirectory(prefix="avo_rebench_") as td:
            snap = Path(td) / "workspace"
            snap.mkdir()
            archive = subprocess.run(
                ["git", "-C", str(lineage.workspace), "archive", e.version],
                capture_output=True, check=True)
            subprocess.run(["tar", "-x", "-C", str(snap)],
                           input=archive.stdout, check=True)
            for i in range(rounds):
                r = runner.score(snap, harness, score_entry,
                                 {**params, "_rebench_round": i})
                if r.correct:
                    scores.append(r.score)
                log(f"[rebench] {e.version} round {i + 1}/{rounds}: "
                    f"{r.score:.4f}{'' if r.correct else ' (INCORRECT)'}")
        mean = statistics.mean(scores) if scores else 0.0
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        rows.append(f"| {e.version} | {mean:.4f} | {std:.4f} | "
                    f"{e.message.splitlines()[0][:60]} |")
    out = run_dir / "report" / "rebench.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"# Rebench ({rounds} rounds per version)\n\n" + "\n".join(rows))
    return out
