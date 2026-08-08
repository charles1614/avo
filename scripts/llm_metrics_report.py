#!/usr/bin/env python3
"""Summarize per-call LLM metrics from one or more runs for cross-model /
cross-environment comparison (e.g. deepseek-v4-flash on this box vs GLM-5.2
through a gateway on the H100).

Usage:
  python scripts/llm_metrics_report.py runs/<idA>/logs/llm_metrics.jsonl \
                                       runs/<idB>/logs/llm_metrics.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        print(f"{path}: empty")
        return
    models = sorted({r.get("model", "?") for r in recs})
    reasoning = [r.get("reasoning_chars", 0) for r in recs]
    text = [r.get("text_chars", 0) for r in recs]
    ctx = [r.get("context_chars", 0) for r in recs]
    lat = [r.get("latency_s", 0.0) for r in recs]
    with_tools = sum(1 for r in recs if r.get("n_tool_calls", 0) > 0)
    empty = sum(1 for r in recs
                if not r.get("n_tool_calls") and not r.get("text_chars"))
    capped = sum(1 for r in recs if r.get("finish_reason") == "length")
    usages = [r["usage"] for r in recs if r.get("usage")]
    tok_out = [u.get("completion_tokens", u.get("output_tokens", 0))
               for u in usages]
    tok_reason = [((u.get("completion_tokens_details") or {})
                   .get("reasoning_tokens", 0)) for u in usages]

    print(f"\n== {path} ==")
    print(f"model(s): {', '.join(models)}   calls: {len(recs)}")
    print(f"reasoning chars   mean {statistics.mean(reasoning):8.0f}  "
          f"p50 {pct(reasoning, 50):8.0f}  p90 {pct(reasoning, 90):8.0f}  "
          f"max {max(reasoning):8.0f}")
    print(f"visible chars     mean {statistics.mean(text):8.0f}  "
          f"p50 {pct(text, 50):8.0f}  p90 {pct(text, 90):8.0f}")
    print(f"context chars     mean {statistics.mean(ctx):8.0f}  "
          f"p90 {pct(ctx, 90):8.0f}  max {max(ctx):8.0f}")
    print(f"latency s         p50 {pct(lat, 50):8.1f}  p90 {pct(lat, 90):8.1f}  "
          f"max {max(lat):8.1f}")
    print(f"tool-call rate    {with_tools}/{len(recs)} "
          f"({100 * with_tools / len(recs):.0f}%)   "
          f"empty turns: {empty}   finish=length (capped): {capped}")
    if any(tok_out):
        print(f"output tokens     mean {statistics.mean(tok_out):8.0f}  "
              f"p90 {pct(tok_out, 90):8.0f}")
    if any(tok_reason):
        print(f"reasoning tokens  mean {statistics.mean(tok_reason):8.0f}  "
              f"p90 {pct(tok_reason, 90):8.0f}  (server-reported)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for p in sys.argv[1:]:
        summarize(Path(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
