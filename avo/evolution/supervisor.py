"""Self-supervision: detect stagnation, review the trajectory, steer the search.

The paper describes the mechanism but gives no algorithm; this is our design:
trigger a reflection LLM call when (a) N consecutive steps ended without a
commit, or (b) the last `window` commits improved by less than
min_rel_improvement in total. The reflection output is injected verbatim into
the next step prompt as SUPERVISOR GUIDANCE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from avo.agent.prompts import SUPERVISOR_PROMPT, lineage_table
from avo.config import SupervisorConfig
from avo.llm.base import LLMClient
from avo.types import ChatMessage, LineageEntry, TextBlock


class Supervisor:
    def __init__(self, cfg: SupervisorConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = Path(log_path)
        self._last_trigger_stagnation = 0

    def should_reflect(self, stagnation_count: int,
                       entries: list[LineageEntry]) -> str | None:
        """Return a trigger reason, or None."""
        n = self.cfg.stagnation_steps
        if stagnation_count >= n and stagnation_count - self._last_trigger_stagnation >= n:
            return f"{stagnation_count} consecutive steps without a commit"
        w = self.cfg.window
        if stagnation_count == 0 and len(entries) >= w + 1:
            recent = entries[-(w + 1):]
            if recent[0].score > 0:
                rel = (recent[-1].score - recent[0].score) / recent[0].score
                if rel < self.cfg.min_rel_improvement:
                    return (f"last {w} commits improved only {rel * 100:.2f}% "
                            f"(< {self.cfg.min_rel_improvement * 100:.1f}%)")
        return None

    def note_triggered(self, stagnation_count: int) -> None:
        self._last_trigger_stagnation = stagnation_count

    def note_commit(self) -> None:
        self._last_trigger_stagnation = 0

    def reflect(self, llm: LLMClient, entries: list[LineageEntry],
                failure_summaries: list[str], source_snapshot: str,
                reason: str) -> str:
        prompt = SUPERVISOR_PROMPT.format(
            lineage_table=lineage_table(entries),
            failures="\n\n".join(f"- {s}" for s in failure_summaries) or "(none recorded)",
            source=source_snapshot[:24_000],
        )
        turn = llm.chat(system="You are a precise engineering strategist.",
                        messages=[ChatMessage("user", [TextBlock(prompt)])])
        guidance = turn.message.text().strip()
        self._log({"ts": time.time(), "reason": reason, "guidance": guidance})
        return guidance

    def _log(self, payload: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
