"""Full-fidelity JSONL transcripts on disk + in-context truncation policy.

The transcript file keeps everything; the *context* sent to the model elides
old tool results once the conversation grows, always preserving the step
prompt (first message) and the most recent turns.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from avo.types import ChatMessage, ToolResultBlock

TOOL_RESULT_CAP = 20_000        # bytes, applied at creation time by tools
CONTEXT_CHAR_BUDGET = 400_000   # ~100k tokens; beyond this, elide old results
KEEP_RECENT_MESSAGES = 12       # never elide anything in the last N messages
ELIDE_THRESHOLD = 1_500         # only elide results bigger than this


class Transcript:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, payload) -> None:
        if isinstance(payload, ChatMessage):
            payload = payload.to_dict()
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind,
                                "payload": payload}) + "\n")


def _total_chars(messages: list[ChatMessage]) -> int:
    return sum(len(json.dumps(m.to_dict())) for m in messages)


def truncate_context(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return a (possibly) reduced copy of the conversation for the next model
    call. The first message (step prompt) and the last KEEP_RECENT_MESSAGES
    are always intact; older large tool results are elided oldest-first."""
    if _total_chars(messages) <= CONTEXT_CHAR_BUDGET:
        return messages
    out = [copy.deepcopy(m) for m in messages]
    cutoff = max(1, len(out) - KEEP_RECENT_MESSAGES)
    for m in out[1:cutoff]:
        if _total_chars(out) <= CONTEXT_CHAR_BUDGET:
            break
        for b in m.blocks:
            if isinstance(b, ToolResultBlock) and len(b.content) > ELIDE_THRESHOLD:
                b.content = f"[tool result elided to save context, was {len(b.content)} bytes]"
    return out
