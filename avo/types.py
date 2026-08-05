"""Core data types shared across the framework.

The canonical chat message format is provider-neutral; provider adapters in
avo/llm/ translate to/from it at the edge.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Union


# ---------------------------------------------------------------------------
# Canonical chat/message types
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    # Set when a provider returned unparseable tool arguments; the agent loop
    # returns this to the model as a tool error instead of executing.
    parse_error: str | None = None
    type: str = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


Block = Union[TextBlock, ToolUseBlock, ToolResultBlock]

_BLOCK_TYPES = {"text": TextBlock, "tool_use": ToolUseBlock, "tool_result": ToolResultBlock}


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    blocks: list[Block]

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]

    def to_dict(self) -> dict:
        return {"role": self.role, "blocks": [asdict(b) for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMessage":
        blocks = []
        for b in d["blocks"]:
            b = dict(b)
            block_cls = _BLOCK_TYPES[b.pop("type")]
            blocks.append(block_cls(**b))
        return cls(role=d["role"], blocks=blocks)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict  # JSON Schema (kept flat: type/properties/required only)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AssistantTurn:
    message: ChatMessage
    stop_reason: str  # normalized: "tool_use" | "end_turn" | "max_tokens" | other
    usage: Usage


# ---------------------------------------------------------------------------
# Scoring / lineage types
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Parsed result.json from a harness run. This schema is the contract."""

    correct: bool
    score: float
    error: dict | None = None  # {"stage": "compile"|"correctness"|"bench"|"harness", "detail", "log_tail"}
    configs: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    eval_hash: str | None = None
    cached: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreResult":
        return cls(
            correct=bool(d.get("correct", False)),
            score=float(d.get("score", 0.0)),
            error=d.get("error"),
            configs=d.get("configs", []),
            meta=d.get("meta", {}),
            eval_hash=d.get("eval_hash"),
            cached=bool(d.get("cached", False)),
        )

    @classmethod
    def failure(cls, stage: str, detail: str, log_tail: str = "") -> "ScoreResult":
        return cls(correct=False, score=0.0,
                   error={"stage": stage, "detail": detail, "log_tail": log_tail})

    def brief(self) -> str:
        """Compact human-readable summary used in prompts and logs."""
        if self.correct:
            per_cfg = ", ".join(
                f"{c.get('seqlen', c.get('size', '?'))}{'c' if c.get('causal') else ''}:"
                f"{c.get('tflops', c.get('throughput', 0)):.4g}"
                for c in self.configs[:12]
            )
            return f"correct=True score={self.score:.4f} [{per_cfg}]"
        err = self.error or {}
        return (f"correct=False score=0 stage={err.get('stage', '?')} "
                f"detail={err.get('detail', '')[:500]}")


@dataclass
class LineageEntry:
    version: str  # "v0000", "v0001", ...
    step: int
    commit: str
    score: float
    message: str
    eval_hash: str
    parent: str | None
    timestamp: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "LineageEntry":
        return cls(**{k: d[k] for k in
                      ("version", "step", "commit", "score", "message",
                       "eval_hash", "parent", "timestamp")})


@dataclass
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def render(self, max_bytes: int = 20_000) -> str:
        out = self.stdout or ""
        err = self.stderr or ""
        body = out + (("\n--- stderr ---\n" + err) if err else "")
        body = truncate_middle(body, max_bytes)
        status = "TIMED OUT" if self.timed_out else f"exit code {self.exit_code}"
        return f"[{status}]\n{body}" if body else f"[{status}]"


def truncate_middle(s: str, max_bytes: int) -> str:
    """Keep head and tail of oversized text (compiler logs etc.)."""
    raw = s.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return s
    head = raw[: int(max_bytes * 0.6)].decode("utf-8", errors="replace")
    tail = raw[-int(max_bytes * 0.4):].decode("utf-8", errors="replace")
    return f"{head}\n... [{len(raw) - max_bytes} bytes elided] ...\n{tail}"


def geomean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    if any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
