"""Anthropic Messages API adapter.

Translation functions are pure (dicts in/out) so tests cover them without
network or SDK objects.
"""
from __future__ import annotations

from typing import Any

from avo.config import LLMConfig
from avo.llm.base import LLMClient
from avo.types import (AssistantTurn, ChatMessage, TextBlock, ToolResultBlock,
                       ToolSpec, ToolUseBlock, Usage)


def to_anthropic_messages(messages: list[ChatMessage]) -> list[dict]:
    out = []
    for m in messages:
        content: list[dict] = []
        for b in m.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                content.append({"type": "tool_use", "id": b.id, "name": b.name,
                                "input": b.input})
            elif isinstance(b, ToolResultBlock):
                content.append({"type": "tool_result", "tool_use_id": b.tool_use_id,
                                "content": b.content, "is_error": b.is_error})
        out.append({"role": m.role, "content": content})
    return out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description,
             "input_schema": t.input_schema} for t in tools]


def parse_anthropic_response(resp: dict) -> AssistantTurn:
    blocks: list[Any] = []
    for c in resp.get("content", []):
        if c["type"] == "text":
            blocks.append(TextBlock(text=c["text"]))
        elif c["type"] == "tool_use":
            blocks.append(ToolUseBlock(id=c["id"], name=c["name"], input=c["input"]))
    usage = resp.get("usage", {})
    return AssistantTurn(
        message=ChatMessage(role="assistant", blocks=blocks),
        stop_reason=resp.get("stop_reason") or "end_turn",
        usage=Usage(input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0)),
    )


class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig, api_key: str):
        super().__init__(cfg)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    def chat(self, system: str, messages: list[ChatMessage],
             tools: list[ToolSpec] | None = None) -> AssistantTurn:
        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=to_anthropic_messages(messages),
        )
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        kwargs.update(self.cfg.extra_body)  # e.g. thinking config
        resp = self._client.messages.create(**kwargs)
        turn = parse_anthropic_response(resp.model_dump())
        self.usage.add(turn.usage)
        return turn
