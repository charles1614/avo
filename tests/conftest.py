"""Shared fixtures. Tests are strictly offline: no API keys are read, no
network calls, no GPU. A FakeLLM plays scripted turns through the real loop."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avo.config import LLMConfig
from avo.llm.base import LLMClient
from avo.types import AssistantTurn, ChatMessage, TextBlock, ToolUseBlock, Usage


class FakeLLM(LLMClient):
    """Replays a scripted list of assistant turns. Each script entry is a list
    of blocks (TextBlock/ToolUseBlock)."""

    def __init__(self, script: list[list]):
        super().__init__(LLMConfig(provider="anthropic", model="fake",
                                   price_input_per_mtok=1.0,
                                   price_output_per_mtok=1.0))
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools=None, max_tokens=None) -> AssistantTurn:
        # snapshot: the agent mutates its message list across turns
        self.calls.append({"system": system, "messages": list(messages),
                           "tools": tools, "max_tokens": max_tokens})
        if not self.script:
            blocks = [TextBlock("(script exhausted)")]
        else:
            blocks = self.script.pop(0)
        usage = Usage(input_tokens=100, output_tokens=50)
        self.usage.add(usage)
        has_tool = any(isinstance(b, ToolUseBlock) for b in blocks)
        return AssistantTurn(
            message=ChatMessage("assistant", list(blocks)),
            stop_reason="tool_use" if has_tool else "end_turn",
            usage=usage)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


def tool_use(name: str, tid: str = "t1", **kwargs) -> ToolUseBlock:
    return ToolUseBlock(id=tid, name=name, input=kwargs)


@pytest.fixture
def make_tool_use():
    return tool_use
