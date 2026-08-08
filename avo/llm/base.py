"""Provider-neutral LLM client interface.

Everything outside avo/llm/ speaks only the canonical types from avo.types.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from avo.config import LLMConfig
from avo.types import AssistantTurn, ChatMessage, ToolSpec, Usage


class LLMClient(ABC):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.usage = Usage()
        # set by the controller: per-call JSONL (reasoning length, context
        # size, usage, latency) for cross-model/cross-env comparison
        self.metrics_path: Path | None = None

    def _record_call(self, record: dict) -> None:
        if self.metrics_path is None:
            return
        try:
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "model": self.cfg.model,
                                    **record}) + "\n")
        except OSError:
            pass  # metrics must never break a run

    @abstractmethod
    def chat(self, system: str, messages: list[ChatMessage],
             tools: list[ToolSpec] | None = None,
             max_tokens: int | None = None) -> AssistantTurn:
        """One model call. Implementations must add to self.usage.
        max_tokens overrides the config for this call only — the agent loop
        escalates it when reasoning models truncate before acting."""

    @property
    def cost_usd(self) -> float:
        return (self.usage.input_tokens * self.cfg.price_input_per_mtok
                + self.usage.output_tokens * self.cfg.price_output_per_mtok) / 1e6


def make_client(cfg: LLMConfig) -> LLMClient:
    """Instantiate a provider client. Imports lazily so offline tests never
    touch provider SDK network setup, and raises early if the key is missing."""
    key = cfg.resolve_api_key()
    if not key:
        raise RuntimeError(
            f"No API key found in ${cfg.key_env_name()} for provider "
            f"'{cfg.provider}'. Export it before running LLM commands.")
    if cfg.provider == "anthropic":
        from avo.llm.anthropic_client import AnthropicClient
        return AnthropicClient(cfg, api_key=key)
    if cfg.provider == "openai_compat":
        from avo.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(cfg, api_key=key)
    raise ValueError(f"unknown provider: {cfg.provider}")
