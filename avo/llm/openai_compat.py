"""OpenAI chat-completions adapter.

With base_url override this covers OpenAI, Ollama, vLLM, DeepSeek, and any
other OpenAI-compatible server. Tool arguments arrive as JSON strings and are
parsed defensively: malformed JSON becomes ToolUseBlock.parse_error, which the
agent loop reflects back to the model as a tool error instead of executing.
"""
from __future__ import annotations

import json

from avo.config import LLMConfig
from avo.llm.base import LLMClient
from avo.types import (AssistantTurn, ChatMessage, TextBlock, ToolResultBlock,
                       ToolSpec, ToolUseBlock, Usage)


def to_openai_messages(system: str, messages: list[ChatMessage]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "assistant":
            msg: dict = {"role": "assistant"}
            text = m.text()
            tool_calls = [
                {"id": b.id, "type": "function",
                 "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                for b in m.tool_uses()
            ]
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = text if text else None
            else:
                # servers reject assistant messages with neither content nor
                # tool_calls; never emit a bare one
                msg["content"] = text if text else "(empty)"
            out.append(msg)
        else:  # user message: text and/or tool results
            results = [b for b in m.blocks if isinstance(b, ToolResultBlock)]
            for r in results:
                content = r.content
                if r.is_error:
                    content = f"[TOOL ERROR]\n{content}"
                out.append({"role": "tool", "tool_call_id": r.tool_use_id,
                            "content": content})
            text = m.text()
            if text:
                out.append({"role": "user", "content": text})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.input_schema}} for t in tools]


def parse_openai_choice(resp: dict) -> AssistantTurn:
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    blocks: list = []
    if msg.get("content"):
        blocks.append(TextBlock(text=msg["content"]))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        parse_error = None
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                args, parse_error = {}, f"tool arguments not an object: {raw_args[:300]}"
        except json.JSONDecodeError as e:
            args, parse_error = {}, f"malformed JSON arguments ({e}): {raw_args[:300]}"
        blocks.append(ToolUseBlock(id=tc.get("id", ""), name=fn.get("name", ""),
                                   input=args, parse_error=parse_error))
    finish = choice.get("finish_reason") or "stop"
    stop_reason = {"tool_calls": "tool_use", "stop": "end_turn",
                   "length": "max_tokens"}.get(finish, finish)
    usage = resp.get("usage") or {}
    return AssistantTurn(
        message=ChatMessage(role="assistant", blocks=blocks),
        stop_reason=stop_reason,
        usage=Usage(input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0)),
    )


class OpenAICompatClient(LLMClient):
    def __init__(self, cfg: LLMConfig, api_key: str):
        super().__init__(cfg)
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url=cfg.base_url,
                                     max_retries=3)

    def chat(self, system: str, messages: list[ChatMessage],
             tools: list[ToolSpec] | None = None) -> AssistantTurn:
        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=to_openai_messages(system, messages),
        )
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        resp = self._client.chat.completions.create(**kwargs)
        turn = parse_openai_choice(resp.model_dump())
        self.usage.add(turn.usage)
        return turn
