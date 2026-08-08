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
from avo.types import (AssistantTurn, ChatMessage, TextBlock, ThinkingBlock,
                       ToolResultBlock, ToolSpec, ToolUseBlock, Usage)

REASONING_KEEP_CHARS = 8000  # keep enough truncated reasoning to be diagnosable


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


def assemble_stream(chunks: list[dict]) -> dict:
    """Fold streaming chunk dicts into one non-streaming-shaped response so
    parse_openai_choice handles both paths identically. Tool-call deltas are
    keyed by index; the final usage-only chunk (stream_options include_usage)
    carries token counts."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        finish_reason = ch.get("finish_reason") or finish_reason
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        if rc:
            reasoning_parts.append(rc)
        for tc in delta.get("tool_calls") or []:
            slot = tool_calls.setdefault(
                tc.get("index", 0),
                {"id": "", "type": "function",
                 "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
    message: dict = {"content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage}


def extract_call_metrics(resp_dict: dict) -> dict:
    """Per-call metrics from a (non-streaming-shaped) response dict.
    reasoning_chars is the FULL length, before any in-context truncation."""
    choice = (resp_dict.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {"reasoning_chars": len(reasoning),
            "text_chars": len(msg.get("content") or ""),
            "n_tool_calls": len(msg.get("tool_calls") or []),
            "finish_reason": choice.get("finish_reason"),
            "usage": resp_dict.get("usage")}


def parse_openai_choice(resp: dict) -> AssistantTurn:
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    blocks: list = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        # visible to the loop (truncation diagnosis) and transcripts; never
        # replayed to OpenAI-compat servers (serializer skips ThinkingBlock)
        blocks.append(ThinkingBlock(thinking=reasoning[-REASONING_KEEP_CHARS:]))
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
             tools: list[ToolSpec] | None = None,
             max_tokens: int | None = None) -> AssistantTurn:
        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=max_tokens or self.cfg.max_tokens,
            messages=to_openai_messages(system, messages),
        )
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        import time as _time
        t0 = _time.monotonic()
        if self.cfg.stream:
            resp_dict = self._create_streamed(kwargs)
        else:
            resp_dict = self._client.chat.completions.create(**kwargs).model_dump()
        self._record_call({
            **extract_call_metrics(resp_dict),
            "latency_s": round(_time.monotonic() - t0, 2),
            "n_messages": len(kwargs["messages"]),
            "context_chars": sum(len(str(m)) for m in kwargs["messages"]),
            "max_tokens": kwargs["max_tokens"],
        })
        turn = parse_openai_choice(resp_dict)
        if turn.usage.total_tokens == 0:  # server omitted usage: estimate
            turn.usage = Usage(
                input_tokens=sum(len(str(m)) for m in kwargs["messages"]) // 4,
                output_tokens=len(str(resp_dict["choices"][0]["message"])) // 4)
        self.usage.add(turn.usage)
        return turn

    def _create_streamed(self, kwargs: dict) -> dict:
        """Stream and assemble locally: gateways buffer non-streamed responses
        and time out (504) on long reasoning output."""
        import openai
        try:
            stream = self._client.chat.completions.create(
                **kwargs, stream=True,
                stream_options={"include_usage": True})
        except openai.BadRequestError:  # server rejects stream_options
            stream = self._client.chat.completions.create(**kwargs, stream=True)
        return assemble_stream([c.model_dump() for c in stream])
