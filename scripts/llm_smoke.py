#!/usr/bin/env python3
"""Smoke-test tool-calling against the live provider in a config: one forced
tool call round-trip (echo tool), then a final text turn. Costs a few hundred
tokens — requires --confirm-spend like every LLM-calling entry point.

Usage: python scripts/llm_smoke.py --config configs/sort_py.yaml --confirm-spend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avo.config import load_run_config
from avo.types import ChatMessage, TextBlock, ToolResultBlock, ToolSpec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--confirm-spend", action="store_true")
    args = ap.parse_args()

    config = load_run_config(args.config)
    llm_cfg = config.llm
    if not args.confirm_spend:
        print(f"Would call {llm_cfg.provider}/{llm_cfg.model}"
              + (f" @ {llm_cfg.base_url}" if llm_cfg.base_url else "")
              + f" (key env: {llm_cfg.key_env_name()}, "
              + ("set" if llm_cfg.resolve_api_key() else "NOT SET") + ").")
        print("Re-run with --confirm-spend to proceed.")
        return 1

    from avo.llm.base import make_client
    client = make_client(llm_cfg)
    tool = ToolSpec(
        name="echo",
        description="Echo the given text back. You MUST call this tool.",
        input_schema={"type": "object",
                      "properties": {"text": {"type": "string"}},
                      "required": ["text"]})

    messages = [ChatMessage("user", [TextBlock(
        "Call the echo tool with text='ping'. After you receive the result, "
        "reply with exactly the word DONE.")])]
    turn = client.chat("You are a tool-calling test harness.", messages, [tool])
    uses = turn.message.tool_uses()
    print(f"turn 1: stop_reason={turn.stop_reason}, tool_calls={[(u.name, u.input) for u in uses]}")
    if not uses or uses[0].name != "echo":
        print("FAIL: model did not call the echo tool")
        return 2

    messages.append(turn.message)
    messages.append(ChatMessage("user", [ToolResultBlock(
        tool_use_id=uses[0].id, content=str(uses[0].input.get("text", "")))]))
    turn2 = client.chat("You are a tool-calling test harness.", messages, [tool])
    print(f"turn 2: text={turn2.message.text()!r}")
    print(f"usage: {client.usage.input_tokens} in / {client.usage.output_tokens} out"
          f" tokens, est ${client.cost_usd:.4f}")
    ok = "DONE" in turn2.message.text().upper()
    print("PASS" if ok else "FAIL: expected DONE in final reply")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
