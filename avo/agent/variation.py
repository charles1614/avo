"""The agentic variation operator: Vary(P_t) = Agent(P_t, K, f).

One `run_step` = one variation step = an autonomous edit-evaluate-diagnose
loop that ends when the agent commits a gate-passing version or exhausts its
turn/eval budget.
"""
from __future__ import annotations

from dataclasses import dataclass

from avo.agent.tools import ToolRegistry
from avo.agent.transcript import Transcript, truncate_context
from avo.llm.base import LLMClient
from avo.types import ChatMessage, TextBlock, ToolResultBlock

NUDGE = ("Reminder: you must act through tools. Inspect the code, make an "
         "edit, `evaluate`, and `submit` when you have a verified improvement.")


@dataclass
class StepResult:
    committed: bool
    submit_message: str | None
    turns_used: int
    evals_used: int
    last_eval_brief: str
    stop_cause: str  # "committed" | "max_turns" | "budget_abort"


class VariationAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry,
                 transcript: Transcript, max_turns: int,
                 budget_abort_fn=None):
        self.llm = llm
        self.registry = registry
        self.transcript = transcript
        self.max_turns = max_turns
        # controller hook: returns a reason string when the global budget
        # (usd/tokens/wall-clock) is exhausted, else None
        self.budget_abort_fn = budget_abort_fn or (lambda: None)

    def run_step(self, system_prompt: str, step_prompt: str) -> StepResult:
        messages = [ChatMessage("user", [TextBlock(step_prompt)])]
        self.transcript.log("step_prompt", messages[0])
        committed = False
        submit_message: str | None = None
        last_eval_brief = ""
        turns = 0
        stop_cause = "max_turns"

        while turns < self.max_turns:
            abort = self.budget_abort_fn()
            if abort:
                stop_cause = f"budget_abort:{abort}"
                break
            turns += 1
            turn = self.llm.chat(system_prompt, truncate_context(messages),
                                 self.registry.specs())
            if not turn.message.blocks:
                # degenerate empty turn (some providers emit these); a bare
                # assistant message is invalid in replayed history
                turn.message.blocks.append(TextBlock("(empty model turn)"))
            self.transcript.log("assistant", turn.message)
            messages.append(turn.message)

            tool_uses = turn.message.tool_uses()
            if not tool_uses:
                messages.append(ChatMessage("user", [TextBlock(NUDGE)]))
                self.transcript.log("nudge", NUDGE)
                continue

            results: list[ToolResultBlock] = []
            for tu in tool_uses:
                if tu.parse_error:
                    outcome_content, is_error, signal = tu.parse_error, True, None
                    if turn.stop_reason == "max_tokens":
                        outcome_content = (
                            "Your tool call was TRUNCATED by the output-token "
                            "limit before the arguments finished. Do not retry "
                            "the same call. Split the work: write_file a short "
                            "skeleton first, then extend it with several "
                            "edit_file calls of at most ~150 lines each.")
                else:
                    o = self.registry.dispatch(tu.name, tu.input)
                    outcome_content, is_error, signal = o.content, o.is_error, o.signal
                if tu.name in ("evaluate", "submit"):
                    last_eval_brief = outcome_content[:600]
                if tu.name == "submit" and signal == "committed":
                    committed = True
                    submit_message = str(tu.input.get("message", ""))
                results.append(ToolResultBlock(tool_use_id=tu.id,
                                               content=outcome_content,
                                               is_error=is_error))
                self.transcript.log("tool", {"name": tu.name, "input": tu.input,
                                             "is_error": is_error,
                                             "content": outcome_content[:4000]})
            blocks: list = list(results)
            remaining = self.max_turns - turns
            if not committed and 0 < remaining <= max(2, self.max_turns // 4):
                blocks.append(TextBlock(
                    f"[only {remaining} turns left this step — evaluate and "
                    "submit a verified improvement before running out]"))
            messages.append(ChatMessage("user", blocks))
            if committed:
                stop_cause = "committed"
                break

        return StepResult(committed=committed, submit_message=submit_message,
                          turns_used=turns,
                          evals_used=self.registry.ctx.evals_used,
                          last_eval_brief=last_eval_brief,
                          stop_cause=stop_cause)
