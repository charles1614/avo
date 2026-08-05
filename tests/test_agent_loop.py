from pathlib import Path

from avo.agent.tools import ToolContext, ToolRegistry
from avo.agent.transcript import Transcript, truncate_context
from avo.agent.variation import VariationAgent
from avo.knowledge.kb import KnowledgeBase
from avo.types import (ChatMessage, ScoreResult, TextBlock, ToolResultBlock,
                       ToolUseBlock)
from tests.conftest import FakeLLM, tool_use


def build_agent(tmp_path, script, max_evals=4, max_turns=6, gate_pass=True):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "f.py").write_text("x = 1\n")
    ctx = ToolContext(
        workspace=ws, kb=KnowledgeBase([]),
        evaluate_fn=lambda: ScoreResult(correct=True, score=2.0),
        submit_fn=lambda msg: (gate_pass, "ACCEPTED" if gate_pass else "REJECTED"),
        max_evals=max_evals)
    llm = FakeLLM(script)
    agent = VariationAgent(llm, ToolRegistry(ctx),
                           Transcript(tmp_path / "t.jsonl"), max_turns=max_turns)
    return agent, llm


def test_edit_evaluate_submit_flow(tmp_path):
    script = [
        [tool_use("edit_file", "t1", path="f.py", old_string="x = 1",
                  new_string="x = 2")],
        [tool_use("evaluate", "t2")],
        [tool_use("submit", "t3", message="set x to 2")],
    ]
    agent, llm = build_agent(tmp_path, script)
    result = agent.run_step("sys", "step")
    assert result.committed and result.stop_cause == "committed"
    assert result.submit_message == "set x to 2"
    assert result.turns_used == 3 and result.evals_used == 2
    # tool results were fed back as user messages
    fed_back = llm.calls[1]["messages"][-1]
    assert isinstance(fed_back.blocks[0], ToolResultBlock)


def test_nudge_on_pure_text_turn(tmp_path):
    script = [
        [TextBlock("thinking out loud, no tools")],
        [tool_use("submit", "t1", message="done")],
    ]
    agent, llm = build_agent(tmp_path, script)
    result = agent.run_step("sys", "step")
    assert result.committed
    nudge_msg = llm.calls[1]["messages"][-1]
    assert "tools" in nudge_msg.text()


def test_max_turns_exhaustion(tmp_path):
    script = [[tool_use("evaluate", f"t{i}")] for i in range(10)]
    agent, _ = build_agent(tmp_path, script, max_evals=100, max_turns=3)
    result = agent.run_step("sys", "step")
    assert not result.committed and result.stop_cause == "max_turns"
    assert result.turns_used == 3


def test_rejected_submit_continues(tmp_path):
    script = [
        [tool_use("submit", "t1", message="try 1")],
        [tool_use("submit", "t2", message="try 2")],
    ]
    agent, _ = build_agent(tmp_path, script, gate_pass=False, max_turns=2)
    result = agent.run_step("sys", "step")
    assert not result.committed and result.turns_used == 2


def test_parse_error_reflected_not_executed(tmp_path):
    bad = ToolUseBlock(id="t1", name="edit_file", input={},
                       parse_error="malformed JSON arguments")
    script = [[bad], [tool_use("submit", "t2", message="ok")]]
    agent, llm = build_agent(tmp_path, script)
    result = agent.run_step("sys", "step")
    assert result.committed
    reflected = llm.calls[1]["messages"][-1].blocks[0]
    assert reflected.is_error and "malformed" in reflected.content


def test_budget_abort_stops_loop(tmp_path):
    script = [[tool_use("evaluate", "t1")]] * 5
    agent, _ = build_agent(tmp_path, script, max_turns=10)
    agent.budget_abort_fn = lambda: "max_usd"
    result = agent.run_step("sys", "step")
    assert not result.committed and result.stop_cause == "budget_abort:max_usd"
    assert result.turns_used == 0


def test_truncate_context_preserves_recent_and_first():
    big = "y" * 10_000
    messages = [ChatMessage("user", [TextBlock("step prompt")])]
    for i in range(60):
        messages.append(ChatMessage("assistant",
                                    [ToolUseBlock(id=f"t{i}", name="shell",
                                                  input={"command": "x"})]))
        messages.append(ChatMessage("user",
                                    [ToolResultBlock(tool_use_id=f"t{i}",
                                                     content=big)]))
    out = truncate_context(messages)
    assert out[0].text() == "step prompt"
    # oldest big results elided, newest intact
    assert "elided" in out[2].blocks[0].content
    assert out[-1].blocks[0].content == big
    # original untouched
    assert messages[2].blocks[0].content == big
