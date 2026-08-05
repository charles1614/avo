from avo.llm.anthropic_client import (parse_anthropic_response,
                                      to_anthropic_messages, to_anthropic_tools)
from avo.llm.openai_compat import (parse_openai_choice, to_openai_messages,
                                   to_openai_tools)
from avo.types import (ChatMessage, TextBlock, ToolResultBlock, ToolSpec,
                       ToolUseBlock)

TOOLS = [ToolSpec(name="edit", description="edit a file",
                  input_schema={"type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"]})]

CONVO = [
    ChatMessage("user", [TextBlock("fix it")]),
    ChatMessage("assistant", [TextBlock("editing"),
                              ToolUseBlock(id="t1", name="edit",
                                           input={"path": "a.py"})]),
    ChatMessage("user", [ToolResultBlock(tool_use_id="t1", content="done"),
                         TextBlock("continue")]),
]


def test_anthropic_translation():
    msgs = to_anthropic_messages(CONVO)
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "t1",
                                     "name": "edit", "input": {"path": "a.py"}}
    assert msgs[2]["content"][0]["type"] == "tool_result"
    tools = to_anthropic_tools(TOOLS)
    assert tools[0]["input_schema"]["required"] == ["path"]


def test_anthropic_parse():
    turn = parse_anthropic_response({
        "content": [{"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "x", "name": "edit",
                     "input": {"path": "b.py"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5}})
    assert turn.stop_reason == "tool_use"
    assert turn.message.tool_uses()[0].input == {"path": "b.py"}
    assert turn.usage.input_tokens == 10


def test_openai_translation():
    msgs = to_openai_messages("sys", CONVO)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assistant = msgs[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "edit"
    assert '"path"' in assistant["tool_calls"][0]["function"]["arguments"]
    tool_msg = msgs[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "done"}
    assert msgs[4] == {"role": "user", "content": "continue"}
    tools = to_openai_tools(TOOLS)
    assert tools[0]["function"]["parameters"]["required"] == ["path"]


def test_openai_parse_ok():
    turn = parse_openai_choice({
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "edit", "arguments": '{"path": "a.py"}'}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert turn.stop_reason == "tool_use"
    tu = turn.message.tool_uses()[0]
    assert tu.input == {"path": "a.py"} and tu.parse_error is None


def test_openai_parse_malformed_arguments():
    turn = parse_openai_choice({
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "edit", "arguments": '{"path": '}}]},
            "finish_reason": "tool_calls"}]})
    tu = turn.message.tool_uses()[0]
    assert tu.parse_error is not None and tu.input == {}


def test_openai_tool_error_prefixed():
    convo = [ChatMessage("user", [ToolResultBlock(tool_use_id="t9",
                                                  content="boom", is_error=True)])]
    msgs = to_openai_messages("s", convo)
    assert msgs[1]["content"].startswith("[TOOL ERROR]")
