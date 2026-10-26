"""Agent loop driven by a scripted model and an in-memory tool session."""

from __future__ import annotations

import pytest
from tests.fakes import FakeSession, ScriptedModel

from ai_agent_mcp.agent.loop import build_system_prompt, run_turn
from ai_agent_mcp.agent.models import ModelTurn, ToolCall, ToolSpec

CALCULATE = ToolSpec(
    name="calculate",
    description="Safe arithmetic.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
CLOCK = ToolSpec(
    name="current_time",
    description="Current time.",
    parameters={"type": "object", "properties": {"timezone": {"type": "string"}}},
)


def _answer(text: str) -> ModelTurn:
    return ModelTurn(content=text, tool_calls=())


def _call(call_id: str, name: str, arguments: object) -> ModelTurn:
    return ModelTurn(content=None, tool_calls=(ToolCall(call_id, name, arguments),))


async def test_answers_without_tools() -> None:
    model = ScriptedModel([_answer("Just prose.")])
    session = FakeSession([CALCULATE], lambda _name, _args: "unused")
    messages = [{"role": "user", "content": "hi"}]
    result = await run_turn(messages, session, model, max_steps=3)
    assert result.answer == "Just prose."
    assert result.trace == ()
    assert session.calls == []
    assert model.seen[0][1][0]["function"]["name"] == "calculate"
    assert messages[-1] == {"role": "assistant", "content": "Just prose."}


async def test_calls_a_tool_then_answers() -> None:
    model = ScriptedModel(
        [
            _call("c1", "calculate", {"expression": "2+2"}),
            _answer("The answer is 4."),
        ]
    )

    def handler(name: str, arguments: dict[str, object]) -> str:
        assert name == "calculate"
        assert arguments == {"expression": "2+2"}
        return "4"

    session = FakeSession([CALCULATE], handler)
    result = await run_turn([{"role": "user", "content": "2+2"}], session, model)
    assert result.answer == "The answer is 4."
    assert result.trace[0].result == "4"
    assert result.steps == 2
    follow_up = model.seen[1][0]
    assert follow_up[-1] == {"role": "tool", "tool_call_id": "c1", "content": "4"}
    assistant = follow_up[-2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"expression": "2+2"}'


async def test_accepts_json_string_arguments() -> None:
    model = ScriptedModel(
        [
            _call("c1", "calculate", '{"expression": "6*7"}'),
            _answer("42"),
        ]
    )
    session = FakeSession([CALCULATE], lambda _name, arguments: str(arguments["expression"]))
    result = await run_turn([{"role": "user", "content": "math"}], session, model)
    assert session.calls == [("calculate", {"expression": "6*7"})]
    assert result.answer == "42"


async def test_invalid_json_is_returned_to_the_model() -> None:
    model = ScriptedModel(
        [
            _call("c1", "calculate", "{not json"),
            _answer("I could not run that."),
        ]
    )
    session = FakeSession([CALCULATE], lambda _name, _args: pytest.fail("tool should not run"))
    result = await run_turn([{"role": "user", "content": "math"}], session, model)
    assert session.calls == []
    assert "not valid JSON" in model.seen[1][0][-1]["content"]
    assert result.answer == "I could not run that."


async def test_tool_exception_is_fed_back() -> None:
    model = ScriptedModel(
        [
            _call("c1", "calculate", {"expression": "1/0"}),
            _answer("That divides by zero."),
        ]
    )

    def handler(_name: str, _arguments: dict[str, object]) -> str:
        raise RuntimeError("Division by zero.")

    session = FakeSession([CALCULATE], handler)
    result = await run_turn([{"role": "user", "content": "1/0"}], session, model)
    assert "Division by zero" in model.seen[1][0][-1]["content"]
    assert result.answer == "That divides by zero."


async def test_several_tools_in_one_step() -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                content=None,
                tool_calls=(
                    ToolCall("a", "calculate", {"expression": "1+1"}),
                    ToolCall("b", "current_time", {"timezone": "UTC"}),
                ),
            ),
            _answer("2, and the clock was checked."),
        ]
    )

    def handler(name: str, _arguments: dict[str, object]) -> str:
        return "2" if name == "calculate" else "2026-05-04T03:02:01+00:00"

    session = FakeSession([CALCULATE, CLOCK], handler)
    result = await run_turn([{"role": "user", "content": "both"}], session, model)
    assert [call[0] for call in session.calls] == ["calculate", "current_time"]
    roles = [message["role"] for message in model.seen[1][0]]
    assert roles[-3:] == ["assistant", "tool", "tool"]
    assert result.answer.startswith("2,")


async def test_stops_at_max_steps() -> None:
    model = ScriptedModel(
        [
            _call("c1", "calculate", {"expression": "1+1"}),
            _call("c2", "calculate", {"expression": "1+1"}),
        ]
    )
    session = FakeSession([CALCULATE], lambda _name, _args: "2")
    result = await run_turn(
        [{"role": "user", "content": "loop"}],
        session,
        model,
        max_steps=2,
    )
    assert result.steps == 2
    assert "Stopped after 2 steps" in result.answer
    assert len(result.trace) == 2


def test_system_prompt_includes_server_notes() -> None:
    prompt = build_system_prompt("calculate evaluates arithmetic and never runs arbitrary code.")
    assert "Server notes:" in prompt
    assert "never runs arbitrary code" in prompt
    assert "web_search" in prompt
