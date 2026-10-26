"""Tool-calling loop.

The loop does not know whether tools are served over stdio or a fake. It
lists tools, asks the model, executes every requested call, and appends the
results until the model answers or the step budget runs out.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ai_agent_mcp.agent.models import (
    AgentResult,
    LanguageModel,
    ModelTurn,
    ToolCall,
    ToolSession,
    ToolTrace,
)
from ai_agent_mcp.agent.prompt import build_system_prompt

ToolObserver = Callable[[ToolTrace], None]

__all__ = ["build_system_prompt", "run_turn"]


async def run_turn(
    messages: list[dict[str, Any]],
    session: ToolSession,
    model: LanguageModel,
    *,
    max_steps: int = 8,
    on_tool: ToolObserver | None = None,
) -> AgentResult:
    """Run one user turn that is already the last entry in ``messages``.

    Assistant and tool messages are appended to ``messages`` so a chat can
    continue with the same list. ``max_steps`` counts model calls, not tool
    calls. A step that requests tools consumes one step even if it calls
    several tools.

    Raises:
        ValueError: ``max_steps`` is less than 1. Model and transport errors
        propagate to the caller.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    specs = await session.list_tools()
    tools = [spec.to_openai() for spec in specs]
    trace: list[ToolTrace] = []

    for step in range(1, max_steps + 1):
        turn = model.complete(messages, tools)
        if not turn.tool_calls:
            answer = (turn.content or "").strip() or "(The model returned an empty answer.)"
            messages.append({"role": "assistant", "content": answer})
            return AgentResult(answer=answer, trace=tuple(trace), steps=step)

        messages.append(_assistant_message(turn))
        for call in turn.tool_calls:
            result_text, arguments = await _execute(session, call)
            event = ToolTrace(name=call.name, arguments=arguments, result=result_text)
            trace.append(event)
            if on_tool is not None:
                on_tool(event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id or f"call_{len(trace)}",
                    "content": result_text,
                }
            )

    answer = (
        f"Stopped after {max_steps} steps without a final answer. "
        "Try a narrower question or raise --max-steps."
    )
    messages.append({"role": "assistant", "content": answer})
    return AgentResult(answer=answer, trace=tuple(trace), steps=max_steps)


async def _execute(session: ToolSession, call: ToolCall) -> tuple[str, dict[str, Any]]:
    try:
        arguments = coerce_arguments(call.arguments)
    except ValueError as exc:
        return f"Tool error: {exc}", {}
    try:
        result = await session.call_tool(call.name, arguments)
    except Exception as exc:
        return f"Tool error: {exc}", arguments
    return result, arguments


def coerce_arguments(raw: Any) -> dict[str, Any]:
    """Accept either a dict or a JSON object string from the model."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Arguments are not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be a JSON object.")
        return parsed
    raise ValueError("Arguments must be a JSON object.")


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    tool_calls = []
    for index, call in enumerate(turn.tool_calls):
        tool_calls.append(
            {
                "id": call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": _arguments_json(call.arguments),
                },
            }
        )
    return {
        "role": "assistant",
        "content": turn.content or "",
        "tool_calls": tool_calls,
    }


def _arguments_json(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return json.dumps(raw, ensure_ascii=False)
