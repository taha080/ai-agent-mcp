"""Offline stand-ins for the model and the MCP session."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from ai_agent_mcp.agent.models import ModelTurn, ToolSpec


class ScriptedModel:
    """Return a prepared sequence of model turns and record what it was shown."""

    def __init__(self, script: list[ModelTurn]) -> None:
        self.script = list(script)
        self.seen: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        self.seen.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if not self.script:
            raise AssertionError("ScriptedModel received an unexpected model call.")
        return self.script.pop(0)


class FakeSession:
    """In-memory tools with a handler the test controls."""

    def __init__(
        self,
        tools: list[ToolSpec],
        handler: Callable[[str, dict[str, Any]], str],
    ) -> None:
        self._tools = tools
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self._handler(name, arguments)
