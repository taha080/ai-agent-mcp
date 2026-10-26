"""MCP client that launches this project's server over stdio."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, Tool

from ai_agent_mcp.agent.models import ToolSpec


class StdioToolSession:
    """Talk to ``python -m ai_agent_mcp.server`` through stdin and stdout.

    The server is a child process. Leaving the ``async with`` block closes
    its stdin and waits for it to exit, which is the MCP shutdown sequence
    implemented by the Python SDK.
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._command = command or sys.executable
        self._args = args if args is not None else ["-m", "ai_agent_mcp.server"]
        self._timeout = timeout
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.instructions: str | None = None

    async def __aenter__(self) -> StdioToolSession:
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env={"PYTHONUNBUFFERED": "1"},
        )
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(
            Client(params, read_timeout_seconds=self._timeout)
        )
        self.instructions = self._client.instructions
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        if self._stack is not None:
            await self._stack.aclose()
        self._client = None
        self._stack = None

    async def list_tools(self) -> list[ToolSpec]:
        """List every tool, following pagination if the server uses it."""
        client = self._require_client()
        specs: list[ToolSpec] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            specs.extend(_spec(tool) for tool in page.tools)
            cursor = page.next_cursor
            if not cursor:
                return specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the text the model should read."""
        client = self._require_client()
        result = await client.call_tool(name, arguments)
        return render_tool_result(result)

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("The MCP session is not open.")
        return self._client


def render_tool_result(result: CallToolResult) -> str:
    """Flatten an MCP tool result into the string stored in the chat."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    body = "\n".join(parts).strip()
    if not body and result.structured_content is not None:
        body = json.dumps(result.structured_content, ensure_ascii=False, sort_keys=True)
    if result.is_error:
        return body or "Tool error."
    return body or "(empty result)"


def _spec(tool: Tool) -> ToolSpec:
    schema = copy.deepcopy(dict(tool.input_schema))
    schema.pop("title", None)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict):
                prop.pop("title", None)
    return ToolSpec(
        name=tool.name,
        description=inspect.cleandoc(tool.description or "").strip(),
        parameters=schema,
    )
