"""MCP server registration, in-memory calls, and one real stdio session."""

from __future__ import annotations

import pytest
from mcp import Client
from tests.fakes import ScriptedModel

from ai_agent_mcp.agent.loop import run_turn
from ai_agent_mcp.agent.models import ModelTurn, ToolCall
from ai_agent_mcp.agent.session import StdioToolSession
from ai_agent_mcp.server import build_server

EXPECTED_TOOLS = {"calculate", "current_time", "fetch_url", "web_search"}


async def test_server_lists_tools_and_calculates() -> None:
    async with Client(build_server()) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert names == EXPECTED_TOOLS
        assert client.instructions
        assert "calculate" in client.instructions
        result = await client.call_tool("calculate", {"expression": "(17 * 23) + sqrt(144)"})
    assert result.is_error is False
    assert result.content[0].text == "403"


async def test_calculate_rejects_code_execution() -> None:
    async with Client(build_server()) as client:
        result = await client.call_tool(
            "calculate",
            {"expression": "__import__('os').system('echo pwned')"},
        )
    assert result.is_error is True
    assert "not allowed" in result.content[0].text.lower() or "Unknown" in result.content[0].text


async def test_fetch_tool_refuses_file_urls() -> None:
    async with Client(build_server()) as client:
        result = await client.call_tool("fetch_url", {"url": "file:///etc/passwd"})
    assert result.is_error is True
    assert "http" in result.content[0].text.lower()


async def test_search_tool_uses_the_injected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def backend(query: str, max_results: int) -> list[dict[str, str]]:
        assert max_results == 1
        return [{"title": "Spec", "href": "https://example.com/mcp", "body": query}]

    monkeypatch.setattr("ai_agent_mcp.tools.search.default_search", backend)
    async with Client(build_server()) as client:
        result = await client.call_tool("web_search", {"query": "mcp", "max_results": 1})
    assert result.is_error is False
    assert "Spec" in result.content[0].text
    assert "https://example.com/mcp" in result.content[0].text


async def test_unknown_timezone_is_a_tool_error() -> None:
    async with Client(build_server()) as client:
        result = await client.call_tool("current_time", {"timezone": "Not/AZone"})
    assert result.is_error is True
    assert "Unknown timezone" in result.content[0].text


async def test_agent_loop_over_stdio() -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                content=None,
                tool_calls=(ToolCall("c1", "calculate", {"expression": "6 * 7"}),),
            ),
            ModelTurn(content="42", tool_calls=()),
        ]
    )
    async with StdioToolSession() as session:
        names = {spec.name for spec in await session.list_tools()}
        assert names == EXPECTED_TOOLS
        calculate = next(spec for spec in await session.list_tools() if spec.name == "calculate")
        assert "title" not in calculate.parameters
        result = await run_turn(
            [{"role": "user", "content": "what is 6 * 7?"}],
            session,
            model,
            max_steps=4,
        )
    assert result.answer == "42"
    assert result.trace[0].name == "calculate"
    assert result.trace[0].result == "42"
