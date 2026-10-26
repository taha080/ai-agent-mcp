"""Web search formatting with a fake backend. No network."""

from __future__ import annotations

import pytest

from ai_agent_mcp.tools import search as search_mod
from ai_agent_mcp.tools.search import SearchError, search_web


def test_formats_backend_results() -> None:
    def backend(query: str, max_results: int) -> list[dict[str, str]]:
        assert query == "model context protocol"
        assert max_results == 2
        return [
            {
                "title": "MCP",
                "href": "https://modelcontextprotocol.io",
                "body": "An open protocol.",
            }
        ]

    text = search_web("model context protocol", max_results=2, backend=backend)
    assert "1. MCP" in text
    assert "https://modelcontextprotocol.io" in text
    assert "An open protocol." in text


def test_empty_backend_reports_no_results() -> None:
    assert search_web("anything", backend=lambda _query, _limit: []) == "No results found."


def test_clamps_max_results() -> None:
    seen: dict[str, int] = {}

    def backend(_query: str, max_results: int) -> list[dict[str, str]]:
        seen["n"] = max_results
        return []

    search_web("query", max_results=100, backend=backend)
    assert seen["n"] == 10
    search_web("query", max_results=0, backend=backend)
    assert seen["n"] == 1


def test_empty_query_is_rejected() -> None:
    with pytest.raises(SearchError, match="empty"):
        search_web("   ")


def test_injected_backend_does_not_construct_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("DDGS should not be constructed")

    monkeypatch.setattr(search_mod, "DDGS", boom)
    text = search_web(
        "offline",
        backend=lambda _query, _limit: [
            {"title": "Local", "href": "https://a.example", "body": "x"}
        ],
    )
    assert "Local" in text


def test_backend_failure_is_a_search_error() -> None:
    def backend(_query: str, _limit: int) -> list[dict[str, str]]:
        raise RuntimeError("offline")

    with pytest.raises(SearchError, match="offline"):
        search_web("query", backend=backend)
