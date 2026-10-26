"""Web search through the keyless ``ddgs`` metasearch client.

DuckDuckGo (and the other backends ``ddgs`` may fall back to) do not require
an API key. Tests inject a ``backend`` callable so the suite never touches
the network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ddgs import DDGS

from ai_agent_mcp.tools.errors import ToolFailure

MAX_QUERY_LENGTH = 400
MAX_RESULTS = 10
DEFAULT_RESULTS = 5

SearchBackend = Callable[[str, int], list[dict[str, Any]]]


class SearchError(ToolFailure):
    """The query was unusable or the search backend failed."""


def search_web(
    query: str,
    max_results: int = DEFAULT_RESULTS,
    *,
    backend: SearchBackend | None = None,
) -> str:
    """Search the public web and return a numbered plain-text list.

    Args:
        query: Words to search for.
        max_results: Desired result count. Values outside 1..10 are clamped.
        backend: Optional replacement for the live ``ddgs`` client.

    Raises:
        SearchError: The query is empty or too long, or the backend fails.
    """
    if not isinstance(query, str):
        raise SearchError("Query must be a string.")
    text = query.strip()
    if not text:
        raise SearchError("Query is empty.")
    if len(text) > MAX_QUERY_LENGTH:
        raise SearchError("Query is too long.")
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise SearchError("max_results must be an integer.")
    capped = min(max(max_results, 1), MAX_RESULTS)
    search = backend or default_search
    try:
        raw_results = search(text, capped)
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f"Search failed: {exc}") from exc
    results = [item for item in list(raw_results or []) if isinstance(item, dict)][:capped]
    if not results:
        return "No results found."
    return _format_results(results)


def default_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run a live text search. This is the only function that uses the network."""
    try:
        with DDGS(timeout=15) as ddgs:
            results = ddgs.text(query, max_results=max_results, safesearch="moderate")
    except Exception as exc:
        raise SearchError(f"Search failed: {exc}") from exc
    return [dict(item) for item in results or []]


def _format_results(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "(no title)").strip()
        href = str(item.get("href") or item.get("url") or "").strip()
        body = " ".join(str(item.get("body") or "").split())
        blocks.append(f"{index}. {title}\n   {href}\n   {body}".rstrip())
    return "\n".join(blocks)
