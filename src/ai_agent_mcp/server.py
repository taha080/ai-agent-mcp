"""MCP server that exposes the agent's free tools over stdio.

Run it directly with ``python -m ai_agent_mcp.server``. The agent does that
itself; any other MCP host can use the same command. Stdout is the protocol
stream, so this module does not print. Logs go to stderr.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from ai_agent_mcp import __version__
from ai_agent_mcp.tools.calculator import evaluate
from ai_agent_mcp.tools.clock import current_time as lookup_time
from ai_agent_mcp.tools.errors import ToolFailure
from ai_agent_mcp.tools.fetch import fetch_page
from ai_agent_mcp.tools.search import DEFAULT_RESULTS, search_web

_INSTRUCTIONS = (
    "Free tools only. web_search uses DuckDuckGo and needs no API key. "
    "calculate evaluates arithmetic and never runs arbitrary code. "
    "fetch_url reads a public http(s) page with a timeout and a size cap. "
    "current_time reports the clock in an IANA timezone."
)


def build_server() -> MCPServer:
    """Create a server instance with the four built-in tools registered."""
    mcp = MCPServer(
        "ai-agent-mcp",
        instructions=_INSTRUCTIONS,
        version=__version__,
    )

    @mcp.tool()
    def calculate(
        expression: Annotated[
            str,
            Field(description="Arithmetic expression, for example (17 * 23) + sqrt(144)."),
        ],
    ) -> str:
        """Evaluate an arithmetic expression.

        Supports +, -, *, /, //, %, **, parentheses, and the functions abs,
        round, sqrt, min, max, and pow. Names, attributes, and other calls
        are rejected. This does not use Python's eval.
        """
        return _guard(evaluate, expression)

    @mcp.tool()
    def web_search(
        query: Annotated[str, Field(description="Words or a question to search for.")],
        max_results: Annotated[
            int,
            Field(description="How many results to return, from 1 to 10.", ge=1, le=10),
        ] = DEFAULT_RESULTS,
    ) -> str:
        """Search the public web without an API key.

        Returns titles, URLs, and snippets. Use fetch_url to read a page when
        a snippet is not enough.
        """
        return _guard(search_web, query, max_results)

    @mcp.tool()
    def fetch_url(
        url: Annotated[str, Field(description="Public http or https URL to read.")],
    ) -> str:
        """Fetch a public web page and return readable text.

        HTML is converted to plain text. The request has a timeout and a size
        cap. Non-HTTP URLs and non-public addresses are refused.
        """
        return _guard(fetch_page, url)

    @mcp.tool()
    def current_time(
        timezone: Annotated[
            str,
            Field(description="IANA timezone name such as UTC or America/New_York."),
        ] = "UTC",
    ) -> str:
        """Return the current date and time in the given IANA timezone."""
        return _guard(lookup_time, timezone)

    return mcp


def _guard(func: Callable[..., str], *args: object) -> str:
    try:
        return func(*args)
    except ToolFailure as exc:
        raise ToolError(str(exc)) from exc


mcp = build_server()


def main() -> None:
    """Serve MCP on stdio until the client disconnects."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
