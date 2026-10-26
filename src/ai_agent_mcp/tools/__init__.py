"""Tool implementations used by the MCP server.

The functions in this package do the actual work. The server module is a thin
MCP adapter around them, which keeps the logic easy to test without a client.
"""

from ai_agent_mcp.tools.calculator import CalculatorError, evaluate
from ai_agent_mcp.tools.clock import TimeError, current_time
from ai_agent_mcp.tools.errors import ToolFailure
from ai_agent_mcp.tools.fetch import FetchError, fetch_page
from ai_agent_mcp.tools.search import SearchError, search_web

__all__ = [
    "CalculatorError",
    "FetchError",
    "SearchError",
    "TimeError",
    "ToolFailure",
    "current_time",
    "evaluate",
    "fetch_page",
    "search_web",
]
