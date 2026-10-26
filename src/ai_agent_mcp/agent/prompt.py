"""System prompt for the tool-calling loop."""

from __future__ import annotations

_RULES = """You are a concise assistant with tools exposed over the Model Context Protocol.

Use the tools when they improve the answer:
- calculate for every arithmetic step. Do not do multi-step math yourself.
- web_search for facts that may have changed or that you are not sure about.
- fetch_url to read one public http or https page when a snippet is not enough.
- current_time when the answer depends on the date or the clock.

If a tool returns an error, change the arguments or explain the failure.
Never invent tool output. When you can answer, reply in plain prose and do not call a tool.
"""


def build_system_prompt(server_instructions: str | None) -> str:
    """Combine the agent rules with the instructions published by the MCP server."""
    notes = (server_instructions or "").strip()
    if not notes:
        return _RULES.strip()
    return f"{_RULES.strip()}\n\nServer notes:\n{notes}"
