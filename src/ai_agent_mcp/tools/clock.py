"""Current time in an IANA timezone. No network access."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_agent_mcp.tools.errors import ToolFailure


class TimeError(ToolFailure):
    """The timezone name was missing or not a known IANA timezone."""


def current_time(timezone: str = "UTC") -> str:
    """Return the current time as an ISO-8601 string.

    Args:
        timezone: IANA name such as ``UTC`` or ``America/New_York``.

    Raises:
        TimeError: ``timezone`` is empty or unknown on this machine.
    """
    if not isinstance(timezone, str):
        raise TimeError("Timezone must be a string.")
    name = timezone.strip()
    if not name:
        raise TimeError("Timezone is empty.")
    try:
        zone = ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise TimeError(f"Unknown timezone {name!r}.") from exc
    return datetime.now(zone).isoformat(timespec="seconds")
