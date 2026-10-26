"""Clock tool with a frozen datetime. No network."""

from __future__ import annotations

from datetime import datetime

import pytest

from ai_agent_mcp.tools.clock import TimeError, current_time


class _Frozen(datetime):
    @classmethod
    def now(cls, tz: object = None) -> _Frozen:
        return cls(2026, 5, 4, 3, 2, 1, tzinfo=tz)  # type: ignore[arg-type]


def test_utc_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_agent_mcp.tools.clock.datetime", _Frozen)
    assert current_time("UTC") == "2026-05-04T03:02:01+00:00"


def test_named_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_agent_mcp.tools.clock.datetime", _Frozen)
    assert current_time("America/New_York") == "2026-05-04T03:02:01-04:00"


def test_unknown_timezone() -> None:
    with pytest.raises(TimeError, match="Unknown timezone"):
        current_time("Not/AZone")


def test_empty_timezone() -> None:
    with pytest.raises(TimeError, match="empty"):
        current_time("  ")
