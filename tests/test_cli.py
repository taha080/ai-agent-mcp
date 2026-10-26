"""CLI parsing. The help text and missing-key path do not call a model."""

from __future__ import annotations

import pytest

from ai_agent_mcp.cli import main


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "ask" in output
    assert "chat" in output
    assert "serve" in output


def test_ask_without_a_groq_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_PROVIDER", "groq")
    with pytest.raises(SystemExit) as exc:
        main(["--provider", "groq", "ask", "What is 2 + 2?"])
    assert exc.value.code == 2
