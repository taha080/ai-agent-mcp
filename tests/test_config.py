"""Settings loading. These tests do not read a developer .env file."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_agent_mcp.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ConfigError,
    load_settings,
)

_ENV_KEYS = (
    "AGENT_PROVIDER",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "AGENT_MAX_STEPS",
    "AGENT_TEMPERATURE",
    "AGENT_LLM_TIMEOUT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_groq_defaults(clean_env: None) -> None:
    del clean_env
    settings = load_settings(dotenv=False)
    assert settings.provider == "groq"
    assert settings.model == DEFAULT_GROQ_MODEL
    assert settings.groq_api_key is None
    assert settings.max_steps == 8


def test_ollama_model_from_the_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del clean_env
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
    settings = load_settings(dotenv=False)
    assert settings.provider == "ollama"
    assert settings.model == "qwen2.5"


def test_cli_overrides_environment(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del clean_env
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    settings = load_settings(provider="groq", model="llama-3.3-70b-versatile", dotenv=False)
    assert settings.provider == "groq"
    assert settings.model == "llama-3.3-70b-versatile"


def test_rejects_unknown_provider(clean_env: None) -> None:
    del clean_env
    with pytest.raises(ConfigError, match="groq"):
        load_settings(provider="openai", dotenv=False)


def test_rejects_bad_step_count(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del clean_env
    monkeypatch.setenv("AGENT_MAX_STEPS", "0")
    with pytest.raises(ConfigError, match="at least 1"):
        load_settings(dotenv=False)


def test_default_ollama_model_name(clean_env: None) -> None:
    del clean_env
    settings = load_settings(provider="ollama", dotenv=False)
    assert settings.model == DEFAULT_OLLAMA_MODEL


def test_env_example_has_no_secret() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("GROQ_API_KEY="):
            assert line == "GROQ_API_KEY="
            break
    else:
        raise AssertionError("GROQ_API_KEY is missing from .env.example")


def test_env_is_gitignored_and_example_is_not() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", ".env"],
        check=False,
        capture_output=True,
        text=True,
    )
    example = subprocess.run(
        ["git", "check-ignore", ".env.example"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    assert example.returncode != 0
