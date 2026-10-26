"""Runtime settings from the environment, an optional .env file, and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

ProviderName = Literal["groq", "ollama"]

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MAX_STEPS = 8
DEFAULT_TEMPERATURE = 0.2
DEFAULT_LLM_TIMEOUT = 120.0


class ConfigError(Exception):
    """The process cannot start with the settings it was given."""


@dataclass(frozen=True)
class Settings:
    """Resolved agent settings. Secrets stay in ``groq_api_key`` and are not logged."""

    provider: ProviderName
    model: str
    groq_api_key: str | None
    groq_base_url: str
    ollama_base_url: str
    max_steps: int
    temperature: float
    llm_timeout: float
    show_trace: bool = False


def load_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    show_trace: bool = False,
    dotenv: bool = True,
) -> Settings:
    """Load settings, letting explicit arguments override the environment.

    Args:
        provider: ``groq`` or ``ollama``. Falls back to ``AGENT_PROVIDER``.
        model: Model id. Falls back to the provider's ``*_MODEL`` variable.
        max_steps: Tool-calling round limit.
        temperature: Sampling temperature from 0 to 2.
        show_trace: When true, the CLI prints tool traffic to stderr.
        dotenv: When true, read a ``.env`` file without overriding real env vars.

    Raises:
        ConfigError: A value is missing or out of range.
    """
    if dotenv:
        load_dotenv(override=False)

    chosen = (provider or os.getenv("AGENT_PROVIDER") or "groq").strip().lower()
    if chosen not in {"groq", "ollama"}:
        raise ConfigError("Provider must be 'groq' or 'ollama'.")
    provider_name: ProviderName = "groq" if chosen == "groq" else "ollama"

    if model is not None and model.strip():
        chosen_model = model.strip()
    elif provider_name == "groq":
        chosen_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    else:
        chosen_model = (
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
        )

    if max_steps is None:
        raw_steps = os.getenv("AGENT_MAX_STEPS", str(DEFAULT_MAX_STEPS))
        resolved_steps = _positive_int(raw_steps, "AGENT_MAX_STEPS")
    else:
        resolved_steps = _positive_int(str(max_steps), "--max-steps")

    if temperature is None:
        raw_temperature = os.getenv("AGENT_TEMPERATURE", str(DEFAULT_TEMPERATURE))
        resolved_temperature = _temperature(raw_temperature, "AGENT_TEMPERATURE")
    else:
        resolved_temperature = _temperature(str(temperature), "--temperature")

    raw_timeout = os.getenv("AGENT_LLM_TIMEOUT", str(DEFAULT_LLM_TIMEOUT))
    timeout = _positive_float(raw_timeout, "AGENT_LLM_TIMEOUT")

    api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    groq_base = os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL).strip() or DEFAULT_GROQ_BASE_URL
    ollama_base = (
        os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip() or DEFAULT_OLLAMA_BASE_URL
    )

    return Settings(
        provider=provider_name,
        model=chosen_model,
        groq_api_key=api_key,
        groq_base_url=groq_base,
        ollama_base_url=ollama_base,
        max_steps=resolved_steps,
        temperature=resolved_temperature,
        llm_timeout=timeout,
        show_trace=show_trace,
    )


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1.")
    return value


def _positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number.") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive.")
    return value


def _temperature(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number.") from exc
    if not 0 <= value <= 2:
        raise ConfigError(f"{name} must be between 0 and 2.")
    return value
