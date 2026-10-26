"""OpenAI-compatible chat clients for Groq and local Ollama.

Both hosts speak the same chat-completions payload. Groq is the free hosted
API. Ollama is a local server and ignores the API key. Tests pass an ``httpx``
transport so no request leaves the machine.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai_agent_mcp.agent.models import ModelTurn, ToolCall
from ai_agent_mcp.config import ConfigError, Settings


class LLMError(Exception):
    """The model host returned an error or an unusable payload."""


class OpenAICompatibleModel:
    """POST ``/chat/completions`` and parse tool calls from the response."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 120.0,
        send_tool_choice: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = openai_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature if temperature > 0 else 1e-8
        self.timeout = timeout
        self.send_tool_choice = send_tool_choice
        self._transport = transport

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        """Send one non-streaming chat request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if self.send_tool_choice:
                payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(transport=self._transport, timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMError("The model request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"The model request failed: {exc.__class__.__name__}.") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"The model request failed ({response.status_code}): {_safe_body(response)}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc
        if not isinstance(body, dict):
            raise LLMError("The model returned an unexpected payload.")
        return parse_chat_response(body)


def build_model(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> OpenAICompatibleModel:
    """Build the provider selected in ``settings``.

    Raises:
        ConfigError: Groq was selected and ``GROQ_API_KEY`` is missing.
    """
    if settings.provider == "groq":
        if not settings.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
                "from https://console.groq.com/keys, or run with --provider ollama."
            )
        return OpenAICompatibleModel(
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key,
            model=settings.model,
            temperature=settings.temperature,
            timeout=settings.llm_timeout,
            send_tool_choice=True,
            transport=transport,
        )
    return OpenAICompatibleModel(
        base_url=settings.ollama_base_url,
        api_key="ollama",
        model=settings.model,
        temperature=settings.temperature,
        timeout=settings.llm_timeout,
        send_tool_choice=False,
        transport=transport,
    )


def openai_base_url(url: str) -> str:
    """Ensure the base URL ends in ``/v1`` exactly once."""
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        raise ConfigError("The model base URL is empty.")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def parse_chat_response(payload: dict[str, Any]) -> ModelTurn:
    """Parse a chat-completions JSON body into a ``ModelTurn``."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("The model response did not contain a message.") from exc
    if not isinstance(message, dict):
        raise LLMError("The model response did not contain a message.")
    content = _content_text(message.get("content"))
    tool_calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise LLMError("The model returned malformed tool calls.")
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call_{index}"),
                name=str(name),
                arguments=function.get("arguments"),
            )
        )
    return ModelTurn(content=content, tool_calls=tuple(tool_calls))


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        text = "".join(parts).strip()
        return text or None
    text = str(content).strip()
    return text or None


def _safe_body(response: httpx.Response) -> str:
    text = " ".join(response.text.split())
    if len(text) > 300:
        return text[:300] + "..."
    return text or response.reason_phrase
