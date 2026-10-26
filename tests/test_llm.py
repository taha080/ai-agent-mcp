"""Chat-completions parsing and provider requests. The transport is mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from ai_agent_mcp.agent.llm import (
    LLMError,
    OpenAICompatibleModel,
    build_model,
    openai_base_url,
    parse_chat_response,
)
from ai_agent_mcp.config import ConfigError, Settings


def _settings(**overrides: object) -> Settings:
    data: dict[str, object] = {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "groq_api_key": "gsk-test-SHOULD-NOT-LEAK",
        "groq_base_url": "https://api.groq.com/openai/v1",
        "ollama_base_url": "http://127.0.0.1:11434",
        "max_steps": 4,
        "temperature": 0.2,
        "llm_timeout": 5.0,
        "show_trace": False,
    }
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


def test_parse_plain_answer() -> None:
    turn = parse_chat_response(
        {"choices": [{"message": {"role": "assistant", "content": " hello "}}]}
    )
    assert turn.content == "hello"
    assert turn.tool_calls == ()


def test_parse_tool_call_with_string_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression": "2+2"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    turn = parse_chat_response(payload)
    assert turn.tool_calls[0].name == "calculate"
    assert turn.tool_calls[0].arguments == '{"expression": "2+2"}'


def test_parse_tool_call_with_object_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "checking"}],
                    "tool_calls": [
                        {"function": {"name": "current_time", "arguments": {"timezone": "UTC"}}}
                    ],
                }
            }
        ]
    }
    turn = parse_chat_response(payload)
    assert turn.content == "checking"
    assert turn.tool_calls[0].id == "call_0"
    assert turn.tool_calls[0].arguments == {"timezone": "UTC"}


def test_parse_rejects_a_payload_without_choices() -> None:
    with pytest.raises(LLMError, match="did not contain a message"):
        parse_chat_response({"choices": []})


def test_groq_request_shape_and_secret_is_not_in_the_error() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(401, json={"error": {"message": "nope"}})

    model = build_model(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="401") as exc:
        model.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert "gsk-test-SHOULD-NOT-LEAK" not in str(exc.value)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk-test-SHOULD-NOT-LEAK"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["tool_choice"] == "auto"
    assert "gsk-test-SHOULD-NOT-LEAK" not in json.dumps(body)


def test_ollama_omits_tool_choice_and_normalizes_the_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert "tool_choice" not in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "local"}}]},
        )

    settings = _settings(provider="ollama", model="llama3.2", groq_api_key=None)
    model = build_model(settings, transport=httpx.MockTransport(handler))
    turn = model.complete([{"role": "user", "content": "hi"}], [])
    assert turn.content == "local"


def test_missing_groq_key() -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        build_model(_settings(groq_api_key=None))


def test_openai_base_url() -> None:
    assert openai_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/v1"
    assert openai_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434/v1"


def test_direct_model_can_parse_a_successful_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "abc",
                                    "function": {"name": "calculate", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    model = OpenAICompatibleModel(
        base_url="https://api.groq.com/openai/v1",
        api_key="test",
        model="demo",
        transport=httpx.MockTransport(handler),
    )
    turn = model.complete([], [])
    assert turn.tool_calls[0].id == "abc"
