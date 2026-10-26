"""Value types shared by the agent loop, the MCP session, and LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class ModelTurn:
    """A single model response, which may include tool calls."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class ToolSpec:
    """A tool the agent is allowed to offer the model."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


@dataclass(frozen=True)
class ToolTrace:
    """A tool call that actually ran, with the text handed back to the model."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentResult:
    """The final answer and the tool calls that produced it."""

    answer: str
    trace: tuple[ToolTrace, ...]
    steps: int


class ToolSession(Protocol):
    """The slice of an MCP client the loop needs."""

    async def list_tools(self) -> list[ToolSpec]:
        """Return the tools currently offered by the server."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call ``name`` and return its text result, including tool errors."""


class LanguageModel(Protocol):
    """A chat model that can request tool calls."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        """Return the next assistant turn for ``messages`` and ``tools``."""
