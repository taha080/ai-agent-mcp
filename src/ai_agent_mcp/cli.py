"""Command-line interface: one question, an interactive chat, or the MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

from ai_agent_mcp import __version__
from ai_agent_mcp.agent.llm import LLMError, build_model
from ai_agent_mcp.agent.loop import build_system_prompt, run_turn
from ai_agent_mcp.agent.models import ToolSpec, ToolTrace
from ai_agent_mcp.agent.session import StdioToolSession
from ai_agent_mcp.config import ConfigError, Settings, load_settings


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser used by the ``ai-agent-mcp`` command."""
    parser = argparse.ArgumentParser(
        prog="ai-agent-mcp",
        description="Free AI agent that calls MCP tools using Groq or local Ollama.",
    )
    parser.add_argument("--version", action="version", version=f"ai-agent-mcp {__version__}")
    parser.add_argument(
        "--provider",
        choices=("groq", "ollama"),
        help="Model host. Overrides AGENT_PROVIDER.",
    )
    parser.add_argument("--model", help="Model id. Overrides GROQ_MODEL or OLLAMA_MODEL.")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum model calls per question. Overrides AGENT_MAX_STEPS.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print each tool call and its full result to stderr.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask one question and print the answer.")
    ask.add_argument("question", nargs="+", help="The question to answer.")

    commands.add_parser("chat", help="Start an interactive chat.")
    commands.add_parser("tools", help="List tools exposed by the MCP server.")
    commands.add_parser("serve", help="Run the MCP server on stdio.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from ai_agent_mcp.server import main as serve_main

            serve_main()
            return
        settings = load_settings(
            provider=args.provider,
            model=args.model,
            max_steps=args.max_steps,
            show_trace=args.show_trace,
        )
        if args.command == "tools":
            asyncio.run(_print_tools())
            return
        model = build_model(settings)
        if args.command == "ask":
            question = " ".join(args.question).strip()
            if not question:
                raise ConfigError("Question is empty.")
            asyncio.run(_ask(question, settings, model))
            return
        if args.command == "chat":
            asyncio.run(_chat(settings, model))
            return
        parser.error(f"Unknown command {args.command!r}.")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print(file=sys.stderr)
        raise SystemExit(130) from None


async def _print_tools() -> None:
    async with StdioToolSession() as session:
        specs = await session.list_tools()
    _print_tool_list(specs)


async def _ask(question: str, settings: Settings, model: Any) -> None:
    async with StdioToolSession() as session:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(session.instructions)},
            {"role": "user", "content": question},
        ]
        result = await run_turn(
            messages,
            session,
            model,
            max_steps=settings.max_steps,
            on_tool=_full_trace if settings.show_trace else None,
        )
    print(result.answer)


async def _chat(settings: Settings, model: Any) -> None:
    async with StdioToolSession() as session:
        specs = await session.list_tools()
        names = ", ".join(sorted(spec.name for spec in specs))
        print(f"ai-agent-mcp {__version__}  ({settings.provider}, {settings.model})")
        print(f"tools: {names}")
        print("Type a question. /tools lists them again. /exit quits.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(session.instructions)}
        ]
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                print(file=sys.stderr)
                return
            except KeyboardInterrupt:
                print(file=sys.stderr)
                return
            if not line:
                continue
            if line.lower() in {"/exit", "/quit", "exit", "quit"}:
                return
            if line == "/tools":
                _print_tool_list(specs)
                continue
            start = len(messages)
            messages.append({"role": "user", "content": line})
            try:
                result = await run_turn(
                    messages,
                    session,
                    model,
                    max_steps=settings.max_steps,
                    on_tool=_preview_trace,
                )
            except (LLMError, ConfigError) as exc:
                del messages[start:]
                print(f"error: {exc}", file=sys.stderr)
                continue
            print(f"agent> {result.answer}")


def _print_tool_list(specs: list[ToolSpec]) -> None:
    if not specs:
        print("No tools registered.")
        return
    for spec in sorted(specs, key=lambda item: item.name):
        print(spec.name)
        if spec.description:
            print(textwrap.indent(spec.description, "  "))
        print()


def _full_trace(event: ToolTrace) -> None:
    print(f"tool> {event.name} {_dump(event.arguments)}", file=sys.stderr)
    print(event.result, file=sys.stderr)


def _preview_trace(event: ToolTrace) -> None:
    print(f"tool> {event.name} {_dump(event.arguments)}", file=sys.stderr)
    preview = " ".join(event.result.split())
    if len(preview) > 160:
        preview = preview[:159] + "…"
    if preview:
        print(f"      {preview}", file=sys.stderr)


def _dump(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
