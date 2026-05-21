"""Interactive CLI test bed for the Ollama provider.

Exercises every knob that makes Ollama's native API better than the
OpenAI-compatible shim — ``think`` on/off, streaming on/off, and
MCP-discovered tools — in one place. Useful for verifying the
provider end-to-end before wiring it into a larger application.

Examples:
    # Streaming chat with whatever the model's default for thinking is
    uv run python examples/ollama_cli.py --model qwen3:8b

    # Force thinking off (fast non-reasoning behavior)
    uv run python examples/ollama_cli.py --model qwen3:8b --no-think

    # Disable streaming (one big response at the end)
    uv run python examples/ollama_cli.py --model qwen3:8b --no-stream

    # Wire in MCP tools from a running server
    uv run python examples/ollama_cli.py --model qwen3:8b \\
        --mcp http://localhost:8080/mcp

Type a message at the prompt. ``/help`` lists commands. ``/quit`` exits.

Run with:
    uv run python examples/ollama_cli.py [flags]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging  # noqa: E402

from roomkit.providers.ai.base import (  # noqa: E402
    AIContext,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig  # noqa: E402

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Output helpers — rich if available, plain otherwise
# ---------------------------------------------------------------------------


class _Renderer:
    """Thin wrapper so the loop doesn't fork on rich-vs-plain everywhere."""

    def __init__(self) -> None:
        self._console: Any = Console() if _HAS_RICH else None
        self._thinking_open = False
        self._answer_open = False

    def header(self, text: str) -> None:
        if self._console:
            self._console.print(Panel.fit(text, style="bold cyan"))
        else:
            print(f"\n=== {text} ===")

    def info(self, text: str) -> None:
        if self._console:
            self._console.print(f"[dim]{text}[/dim]")
        else:
            print(text)

    def warn(self, text: str) -> None:
        if self._console:
            self._console.print(f"[yellow]{text}[/yellow]")
        else:
            print(f"WARN: {text}")

    def error(self, text: str) -> None:
        if self._console:
            self._console.print(f"[red]{text}[/red]")
        else:
            print(f"ERROR: {text}")

    def thinking_delta(self, delta: str) -> None:
        if not self._thinking_open:
            self._close_answer()
            if self._console:
                self._console.print("💭 thinking: ", style="dim italic", end="")
            else:
                print("💭 thinking: ", end="", flush=True)
            self._thinking_open = True
        if self._console:
            # Pass style as a kwarg so brackets/backslashes in the delta
            # (LaTeX, code snippets) aren't reinterpreted as Rich markup.
            self._console.print(delta, style="dim italic", end="", markup=False)
        else:
            print(delta, end="", flush=True)

    def text_delta(self, delta: str) -> None:
        if not self._answer_open:
            self._close_thinking()
            if self._console:
                self._console.print("🤖 ", style="bold green", end="")
            else:
                print("🤖 ", end="", flush=True)
            self._answer_open = True
        if self._console:
            self._console.print(delta, end="", markup=False)
        else:
            print(delta, end="", flush=True)

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._close_thinking()
        self._close_answer()
        payload = json.dumps(args, ensure_ascii=False)
        if self._console:
            self._console.print(f"[bold magenta]🔧 {name}[/bold magenta]({payload})")
        else:
            print(f"🔧 {name}({payload})")

    def tool_result(self, name: str, result: str) -> None:
        snippet = result if len(result) <= 400 else result[:400] + "…"
        if self._console:
            self._console.print(f"[magenta]   ↳ {name} →[/magenta] {snippet}")
        else:
            print(f"   ↳ {name} → {snippet}")

    def done(self, finish_reason: str | None, usage: dict[str, int], elapsed: float) -> None:
        self._close_thinking()
        self._close_answer()
        if self._console:
            self._console.print()
            self._console.print(
                f"[dim](finish={finish_reason!s} • "
                f"tokens in={usage.get('input_tokens', 0)} "
                f"out={usage.get('output_tokens', 0)} • "
                f"elapsed={elapsed:.1f}s)[/dim]"
            )
        else:
            print()
            print(
                f"(finish={finish_reason!s} "
                f"tokens in={usage.get('input_tokens', 0)} "
                f"out={usage.get('output_tokens', 0)} "
                f"elapsed={elapsed:.1f}s)"
            )

    def assistant_block(self, text: str) -> None:
        """Render a non-streamed assistant reply."""
        self._close_thinking()
        self._close_answer()
        if self._console and text:
            self._console.print(Markdown(text))
        elif text:
            print(text)

    def _close_thinking(self) -> None:
        if self._thinking_open:
            if self._console:
                self._console.print()
            else:
                print()
            self._thinking_open = False

    def _close_answer(self) -> None:
        if self._answer_open:
            if self._console:
                self._console.print()
            else:
                print()
            self._answer_open = False


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _exec_tool(
    name: str,
    args: dict[str, Any],
    mcp: Any | None,
) -> str:
    if mcp is not None and name in mcp.tool_names():
        return await mcp.call_tool(name, args)
    # Built-in fallback so the CLI works without MCP for quick sanity checks.
    if name == "get_time":
        return json.dumps({"now": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return json.dumps({"error": f"No handler registered for tool {name!r}"})


# ---------------------------------------------------------------------------
# One conversation turn
# ---------------------------------------------------------------------------


async def _run_turn(
    *,
    provider: OllamaAIProvider,
    history: list[AIMessage],
    system_prompt: str,
    tools: list[AITool],
    thinking_budget: int | None,
    use_stream: bool,
    mcp: Any | None,
    renderer: _Renderer,
    max_tool_rounds: int = 8,
) -> None:
    """Run one user turn, looping for tool calls until the model is done."""
    for _tool_round in range(max_tool_rounds):
        context = AIContext(
            messages=history,
            system_prompt=system_prompt,
            tools=tools,
            thinking_budget=thinking_budget,
            max_tokens=2048,
        )
        t0 = time.monotonic()

        if use_stream:
            stream = provider.generate_structured_stream(context)
        else:
            # Wrap the non-streaming generate() as a single-event "stream"
            # so the rest of the loop is uniform.
            stream = _wrap_generate_as_stream(provider, context)

        tool_calls: list[StreamToolCall] = []
        assistant_text = ""
        assistant_thinking = ""
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        async for event in stream:
            if isinstance(event, StreamThinkingDelta):
                assistant_thinking += event.thinking
                renderer.thinking_delta(event.thinking)
            elif isinstance(event, StreamTextDelta):
                assistant_text += event.text
                renderer.text_delta(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(event)
                renderer.tool_call(event.name, event.arguments)
            elif isinstance(event, StreamDone):
                finish_reason = event.finish_reason
                usage = event.usage

        renderer.done(finish_reason, usage, time.monotonic() - t0)

        # Persist assistant turn (thinking + text + tool calls together).
        assistant_parts: list[Any] = []
        if assistant_thinking:
            assistant_parts.append(AIThinkingPart(thinking=assistant_thinking))
        if assistant_text:
            assistant_parts.append(AITextPart(text=assistant_text))
        for tc in tool_calls:
            assistant_parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
        if assistant_parts:
            history.append(AIMessage(role="assistant", content=assistant_parts))

        if not tool_calls:
            return

        # Execute tools and feed the results back.
        for tc in tool_calls:
            try:
                result = await _exec_tool(tc.name, tc.arguments, mcp)
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            renderer.tool_result(tc.name, result)
            history.append(
                AIMessage(
                    role="tool",
                    content=[AIToolResultPart(tool_call_id=tc.id, name=tc.name, result=result)],
                )
            )

    renderer.warn(f"Stopped after {max_tool_rounds} tool rounds (cap reached).")


async def _wrap_generate_as_stream(
    provider: OllamaAIProvider, context: AIContext
) -> AsyncIterator[StreamEvent]:
    """Adapter for --no-stream that still yields the same event types."""
    response = await provider.generate(context)
    if response.thinking:
        yield StreamThinkingDelta(thinking=response.thinking)
    if response.content:
        yield StreamTextDelta(text=response.content)
    for tc in response.tool_calls:
        yield StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
    yield StreamDone(
        finish_reason=response.finish_reason,
        usage=response.usage,
    )


# ---------------------------------------------------------------------------
# MCP setup
# ---------------------------------------------------------------------------


async def _setup_mcp(url: str, stack: AsyncExitStack) -> Any:
    """Connect to an MCP server and return the entered provider."""
    from roomkit.tools.mcp import MCPToolProvider

    mcp = MCPToolProvider.from_url(url)
    await stack.enter_async_context(mcp)
    return mcp


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


_HELP = """
Commands:
  /help            Show this help.
  /think on|off    Enable/disable thinking for upcoming turns.
  /stream on|off   Enable/disable streaming for upcoming turns.
  /reset           Clear the conversation history.
  /system <text>   Replace the system prompt for upcoming turns.
  /tools           List currently available tools.
  /quit            Exit.

Anything else is sent to the model as a user message.
""".strip()


async def _repl(args: argparse.Namespace) -> None:
    renderer = _Renderer()

    provider = OllamaAIProvider(
        OllamaConfig(
            host=args.host,
            model=args.model,
            timeout=args.timeout,
            think=_initial_think(args),
            num_ctx=args.num_ctx,
            keep_alive=args.keep_alive,
        )
    )

    system_prompt = args.system
    use_stream = args.stream
    thinking_budget: int | None = args.thinking_budget

    history: list[AIMessage] = []

    async with AsyncExitStack() as stack:
        mcp = None
        tools: list[AITool] = []
        if args.mcp:
            renderer.info(f"Connecting MCP at {args.mcp}…")
            mcp = await _setup_mcp(args.mcp, stack)
            tools = mcp.get_tools()
            renderer.info(f"MCP connected — {len(tools)} tools: {[t.name for t in tools]}")
        else:
            # Built-in get_time so --no-mcp runs still have something to call
            # when verifying tool plumbing.
            tools = [
                AITool(
                    name="get_time",
                    description="Return the current local time.",
                    parameters={"type": "object", "properties": {}},
                )
            ]

        renderer.header(
            f"Ollama CLI · {args.model} @ {args.host}\n"
            f"think={_initial_think(args)} stream={use_stream} "
            f"tools={len(tools)} thinking_budget={thinking_budget}"
        )
        renderer.info("Type /help for commands.")

        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, lambda: input("\nyou › "))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue

            if line == "/quit":
                break
            if line == "/help":
                print(_HELP)
                continue
            if line == "/reset":
                history.clear()
                renderer.info("History cleared.")
                continue
            if line == "/tools":
                if not tools:
                    renderer.info("No tools registered.")
                else:
                    for t in tools:
                        renderer.info(f"  • {t.name} — {t.description}")
                continue
            if line.startswith("/system"):
                _, _, rest = line.partition(" ")
                system_prompt = rest.strip() or system_prompt
                renderer.info(f"System prompt set to: {system_prompt!r}")
                continue
            if line.startswith("/think"):
                _, _, rest = line.partition(" ")
                rest = rest.strip().lower()
                if rest == "on":
                    thinking_budget = 4096
                    renderer.info("Thinking ENABLED for upcoming turns.")
                elif rest == "off":
                    thinking_budget = 0
                    renderer.info("Thinking DISABLED for upcoming turns.")
                else:
                    renderer.warn("Usage: /think on|off")
                continue
            if line.startswith("/stream"):
                _, _, rest = line.partition(" ")
                rest = rest.strip().lower()
                if rest == "on":
                    use_stream = True
                    renderer.info("Streaming ENABLED.")
                elif rest == "off":
                    use_stream = False
                    renderer.info("Streaming DISABLED.")
                else:
                    renderer.warn("Usage: /stream on|off")
                continue

            history.append(AIMessage(role="user", content=line))
            try:
                await _run_turn(
                    provider=provider,
                    history=history,
                    system_prompt=system_prompt,
                    tools=tools,
                    thinking_budget=thinking_budget,
                    use_stream=use_stream,
                    mcp=mcp,
                    renderer=renderer,
                )
            except Exception as exc:
                renderer.error(f"Turn failed: {exc}")

        await provider.close()


def _initial_think(args: argparse.Namespace) -> bool | None:
    if args.think is True:
        return True
    if args.no_think:
        return False
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Ollama CLI test bed.")
    p.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL. Env: OLLAMA_HOST.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
        help="Model identifier. Env: OLLAMA_MODEL.",
    )
    p.add_argument(
        "--system",
        default="You are a helpful assistant. Be concise.",
        help="System prompt.",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    p.add_argument("--num-ctx", type=int, default=None, help="Override context window.")
    p.add_argument("--keep-alive", default=None, help="Ollama keep_alive (e.g. '5m').")
    p.add_argument("--mcp", default=None, help="MCP server URL to import tools from.")
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Initial thinking budget (>0 enables thinking, 0 disables).",
    )

    think_group = p.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Force the model to think (overrides its default).",
    )
    think_group.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="Force the model NOT to think (overrides its default).",
    )

    stream_group = p.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream tokens as they arrive (default).",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Wait for the full response before printing.",
    )

    return p.parse_args()


def main() -> None:
    setup_logging("ollama_cli")
    args = _parse_args()
    try:
        asyncio.run(_repl(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
