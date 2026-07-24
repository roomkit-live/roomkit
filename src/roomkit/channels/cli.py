"""CLI channel — interactive terminal transport for RoomKit.

Provides a text-based human-in-the-loop channel that reads from stdin
and prints agent responses to stdout.  Designed for quick prototyping,
examples, and testing multi-agent workflows without a web frontend.

Usage::

    from roomkit import CLIChannel, RoomKit

    kit = RoomKit(...)
    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="my-room")
    await kit.attach_channel("my-room", "cli")

    await cli.run(kit, room_id="my-room")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType, EventType
from roomkit.models.event import EventContent, EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import ThinkingDeltaMarker

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit


class CLIChannel(Channel):
    """Interactive terminal channel.

    Reads user input from stdin and prints agent responses to stdout
    with optional ANSI color formatting.

    Args:
        channel_id: Unique channel identifier.
        prompt: Input prompt shown to the user.
        user_color: ANSI code for user prompt (default: yellow).
        agent_color: ANSI code for agent output (default: cyan).
        use_color: Enable ANSI colors. Auto-detected from terminal.
        agent_label: Callable that maps ``channel_id`` to a display name
            for agent responses.  Defaults to the raw channel ID.
        markdown: Render agent output as progressively updated Markdown.
            Requires the ``console`` extra.
    """

    channel_type = ChannelType.CLI

    def __init__(
        self,
        channel_id: str = "cli",
        *,
        prompt: str = "You: ",
        user_color: str = "\033[33m",
        agent_color: str = "\033[36m",
        thinking_color: str = "\033[2;3m",
        use_color: bool | None = None,
        agent_label: Callable[[str], str] | None = None,
        show_thinking: bool = False,
        markdown: bool = False,
    ) -> None:
        super().__init__(channel_id)
        if markdown:
            from roomkit.channels._cli_markdown import require_markdown_support

            require_markdown_support()
        self._prompt = prompt
        self._user_color = user_color
        self._agent_color = agent_color
        self._thinking_color = thinking_color
        self._use_color = use_color if use_color is not None else _is_tty()
        self._agent_label = agent_label or _default_agent_label
        self._show_thinking = show_thinking
        self._markdown = markdown
        self._reset = "\033[0m"

    # -- Channel interface ----------------------------------------------------

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                provider=self.provider_name,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        # Skip echoing back the user's own messages
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        text = self.extract_text(event)
        if not text:
            return ChannelOutput.empty()

        label = self._agent_label(event.source.channel_id)
        if self._markdown:
            from roomkit.channels._cli_markdown import print_markdown

            print_markdown(label, text, file=sys.stdout, use_color=self._use_color)
        else:
            self._print_agent(label, text)
        return ChannelOutput.empty()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Stream tokens to stdout as they arrive.

        Renders text deltas as they arrive. When ``show_thinking`` is
        enabled, :class:`ThinkingDeltaMarker` chunks are rendered in
        ``thinking_color`` with a leading ``💭`` and a trailing newline
        before the first text delta, so the reasoning appears coherently
        above the answer. Persisted tool-call events are rendered inline so
        long-running agent work remains visible before the final answer.
        """
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        label = self._agent_label(event.source.channel_id)
        if self._markdown:
            return await self._deliver_markdown_stream(text_stream, label)

        agent_prefix = self._colorize(self._agent_color, f"{label}: ")

        thinking_open = False
        thinking_has_text = False
        answer_started = False
        tool_activity_rendered = False

        async for chunk in text_stream:
            if self._show_thinking and isinstance(chunk, ThinkingDeltaMarker):
                # Trim whitespace before the first reasoning character so the
                # 💭 sits on the same line as the text — reasoning models
                # (qwen, etc.) open their <think> block with a newline.
                text = chunk.thinking if thinking_has_text else chunk.thinking.lstrip()
                if not text:
                    continue
                if not thinking_open:
                    sys.stdout.write(f"\n{self._colorize(self._thinking_color, '💭 ')}")
                    thinking_open = True
                thinking_has_text = True
                sys.stdout.write(text)
                sys.stdout.flush()
            elif isinstance(chunk, str):
                if thinking_open:
                    sys.stdout.write(f"{self._reset}\n")
                    thinking_open = False
                    # Next thinking block (e.g. after a tool round) trims its
                    # own leading whitespace, so an empty one shows no icon.
                    thinking_has_text = False
                # Defer the agent prefix until there's real answer text. A
                # tool-call round emits a whitespace-only delta before the
                # final answer; printing "Assistant:" on it would dangle the
                # prefix above the next thinking block.
                text = chunk if answer_started else chunk.lstrip()
                if not text:
                    continue
                if not answer_started:
                    sys.stdout.write(f"\n{agent_prefix}")
                    answer_started = True
                sys.stdout.write(text)
                sys.stdout.flush()
            elif isinstance(chunk, RoomEvent) and isinstance(chunk.content, ToolCallContent):
                if thinking_open:
                    sys.stdout.write(f"{self._reset}\n")
                    thinking_open = False
                    thinking_has_text = False
                self._print_tool_event(chunk)
                tool_activity_rendered = True

        if thinking_open:
            sys.stdout.write(f"{self._reset}\n")
        if not answer_started and not tool_activity_rendered:
            # No text — at least put the prefix so the user sees something.
            sys.stdout.write(f"\n{agent_prefix}")
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        return ChannelOutput.empty()

    async def _deliver_markdown_stream(
        self,
        stream: AsyncIterator[Any],
        label: str,
    ) -> ChannelOutput:
        from roomkit.channels._cli_markdown import MarkdownStreamRenderer

        renderer = MarkdownStreamRenderer(
            label,
            file=sys.stdout,
            use_color=self._use_color,
        )
        try:
            async for chunk in stream:
                if isinstance(chunk, str):
                    renderer.add_text(chunk)
                elif self._show_thinking and isinstance(chunk, ThinkingDeltaMarker):
                    renderer.add_thinking(chunk.thinking)
                elif isinstance(chunk, RoomEvent) and isinstance(chunk.content, ToolCallContent):
                    renderer.add_tool_event(chunk)
        finally:
            renderer.close()
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])

    # -- Interactive loop -----------------------------------------------------

    async def run(
        self,
        kit: RoomKit,
        room_id: str,
        *,
        sender_id: str = "user",
        welcome: str | None = None,
        content_factory: Callable[[str], EventContent | None] | None = None,
    ) -> None:
        """Run an interactive input loop.

        Reads lines from stdin and feeds them into the room as inbound
        messages.  Agent responses are printed by :meth:`deliver`.

        Args:
            kit: The RoomKit instance (channel must already be registered
                and attached to the room).
            room_id: Target room ID.
            sender_id: Participant ID for the human user.
            welcome: Optional welcome message printed before the loop.
            content_factory: Optional hook mapping a raw input line to the
                inbound content. Defaults to ``TextContent(body=line)``; an
                example can return richer content (e.g. an image attachment)
                without reimplementing this loop. Returning ``None`` skips the
                line (e.g. a local slash-command already handled by the hook).
        """
        if welcome:
            print(welcome)

        loop = asyncio.get_running_loop()
        prompt = self._colorize(self._user_color, self._prompt)

        # Use a daemon thread so Ctrl+C doesn't hang waiting for
        # the blocked input() call during asyncio shutdown.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cli-input",
        )
        # Mark the thread as daemon so it dies with the process
        executor._thread_name_prefix = "cli-input"

        while True:
            try:
                line = await loop.run_in_executor(
                    executor,
                    lambda p=prompt: input(p),
                )
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except asyncio.CancelledError:
                break

            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower() in ("quit", "exit", "q"):
                break

            if content_factory:
                content = content_factory(stripped)
                if content is None:
                    continue
            else:
                content = TextContent(body=stripped)
            try:
                await kit.process_inbound(
                    InboundMessage(
                        channel_id=self.channel_id,
                        sender_id=sender_id,
                        content=content,
                    )
                )
            except asyncio.CancelledError:
                break

        executor.shutdown(wait=False, cancel_futures=True)

    # -- Internal helpers -----------------------------------------------------

    def _colorize(self, color: str, text: str) -> str:
        if self._use_color:
            return f"{color}{text}{self._reset}"
        return text

    def _print_agent(self, label: str, text: str) -> None:
        prefix = self._colorize(self._agent_color, f"{label}:")
        print(f"\n{prefix} {text}\n")

    def _print_tool_event(self, event: RoomEvent) -> None:
        content = event.content
        if not isinstance(content, ToolCallContent):
            return

        if event.type == EventType.TOOL_CALL_START:
            arguments = _format_tool_arguments(content.arguments)
            sys.stdout.write(f"\n🔧 {content.tool_name}{arguments}\n")
        elif event.type == EventType.TOOL_CALL_END:
            symbol = "✗" if content.status == "failed" else "✓"
            duration = (
                f" ({content.duration_ms} ms)"
                if content.duration_ms is not None and content.duration_ms > 0
                else ""
            )
            sys.stdout.write(f"\n{symbol} {content.tool_name}{duration}\n")
        sys.stdout.flush()


def _default_agent_label(channel_id: str) -> str:
    """Convert ``agent-researcher`` to ``Researcher``."""
    name = channel_id.removeprefix("agent-")
    return name.replace("-", " ").replace("_", " ").title()


def _is_tty() -> bool:
    """Check if stdout is connected to a terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _format_tool_arguments(arguments: dict[str, Any], *, max_length: int = 240) -> str:
    if not arguments:
        return ""
    rendered = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
    if len(rendered) > max_length:
        rendered = f"{rendered[: max_length - 1]}…"
    return f" {rendered}"
