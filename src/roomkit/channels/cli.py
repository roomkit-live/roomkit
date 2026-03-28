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
import sys
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent

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
    """

    channel_type = ChannelType.CLI

    def __init__(
        self,
        channel_id: str = "cli",
        *,
        prompt: str = "You: ",
        user_color: str = "\033[33m",
        agent_color: str = "\033[36m",
        use_color: bool | None = None,
        agent_label: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._prompt = prompt
        self._user_color = user_color
        self._agent_color = agent_color
        self._use_color = use_color if use_color is not None else _is_tty()
        self._agent_label = agent_label or _default_agent_label
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
        self._print_agent(label, text)
        return ChannelOutput.empty()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Stream tokens to stdout as they arrive."""
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        label = self._agent_label(event.source.channel_id)
        prefix = self._colorize(self._agent_color, f"{label}: ")
        sys.stdout.write(f"\n{prefix}")

        async for chunk in text_stream:
            sys.stdout.write(chunk)
            sys.stdout.flush()

        sys.stdout.write("\n\n")
        sys.stdout.flush()
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

            try:
                await kit.process_inbound(
                    InboundMessage(
                        channel_id=self.channel_id,
                        sender_id=sender_id,
                        content=TextContent(body=stripped),
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


def _default_agent_label(channel_id: str) -> str:
    """Convert ``agent-researcher`` to ``Researcher``."""
    name = channel_id.removeprefix("agent-")
    return name.replace("-", " ").replace("_", " ").title()


def _is_tty() -> bool:
    """Check if stdout is connected to a terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
