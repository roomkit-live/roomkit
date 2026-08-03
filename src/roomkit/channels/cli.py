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
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType, EventType, Visibility
from roomkit.models.event import EventContent, EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import ThinkingDeltaMarker

if TYPE_CHECKING:
    from roomkit.channels._cli_markdown import MarkdownStreamRenderer
    from roomkit.console._activity import ActivityTracker
    from roomkit.console._chat import PinnedStreamRenderer
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.channels.cli")

CommandHandler = Callable[[str], Awaitable[None]]
"""A local console command. Receives the rest of the line; awaited by the loop."""

AddressFactory = Callable[[str], Sequence[str] | None]
"""Names the intelligence channels a submitted line asks to act (RFC §19.3)."""

StatusFactory = Callable[[], str | None]
"""The application's own status-bar segment, asked fresh on every render."""

VisibilityFactory = Callable[[str], str | Sequence[str] | None]
"""Scopes a submitted line: a keyword, channel ids, or None for no restriction."""


def resolve_visibility(factory: VisibilityFactory | None, line: str) -> str | None:
    """Turn what *factory* returns into a visibility spec.

    A string is a spec verbatim — a keyword (``"transport"``) or an already
    written id list. A sequence is joined into an id list, which is the form
    worth encouraging: ``"transport,codex"`` looks like "the transports plus
    codex" and means nothing of the sort, because the comma form matches
    channel ids only. Handing ids avoids inventing that sentence at all.
    """
    if factory is None:
        return None
    scope = factory(line)
    if scope is None:
        return None
    if isinstance(scope, str):
        return scope or None
    ids = [str(item).strip() for item in scope if str(item).strip()]
    return ",".join(ids) if ids else None


def resolve_address(factory: AddressFactory | None, line: str) -> list[str] | None:
    """Ask *factory* who this line addresses, tolerating no factory at all."""
    if factory is None:
        return None
    addressed = factory(line)
    return None if addressed is None else list(addressed)


def match_command(
    commands: Mapping[str, CommandHandler] | None,
    line: str,
) -> tuple[CommandHandler, str] | None:
    """Match *line* against *commands* on its first word.

    Returns the handler and the remaining argument, or ``None`` when the line
    is not a command and should travel to the room as a message.
    """
    if not commands:
        return None
    name, _, argument = line.partition(" ")
    handler = commands.get(name)
    if handler is None:
        return None
    return handler, argument.strip()


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
        console: Full branded console mode — startup banner (logo, RoomKit
            version, AI models, room, channels), brand-palette styling and
            progressive Markdown, rendered inline in the normal scrollback.
            Subsumes ``markdown``. Requires the ``console`` extra.
    """

    channel_type = ChannelType.CLI

    sender_is_participant = True
    """The human at the terminal is named by the room, not addressed.

    A transport usually reads its ``sender_id`` off the wire — a number, an
    address, a handle the remote network chose. This one is the channel's own
    default: :meth:`run` names the typist, and calls what it names a Participant
    ID. No resolver can match that, so resolving it returns UNKNOWN for every
    line typed, ``ON_IDENTITY_UNKNOWN`` fires per line, and a hook written to
    refuse unknown senders — the pattern RFC §11.2 provides for — makes
    everything typed at the keyboard vanish without trace.

    ``ensure_participant`` would not settle it either: the record it creates is
    PENDING, which stays deliberately resolvable so a PENDING sender on a text
    channel can still be challenged or refused.
    """

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
        console: bool = False,
    ) -> None:
        super().__init__(channel_id)
        if console:
            from roomkit.channels._cli_markdown import require_console_support

            require_console_support()
            # Branded defaults; explicit values always win.
            if prompt == "You: ":
                prompt = "❯ "
            if user_color == "\033[33m":
                user_color = "\033[38;2;99;102;241m"
        elif markdown:
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
        self._console = console
        self._reset = "\033[0m"
        # Set by the pinned-bar shell while it owns the terminal.
        self._pinned_shell_active = False
        self._shell_width: int | None = None
        self._activity: ActivityTracker | None = None

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

        label = _speaker_label(event, context, self._agent_label)
        if self._console:
            from roomkit.console._chat import print_message

            print_message(
                label,
                text,
                file=sys.stdout,
                use_color=self._use_color,
                # Under the shell, stdout is a proxy: not a TTY, no size.
                force_terminal=self._pinned_shell_active and self._use_color,
                width=self._shell_width if self._pinned_shell_active else None,
            )
        elif self._markdown:
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

        label = _speaker_label(event, context, self._agent_label)
        if self._console or self._markdown:
            return await self._deliver_rendered_stream(text_stream, label, event.source.channel_id)

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

    async def _deliver_rendered_stream(
        self,
        stream: AsyncIterator[Any],
        label: str,
        source_channel_id: str,
    ) -> ChannelOutput:
        renderer: MarkdownStreamRenderer | PinnedStreamRenderer
        if self._console and self._pinned_shell_active:
            from roomkit.console._chat import PinnedStreamRenderer as _Pinned

            renderer = _Pinned(
                label,
                use_color=self._use_color,
                width=self._shell_width,
            )
        elif self._console:
            from roomkit.console._chat import ConsoleStreamRenderer

            renderer = ConsoleStreamRenderer(
                label,
                file=sys.stdout,
                use_color=self._use_color,
            )
        else:
            from roomkit.channels._cli_markdown import MarkdownStreamRenderer as _Renderer

            renderer = _Renderer(
                label,
                file=sys.stdout,
                use_color=self._use_color,
            )
        # The status bar names who is working: the stream tells us what the
        # agent is doing, and the tracker times it. Absent outside the pinned
        # shell, where there is no status bar to feed.
        activity = self._activity
        if activity is not None:
            activity.start(source_channel_id, label)
        try:
            async for chunk in stream:
                if isinstance(chunk, str):
                    renderer.add_text(chunk)
                    if activity is not None and chunk.strip():
                        activity.note(source_channel_id, "responding")
                elif isinstance(chunk, ThinkingDeltaMarker):
                    if activity is not None:
                        activity.note(source_channel_id, "thinking")
                    if self._show_thinking:
                        renderer.add_thinking(chunk.thinking)
                elif isinstance(chunk, RoomEvent) and isinstance(chunk.content, ToolCallContent):
                    renderer.add_tool_event(chunk)
                    if activity is not None:
                        activity.note(source_channel_id, _tool_detail(chunk.content))
        finally:
            renderer.close()
            if activity is not None:
                activity.finish(source_channel_id)
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
        commands: Mapping[str, CommandHandler] | None = None,
        addressed_to: AddressFactory | None = None,
        visibility: VisibilityFactory | None = None,
        status_extra: StatusFactory | None = None,
    ) -> None:
        """Run an interactive input loop.

        Reads lines from stdin and feeds them into the room as inbound
        messages.  Agent responses are printed by :meth:`deliver`.

        Args:
            kit: The RoomKit instance (channel must already be registered
                and attached to the room).
            room_id: Target room ID.
            sender_id: Participant ID for the human user — a room
                ``Participant.id``, not an address. This channel declares it as
                such (``sender_is_participant``), so identity resolution never
                runs on it; passing an address here would leave that address
                unresolved.
            welcome: Optional welcome message printed before the loop. In
                console mode it becomes the notes line under the banner.
            content_factory: Optional hook mapping a raw input line to the
                inbound content. Defaults to ``TextContent(body=line)``; an
                example can return richer content (e.g. an image attachment)
                without reimplementing this loop. Returning ``None`` skips the
                line (e.g. a local slash-command already handled by the hook).
            commands: Local commands keyed by their first word (``"/model"``,
                ``":q"`` — the prefix is yours). A matching line never reaches
                ``content_factory`` or the room; its handler is **awaited by
                the loop**, in submission order, with the rest of the line as
                its argument. That ordering is the point: a handler may prompt
                (:func:`roomkit.console.terminal_input`,
                :func:`~roomkit.console.terminal_select`) without racing the
                loop for stdin, and a command queued behind a message runs
                after that message's turn, never inside it.
            addressed_to: Hook naming the intelligence channels each
                submission asks to act (RFC §19.3). Receives the submitted
                line and returns channel ids, or ``None`` to leave the
                message unaddressed. Called **after** ``content_factory``, so
                a factory that switched which agent you are talking to is
                already reflected. RoomKit wants ids, never a syntax — how a
                user names an agent (``@codex``, ``/agent codex``, a picker)
                is yours to decide.
            visibility: Hook scoping each submission — who may see it, and
                where its answer may go. Returns a visibility keyword, a
                sequence of channel ids, or ``None`` for no restriction (the
                default: everything is visible to the whole room). It sets
                both the message's ``visibility`` and its
                ``response_visibility``, so a private question does not
                publish the answer it gets — the scope covers the whole turn,
                tool activity included. Called after ``content_factory``, like
                ``addressed_to``.
            status_extra: A segment for the pinned status bar, asked fresh on
                every render — who the next message addresses, a mode, a
                counter. Console mode on a real terminal only; the classic
                loop has no bar. Return ``None`` to show nothing.

        In console mode on a real terminal, this runs the pinned-bar shell:
        the input bar stays at the bottom, responses stream above it, and the
        user can keep typing while a turn is in flight (submissions queue and
        process one at a time). Non-TTY sessions (pipes, CI) fall back to
        this plain sequential loop.
        """
        if self._console:
            from roomkit.console._chat import collect_banner_data, print_banner

            banner = await collect_banner_data(kit, room_id)
            print_banner(banner, file=sys.stdout, use_color=self._use_color, notes=welcome)
            if _is_tty() and _stdin_is_tty():
                from roomkit.console._shell import run_console_shell

                await run_console_shell(
                    self,
                    kit,
                    room_id,
                    sender_id=sender_id,
                    banner=banner,
                    content_factory=content_factory,
                    commands=commands,
                    addressed_to=addressed_to,
                    visibility=visibility,
                    status_extra=status_extra,
                )
                return
        elif welcome:
            print(welcome)

        await self._run_classic(
            kit,
            sender_id=sender_id,
            content_factory=content_factory,
            commands=commands,
            addressed_to=addressed_to,
            visibility=visibility,
        )

    async def _run_classic(
        self,
        kit: RoomKit,
        *,
        sender_id: str,
        content_factory: Callable[[str], EventContent | None] | None,
        commands: Mapping[str, CommandHandler] | None = None,
        addressed_to: AddressFactory | None = None,
        visibility: VisibilityFactory | None = None,
    ) -> None:
        """Blocking-input sequential loop (classic mode and non-TTY fallback)."""
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

            scope = resolve_visibility(visibility, stripped)
            match = match_command(commands, stripped)
            if match is not None:
                handler, argument = match
                try:
                    await handler(argument)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("Console command failed: %s", stripped)
                continue

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
                        addressed_to=resolve_address(addressed_to, stripped),
                        # A scope that hid the question and published the
                        # answer would be worse than none, so both move
                        # together.
                        visibility=scope or Visibility.ALL,
                        response_visibility=scope,
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


def _speaker_label(
    event: RoomEvent,
    context: RoomContext,
    agent_label: Callable[[str], str],
) -> str:
    """Who is speaking, as the transcript should name them.

    A person gets their own name and the channel they speak through —
    ``"Marie · sms"`` — because in a room holding several humans, the channel
    id names none of them: two colleagues texting in would otherwise share
    one handle. Anything without a participant (an agent, a system event)
    keeps the channel-derived label, so nothing that works today changes.

    ``source.participant_id`` holds a ``Participant.id`` when the channel
    names its own sender, and an ``Identity.id`` when the identity pipeline
    resolved one (RFC §11) — two namespaces in one field, so both are tried.
    """
    participant_id = event.source.participant_id
    if not participant_id:
        return agent_label(event.source.channel_id)

    person = next(
        (p for p in context.participants if p.id == participant_id),
        None,
    ) or next(
        (p for p in context.participants if p.identity_id == participant_id),
        None,
    )
    if person is None:
        return agent_label(event.source.channel_id)
    return f"{person.display_name or person.id} · {event.source.channel_id}"


def _default_agent_label(channel_id: str) -> str:
    """Convert ``agent-researcher`` to ``Researcher``."""
    name = channel_id.removeprefix("agent-")
    return name.replace("-", " ").replace("_", " ").title()


def _tool_detail(content: ToolCallContent) -> str | None:
    """What the status bar says while a tool runs — its name, then nothing.

    A finished tool is no longer what the agent is doing, and the next chunk
    (thinking, text, another tool) says what took its place.
    """
    if content.status in ("completed", "failed"):
        return None
    return content.tool_name or None


def _is_tty() -> bool:
    """Check if stdout is connected to a terminal."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _stdin_is_tty() -> bool:
    """Check if stdin is connected to a terminal."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _format_tool_arguments(arguments: dict[str, Any], *, max_length: int = 240) -> str:
    if not arguments:
        return ""
    rendered = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
    if len(rendered) > max_length:
        rendered = f"{rendered[: max_length - 1]}…"
    return f" {rendered}"
