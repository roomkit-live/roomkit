"""Pinned-bar interactive shell for ``CLIChannel(console=True)``.

The Claude Code layout: an input bar with a status toolbar under it, and the
conversation scrolling above in normal scrollback. prompt_toolkit lays the
two out as one container and draws it where the cursor is, so the zone opens
under the banner and settles at the bottom of the screen as the transcript
grows into it. The user can type while the agent streams; submitted messages
queue and process strictly one at a time.

prompt_toolkit is imported at module top — this module itself is imported
lazily, behind ``require_console_support()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from roomkit.channels.cli import (
    AddressFactory,
    CommandHandler,
    StatusFactory,
    VisibilityFactory,
    match_command,
    resolve_address,
    resolve_visibility,
)
from roomkit.console._activity import (
    FRAME_SECONDS,
    ActivityTracker,
    format_activity,
    format_elapsed,
    spinner_frame,
)
from roomkit.console._chat import print_user_line
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import Visibility
from roomkit.models.event import TextContent
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.input import Input
    from prompt_toolkit.output import Output

    from roomkit.channels.cli import CLIChannel
    from roomkit.console._chat import ConsoleBannerData
    from roomkit.core.framework import RoomKit
    from roomkit.models.event import EventContent

logger = logging.getLogger("roomkit.console.shell")

# Brand colors as prompt_toolkit style strings (PT syntax ≠ rich syntax;
# hex values mirror roomkit.console._brand).
_PT_STYLE = Style.from_dict(
    {
        "prompt": "#6366f1 bold",
        "rule": "#373a82",  # PRIMARY_DIM
        # The input zone is framed by rules instead of reverse-video.
        "bottom-toolbar": "noreverse",
        "toolbar-text": "#64748b",
    }
)

# Colorless variant: still kill the toolbar's default reverse-video so the
# framed layout reads the same.
_PT_STYLE_PLAIN = Style.from_dict({"bottom-toolbar": "noreverse"})

_QUIT_COMMANDS = ("quit", "exit", "q")

_active_app: Application[Any] | None = None


def active_shell_app() -> Application[Any] | None:
    """The running shell's prompt_toolkit Application, if any.

    Used by :func:`roomkit.console.terminal_input` to suspend the bar for
    plain terminal reads (e.g. tool-permission prompts fired mid-stream).
    A module-level accessor rather than prompt_toolkit's ``get_app()``
    because those callers run in tasks created before the shell existed.
    """
    return _active_app


@dataclass(slots=True)
class _QueuedCommand:
    """A local command waiting its turn in the submission queue."""

    handler: CommandHandler
    argument: str


@dataclass
class _ShellState:
    room_id: str
    model_label: str | None
    activity: ActivityTracker
    working: bool = False
    working_since: float | None = None
    """When the current turn was submitted — the wait the user is living."""

    in_flight: asyncio.Task[Any] | None = None
    """The current ``process_inbound`` task — what an Esc interrupt cancels."""

    status_extra: StatusFactory | None = None
    """The application's own segment — who the next message addresses, a mode,
    anything the shell cannot know."""

    @property
    def busy(self) -> bool:
        """A turn is in flight — routing, or an agent actually streaming."""
        return self.working or bool(self.activity)


async def run_console_shell(
    channel: CLIChannel,
    kit: RoomKit,
    room_id: str,
    *,
    sender_id: str,
    banner: ConsoleBannerData,
    content_factory: Callable[[str], EventContent | None] | None = None,
    commands: Mapping[str, CommandHandler] | None = None,
    addressed_to: AddressFactory | None = None,
    visibility: VisibilityFactory | None = None,
    status_extra: StatusFactory | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> None:
    """Run the pinned-bar input loop until quit/EOF/interrupt.

    Submissions land in a queue consumed strictly one at a time — the
    framework does not serialize concurrent ``deliver_stream`` calls to one
    channel, so the shell must. Local commands ride the same queue, so a
    command typed behind a message runs after that message's turn, and runs
    while the bar is up — which is what lets a handler prompt through
    ``terminal_input``/``terminal_select``. On exit, the in-flight turn is
    cancelled and queued submissions are dropped; a turn cancelled mid-stream
    keeps the partial segment the pipeline already persisted, so what the
    reader saw stays in the room's record.

    ``input``/``output`` inject prompt_toolkit pipe/dummy IO for tests.
    """
    global _active_app

    model_label: str | None = None
    if banner.models:
        entry = banner.models[0]
        model_label = entry.model or entry.channel_id
        if entry.provider:
            model_label = f"{model_label} ({entry.provider})"
    tracker = ActivityTracker()
    state = _ShellState(
        room_id=room_id,
        model_label=model_label,
        activity=tracker,
        status_extra=status_extra,
    )

    queue: asyncio.Queue[InboundMessage | _QueuedCommand] = asyncio.Queue()

    # The input zone is framed Claude Code-style: a rule above the input
    # line (part of the prompt message) and one below it (first line of the
    # toolbar). Both are callables so the rules follow terminal resizes.
    # ``erase_when_done`` removes the accepted line so the loop can re-echo
    # it into the transcript with a distinct background (print_user_line).
    session: PromptSession[str] = PromptSession(
        message=lambda: [
            ("class:rule", f"{_rule(session)}\n"),
            ("class:prompt", channel._prompt),
        ],
        bottom_toolbar=lambda: [
            ("class:rule", f"{_rule(session)}\n"),
            ("class:toolbar-text", _toolbar_text(state, queue.qsize())),
        ],
        style=_PT_STYLE if channel._use_color else _PT_STYLE_PLAIN,
        key_bindings=_interrupt_binding(state),
        erase_when_done=True,
        input=input,
        output=output,
    )

    channel._shell_width = _terminal_width(session)
    if _active_app is not None:
        logger.warning("A console shell is already active; replacing the registration")
    _active_app = session.app
    channel._pinned_shell_active = True
    # The channel reports who is streaming; the tracker times it and the
    # toolbar spins on it.
    channel._activity = tracker
    spin_wake = asyncio.Event()

    def _activity_changed() -> None:
        session.app.invalidate()
        spin_wake.set()

    tracker.on_change = _activity_changed

    consumer: asyncio.Task[None] | None = None
    spinner: asyncio.Task[None] | None = None
    subscription = await _subscribe_agent_telemetry(kit, room_id, state, _activity_changed)
    try:
        with patch_stdout(raw=True):
            consumer = asyncio.create_task(
                _consume(queue, kit, state, _activity_changed),
                name="roomkit-cli-shell-consumer",
            )
            spinner = asyncio.create_task(
                _spin(state, session.app.invalidate, spin_wake),
                name="roomkit-cli-shell-spinner",
            )
            while True:
                try:
                    line = await session.prompt_async()
                except (EOFError, KeyboardInterrupt):
                    break

                stripped = line.strip()
                if not stripped:
                    continue
                print_user_line(
                    stripped,
                    prompt=channel._prompt,
                    use_color=channel._use_color,
                    width=channel._shell_width,
                )
                if stripped.lower() in _QUIT_COMMANDS:
                    break

                scope = resolve_visibility(visibility, stripped)
                match = match_command(commands, stripped)
                if match is not None:
                    handler, argument = match
                    queue.put_nowait(_QueuedCommand(handler=handler, argument=argument))
                    session.app.invalidate()
                    continue

                if content_factory:
                    content = content_factory(stripped)
                    if content is None:
                        continue
                else:
                    content = TextContent(body=stripped)

                queue.put_nowait(
                    InboundMessage(
                        channel_id=channel.channel_id,
                        sender_id=sender_id,
                        content=content,
                        addressed_to=resolve_address(addressed_to, stripped),
                        # Both together: a scope that hid the question and
                        # published the answer would be worse than none.
                        visibility=scope or Visibility.ALL,
                        response_visibility=scope,
                    )
                )
                session.app.invalidate()
    finally:
        # Drop queued submissions, then stop the consumer (its cancellation
        # handler cancels and awaits the in-flight turn first).
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if consumer is not None:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
        if spinner is not None:
            spinner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await spinner
        if subscription is not None:
            with contextlib.suppress(Exception):
                await kit.realtime.unsubscribe(subscription)
        _active_app = None
        channel._pinned_shell_active = False
        channel._activity = None
        tracker.on_change = None
        tracker.clear()


async def _spin(
    state: _ShellState,
    invalidate: Callable[[], None],
    wake: asyncio.Event,
) -> None:
    """Repaint the toolbar while work is in flight, and only then.

    An idle console must not redraw: the animation exists to show that the
    wait is alive, and a spinner turning over an idle prompt is noise (and
    wasted wakeups on a laptop).
    """
    while True:
        if not state.busy:
            await wake.wait()
            wake.clear()
            continue
        invalidate()
        await asyncio.sleep(FRAME_SECONDS)


async def _subscribe_agent_telemetry(
    kit: RoomKit,
    room_id: str,
    state: _ShellState,
    changed: Callable[[], None],
) -> str | None:
    """Follow what agents report about themselves — model, context usage.

    Ephemeral events, not the delivery stream: an agent's model can change
    without producing a single token (the user running ``/model`` inside a
    coding agent), and context usage is reported between turns too.
    """

    async def _on_event(event: EphemeralEvent) -> None:
        if event.type is not EphemeralEventType.CUSTOM:
            return
        data = event.data
        channel_id = event.channel_id or str(data.get("channel_id") or "")
        if not channel_id:
            return
        kind = data.get("type")
        if kind == "acp_config_options":
            # The label reads for humans ("Opus"); the raw value can be as
            # opaque as "default". Fall back to it when unlabelled.
            for key in ("labels", "values"):
                mapping = data.get(key)
                model = mapping.get("model") if isinstance(mapping, Mapping) else None
                if isinstance(model, str) and model:
                    state.activity.set_model(channel_id, model)
                    break
        elif kind == "acp_usage":
            usage = data.get("usage")
            if isinstance(usage, Mapping):
                used = usage.get("used")
                size = usage.get("size")
                state.activity.observe_usage(
                    channel_id,
                    used=used if isinstance(used, int) else None,
                    size=size if isinstance(size, int) else None,
                )

    try:
        return await kit.realtime.subscribe_to_room(room_id, _on_event)
    except Exception:  # telemetry is a nicety — never fail the shell over it
        logger.debug("Console telemetry subscription failed", exc_info=True)
        return None


async def _consume(
    queue: asyncio.Queue[InboundMessage | _QueuedCommand],
    kit: RoomKit,
    state: _ShellState,
    invalidate: Callable[[], None],
) -> None:
    """Process queued submissions strictly sequentially."""
    while True:
        message = await queue.get()
        if isinstance(message, _QueuedCommand):
            # Commands run here, not in the prompt loop: the bar is up, so a
            # handler that prompts can suspend it, and the command lands
            # between turns instead of inside one.
            try:
                await message.handler(message.argument)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Console command failed")
            continue
        state.working = True
        state.working_since = state.activity.clock()
        invalidate()
        task = asyncio.create_task(kit.process_inbound(message))
        state.in_flight = task
        try:
            await task
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # shell shutdown cancelled the consumer itself
            # Only the in-flight turn was cancelled (an Esc interrupt):
            # keep serving the queue.
        except Exception:
            # ON_ERROR hooks already fired inside the pipeline.
            logger.debug("process_inbound failed", exc_info=True)
        finally:
            state.in_flight = None
            state.working = False
            state.working_since = None
            invalidate()


def _interrupt_binding(state: _ShellState) -> KeyBindings:
    """Esc stops the turn in flight, and only that.

    Not Ctrl-C, which already ends the session: interrupting a long answer
    and leaving are different intentions and deserve different keys. What the
    agent had already said stays — it is on screen, and the pipeline persists
    the partial segment on cancellation. The queue keeps draining, so a
    message typed behind the interrupted turn still runs.

    ``eager`` because a bare Escape is otherwise held as the prefix of a meta
    sequence: without it the interrupt waits for a key that never comes.
    """
    keys = KeyBindings()

    @keys.add("escape", eager=True)
    def _interrupt(event: Any) -> None:
        task = state.in_flight
        if task is None or task.done():
            return
        logger.debug("Console interrupt: cancelling the in-flight turn")
        task.cancel()

    return keys


def _toolbar_text(state: _ShellState, queued: int) -> str:
    parts = [state.room_id]
    model = _model_text(state)
    if model:
        parts.append(model)
    extra = _extra_text(state)
    if extra:
        parts.append(extra)
    parts.append(_status_text(state, queued))
    return " " + " · ".join(parts)


def _extra_text(state: _ShellState) -> str | None:
    """The application's segment, asked fresh on every render.

    Rendering must not be a place an application can crash: a bad segment
    costs its own line, never the bar.
    """
    if state.status_extra is None:
        return None
    try:
        extra = state.status_extra()
    except Exception:
        logger.debug("Console status segment failed", exc_info=True)
        return None
    return extra.strip() if extra else None


def _model_text(state: _ShellState) -> str | None:
    """The model to show: what the agents report, else the banner's guess.

    Agents report their model only once a session exists, so the banner
    label (a provider's configured model, or nothing for an ACP agent)
    carries the bar until the first turn — then the live value takes over,
    including a model the user switched from inside the agent.
    """
    reported = {model for model in state.activity.models.values() if model}
    if len(reported) == 1:
        return next(iter(reported))
    if len(reported) > 1:
        return f"{len(reported)} models"
    return state.model_label


def _status_text(state: _ShellState, queued: int) -> str:
    suffix = f" ({queued} queued)" if queued else ""
    activity = format_activity(state.activity)
    if activity is not None:
        return f"{activity}{suffix}"
    if state.working:
        # Submitted, but no agent is streaming yet — routing, hooks, or an
        # agent still thinking before its first token.
        elapsed = state.activity.clock() - (state.working_since or state.activity.clock())
        return f"{spinner_frame(elapsed)} working {format_elapsed(elapsed)}{suffix}"
    return f"idle{suffix}"


def _terminal_width(session: PromptSession[str]) -> int | None:
    """Real terminal width, captured before patch_stdout hides the TTY."""
    with contextlib.suppress(NotImplementedError, OSError, ValueError):
        return int(session.output.get_size().columns)
    try:
        return os.get_terminal_size().columns
    except OSError:
        return None


def _rule(session: PromptSession[str]) -> str:
    """A full-width horizontal rule, recomputed per render (follows resizes)."""
    columns = 80
    with contextlib.suppress(NotImplementedError, OSError, ValueError):
        columns = int(session.output.get_size().columns)
    return "─" * max(1, columns)


__all__ = ["active_shell_app", "run_console_shell"]
