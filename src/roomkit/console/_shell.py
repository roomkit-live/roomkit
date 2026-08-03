"""Pinned-bar interactive shell for ``CLIChannel(console=True)``.

The Claude Code layout: an input bar pinned to the bottom of the terminal,
a status toolbar under it, and the conversation scrolling above in normal
scrollback. The user can type while the agent streams; submitted messages
queue and process strictly one at a time.

prompt_toolkit is imported at module top — this module itself is imported
lazily, behind ``require_console_support()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from roomkit.console._chat import print_user_line
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent

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


@dataclass
class _ShellState:
    room_id: str
    model_label: str | None
    working: bool = False
    in_flight: asyncio.Task[Any] | None = None
    """The current ``process_inbound`` task — the phase-2b interrupt hook."""


async def run_console_shell(
    channel: CLIChannel,
    kit: RoomKit,
    room_id: str,
    *,
    sender_id: str,
    banner: ConsoleBannerData,
    content_factory: Callable[[str], EventContent | None] | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> None:
    """Run the pinned-bar input loop until quit/EOF/interrupt.

    Submissions land in a queue consumed strictly one at a time — the
    framework does not serialize concurrent ``deliver_stream`` calls to one
    channel, so the shell must. On exit, the in-flight turn is cancelled and
    queued submissions are dropped; a turn cancelled mid-stream does not
    persist its partial text (graceful interruption is phase 2b).

    ``input``/``output`` inject prompt_toolkit pipe/dummy IO for tests.
    """
    global _active_app

    model_label: str | None = None
    if banner.models:
        entry = banner.models[0]
        model_label = entry.model or entry.channel_id
        if entry.provider:
            model_label = f"{model_label} ({entry.provider})"
    state = _ShellState(room_id=room_id, model_label=model_label)

    queue: asyncio.Queue[InboundMessage] = asyncio.Queue()

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
        erase_when_done=True,
        input=input,
        output=output,
    )

    channel._shell_width = _terminal_width(session)
    if _active_app is not None:
        logger.warning("A console shell is already active; replacing the registration")
    _active_app = session.app
    channel._pinned_shell_active = True

    if output is None:
        # Real terminal: start the bar at the BOTTOM of the screen, Claude
        # Code-style. prompt_toolkit renders the input line wherever the
        # cursor is (only the toolbar is bottom-anchored), so without this
        # the bar floats mid-screen under the banner, detached from the
        # toolbar. Moving the cursor to the last row pins input + toolbar
        # together; every subsequent write scrolls the transcript above.
        _pin_to_bottom(session)

    consumer: asyncio.Task[None] | None = None
    try:
        with patch_stdout(raw=True):
            consumer = asyncio.create_task(
                _consume(queue, kit, state, session.app.invalidate),
                name="roomkit-cli-shell-consumer",
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
        _active_app = None
        channel._pinned_shell_active = False


async def _consume(
    queue: asyncio.Queue[InboundMessage],
    kit: RoomKit,
    state: _ShellState,
    invalidate: Callable[[], None],
) -> None:
    """Process queued submissions strictly sequentially."""
    while True:
        message = await queue.get()
        state.working = True
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
            # Only the in-flight turn was cancelled (phase-2b interrupt):
            # keep serving the queue.
        except Exception:
            # ON_ERROR hooks already fired inside the pipeline.
            logger.debug("process_inbound failed", exc_info=True)
        finally:
            state.in_flight = None
            state.working = False
            invalidate()


def _toolbar_text(state: _ShellState, queued: int) -> str:
    parts = [state.room_id]
    if state.model_label:
        parts.append(state.model_label)
    if state.working:
        status = "working" if queued == 0 else f"working ({queued} queued)"
    else:
        status = "idle"
    parts.append(status)
    return " " + " · ".join(parts)


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


def _pin_to_bottom(session: PromptSession[str]) -> None:
    """Move the cursor to the last screen row so the bar renders there.

    Plain cursor positioning — no scroll, existing content stays put. If the
    prompt + toolbar need more rows than remain, prompt_toolkit scrolls the
    difference itself.
    """
    rows: int | None = None
    with contextlib.suppress(NotImplementedError, OSError, ValueError):
        rows = int(session.output.get_size().rows)
    if rows is None:
        try:
            rows = os.get_terminal_size().lines
        except OSError:
            return
    sys.stdout.write(f"\x1b[{rows};1H")
    sys.stdout.flush()


__all__ = ["active_shell_app", "run_console_shell"]
