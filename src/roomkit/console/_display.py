"""RoomKitConsole — full-screen terminal dashboard for voice agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.console import Console as RichConsole
from rich.console import ConsoleOptions, RenderableType, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from roomkit.console._hooks import register_console_hooks, unregister_console_hooks
from roomkit.console._state import ConsoleState, LogRingBuffer

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.console")

# ---------------------------------------------------------------------------
# Brand palette (from roomkit.live website)
# ---------------------------------------------------------------------------

_PRIMARY = "rgb(99,102,241)"
_PRIMARY_LIGHT = "rgb(129,140,248)"
_PRIMARY_DIM = "rgb(55,58,130)"
_ACCENT = "rgb(6,182,212)"
_MUTED = "rgb(100,116,139)"

# Block characters for audio meter (index 0 = silence, 8 = max).
_BLOCKS = " ▁▂▃▄▅▆▇█"

# Level styles for log records.
_LOG_LEVEL_STYLES: dict[str, str] = {
    "DEBUG": _MUTED,
    "INFO": _ACCENT,
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}

# Colors for voice state indicator.
_STATE_STYLES: dict[str, tuple[str, str]] = {
    "idle": (_MUTED, "IDLE"),
    "listening": ("bold green", "LISTENING"),
    "processing": ("bold yellow", "PROCESSING"),
    "speaking": (f"bold {_ACCENT}", "SPEAKING"),
}


# ---------------------------------------------------------------------------
# Bottom-aligned renderable (crops from TOP, keeps newest at bottom)
# ---------------------------------------------------------------------------


class _BottomAligned:
    """Wraps a renderable and keeps only the last *height* lines.

    Rich always clips overflow from the bottom. This renderable
    pre-renders the inner content using Rich's own engine, splits
    into lines, and yields only the tail — so the newest content
    is always visible.
    """

    def __init__(self, inner: RenderableType, height: int) -> None:
        self.inner = inner
        self.height = height

    def __rich_console__(self, console: RichConsole, options: ConsoleOptions) -> RenderResult:
        # Render the inner content with Rich's engine (preserves styling).
        rendered = list(console.render(self.inner, options))

        # Split segments into lines at newline boundaries.
        lines: list[list[Segment]] = [[]]
        for segment in rendered:
            text = segment.text
            if "\n" not in text:
                lines[-1].append(segment)
                continue
            parts = text.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append(Segment(part, segment.style, segment.control))

        # Keep only the last N lines.
        visible = lines[-self.height :] if len(lines) > self.height else lines

        # Yield the visible lines with newlines between them.
        for i, line_segs in enumerate(visible):
            yield from line_segs
            if i < len(visible) - 1:
                yield Segment("\n")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _db_to_ratio(level_db: float) -> float:
    """Convert dB level (-60..0) to 0.0..1.0 ratio."""
    clamped = max(-60.0, min(0.0, level_db))
    return (clamped + 60.0) / 60.0


def _render_meter(history: deque[float] | list[float], width: int = 20) -> Text:
    """Render an audio level waveform from a history of dB samples."""
    text = Text()
    samples = list(history)[-width:]
    for level_db in samples:
        ratio = _db_to_ratio(level_db)
        idx = min(len(_BLOCKS) - 1, int(ratio * (len(_BLOCKS) - 1)))
        char = _BLOCKS[idx]
        if ratio < 0.5:
            style = _PRIMARY_LIGHT
        elif ratio < 0.8:
            style = _ACCENT
        else:
            style = "bold red"
        text.append(char, style=style)
    remaining = width - len(samples)
    if remaining > 0:
        text.append(_BLOCKS[0] * remaining, style="dim")
    return text


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_uptime(started_at: datetime) -> str:
    """Format elapsed time since start as H:MM:SS."""
    delta = datetime.now(UTC) - started_at
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _build_header(state: ConsoleState) -> Table:
    """Build the header with logo and info."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    uptime = _format_uptime(state.started_at)

    table = Table.grid(padding=(0, 1))
    table.add_column(width=6)
    table.add_column()

    logo_top = Text()
    logo_top.append("██", style=_PRIMARY)
    logo_top.append(" ", style="")
    logo_top.append("██", style=_PRIMARY_LIGHT)

    logo_bot = Text()
    logo_bot.append("██", style=_PRIMARY_LIGHT)
    logo_bot.append(" ", style="")
    logo_bot.append("██", style=_PRIMARY_DIM)

    info_top = Text()
    info_top.append("RoomKit", style=f"bold {_PRIMARY}")
    info_top.append(" Console", style=f"bold {_PRIMARY_LIGHT}")
    info_top.append(f"  |  {now} UTC", style=_MUTED)

    info_bot = Text()
    info_bot.append(f"Uptime: {uptime}", style=_MUTED)
    if state.room_id:
        info_bot.append(f"  |  Room: {state.room_id}", style=_ACCENT)

    table.add_row(logo_top, info_top)
    table.add_row(logo_bot, info_bot)
    return table


def _build_voice_status_panel(state: ConsoleState) -> Panel:
    """Build the Voice Status panel (top-left)."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {_PRIMARY_LIGHT}", width=10)
    table.add_column()

    style, label = _STATE_STYLES.get(state.voice_state, ("dim", state.voice_state.upper()))
    table.add_row("State", Text(label, style=style))
    table.add_row("Room", Text(state.room_id or "-", style=_ACCENT))
    table.add_row("Channel", Text(state.channel_id or "-"))
    table.add_row("Session", Text(_truncate(state.session_id or "-", 20)))
    table.add_row("User", Text(state.participant_id or "-"))

    # Barge-in status
    if state.barge_in_enabled is True:
        table.add_row("Barge-in", Text("ON", style="bold green"))
    elif state.barge_in_enabled is False:
        table.add_row("Barge-in", Text("OFF (no AEC)", style="bold red"))

    # Skills
    if state.skill_names:
        names = ", ".join(state.skill_names[:3])
        table.add_row("Skills", Text(f"{len(state.skill_names)}: {names}", style=_PRIMARY_LIGHT))

    return Panel(table, title="Voice Status", border_style=_PRIMARY)


def _build_audio_panel(state: ConsoleState) -> Panel:
    """Build the Audio Meters + Voice Activity panel (top-right)."""
    outer = Table.grid(padding=(0, 2))
    outer.add_column(ratio=1)  # meters
    outer.add_column(ratio=1)  # activity

    # --- Left: Audio meters ---
    meters = Table.grid(padding=(0, 1))
    meters.add_column(justify="right", width=4)
    meters.add_column(width=22)
    meters.add_column(width=10)
    meters.add_column()

    in_meter = _render_meter(state.input_level_history)
    in_db = Text(f"{state.input_level_db:+6.1f} dB", style=_MUTED)
    speech = Text()
    if state.is_speech:
        speech.append(" ● SPEECH", style="bold green")
    meters.add_row(Text("IN", style="bold green"), in_meter, in_db, speech)

    out_meter = _render_meter(state.output_level_history)
    out_db = Text(f"{state.output_level_db:+6.1f} dB", style=_MUTED)
    meters.add_row(Text("OUT", style=f"bold {_ACCENT}"), out_meter, out_db)

    # --- Right: Voice activity timeline ---
    activity = Text()
    events = list(state.voice_events)[-6:]
    for ve in events:
        ts = ve.timestamp.strftime("%H:%M:%S")
        activity.append(f"{ts} ", style=_MUTED)
        activity.append(f"● {ve.label}\n", style=ve.style)

    if not events:
        activity.append("Waiting for events...", style=f"italic {_MUTED}")

    outer.add_row(meters, activity)

    return Panel(outer, title="Audio & Activity", border_style=_PRIMARY)


def _build_conversation_text(state: ConsoleState) -> Text:
    """Build the full conversation Text (may exceed panel height)."""
    text = Text()
    for turn in state.conversation:
        if turn.role == "user":
            text.append("[User] ", style="bold yellow")
        else:
            text.append("[AI]   ", style=f"bold {_ACCENT}")
        text.append(f"{turn.text}\n")

    # Streaming partial text (user typing or AI generating).
    if state.partial_assistant_text:
        text.append("[AI]   ", style=f"bold {_ACCENT}")
        text.append(f"{state.partial_assistant_text}", style=f"italic {_ACCENT}")
        text.append("▍\n", style=f"bold {_PRIMARY}")
    if state.partial_text:
        text.append("[User] ", style="bold yellow")
        text.append(f"{state.partial_text}...", style="italic dim")

    if not state.conversation and not state.partial_text and not state.partial_assistant_text:
        text.append("Waiting for conversation...", style=f"dim italic {_MUTED}")

    return text


def _build_conversation_panel(state: ConsoleState, content_lines: int) -> Panel:
    """Build the Conversation panel with bottom-aligned auto-scroll."""
    inner = _build_conversation_text(state)
    return Panel(
        _BottomAligned(inner, height=content_lines),
        title="Conversation",
        border_style=_PRIMARY,
    )


def _build_log_text(log_buffer: LogRingBuffer, max_records: int = 100) -> Text:
    """Build the full log Text (may exceed panel height)."""
    text = Text()
    records = list(log_buffer.records_buffer)[-max_records:]
    for record in records:
        time_str = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        level_style = _LOG_LEVEL_STYLES.get(record.levelname, "")

        text.append(f"{time_str} ", style=_MUTED)
        text.append(f"{record.levelname[:5]:5s}", style=level_style)
        text.append(f" {record.getMessage()}\n")

    if not records:
        text.append("Waiting for logs...", style=f"dim italic {_MUTED}")

    return text


def _build_log_panel(log_buffer: LogRingBuffer, content_lines: int) -> Panel:
    """Build the Debug Log panel with bottom-aligned auto-scroll."""
    inner = _build_log_text(log_buffer)
    return Panel(
        _BottomAligned(inner, height=content_lines),
        title="Debug Log",
        border_style=_PRIMARY,
    )


def _build_stats_panel(state: ConsoleState) -> Panel:
    """Build the Stats footer panel."""
    parts = Text()
    parts.append(f" STT: {state.transcription_count}", style=_ACCENT)
    parts.append(f"  |  TTS: {state.tts_count}", style=_PRIMARY_LIGHT)
    parts.append(f"  |  Tools: {state.tool_call_count}", style="yellow")
    parts.append(f"  |  Barge-ins: {state.barge_in_count}", style="red")
    return Panel(parts, border_style=_PRIMARY)


def _build_footer() -> Text:
    """Build the bottom control hint bar."""
    footer = Text()
    footer.append(" Controls: ", style=_MUTED)
    footer.append("Ctrl+C", style=f"bold {_PRIMARY_LIGHT}")
    footer.append(" to quit", style=_MUTED)
    return footer


# ---------------------------------------------------------------------------
# Full dashboard layout
# ---------------------------------------------------------------------------


def _build_dashboard(
    state: ConsoleState,
    log_buffer: LogRingBuffer,
    terminal_height: int = 40,
    terminal_width: int = 120,
) -> Layout:
    """Build the full-screen dashboard layout."""
    layout = Layout()

    header_h, top_h, stats_h, footer_h = 2, 9, 3, 1

    layout.split_column(
        Layout(name="header", size=header_h),
        Layout(name="top", size=top_h),
        Layout(name="bottom"),
        Layout(name="stats", size=stats_h),
        Layout(name="footer", size=footer_h),
    )

    # Available content lines inside bottom panels.
    # Panel border takes 2 lines (top + bottom).
    bottom_h = terminal_height - header_h - top_h - stats_h - footer_h
    content_lines = max(5, bottom_h - 2)

    # Header
    layout["header"].update(_build_header(state))

    # Top row
    layout["top"].split_row(
        Layout(name="voice_status", ratio=1),
        Layout(name="audio_meters", ratio=2),
    )
    layout["voice_status"].update(_build_voice_status_panel(state))
    layout["audio_meters"].update(_build_audio_panel(state))

    # Bottom row — both panels use _BottomAligned for auto-scroll
    layout["bottom"].split_row(
        Layout(name="conversation", ratio=1),
        Layout(name="debug_log", ratio=1),
    )
    layout["conversation"].update(_build_conversation_panel(state, content_lines))
    layout["debug_log"].update(_build_log_panel(log_buffer, content_lines))

    # Stats + footer
    layout["stats"].update(_build_stats_panel(state))
    layout["footer"].update(_build_footer())

    return layout


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RoomKitConsole:
    """Full-screen terminal dashboard for RoomKit voice agent development.

    Provides a rich dashboard with panels for voice status, audio meters,
    conversation history, debug logs, and stats. Enable with a single line::

        console = RoomKitConsole(kit)

    Requires the ``rich`` library::

        pip install roomkit[console]
    """

    def __init__(
        self,
        kit: RoomKit,
        *,
        log_level: int = logging.DEBUG,
        refresh_rate: float = 4.0,
    ) -> None:
        self._kit = kit
        self._state = ConsoleState()
        self._refresh_rate = refresh_rate
        self._hook_names: list[str] = []
        self._original_handlers: list[logging.Handler] = []
        self._log_buffer = LogRingBuffer(max_records=200)
        self._live: Live | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._console = RichConsole(stderr=True)
        self._original_level: int = logging.WARNING

        self._hook_names = register_console_hooks(kit.hook_engine, self._state, kit)
        self._setup_logging(log_level)

        if self._is_tty:
            self._start_live()

    @property
    def state(self) -> ConsoleState:
        """The current console state (read-only access for inspection)."""
        return self._state

    def _setup_logging(self, level: int) -> None:
        """Install the log ring buffer handler."""
        root = logging.getLogger()
        self._original_handlers = list(root.handlers)
        self._original_level = root.level

        for h in self._original_handlers:
            root.removeHandler(h)

        self._log_buffer.setLevel(level)
        root.addHandler(self._log_buffer)
        if root.level > level:
            root.setLevel(level)

    def _start_live(self) -> None:
        """Start the Rich Live full-screen dashboard."""
        self._live = Live(
            _build_dashboard(
                self._state,
                self._log_buffer,
                self._console.height,
                self._console.width,
            ),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=True,
            auto_refresh=False,
        )
        self._live.start()
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(), name="roomkit-console-refresh"
        )

    async def _refresh_loop(self) -> None:
        """Periodically update the dashboard with current state."""
        try:
            while True:
                if self._live is not None:
                    self._live.update(
                        _build_dashboard(
                            self._state,
                            self._log_buffer,
                            self._console.height,
                            self._console.width,
                        ),
                        refresh=True,
                    )
                await asyncio.sleep(1.0 / self._refresh_rate)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the dashboard and restore original logging."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None

        if self._live is not None:
            self._live.stop()
            self._live = None

        unregister_console_hooks(self._kit.hook_engine, self._hook_names)
        self._hook_names.clear()

        root = logging.getLogger()
        root.removeHandler(self._log_buffer)
        for h in self._original_handlers:
            root.addHandler(h)
        root.setLevel(self._original_level)
        self._original_handlers.clear()
