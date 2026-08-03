"""Inline branded chat presenter for ``CLIChannel(console=True)``.

Renders in the normal terminal scrollback — unlike the voice dashboard
(:mod:`roomkit.console._display`), there is no alternate screen and no
full-screen layout, so history stays scrollable and copy-pastable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from roomkit._version import __version__
from roomkit.channels._cli_markdown import MarkdownStreamRenderer, _format_arguments
from roomkit.console._brand import ACCENT, MUTED, PRIMARY, PRIMARY_LIGHT, SURFACE, logo_lines
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import RoomEvent, ToolCallContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit


@dataclass(slots=True)
class BannerModel:
    """One AI model line in the banner."""

    channel_id: str
    model: str | None
    provider: str | None


@dataclass(slots=True)
class BannerChannel:
    """One attached channel in the banner."""

    channel_id: str
    channel_type: str
    category: str


@dataclass(slots=True)
class ConsoleBannerData:
    """Everything the startup banner displays, gathered before rendering."""

    version: str
    room_id: str
    models: list[BannerModel] = field(default_factory=list)
    channels: list[BannerChannel] = field(default_factory=list)


async def collect_banner_data(kit: RoomKit, room_id: str) -> ConsoleBannerData:
    """Gather banner facts from the framework — no rendering.

    Model detection is duck-typed: any intelligence-bound channel exposing a
    ``provider`` with a ``model_name`` contributes a model line. Channels
    whose provider lacks one (e.g. ACP agents) still appear in the channel
    list.
    """
    data = ConsoleBannerData(version=__version__, room_id=room_id)
    bindings = await kit.list_bindings(room_id)
    for binding in bindings:
        data.channels.append(
            BannerChannel(
                channel_id=binding.channel_id,
                channel_type=str(binding.channel_type.value),
                category=str(binding.category.value),
            )
        )
        if binding.category != ChannelCategory.INTELLIGENCE:
            continue
        channel = kit.channels.get(binding.channel_id)
        if channel is None:
            continue
        try:
            provider = getattr(channel, "provider", None)
            model = getattr(provider, "model_name", None)
            label = getattr(channel, "provider_name", None)
            if label is None and provider is not None:
                label = type(provider).__name__
        except Exception:  # a provider property is free to raise — banner must not
            provider, model, label = None, None, None
        if model is not None or label is not None:
            data.models.append(
                BannerModel(channel_id=binding.channel_id, model=model, provider=label)
            )
    return data


def build_banner(data: ConsoleBannerData, *, notes: str | None = None) -> RenderableType:
    """Build the startup banner renderable — a plain inline grid, no Live."""
    lines: list[Text] = []

    title = Text()
    title.append("RoomKit", style=f"bold {PRIMARY}")
    title.append(f" v{data.version}", style=PRIMARY_LIGHT)
    lines.append(title)

    for entry in data.models:
        line = Text()
        line.append("◆ ", style=ACCENT)
        line.append(entry.model or entry.channel_id, style=ACCENT)
        if entry.provider:
            line.append(f" ({entry.provider})", style=MUTED)
        lines.append(line)

    context = Text()
    context.append(f"Room: {data.room_id}", style=MUTED)
    if data.channels:
        rendered = ", ".join(
            ch.channel_id if ch.category == "transport" else f"{ch.channel_id} ({ch.category})"
            for ch in data.channels
        )
        context.append(f"  ·  Channels: {rendered}", style=MUTED)
    lines.append(context)

    if notes:
        for note_line in notes.strip().splitlines():
            lines.append(Text(note_line, style=f"italic {MUTED}"))

    logo_top, logo_bottom = logo_lines()
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=6)
    grid.add_column()
    for index, line in enumerate(lines):
        logo_cell = logo_top if index == 0 else logo_bottom if index == 1 else Text("")
        grid.add_row(logo_cell, line)
    if len(lines) < 2:  # never truncate the logo
        grid.add_row(logo_bottom, Text(""))
    return grid


def print_banner(
    data: ConsoleBannerData,
    *,
    file: IO[str],
    use_color: bool,
    notes: str | None = None,
) -> None:
    """Render the startup banner once, inline."""
    console = Console(file=file, no_color=not use_color)
    console.print()
    console.print(build_banner(data, notes=notes))
    console.print()


def print_message(
    label: str,
    text: str,
    *,
    file: IO[str],
    use_color: bool,
    force_terminal: bool = False,
    width: int | None = None,
) -> None:
    """Render one complete (non-streamed) agent message.

    ``force_terminal``/``width`` matter under the pinned shell: its stdout
    proxy is not a TTY and has no size, so rich must be told both.
    """
    console = Console(
        file=file,
        no_color=not use_color,
        force_terminal=force_terminal or None,
        width=width,
    )
    console.print(Text(f"\n● {label}", style=f"bold {PRIMARY}"))
    console.print(Markdown(text))
    console.print()


def format_tool_line(event: RoomEvent, content: ToolCallContent) -> str | None:
    """Claude Code-style tool activity line (``⏺ tool(args)`` / ``  ⎿ ✓ 42 ms``)."""
    if event.type == EventType.TOOL_CALL_START:
        arguments = _format_arguments(content.arguments, max_length=120)
        return f"⏺ {content.tool_name}({arguments.strip()})"
    if event.type == EventType.TOOL_CALL_END:
        if content.status == "failed":
            return f"  ⎿ ✗ {content.tool_name} failed"
        duration = (
            f" · {content.duration_ms} ms"
            if content.duration_ms is not None and content.duration_ms > 0
            else ""
        )
        return f"  ⎿ ✓ {content.tool_name}{duration}"
    return None


class ConsoleStreamRenderer(MarkdownStreamRenderer):
    """Brand-styled progressive renderer for console mode.

    Inherits the inline Live mechanics (scrollback-preserving) from
    :class:`MarkdownStreamRenderer` and overrides only the styling hooks.
    """

    def _render_label(self) -> Any:
        return Text(f"● {self._label}", style=f"bold {PRIMARY}")

    def _render_thinking(self, text: str) -> Any | None:
        thinking = text.lstrip()
        if not thinking:
            return None
        return Text(f"💭 {thinking}", style=f"dim italic {MUTED}")

    def _render_activity(self, line: str) -> Any:
        style = MUTED if line.startswith("  ⎿") else ACCENT
        return Text(line, style=style)

    def _format_tool_line(self, event: RoomEvent, content: ToolCallContent) -> str | None:
        return format_tool_line(event, content)


def print_user_line(
    line: str,
    *,
    prompt: str = "❯ ",
    use_color: bool,
    width: int | None = None,
    file: IO[str] | None = None,
) -> None:
    """Echo a submitted user line into the transcript with a tinted background.

    The shell erases the input bar on accept and re-prints the line itself,
    so the user's messages stand apart from agent output in the scrollback.
    """
    console = Console(
        file=file if file is not None else sys.stdout,
        force_terminal=use_color or None,
        no_color=not use_color,
        width=width,
    )
    # Muted chevron (unlike the live bar's primary one) marks the line as
    # sent history; the background is padded to the full terminal width so
    # the tint reads as a block, not a text highlight.
    text = Text()
    text.append(prompt, style=f"bold {MUTED} on {SURFACE}")
    text.append(line, style=f"on {SURFACE}")
    pad = (-text.cell_len) % console.width
    if pad:
        text.append(" " * pad, style=f"on {SURFACE}")
    console.print(text)


class PinnedStreamRenderer:
    """Append-only streaming renderer for the pinned-bar shell.

    ``rich.Live`` repaints in place, which fights the prompt_toolkit bar for
    cursor control, and ``patch_stdout``'s proxy line-buffers partial writes
    anyway. So under the shell the stream is flushed append-only: completed
    Markdown blocks (``\\n\\n`` boundaries, never inside a code fence),
    completed thinking lines, and tool lines as they arrive. Same duck
    interface as :class:`MarkdownStreamRenderer`.

    ``file=None`` resolves ``sys.stdout`` at each flush so output goes
    through whatever proxy ``patch_stdout`` has installed; tests inject a
    ``StringIO``. ``use_color``/``width`` were captured against the real
    terminal before the proxy hid it.
    """

    def __init__(
        self,
        label: str,
        *,
        file: IO[str] | None = None,
        use_color: bool,
        width: int | None = None,
    ) -> None:
        self._label = label
        self._file = file
        self._use_color = use_color
        self._width = width
        self._buffer = ""
        self._thinking = ""
        self._thinking_prefix_pending = True
        self._label_printed = False
        self._closed = False
        self._update_count = 0

    # -- Duck interface (MarkdownStreamRenderer-compatible) -------------------

    def add_text(self, text: str) -> None:
        """Buffer one text delta; flush any completed Markdown blocks."""
        if not text:
            return
        self._update_count += 1
        self._flush_thinking(force=True)
        self._buffer += text
        self._flush_text()

    def add_thinking(self, thinking: str) -> None:
        """Buffer one reasoning delta; flush completed lines."""
        if not thinking:
            return
        self._update_count += 1
        self._thinking += thinking
        self._flush_thinking()

    def add_tool_event(self, event: RoomEvent) -> None:
        """Print a tool start/completion line immediately."""
        content = event.content
        if not isinstance(content, ToolCallContent):
            return
        line = format_tool_line(event, content)
        if line is None:
            return
        self._update_count += 1
        # Ordering beats typography: everything pending renders first.
        self._flush_thinking(force=True)
        self._flush_text(force=True)
        style = MUTED if line.startswith("  ⎿") else ACCENT
        self._console().print(Text(line, style=style))

    def close(self) -> None:
        """Flush the tails and end the turn with a blank line."""
        if self._closed:
            return
        self._closed = True
        self._flush_thinking(force=True)
        self._flush_text(force=True)
        self._console().print()

    @property
    def update_count(self) -> int:
        """Number of stream segments received; useful for diagnostics."""
        return self._update_count

    # -- Internals ------------------------------------------------------------

    def _console(self) -> Console:
        file = self._file if self._file is not None else sys.stdout
        return Console(
            file=file,
            force_terminal=self._use_color or None,
            no_color=not self._use_color,
            width=self._width,
        )

    def _ensure_label(self) -> None:
        if self._label_printed:
            return
        self._label_printed = True
        self._console().print(Text(f"● {self._label}", style=f"bold {PRIMARY}"))

    def _flush_text(self, *, force: bool = False) -> None:
        if force:
            head, self._buffer = self._buffer, ""
        else:
            head, self._buffer = _split_flushable(self._buffer)
        if not head.strip():
            return
        self._ensure_label()
        self._console().print(Markdown(head))

    def _flush_thinking(self, *, force: bool = False) -> None:
        if force:
            pending, self._thinking = self._thinking, ""
            # Next thinking block (e.g. after a tool round) gets its own 💭.
            reset_prefix = True
        elif "\n" in self._thinking:
            split_at = self._thinking.rfind("\n")
            pending, self._thinking = self._thinking[:split_at], self._thinking[split_at + 1 :]
            reset_prefix = False
        else:
            return
        for raw_line in pending.split("\n"):
            line = raw_line.lstrip() if self._thinking_prefix_pending else raw_line
            if not line:
                continue
            prefix = "💭 " if self._thinking_prefix_pending else ""
            self._thinking_prefix_pending = False
            self._console().print(Text(f"{prefix}{line}", style=f"dim italic {MUTED}"))
        if reset_prefix:
            self._thinking_prefix_pending = True


def _split_flushable(buffer: str) -> tuple[str, str]:
    """Split *buffer* at the last blank-line boundary outside a code fence.

    Returns ``(flushable_head, retained_tail)``. The final (incomplete) line
    is never flushable, and a boundary is only taken where all fences are
    closed — so a fenced block with internal blank lines is never split.
    """
    lines = buffer.split("\n")
    fence_open = False
    flush_upto = 0
    for i, line in enumerate(lines[:-1]):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_open = not fence_open
        if not fence_open and line.strip() == "" and i > 0:
            flush_upto = i + 1
    head = "\n".join(lines[:flush_upto])
    tail = "\n".join(lines[flush_upto:])
    return head, tail


__all__ = [
    "BannerChannel",
    "BannerModel",
    "ConsoleBannerData",
    "ConsoleStreamRenderer",
    "PinnedStreamRenderer",
    "build_banner",
    "collect_banner_data",
    "format_tool_line",
    "print_banner",
    "print_message",
    "print_user_line",
]
