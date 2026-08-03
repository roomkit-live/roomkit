"""Inline branded chat presenter for ``CLIChannel(console=True)``.

Renders in the normal terminal scrollback — unlike the voice dashboard
(:mod:`roomkit.console._display`), there is no alternate screen and no
full-screen layout, so history stays scrollable and copy-pastable.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from collections.abc import Mapping
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
    console.print()
    console.print(agent_handle(label))
    console.print(answer_block(text))
    console.print()


def agent_handle(label: str) -> Text:
    """``@claude code`` — who is speaking, stated quietly above the answer."""
    return Text(f"@{label.lower()}", style=f"dim italic {MUTED}")


def answer_block(markdown: str, *, bullet: bool = True) -> RenderableType:
    """The agent's prose with the turn marker in front of its first line.

    A two-column grid rather than a prefixed string: the body stays real
    Markdown (headings, lists, fenced code keep their formatting) while its
    continuation lines align under the text, not under the marker.
    """
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(overflow="fold")
    marker = Text("●", style=f"bold {PRIMARY}") if bullet else Text(" ")
    grid.add_row(marker, Markdown(markdown))
    return grid


def format_turn_footer(duration_ms: int, tool_calls: int) -> str:
    """``⎿ took 2m 30s · 3 tools`` — what the wait cost, once it is over."""
    parts = [f"took {_format_duration(duration_ms)}"]
    if tool_calls:
        parts.append(f"{tool_calls} tool{'s' if tool_calls > 1 else ''}")
    return f"  ⎿ {' · '.join(parts)}"


def format_tool_start_line(content: ToolCallContent) -> str:
    """``⏺ tool(args)``, or just ``⏺ tool`` when no arguments are known yet.

    ACP agents announce a tool before its title/arguments arrive, so an
    empty-parens render would be pure noise — the enriched name shows on
    the completion line instead.
    """
    if content.arguments:
        arguments = _format_arguments(content.arguments, max_length=120)
        return f"⏺ {content.tool_name}({arguments.strip()})"
    return f"⏺ {content.tool_name}"


def format_tool_end_line(content: ToolCallContent) -> str:
    """``  ⎿ ✓ tool · 3.0s`` / ``  ⎿ ✗ tool failed``."""
    if content.status == "failed":
        return f"  ⎿ ✗ {content.tool_name} failed"
    duration = (
        f" · {_format_duration(content.duration_ms)}"
        if content.duration_ms is not None and content.duration_ms > 0
        else ""
    )
    return f"  ⎿ ✓ {content.tool_name}{duration}"


def _format_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    minutes, rest = divmod(ms, 60_000)
    return f"{minutes}m {rest // 1000}s"


_PREVIEW_MAX_LINES = 5
_PREVIEW_HARD_CAP = 200  # lines collected before slicing — bounds huge diffs


def tool_result_preview(
    content: ToolCallContent,
    *,
    max_lines: int = _PREVIEW_MAX_LINES,
) -> list[tuple[str, str]]:
    """Extract display lines from a tool result — what Claude Code shows.

    Returns ``(kind, line)`` pairs, kind one of ``"dim"`` (plain output),
    ``"add"``/``"del"`` (diff lines), capped at *max_lines* with a trailing
    ``… +N lines`` marker. Understands ACP content blocks (text, ``diff``
    with old/new text), MCP-style ``{"content": [...]}`` payloads, plain
    strings, and falls back to compact JSON for anything else.

    ACP's display-intended payload (``structured_content["acp_content"]``,
    where file diffs live) wins over the raw result; the error text wins on
    failure.
    """
    source: Any = None
    if content.status == "failed" and content.error:
        source = content.error
    else:
        if isinstance(content.structured_content, Mapping):
            source = content.structured_content.get("acp_content")
        if source is None:
            source = content.result
    collected: list[tuple[str, str]] = []
    _collect_preview(source, collected)
    if not collected:
        return []
    if len(collected) > max_lines:
        hidden = len(collected) - max_lines
        collected = collected[:max_lines]
        collected.append(("dim", f"… +{hidden} lines"))
    return collected


def _collect_preview(value: Any, out: list[tuple[str, str]]) -> None:
    if value is None or len(out) >= _PREVIEW_HARD_CAP:
        return
    if isinstance(value, str):
        out.extend(("dim", line) for line in value.splitlines() if line.strip())
        return
    if isinstance(value, Mapping):
        block_type = value.get("type")
        if block_type == "diff":
            _collect_diff_preview(value, out)
            return
        if block_type == "content":
            _collect_preview(value.get("content"), out)
            return
        if block_type == "text":
            _collect_preview(value.get("text"), out)
            return
        if block_type == "terminal":
            return  # terminal blocks carry no inline output
        for key in ("output", "text", "content"):
            if key in value:
                _collect_preview(value[key], out)
                return
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        out.append(("dim", rendered[:200]))
        return
    if isinstance(value, list):
        for item in value:
            if len(out) >= _PREVIEW_HARD_CAP:
                return
            _collect_preview(item, out)
        return
    out.append(("dim", str(value)[:200]))


def _collect_diff_preview(block: Mapping[str, Any], out: list[tuple[str, str]]) -> None:
    path = block.get("path")
    if path:
        out.append(("dim", str(path)))
    # ACP dumps use camelCase aliases (oldText/newText); tolerate snake_case.
    old_text = block.get("oldText", block.get("old_text")) or ""
    new_text = block.get("newText", block.get("new_text")) or ""
    diff = difflib.unified_diff(
        str(old_text).splitlines(),
        str(new_text).splitlines(),
        lineterm="",
        n=1,
    )
    for line in diff:
        if len(out) >= _PREVIEW_HARD_CAP:
            return
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            out.append(("add", line))
        elif line.startswith("-"):
            out.append(("del", line))
        else:
            out.append(("dim", line))


def format_tool_line(event: RoomEvent, content: ToolCallContent) -> str | None:
    """Single-string tool render for the Live console renderer.

    START is one line; END is the status line plus indented result preview
    lines (monochrome — the pinned renderer does the colored version).
    """
    if event.type == EventType.TOOL_CALL_START:
        return format_tool_start_line(content)
    if event.type == EventType.TOOL_CALL_END:
        parts = [format_tool_end_line(content)]
        parts.extend(f"    {line}" for _kind, line in tool_result_preview(content))
        return "\n".join(parts)
    return None


class ConsoleStreamRenderer(MarkdownStreamRenderer):
    """Brand-styled progressive renderer for console mode.

    Inherits the inline Live mechanics (scrollback-preserving) from
    :class:`MarkdownStreamRenderer` and overrides only the styling hooks.
    """

    def _render_label(self) -> Any:
        return agent_handle(self._label)

    def _render_answer(self, markdown_text: str) -> Any:
        return answer_block(markdown_text)

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
    console.print()  # breathing room before the agent's response


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
        # Claude Code-style block spacing: one blank line between sections
        # (label, tool activity, thinking, text resuming after either).
        self._last_was_gap = True  # start of turn: no leading blank needed
        self._text_gap_pending = False
        # The turn marker leads each stretch of prose — a fresh one after
        # every tool round or thinking block, as an agent CLI transcript does.
        self._bullet_pending = True
        self._tool_calls = 0
        self._started_at = time.monotonic()

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
        """Print tool activity immediately — status line plus result preview."""
        content = event.content
        if not isinstance(content, ToolCallContent):
            return
        if event.type not in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END):
            return
        self._update_count += 1
        # Ordering beats typography: everything pending renders first.
        self._flush_thinking(force=True)
        self._flush_text(force=True)
        just_labelled = self._ensure_label()
        if event.type == EventType.TOOL_CALL_START:
            if not just_labelled:
                self._gap()
            self._print(Text(format_tool_start_line(content), style=ACCENT))
            return
        self._tool_calls += 1
        self._print(Text(format_tool_end_line(content), style=MUTED))
        for kind, line in tool_result_preview(content):
            style = {"add": "green", "del": "red"}.get(kind, f"dim {MUTED}")
            self._print(Text(f"    {line}", style=style))
        self._text_gap_pending = True
        self._bullet_pending = True  # prose resuming after a tool leads again

    def close(self) -> None:
        """Flush the tails, report what the turn cost, end with a blank line."""
        if self._closed:
            return
        self._closed = True
        self._flush_thinking(force=True)
        self._flush_text(force=True)
        if self._label_printed:
            elapsed_ms = int((time.monotonic() - self._started_at) * 1000)
            self._print(
                Text(
                    format_turn_footer(elapsed_ms, self._tool_calls),
                    style=f"dim italic {MUTED}",
                )
            )
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

    def _print(self, renderable: Any) -> None:
        self._console().print(renderable)
        self._last_was_gap = False

    def _gap(self) -> None:
        """One blank line between sections; never two in a row."""
        if self._last_was_gap:
            return
        self._console().print()
        self._last_was_gap = True

    def _ensure_label(self) -> bool:
        """Print the handle once per turn. True when this call printed it.

        Callers use the answer to skip their own separator: the handle
        already opens the section, and a blank line under it would detach
        the agent's name from what it introduces.
        """
        if self._label_printed:
            return False
        self._label_printed = True
        self._gap()
        self._print(agent_handle(self._label))
        return True

    def _flush_text(self, *, force: bool = False) -> None:
        if force:
            head, self._buffer = self._buffer, ""
        else:
            head, self._buffer = _split_flushable(self._buffer)
        if not head.strip():
            return
        self._ensure_label()
        if self._text_gap_pending:
            self._gap()
            self._text_gap_pending = False
        self._print(answer_block(head, bullet=self._bullet_pending))
        self._bullet_pending = False

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
        printed = False
        for raw_line in pending.split("\n"):
            line = raw_line.lstrip() if self._thinking_prefix_pending else raw_line
            if not line:
                continue
            if self._thinking_prefix_pending and not self._ensure_label():
                self._gap()
            prefix = "💭 " if self._thinking_prefix_pending else ""
            self._thinking_prefix_pending = False
            self._print(Text(f"{prefix}{line}", style=f"dim italic {MUTED}"))
            printed = True
        if reset_prefix:
            self._thinking_prefix_pending = True
            if printed:
                # Text resuming after a thinking block gets its own gap and
                # leads with the turn marker again.
                self._text_gap_pending = True
                self._bullet_pending = True


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
    "agent_handle",
    "answer_block",
    "format_tool_end_line",
    "format_tool_line",
    "format_tool_start_line",
    "format_turn_footer",
    "print_banner",
    "print_message",
    "print_user_line",
    "tool_result_preview",
]
