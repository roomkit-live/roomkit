"""Progressive Markdown rendering for :mod:`roomkit.channels.cli`.

Rich is imported lazily so the base RoomKit installation keeps no terminal UI
dependency. The renderer rebuilds the accumulated Markdown document for every
real stream delta; it never delays output until the turn is complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import IO, Any, Literal

from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, ToolCallContent


@dataclass(slots=True)
class _Segment:
    kind: Literal["text", "thinking", "activity"]
    content: str


def require_markdown_support() -> None:
    """Raise an actionable error when the optional Rich dependency is absent."""
    try:
        import rich  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CLIChannel(markdown=True) requires Rich. "
            "Install it with `pip install roomkit[console]`."
        ) from exc


def print_markdown(
    label: str,
    text: str,
    *,
    file: IO[str],
    use_color: bool,
) -> None:
    """Render one complete Markdown response."""
    console_type, _, markdown_type, text_type = _rich_types()
    console = console_type(file=file, no_color=not use_color)
    console.print(text_type(f"\n{label}:", style="bold cyan"))
    console.print(markdown_type(text))
    console.print()


class MarkdownStreamRenderer:
    """Render an interleaved agent stream as a progressively updated document."""

    def __init__(self, label: str, *, file: IO[str], use_color: bool) -> None:
        console_type, live_type, markdown_type, text_type = _rich_types()
        self._label = label
        self._markdown_type = markdown_type
        self._text_type = text_type
        self._console = console_type(file=file, no_color=not use_color)
        self._live = live_type(
            console=self._console,
            auto_refresh=False,
            refresh_per_second=30,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="visible",
        )
        self._segments: list[_Segment] = []
        self._update_count = 0
        self._started = False
        self._closed = False

    def add_text(self, text: str) -> None:
        """Append one text delta and refresh immediately."""
        if not text:
            return
        self._append("text", text)
        self._refresh()

    def add_thinking(self, thinking: str) -> None:
        """Append one visible reasoning delta and refresh immediately."""
        if not thinking:
            return
        self._append("thinking", thinking)
        self._refresh()

    def add_tool_event(self, event: RoomEvent) -> None:
        """Append a tool start/completion line and refresh immediately."""
        content = event.content
        if not isinstance(content, ToolCallContent):
            return
        if event.type == EventType.TOOL_CALL_START:
            arguments = _format_arguments(content.arguments)
            line = f"🔧 {content.tool_name}{arguments}"
        elif event.type == EventType.TOOL_CALL_END:
            symbol = "✗" if content.status == "failed" else "✓"
            duration = (
                f" ({content.duration_ms} ms)"
                if content.duration_ms is not None and content.duration_ms > 0
                else ""
            )
            line = f"{symbol} {content.tool_name}{duration}"
        else:
            return
        self._segments.append(_Segment(kind="activity", content=line))
        self._refresh()

    def close(self) -> None:
        """Stop the live display and leave its final render in the terminal."""
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._live.stop()

    @property
    def update_count(self) -> int:
        """Number of real stream segments rendered; useful for diagnostics."""
        return self._update_count

    def _append(
        self,
        kind: Literal["text", "thinking"],
        content: str,
    ) -> None:
        if self._segments and self._segments[-1].kind == kind:
            self._segments[-1].content += content
        else:
            self._segments.append(_Segment(kind=kind, content=content))

    def _refresh(self) -> None:
        self._update_count += 1
        if not self._started:
            self._live.start(refresh=False)
            self._started = True
        self._live.update(self._render(), refresh=True)

    def _render(self) -> Any:
        from rich.console import Group

        renderables: list[Any] = []
        for segment in self._segments:
            if segment.kind == "thinking":
                thinking = segment.content.lstrip()
                if thinking:
                    renderables.append(self._text_type(f"💭 {thinking}", style="dim italic"))
            elif segment.kind == "activity":
                renderables.append(self._text_type(segment.content, style="magenta"))
            else:
                renderables.append(self._text_type(f"{self._label}:", style="bold cyan"))
                renderables.append(self._markdown_type(segment.content))
        return Group(*renderables)


def _rich_types() -> tuple[Any, Any, Any, Any]:
    require_markdown_support()
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text

    return Console, Live, Markdown, Text


def _format_arguments(arguments: dict[str, Any], *, max_length: int = 240) -> str:
    if not arguments:
        return ""
    rendered = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
    if len(rendered) > max_length:
        rendered = f"{rendered[: max_length - 1]}…"
    return f" {rendered}"
