"""The discussion console's full-screen layout: the room, the agents, the queue, your input.

```
┌─ room ──────────────────────────────┬─ agents ─────────────┐
│ @investigator · claude-sonnet → @sre│ ▶ @investigator      │
│   ⚙ query_logs(level='ERROR') → …   │   speaking           │
│   The 500s come from engine v2 …    │ • @sre        next   │
├─────────────────────────────────────┴──────────────────────┤
│ speaking @investigator · next @sre                         │
│ ❯ @sre what do the metrics say?                            │
└────────────────────────────────────────────────────────────┘
```

prompt_toolkit runs it in the event loop the room runs in; the panel and the
status line are drawn from the :class:`DiscussionView` at every refresh.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from roomkit._version import __version__
from roomkit.console._brand import ACCENT, MUTED, PRIMARY, PRIMARY_DIM, PRIMARY_LIGHT, SURFACE
from roomkit.console._discussion_view import DiscussionView

HINTS = "Enter send · PgUp/PgDn scroll · /help · Ctrl-C quit"
_PANEL_WIDTH = 44
_SCROLL_STEP = 10


def _hex(rich_rgb: str) -> str:
    """``rgb(99,102,241)`` (rich) as ``#6366f1`` (prompt_toolkit)."""
    red, green, blue = (int(c) for c in rich_rgb.removeprefix("rgb(").rstrip(")").split(","))
    return f"#{red:02x}{green:02x}{blue:02x}"


_BRAND = {
    "primary": _hex(PRIMARY),
    "light": _hex(PRIMARY_LIGHT),
    "dim": _hex(PRIMARY_DIM),
    "accent": _hex(ACCENT),
    "muted": _hex(MUTED),
    "surface": _hex(SURFACE),
}
# One color per agent, the brand's own first: AGENT_STYLES of them.
_AGENT_COLORS = (_BRAND["accent"], "#f59e0b", "#ec4899", "#10b981", _BRAND["light"])

STYLE = Style.from_dict(
    {
        "frame.border": _BRAND["muted"],
        "frame.label": f"{_BRAND['light']} bold",
        "brand": f"{_BRAND['light']} bold",
        "prompt": f"{_BRAND['accent']} bold",
        "you": f"{_BRAND['accent']} bold",
        "person": "bold",
        "ask": f"bg:{_BRAND['accent']} #000000 bold",
        "dim": _BRAND["muted"],
        "note": f"{_BRAND['muted']} italic",
        "status": f"bg:{_BRAND['surface']} {_BRAND['light']}",
        "speaking": f"bg:{_BRAND['primary']} #ffffff bold",
        "held": "#ef4444",
        **{f"agent{i}": f"{color} bold" for i, color in enumerate(_AGENT_COLORS)},
    }
)


class _RoomLexer(Lexer):
    """Styles each line of the room by the style it was added with."""

    def __init__(self, view: DiscussionView) -> None:
        self._view = view

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        lines = self._view.lines

        def line(number: int) -> StyleAndTextTuples:
            style = lines[number][0] if number < len(lines) else ""
            return [(style, document.lines[number])]

        return line


class DiscussionScreen:
    """The room on the left, the agents on the right, the status and your input below."""

    def __init__(
        self,
        view: DiscussionView,
        *,
        on_submit: Callable[[str], Coroutine[Any, Any, None]],
        title: str = "",
    ) -> None:
        self._view = view
        self._on_submit = on_submit
        self._title = title
        self._follow = True
        self._shown = 0
        self._submitted: set[asyncio.Task[None]] = set()
        self._room = TextArea(
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            focusable=False,
            lexer=_RoomLexer(view),
            get_line_prefix=self._wrap_prefix,
        )
        self._input = TextArea(
            height=1,
            prompt=[("class:prompt", "❯ ")],
            multiline=False,
            accept_handler=self._accept,
        )
        self.app: Application[None] = Application(
            layout=Layout(self._layout(), focused_element=self._input),
            key_bindings=self._keys(),
            style=STYLE,
            full_screen=True,
        )

    def refresh(self) -> None:
        """Show what the view gained since the last refresh."""
        if len(self._view.lines) != self._shown:
            self._show_room()
        self.app.invalidate()

    async def run(self) -> None:
        self._show_room()
        await self.app.run_async()

    def exit(self) -> None:
        if self.app.is_running:
            self.app.exit()

    def _layout(self) -> HSplit:
        header = Window(FormattedTextControl(self._header_fragments), height=2)
        agents = Window(FormattedTextControl(self._agent_fragments), wrap_lines=True)
        status = Window(
            FormattedTextControl(self._status_fragments), height=1, style="class:status"
        )
        return HSplit(
            [
                header,
                VSplit(
                    [
                        Frame(self._room, title="room"),
                        Frame(agents, title="agents", width=_PANEL_WIDTH),
                    ]
                ),
                status,
                self._input,
            ]
        )

    # -- The room pane --

    def _show_room(self) -> None:
        lines = self._view.lines
        self._shown = len(lines)
        text = "\n".join(line for _style, line in lines)
        buffer = self._room.buffer
        cursor = len(text) if self._follow else min(buffer.cursor_position, len(text))
        buffer.set_document(Document(text, cursor), bypass_readonly=True)

    def _wrap_prefix(self, number: int, wrap_count: int) -> StyleAndTextTuples:
        """A wrapped line keeps the indent of the line it continues."""
        lines = self._view.lines
        if wrap_count == 0 or number >= len(lines):
            return []
        line = lines[number][1]
        return [("", " " * min(len(line) - len(line.lstrip(" ")) + 2, 8))]

    def _scroll(self, lines: int) -> None:
        buffer = self._room.buffer
        document = buffer.document
        if lines < 0:
            offset = document.get_cursor_up_position(count=-lines)
        else:
            offset = document.get_cursor_down_position(count=lines)
        buffer.cursor_position += offset
        # Back at the last line: follow the room again.
        self._follow = buffer.document.cursor_position_row >= document.line_count - 1
        self.app.invalidate()

    # -- Header and status --

    def _header_fragments(self) -> StyleAndTextTuples:
        """RoomKit's 2x2 block logo, its version, and what this room is."""
        return [
            (f"fg:{_BRAND['primary']}", "██"),
            ("", " "),
            (f"fg:{_BRAND['light']}", "██"),
            ("class:brand", f"  RoomKit v{__version__}"),
            ("class:dim", f"  ·  {self._title}\n"),
            (f"fg:{_BRAND['light']}", "██"),
            ("", " "),
            (f"fg:{_BRAND['dim']}", "██"),
            ("class:dim", f"  a discussion · you are @{self._view.you}"),
        ]

    def _agent_fragments(self) -> StyleAndTextTuples:
        return [(style, text) for style, text in self._view.agent_fragments()]

    def _status_fragments(self) -> StyleAndTextTuples:
        return [("class:status", f" {self._view.status_text()}  |  {HINTS} ")]

    # -- Input and keys --

    def _accept(self, buffer: object) -> bool:
        text = getattr(buffer, "text", "").strip()
        if text:
            task = asyncio.get_running_loop().create_task(self._on_submit(text))
            self._submitted.add(task)
            task.add_done_callback(self._submitted.discard)
        return False

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("c-c")
        @keys.add("c-d")
        def _quit(event: KeyPressEvent) -> None:
            event.app.exit()

        @keys.add("pageup")
        def _up(event: KeyPressEvent) -> None:
            self._scroll(-_SCROLL_STEP)

        @keys.add("pagedown")
        def _down(event: KeyPressEvent) -> None:
            self._scroll(_SCROLL_STEP)

        return keys
