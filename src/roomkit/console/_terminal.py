"""Terminal prompts that cooperate with a running console shell.

Two questions an application asks mid-session: *answer this* (
:func:`terminal_input`) and *pick one of these* (:func:`terminal_select`).
Both suspend the pinned input bar for the duration and both degrade to plain
stdin when no shell owns the terminal.

Importable without prompt_toolkit installed — the shell lookup is lazy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

Choice = tuple[str, str]
"""One option: ``(value, label)``. The value is returned, the label is shown."""


async def terminal_input(prompt: str = "") -> str:
    """Read one line from the real terminal.

    When the pinned-bar console shell is running, the bar is suspended
    (prompt_toolkit ``in_terminal``) for the duration of the read so the
    prompt and the typed line appear in the normal scrollback — the way a
    tool-permission question should. Without a shell (classic CLI mode, or
    prompt_toolkit not installed) this is simply ``input()`` on a worker
    thread.

    ``EOFError`` and ``KeyboardInterrupt`` propagate to the caller.
    """
    app = _active_app()
    if app is None:
        return await asyncio.to_thread(input, prompt)

    from prompt_toolkit.application import in_terminal

    async with in_terminal():
        return await asyncio.to_thread(input, prompt)


async def terminal_select(
    options: Sequence[Choice | str],
    *,
    title: str = "",
    default: str | None = None,
) -> str | None:
    """Ask the user to pick one option, and return its value.

    Under the pinned-bar shell this renders an inline arrow-key menu (↑/↓,
    Enter to choose, Esc to cancel) where the bar sits — no alternate
    screen, the transcript above stays put, and the menu erases itself once
    answered. Without a shell it falls back to a numbered list read from
    stdin, so piped and CI runs still work.

    Args:
        options: ``(value, label)`` pairs, or bare strings used as both.
        title: Optional heading shown above the list.
        default: Value to start on. Defaults to the first option.

    Returns:
        The chosen value, or ``None`` when the user cancels (Esc, Ctrl-C,
        EOF, or an unparseable answer in the fallback).
    """
    choices = [(item, item) if isinstance(item, str) else item for item in options]
    if not choices:
        return None
    start = next((i for i, (value, _) in enumerate(choices) if value == default), 0)

    app = _active_app()
    if app is None:
        return await asyncio.to_thread(_numbered_fallback, choices, title, start)

    from prompt_toolkit.application import in_terminal

    async with in_terminal():
        # The picker borrows the shell's terminal — same input and output,
        # so it reads the keys the bar would have read and nothing competes.
        picker = _build_picker(choices, title, start, app.input, app.output)
        return await picker.run_async()


def _active_app() -> Any:
    try:
        from roomkit.console._shell import active_shell_app
    except ImportError:
        return None
    return active_shell_app()


def _numbered_fallback(choices: list[Choice], title: str, start: int) -> str | None:
    """Pick by number when no shell owns the terminal (classic mode, pipes)."""
    lines = [""]
    if title:
        lines.append(title)
    for index, (_value, label) in enumerate(choices, start=1):
        marker = "*" if index - 1 == start else " "
        lines.append(f" {marker} {index}) {label}")
    print("\n".join(lines))
    try:
        answer = input(f"Choice [1-{len(choices)}, Enter for {start + 1}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not answer:
        return choices[start][0]
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1][0]
    return None


def _build_picker(
    choices: list[Choice],
    title: str,
    start: int,
    input_: Any,
    output: Any,
) -> Any:
    """A one-shot, non-full-screen prompt_toolkit application."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    cursor = [start]

    def render() -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        if title:
            fragments.append(("class:select-title", f"{title}\n"))
        for index, (_value, label) in enumerate(choices):
            picked = index == cursor[0]
            fragments.append(
                (
                    "class:select-on" if picked else "class:select-off",
                    f"{'❯' if picked else ' '} {label}\n",
                )
            )
        fragments.append(("class:select-hint", "  ↑/↓ to move · enter to choose · esc to cancel"))
        return fragments

    keys = KeyBindings()

    @keys.add("up")
    @keys.add("c-p")
    def _up(event: Any) -> None:
        cursor[0] = (cursor[0] - 1) % len(choices)

    @keys.add("down")
    @keys.add("c-n")
    def _down(event: Any) -> None:
        cursor[0] = (cursor[0] + 1) % len(choices)

    @keys.add("enter")
    def _choose(event: Any) -> None:
        event.app.exit(result=choices[cursor[0]][0])

    @keys.add("escape", eager=True)
    @keys.add("c-c")
    @keys.add("c-d")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    height = len(choices) + (1 if title else 0) + 1  # + the hint line
    window = Window(FormattedTextControl(render, focusable=True), height=height)
    return Application(
        layout=Layout(HSplit([window])),
        key_bindings=keys,
        style=Style.from_dict(
            {
                "select-title": "#64748b",  # MUTED
                "select-on": "#6366f1 bold",  # PRIMARY
                "select-off": "",
                "select-hint": "#64748b italic",
            }
        ),
        full_screen=False,
        erase_when_done=True,
        input=input_,
        output=output,
        # stdin may be a pipe under the shell's patch_stdout proxy; refuse to
        # guess a mouse protocol we cannot read back.
        mouse_support=False,
    )


__all__ = ["Choice", "terminal_input", "terminal_select"]
