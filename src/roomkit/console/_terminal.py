"""Terminal input that cooperates with a running console shell.

Importable without prompt_toolkit installed — the shell lookup is lazy, and
without an active shell this is a plain threaded ``input()``.
"""

from __future__ import annotations

import asyncio


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
    try:
        from roomkit.console._shell import active_shell_app
    except ImportError:
        app = None
    else:
        app = active_shell_app()

    if app is None:
        return await asyncio.to_thread(input, prompt)

    from prompt_toolkit.application import in_terminal

    async with in_terminal():
        return await asyncio.to_thread(input, prompt)


__all__ = ["terminal_input"]
