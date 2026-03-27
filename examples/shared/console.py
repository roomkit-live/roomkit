"""Optional console dashboard for RoomKit examples.

Enable the dashboard by setting ``CONSOLE=1``::

    CONSOLE=1 uv run python examples/realtime_voice_local_gemini.py

When ``CONSOLE`` is not set (or ``0``), ``setup_console`` is a no-op.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from roomkit import RoomKit


def setup_console(kit: RoomKit) -> Callable[[], Awaitable[None]] | None:
    """Enable the console dashboard if ``CONSOLE=1`` is set.

    Returns a cleanup coroutine to pass to ``run_until_stopped``
    (or ``None`` when the console is disabled).

    Usage::

        from shared import run_until_stopped, setup_console

        console_cleanup = setup_console(kit)
        await run_until_stopped(kit, cleanup=console_cleanup)
    """
    if os.environ.get("CONSOLE", "0") not in ("1", "true", "yes"):
        return None

    from roomkit.console import RoomKitConsole

    console = RoomKitConsole(kit)
    return console.stop
