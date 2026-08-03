"""Optional console rendering for RoomKit examples.

Enable it by setting ``CONSOLE=1``::

    CONSOLE=1 uv run python examples/realtime_voice_local_gemini.py   # voice dashboard
    CONSOLE=1 uv run python examples/ollama_ai.py                     # CLI console mode

Voice examples call ``setup_console(kit)`` (full-screen dashboard); CLI
examples pass ``console=console_enabled()`` to :class:`roomkit.CLIChannel`
(inline branded rendering). When ``CONSOLE`` is not set (or ``0``), both
are no-ops.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from roomkit import RoomKit

_TRUTHY = ("1", "true", "yes")


def console_enabled() -> bool:
    """Whether ``CONSOLE=1`` (or ``true``/``yes``) is set."""
    return os.environ.get("CONSOLE", "0") in _TRUTHY


def setup_console(kit: RoomKit) -> Callable[[], Awaitable[None]] | None:
    """Enable the console dashboard if ``CONSOLE=1`` is set.

    Returns a cleanup coroutine to pass to ``run_until_stopped``
    (or ``None`` when the console is disabled).

    Usage::

        from shared import run_until_stopped, setup_console

        console_cleanup = setup_console(kit)
        await run_until_stopped(kit, cleanup=console_cleanup)
    """
    if not console_enabled():
        return None

    from roomkit.console import RoomKitConsole

    console = RoomKitConsole(kit)
    return console.stop
