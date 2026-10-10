"""RoomKit Console — terminal displays for agent development.

Requires the ``console`` extra (``rich`` and ``prompt_toolkit``)::

    pip install roomkit[console]

Usage::

    from roomkit.console import DiscussionConsole, RoomKitConsole

    console = RoomKitConsole(kit)  # voice dashboard
    await DiscussionConsole(kit, room_id, channel_id="oncall").run()  # a discussion
"""

from __future__ import annotations

from typing import Any

from roomkit.console._discussion_view import AgentCard
from roomkit.console._terminal import Choice, terminal_input, terminal_select

try:
    from roomkit.console._display import RoomKitConsole as RoomKitConsole
except ImportError as _exc:
    _import_error: ImportError | None = _exc

    class RoomKitConsole:
        """Stub that raises ``ImportError`` when ``rich`` is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RoomKitConsole requires the 'rich' library. "
                "Install it with: pip install roomkit[console]"
            ) from _import_error


try:
    from roomkit.console._discussion import DiscussionConsole as DiscussionConsole
except ImportError as _exc:
    _discussion_import_error: ImportError | None = _exc

    class DiscussionConsole:  # type: ignore[no-redef]
        """Stub that raises ``ImportError`` when ``prompt_toolkit`` is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "DiscussionConsole requires prompt_toolkit. "
                "Install it with: pip install roomkit[console]"
            ) from _discussion_import_error


__all__ = [
    "AgentCard",
    "Choice",
    "DiscussionConsole",
    "RoomKitConsole",
    "terminal_input",
    "terminal_select",
]
