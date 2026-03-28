"""RoomKit Console — rich terminal display for voice agent development.

Requires the ``rich`` library::

    pip install roomkit[console]

Usage::

    from roomkit.console import RoomKitConsole

    kit = RoomKit()
    console = RoomKitConsole(kit)
"""

from __future__ import annotations

from typing import Any

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


__all__ = ["RoomKitConsole"]
