"""Shared sounddevice import helper.

``sounddevice`` is an optional dependency used by both the local voice backend
and the local capture source.  Both need the same clear failure when it is
missing, so the import lives here rather than in either of them.
"""

from __future__ import annotations

from typing import Any


def import_sounddevice(component: str) -> Any:
    """Import sounddevice, raising a clear error if missing.

    Args:
        component: Name of the class requiring it, used in the error message.

    Returns:
        The imported ``sounddevice`` module.

    Raises:
        ImportError: If sounddevice is not installed.
    """
    try:
        import sounddevice as _sd

        return _sd
    except ImportError as exc:
        raise ImportError(
            f"sounddevice is required for {component}. "
            "Install it with: pip install roomkit[local-audio]"
        ) from exc
