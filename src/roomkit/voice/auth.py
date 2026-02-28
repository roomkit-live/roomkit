"""Pluggable authentication primitives for voice transports."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from typing import Any

AuthCallback = Callable[[Any], Awaitable[dict[str, Any] | None]]
"""Async callback for transport authentication.

Receives the connection context (e.g. WebSocket, HTTP request) and returns
a metadata dict on success or ``None`` to reject the connection.
"""

auth_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "auth_context", default=None
)
"""Context variable holding auth metadata from the most recent authentication.

Set automatically before ``session_factory`` is called so the factory can
read auth metadata without changing its signature::

    from roomkit.voice.auth import auth_context

    async def my_session_factory(websocket_id: str) -> VoiceSession:
        meta = auth_context.get()  # dict or None
        ...
"""
