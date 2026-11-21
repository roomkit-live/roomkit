"""The agent's ACP permission requests still waiting on RoomKit's decision."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass(eq=False, slots=True)
class _Asking:
    """One permission request, its decision being made."""

    deciding: asyncio.Task[Any]
    abandoned: bool = False


def _cancelling() -> bool:
    """Whether the running task itself is being cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class PendingPermissions:
    """The permission requests of each session whose decision is pending.

    ACP: a client that sends ``session/cancel`` MUST answer every pending
    ``session/request_permission`` of that session with the ``cancelled``
    outcome. Cancelling a session abandons the decisions still pending in it,
    the external tool handler's included (RFC §6.4).
    """

    def __init__(self) -> None:
        self._by_session: dict[str, set[_Asking]] = {}

    async def decide[T](self, session_id: str, decision: Coroutine[Any, Any, T]) -> T | None:
        """The result of *decision*, a permission request's decision made in
        *session_id*; ``None`` once the session's prompt was cancelled while
        it was pending, the decision abandoned."""
        asking = _Asking(asyncio.get_running_loop().create_task(decision))
        pending = self._by_session.setdefault(session_id, set())
        pending.add(asking)
        try:
            decided = await asking.deciding
        except asyncio.CancelledError:
            if not asking.abandoned or _cancelling():
                raise
            return None
        finally:
            pending.discard(asking)
            if not pending and self._by_session.get(session_id) is pending:
                del self._by_session[session_id]
        # A decision made as the session was cancelled is abandoned all the same.
        return None if asking.abandoned else decided

    def abandon(self, session_id: str) -> None:
        """Abandon every decision pending in *session_id*: each request is
        answered ``cancelled``, and its decision is cancelled."""
        for asking in tuple(self._by_session.get(session_id, ())):
            asking.abandoned = True
            asking.deciding.cancel()
