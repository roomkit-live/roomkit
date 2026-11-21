"""The agent's ACP permission requests still waiting on RoomKit's decision."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, slots=True)
class _Asking:
    """One permission request, its decision being made."""

    deciding: asyncio.Task[Any]
    abandoned: bool = False
    woken: asyncio.Event = field(default_factory=asyncio.Event)
    """Set once the session's cancellation is out: the request is answered."""


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
        *session_id*; ``None`` once the session's prompt was cancelled while it
        was pending, the decision abandoned."""
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
        if asking.abandoned:
            # Made while the cancellation was being sent: dropped, and the
            # request answered once the cancellation is out.
            await asking.woken.wait()
            return None
        return decided

    @contextmanager
    def abandoning(self, session_id: str) -> Generator[None]:
        """Abandon the decisions pending in *session_id* around the block that
        sends the session's cancellation: a decision made meanwhile is
        dropped, and once the block ends, however it ends, each request is
        answered ``cancelled`` and its decision cancelled."""
        asked = tuple(self._by_session.get(session_id, ()))
        for asking in asked:
            asking.abandoned = True
        try:
            yield
        finally:
            for asking in asked:
                asking.woken.set()
                asking.deciding.cancel()
