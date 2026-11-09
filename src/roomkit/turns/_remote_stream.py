"""The owner's copy of a turn's response stream (RFC §10.6.4).

Its reads grant the executor a window of items; closing it closes the
channel's stream. A stream its reader drops unclosed is closed for it, as a
collected generator would be.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.turns._owner_turn import OwnerTurn

# The closes of streams their reader dropped, held until they end.
_DROPPED_CLOSES: set[asyncio.Task[None]] = set()


def _close_dropped(turn: OwnerTurn, loop: asyncio.AbstractEventLoop) -> None:
    """Close the stream of *turn*, which its reader dropped unclosed: the
    executor closes the channel's, as a collected generator would be."""
    if loop.is_closed():
        return
    if turn.over:
        # Ended before its reader dropped it: the record is still the turn's last.
        loop.call_soon_threadsafe(turn.reader_stopped)
        return

    def start() -> None:
        task = loop.create_task(turn.close_stream())
        _DROPPED_CLOSES.add(task)
        task.add_done_callback(_DROPPED_CLOSES.discard)

    loop.call_soon_threadsafe(start)


class RemoteStream:
    """The owner's copy of a turn's response stream: its reads grant the
    executor items, closing it closes the channel's (RFC §10.6.4)."""

    def __init__(self, turn: OwnerTurn) -> None:
        self._turn = turn
        self._reading = False
        self._done = False
        # A stream its reader drops without closing it is closed for it.
        self._dropped = weakref.finalize(self, _close_dropped, turn, asyncio.get_running_loop())

    def __aiter__(self) -> RemoteStream:
        return self

    @property
    def executor_id(self) -> str | None:
        """The executor that took the turn (RFC §10.6.4)."""
        return self._turn.executor_id

    @property
    def end_owed(self) -> bool:
        """Whether its turn ended for the owner without the executor's end:
        its cleanup is not known to be done (RFC §10.6.8)."""
        return self._turn.end_owed

    async def __anext__(self) -> Any:
        if self._done:
            raise StopAsyncIteration
        if self._reading:
            raise RuntimeError("anext(): asynchronous generator is already running")
        self._reading = True
        try:
            return await self._turn.read_item()
        except BaseException:
            self._done = True
            self._dropped.detach()
            raise
        finally:
            self._reading = False

    def read_as_trace(self) -> None:
        """Say, before reading it, that its reader commits the turn as a
        delegated task's trace, crossing no hook (RFC §23.3 step 5): only such
        a turn's answer text is published (RFC §6.4)."""
        self._turn.read_as_trace()

    def drop(self) -> None:
        """Give the stream up unread: it is closed in the background, as its
        collection would close it."""
        if not self._done and not self._reading:
            self._done = True
            self._dropped.detach()
            _close_dropped(self._turn, asyncio.get_running_loop())

    async def aclose(self) -> None:
        if self._done:
            return
        if self._reading:
            raise RuntimeError("aclose(): asynchronous generator is already running")
        self._done = True
        self._dropped.detach()
        await self._turn.close_stream()
