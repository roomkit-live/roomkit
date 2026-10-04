"""The reader of a streamed turn's response: a running tool outlives a stop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker

logger = logging.getLogger("roomkit.inbound")


class ResponseReader:
    """Reads a response stream, keeping a running tool's read out of a stop.

    A tool round yields every call's start marker, then executes the calls
    inside the read that follows the last one, then yields one end marker per
    read. So a read made while calls are open may be where they execute. A
    barge-in cancels whoever is reading, and cancelling there would abort a
    tool halfway through a side effect: that read runs as its own task
    instead, which a cancelled reader leaves running.

    :meth:`stop` then settles the round (RFC §12.2 step 13s): a round that was
    executing is let finish, a round that had not started executing never
    does, and every call it opened gets its end. A turn cancelled from outside
    still aborts a running tool, through :meth:`abandon`.
    """

    def __init__(self, stream: AsyncIterator[Any]) -> None:
        self._stream = stream
        self._open: dict[str, ToolCallStartMarker] = {}
        self._starts: dict[str, ToolCallStartMarker] = {}
        self._pending: asyncio.Future[Any] | None = None

    async def next(self) -> Any:
        """The next item of the stream; ``StopAsyncIteration`` at its end."""
        if not self._open:
            item = await anext(self._stream)
        else:
            self._pending = asyncio.ensure_future(anext(self._stream))
            item = await asyncio.shield(self._pending)
            self._pending = None
        self._track(item)
        return item

    async def stop(self) -> list[tuple[ToolCallStartMarker, ToolCallEndMarker]]:
        """Settle the open tool round without starting anything.

        Returns each call it closed with its start marker, so a caller whose
        start row was cut off can still write it before the end.

        Only the read already in flight may still execute tools; this method
        never issues one that could start them. Once that read yields its
        first end marker, the round has executed and its other ends follow
        read by read, with nothing further to execute. Stopping when no call
        is open leaves the model's next round unrequested. A call that never
        ran, or whose round failed, is closed as ``failed`` so its start row
        does not stay pending.
        """
        ends: list[tuple[ToolCallStartMarker, ToolCallEndMarker]] = []
        pending, self._pending = self._pending, None
        error = "cancelled"
        if pending is not None:
            try:
                item = await pending
                while isinstance(item, ToolCallEndMarker):
                    self._track(item)
                    if (start := self._starts.get(item.tool_id)) is not None:
                        ends.append((start, item))
                    if not self._open:
                        break
                    item = await anext(self._stream)
                else:
                    # A start marker: the round had not begun executing.
                    self._track(item)
            except StopAsyncIteration:
                pass
            except Exception:
                # The detail goes to the log; the stored row names the outcome.
                logger.exception("A running tool round failed after the response was stopped")
                error = "tool round failed"
        ends.extend(self._close_open(error))
        return ends

    async def abandon(
        self, error: str = "cancelled"
    ) -> list[tuple[ToolCallStartMarker, ToolCallEndMarker]]:
        """Cancel a running tool's read (the turn itself was cancelled or failed).

        Returns every call still open, closed as ``failed`` with *error*, so
        no start row stays pending (RFC §12.2 step 13s).
        """
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        return self._close_open(error)

    def _close_open(self, error: str) -> list[tuple[ToolCallStartMarker, ToolCallEndMarker]]:
        closed = [
            (
                start,
                ToolCallEndMarker(
                    tool_name=start.tool_name,
                    tool_id=start.tool_id,
                    arguments=start.ran_with if start.ran_with is not None else start.arguments,
                    status="failed",
                    error=error,
                    outcome="cancelled",
                ),
            )
            for start in self._open.values()
        ]
        self._open.clear()
        return closed

    def _track(self, item: Any) -> None:
        if isinstance(item, ToolCallStartMarker):
            self._open[item.tool_id] = item
            self._starts[item.tool_id] = item
        elif isinstance(item, ToolCallEndMarker):
            self._open.pop(item.tool_id, None)
