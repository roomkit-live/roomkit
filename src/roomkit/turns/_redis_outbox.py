"""What a process writes to the other processes' inboxes on the Redis turn backend.

Each destination has one writer task: what is posted to a node reaches its
inbox in the order it was posted, a batch per round trip, whatever task posted
it and whether that task is cancelled meanwhile. Posting never waits, so a
message is numbered and queued in one step, and two messages of one turn can
never reach the other side in the other order (RFC §10.6.3). A message of a
batch Redis refuses is written again a moment later, before what was posted
after it, when it may be (an answer, work after a turn, written until they
are); one that may not (a turn's message, whose order its turn counts) is
not, and its sender is told: its turn fails rather than wait for what will
not come. A
writer left idle goes, and the next message to its node starts another;
closing writes what is queued (a turn's ``end`` among it) within a bound, and
tells the senders of the rest.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.turns.broker import BrokerMessage, TurnBroker

logger = logging.getLogger("roomkit.turns")

_BATCH = 256
# How long a writer waits before writing again what Redis refused.
RETRY_DELAY = 1.0

Failed = Callable[[Exception], None]
"""Told when Redis refused the batch a message was in, for its last write."""
Written = Callable[[], None]
"""Told once Redis took the batch a message was in."""

# A message, who to tell when it cannot be written, its writes left (None:
# until it is written), when it is written again no more (None: never), and
# who to tell once it is written.
_Queued = tuple["BrokerMessage", Failed | None, int | None, float | None, Written | None]
# Put on each queue on closing: its writer stops once it wrote what came before.
_CLOSING = None


class Outbox:
    """The writes of this process, in order per destination node.

    Args:
        broker: What carries the messages to the nodes' inboxes.
        ttl_ms: How long a writer left idle stays.
        retry_delay: How long a writer waits before writing again what
            Redis refused.
    """

    def __init__(self, broker: TurnBroker, ttl_ms: int, retry_delay: float = RETRY_DELAY) -> None:
        self._broker = broker
        self._ttl_ms = ttl_ms
        self._retry_delay = retry_delay
        self._queues: dict[str, asyncio.Queue[_Queued | None]] = {}
        self._writers: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def post(
        self,
        node: str,
        envelope: BrokerMessage,
        failed: Failed | None = None,
        *,
        writes: int | None = 1,
        within: float | None = None,
        written: Written | None = None,
    ) -> None:
        """Queue *envelope* for *node*'s inbox, after what was posted to it
        before, written up to *writes* times while Redis refuses it (None:
        until it is written, while the outbox runs), for at most *within*
        seconds when given; *failed* is told if it is never written, and
        *written* once it is."""
        if self._closed:
            if failed is not None:
                failed(ConnectionError("The turn backend closed"))
            return
        queue = self._queues.get(node)
        if queue is None:
            queue = self._queues[node] = asyncio.Queue()
            # A context of its own: the writes belong to no turn's trace.
            self._writers[node] = asyncio.get_running_loop().create_task(
                self._write(node, queue),
                name=f"roomkit-turns-outbox-{node}",
                context=contextvars.Context(),
            )
        deadline = None if within is None else time.monotonic() + within
        queue.put_nowait((envelope, failed, writes, deadline, written))

    async def close(self, drain: float) -> None:
        """Take no more messages, write what is queued within *drain*
        seconds, and tell the senders of what is left."""
        self._closed = True
        for queue in self._queues.values():
            queue.put_nowait(_CLOSING)
        writers = list(self._writers.values())
        if writers:
            _, late = await asyncio.wait(writers, timeout=drain)
            for writer in late:
                writer.cancel()
            await asyncio.gather(*writers, return_exceptions=True)
        left = ConnectionError("The turn backend closed before writing it")
        for queue in self._queues.values():
            _tell([item for item in _emptied(queue) if item is not _CLOSING], left)
        self._writers.clear()

    async def _write(self, node: str, queue: asyncio.Queue[_Queued | None]) -> None:
        """Write what *node* is posted, in order, until the closing mark or
        an idle while; what Redis refused and may be written again goes
        first, a moment later."""
        again: list[_Queued] = []
        closing = False
        try:
            while again or not closing:
                if again:
                    await asyncio.sleep(self._retry_delay)
                    start, again = again, []
                else:
                    try:
                        async with asyncio.timeout(self._ttl_ms / 1000):
                            first = await queue.get()
                    except TimeoutError:
                        if queue.empty():
                            # Idle: the writer goes; the next message starts another.
                            del self._queues[node], self._writers[node]
                            return
                        continue
                    if first is _CLOSING:
                        closing = True
                        continue
                    start = [first]
                batch, closing = _batch(start, queue, closing)
                again = await self._send(node, batch)
        except asyncio.CancelledError:
            _tell(again, ConnectionError("The turn backend closed before writing it"))
            raise

    async def _send(self, node: str, batch: list[_Queued]) -> list[_Queued]:
        """Write *batch*: what Redis refused and has writes left, to write
        again; the senders of the rest are told."""
        try:
            await self._broker.post(node, [envelope for envelope, *_ in batch])
        except asyncio.CancelledError:
            _tell(batch, ConnectionError("The turn backend closed before writing it"))
            raise
        except Exception as exc:
            now = time.monotonic()
            again = [
                (envelope, failed, None if left is None else left - 1, deadline, written)
                for envelope, failed, left, deadline, written in batch
                if _written_again(left, deadline, now)
            ]
            logger.warning(
                "Writing %d turn message(s) to %s failed (%d written again): %s",
                len(batch),
                node,
                len(again),
                exc,
            )
            _tell([item for item in batch if not _written_again(item[2], item[3], now)], exc)
            return again
        _tell_written(batch)
        return []


def _batch(
    start: list[_Queued], queue: asyncio.Queue[_Queued | None], closing: bool
) -> tuple[list[_Queued], bool]:
    """*start* and up to a batch of what follows it in *queue* (what is
    written again never keeps the rest from its one write), and whether the
    closing mark came (nothing is taken after it)."""
    batch = list(start)
    taken = 0
    while not closing and taken < _BATCH and not queue.empty():
        item = queue.get_nowait()
        if item is _CLOSING:
            closing = True
        else:
            batch.append(item)
            taken += 1
    return batch, closing


def _written_again(left: int | None, deadline: float | None, now: float) -> bool:
    """Whether a message Redis refused is written again."""
    return (left is None or left > 1) and (deadline is None or now < deadline)


def _emptied(queue: asyncio.Queue[_Queued | None]) -> list[_Queued | None]:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _tell_written(batch: list[_Queued]) -> None:
    for envelope, *_, written in batch:
        if written is None:
            continue
        try:
            written()
        except Exception:
            logger.exception(
                "Telling the sender of a %s message it was written failed", envelope.kind
            )


def _tell(batch: list[_Queued], error: Exception) -> None:
    for envelope, failed, *_ in batch:
        if failed is None:
            continue
        try:
            failed(error)
        except Exception:
            logger.exception("Telling the sender of a %s message it failed failed", envelope.kind)
