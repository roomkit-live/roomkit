"""The reader of a process's inbox for the turn backend (RFC §10.6).

Each process that runs a
:class:`~roomkit.turns.distributed.DistributedTurnBackend` has one inbox,
which every other process writes to (through its outbox,
:mod:`~roomkit.turns._redis_outbox`) and it alone reads, with one blocking
read for all its turns. An entry carries one message of the turn protocol as
its JSON (a request, a frame, a control message, a channel control, a
message outside a turn, an answer), with the transport's fields only: its
kind, the node that sent it (the answer's address), the executor it is for,
and a reference tying an answer to its question. What was read is let go of.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.turns.base import TurnTimeouts
    from roomkit.turns.broker import BrokerMessage, InboxEntry, TurnBroker, TurnInbox

logger = logging.getLogger("roomkit.turns")

READ_BLOCK_S = 1.0
"""How long the server holds the inbox's blocking read at most, in seconds:
its connection waits that much longer than any other call (RFC §10.6.8)."""
_RETRY_DELAY = 1.0


def read_block(timeouts: TurnTimeouts) -> float:
    """How long the broker holds the inbox's blocking read: a second, never
    over a beat's interval, so the read cut and tried again at once lands
    inside the silence bound too (RFC §10.6.8)."""
    return min(READ_BLOCK_S, timeouts.heartbeat_interval)


class Inbox:
    """The reader of this process's inbox.

    Args:
        broker: What carries the messages to this process's inbox.
        node: This process's node id.
        handle: Called with each message read, in order, on the reader's
            task; it must not wait.
        block: How long each read waits at most, in seconds.
    """

    def __init__(
        self,
        broker: TurnBroker,
        node: str,
        handle: Callable[[BrokerMessage], None],
        *,
        block: float = READ_BLOCK_S,
    ) -> None:
        self._broker = broker
        self._node = node
        self._handle = handle
        self._block = block
        self._inbox: TurnInbox | None = None
        self._reader: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start reading this process's inbox, once."""
        if self._reader is None:
            self._reader = asyncio.get_running_loop().create_task(
                self._read(), name=f"roomkit-turns-inbox-{self._node}"
            )

    async def close(self) -> None:
        """Stop reading; what is left in the inbox expires with it."""
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        if self._inbox is not None:
            await self._inbox.close()
            self._inbox = None

    async def _read(self) -> None:
        """Read the inbox from its start, each entry handed on and let go of."""
        last = ""
        while True:
            ids = []
            for entry in await self._entries_after(last):
                last = entry.id
                ids.append(entry.id)
                self._hand_on(entry.message)
            if ids:
                await self._acknowledge(ids)

    async def _entries_after(self, last: str) -> list[InboxEntry]:
        """The entries after *last*, once read. A read its bound cut is read
        again at once, on a new connection, so it lands inside the silence
        bound (RFC §10.6.8); another failure, or a second cut in a row, is
        retried after a delay. The entries wait where they are meanwhile."""
        again = True
        while True:
            try:
                if self._inbox is None:
                    self._inbox = await self._broker.inbox(self._node, self._block)
                return await self._inbox.read(last)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Reading the turn inbox %s failed; retrying", self._node, exc_info=True
                )
                if again and isinstance(exc, TimeoutError):
                    again = False
                    continue
                again = True
                await asyncio.sleep(_RETRY_DELAY)

    def _hand_on(self, envelope: BrokerMessage | None) -> None:
        if envelope is None:
            logger.warning("A malformed entry in the turn inbox was dropped")
            return
        try:
            self._handle(envelope)
        except Exception:
            logger.exception("A %s message of the turn inbox could not be handled", envelope.kind)

    async def _acknowledge(self, ids: list[str]) -> None:
        try:
            await self._broker.acknowledge(self._node, ids)
        except Exception:
            logger.debug("Deleting read entries of %s failed", self._node, exc_info=True)
