"""The frames of one turn, sent in order (RFC §10.6.4).

Text deltas the channel's stream yields without waiting between them cross as
one item: a model that delivers several chunks of one network read, or a
scripted stream, costs one frame per burst rather than one per delta. The
batch goes before any other frame of the turn, and as soon as the stream
waits; the item says how many deltas it carries, which the reader counts
against what it granted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from roomkit.turns.wire import Item

if TYPE_CHECKING:
    from roomkit.turns.wire import Frame

logger = logging.getLogger("roomkit.turns")


class FrameSender:
    """Sends a turn's frames one at a time, its text deltas by burst, and
    what a generation used among them, so the stream never waits on the link
    between a generation's last text and what the loop does next."""

    def __init__(self, send: Callable[[Frame], Awaitable[int]]) -> None:
        self._send = send
        self._lock = asyncio.Lock()
        self._kept: list[str | Item] = []
        self._flush: asyncio.Task[None] | None = None
        # A burst the link failed to carry: the next send raises it.
        self._failure: Exception | None = None

    def add_text(self, text: str) -> None:
        """Keep *text* for the burst, sent once the stream waits; a text no
        frame can carry (a lone surrogate) raises ``ValueError`` here, before
        it spoils the burst."""
        text.encode()
        self._keep(text)

    def add_item(self, item: Item) -> None:
        """Keep *item* for the burst, in its order among the text."""
        self._keep(item)

    def _keep(self, kept: str | Item) -> None:
        self._kept.append(kept)
        if self._flush is None:
            # Runs at the loop's next turn: once the stream's task waits.
            self._flush = asyncio.get_running_loop().create_task(self._flush_burst())

    async def send(self, frame: Frame) -> int:
        """Send *frame* after the text kept before it: its sequence number. A
        burst the link failed to carry raises here, its deltas being lost:
        the turn fails as it would on any frame that cannot cross."""
        async with self._lock:
            if (failure := self._failure) is not None:
                self._failure = None
                raise failure
            await self._send_kept()
            return await self._send(frame)

    async def _flush_burst(self) -> None:
        try:
            async with self._lock:
                # Text kept while a send suspended goes too: no flush was
                # scheduled for it.
                while self._kept:
                    await self._send_kept()
        except Exception as exc:
            logger.debug("A burst of text deltas could not be sent", exc_info=True)
            self._failure = exc
        finally:
            self._flush = None

    async def _send_kept(self) -> None:
        kept, self._kept = self._kept, []
        texts: list[str] = []
        for part in kept:
            if isinstance(part, str):
                texts.append(part)
                continue
            await self._send_text(texts)
            texts = []
            await self._send(part)
        await self._send_text(texts)

    async def _send_text(self, texts: list[str]) -> None:
        if not texts:
            return
        item: dict[str, object] = {"kind": "text", "text": "".join(texts)}
        if len(texts) > 1:
            item["deltas"] = len(texts)
        await self._send(Item(item=item))
