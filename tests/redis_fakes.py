"""Shared fakes for testing Redis-backed backends without a Redis server."""

from __future__ import annotations

import asyncio


class FakePubSub:
    """Queue-backed PubSub fake.

    A real class rather than AsyncMock: an AsyncMock ``get_message``
    resolves without yielding to the event loop, which starves it and
    hangs the reader-loop tests.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False
        self.confirm_subscribes = True

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)
        if self.confirm_subscribes:
            self._queue.put_nowait(
                {"type": "subscribe", "pattern": None, "channel": channel.encode(), "data": 1}
            )

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed.append(channel)

    async def aclose(self) -> None:
        self.closed = True

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float = 0.0
    ) -> dict | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def inject(self, channel: str | bytes, data: str | bytes) -> None:
        """Simulate a message arriving from Redis."""
        self._queue.put_nowait(
            {"type": "message", "pattern": None, "channel": channel, "data": data}
        )
