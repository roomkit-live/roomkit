"""Where the turn backends of one test meet (RFC §10.6): several backends in
one process stand for several processes, on an in-memory broker and registry
of the test's own, and on the Redis server REDIS_URL names."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from roomkit.turns import TurnTimeouts
from roomkit.turns._memory_ports import InMemoryTurnBroker, InMemoryTurnRegistry
from roomkit.turns.distributed import DistributedTurnBackend
from roomkit.turns.redis import RedisTurnBackend

REDIS = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")
ACROSS = ["in-memory", pytest.param("redis", marks=REDIS)]
"""The services a test's nodes meet on: parametrize a test with these."""


class Nodes:
    """The nodes of one test's backend, on the services *services* names."""

    def __init__(self, services: str) -> None:
        self.prefix = f"roomkit:test:{uuid4().hex[:12]}"
        self._pair = (
            (InMemoryTurnBroker(), InMemoryTurnRegistry()) if services == "in-memory" else None
        )

    def node(self, timeouts: TurnTimeouts) -> DistributedTurnBackend:
        """One more node: a process of its own, as the others see it."""
        if self._pair is not None:
            return DistributedTurnBackend(*self._pair, timeouts=timeouts)
        return RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=self.prefix, timeouts=timeouts)

    async def forget(self) -> None:
        """Delete what the nodes wrote on the server; an in-memory pair goes
        with the test."""
        if self._pair is not None:
            return
        client = pytest.importorskip("redis.asyncio").from_url(os.environ["REDIS_URL"])
        keys = [key async for key in client.scan_iter(f"{self.prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()
