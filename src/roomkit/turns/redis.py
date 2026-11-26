"""Turns carried between processes by Redis (RFC §10.6).

The backend across processes
(:class:`~roomkit.turns.distributed.DistributedTurnBackend`), its two
services kept by a Redis server: each process's inbox is a stream every
other process writes to, and the beats, each room's holder and the presence
last published are kept by scripts on the server's clock.

Requires ``pip install roomkit[redis]`` and a store every process shares
(RFC §10.6.9). Redis is the trust boundary: whoever writes to it can act as
any executor or owner, so restrict it (ACL, TLS) as the rooms it carries.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.turns._redis_broker import RedisTurnBroker
from roomkit.turns._redis_clients import redis_clients
from roomkit.turns._redis_keys import Keys
from roomkit.turns._redis_registry import RedisTurnRegistry
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.distributed import DistributedTurnBackend

_DEFAULT_URL = "redis://localhost:6379"


class RedisTurnBackend(DistributedTurnBackend):
    """Turns carried between processes by Redis (RFC §10.6).

    One backend per process, given to the kit (``RoomKit(turn_backend=...)``)
    and to the executors that process serves (``await backend.serve(executor)``).
    An executor whose channels are given as a collection
    (``TurnExecutor(build, hosts={"meteo"})``) takes turns from every
    process; one given a predicate, as the kit's own, from its process only.

    Args:
        url: The Redis server (``redis://localhost:6379`` by default). With
            *client*, still required: the process's beat thread connects
            with it.
        client: A ``redis.asyncio`` client to use for the backend's writes
            and reads instead of connecting; its calls keep its own timeouts.
        key_prefix: The prefix of every key the backend writes.
        timeouts: The heartbeat bounds (RFC §10.6.8), kept by the kit given
            this backend: give every process, and every executor's
            ``TurnExecutor(timeouts=...)``, the same.

    Raises:
        ValueError: *client* without *url*.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Any = None,
        key_prefix: str = "roomkit:turns",
        timeouts: TurnTimeouts | None = None,
        _loop_back: bool = False,
    ) -> None:
        if client is not None and url is None:
            raise ValueError(
                "RedisTurnBackend(client=...) needs url= too: its beat connects with it"
            )
        timeouts = timeouts or TurnTimeouts()
        clients = redis_clients(url or _DEFAULT_URL, timeouts, client)
        self._owns_client = client is None
        self._client = clients.pooled
        # Redis is reached through these two alone: what carries the
        # backend's messages, and what its processes share.
        keys = Keys(key_prefix)
        ttl_ms = int(timeouts.heartbeat_timeout * 3000)
        self._redis_broker = RedisTurnBroker(clients, keys, ttl_ms)
        super().__init__(
            self._redis_broker,
            RedisTurnRegistry(clients, keys, ttl_ms),
            timeouts=timeouts,
            _loop_back=_loop_back,
            _timeout_errors=clients.timeout_errors,
        )

    async def _shut(self, until: float | None) -> set[asyncio.Task[Any]]:
        """Close the backend, then the connections it opened."""
        if self._closed:
            return set()
        left = await super()._shut(until)
        await self._redis_broker.close()
        if self._owns_client:
            await self._client.aclose()
        return left
