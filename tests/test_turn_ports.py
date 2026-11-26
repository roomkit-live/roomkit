"""The contract of the two ports of a turn backend across processes (RFC
§10.6.7, §10.6.8), kept by the library's own brokers and registries: the
in-memory pair, and Redis's where REDIS_URL names a server."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest

from roomkit.testing.turn_ports import check_turn_broker, check_turn_registry
from roomkit.turns import TurnTimeouts
from roomkit.turns._memory_ports import InMemoryTurnBroker, InMemoryTurnRegistry
from roomkit.turns._redis_broker import RedisTurnBroker
from roomkit.turns._redis_clients import RedisClients, redis_clients
from roomkit.turns._redis_keys import Keys
from roomkit.turns._redis_registry import RedisTurnRegistry
from roomkit.turns.broker import BrokerMessage


async def test_the_broker_of_this_process_keeps_the_port() -> None:
    await check_turn_broker(InMemoryTurnBroker())


async def test_the_registry_of_this_process_keeps_the_port() -> None:
    await check_turn_registry(InMemoryTurnRegistry())


class _Reversing(InMemoryTurnBroker):
    """Adds the messages of one post in the other order."""

    async def post(self, node: str, messages: Sequence[BrokerMessage]) -> None:
        await super().post(node, list(reversed(messages)))


async def test_a_broker_that_loses_the_order_of_a_post_fails_the_check() -> None:
    with pytest.raises(AssertionError, match="in the order posted"):
        await check_turn_broker(_Reversing())


class _Unguarded(InMemoryTurnRegistry):
    """Writes every beat, an older one too."""

    async def beat(self, beat: object) -> None:
        self._generations.clear()
        await super().beat(beat)  # type: ignore[arg-type]


async def test_a_registry_that_writes_an_older_beat_fails_the_check() -> None:
    with pytest.raises(AssertionError, match="older beat"):
        await check_turn_registry(_Unguarded())


@pytest.fixture
async def redis() -> AsyncIterator[tuple[RedisClients, Keys, int]]:
    """The connections of a backend on the Redis server REDIS_URL names,
    under keys of the test's own, deleted after it."""
    url = os.environ.get("REDIS_URL")
    if not url:
        pytest.skip("REDIS_URL not set")
    timeouts = TurnTimeouts()
    clients = redis_clients(url, timeouts)
    keys = Keys(f"roomkit:test:{uuid4().hex[:12]}")
    yield clients, keys, int(timeouts.heartbeat_timeout * 3000)
    left = [key async for key in clients.pooled.scan_iter(f"{keys.prefix}:*")]
    if left:
        await clients.pooled.delete(*left)
    await clients.pooled.aclose()
    await clients.inbox.aclose()


async def test_the_broker_on_redis_keeps_the_port(redis: tuple[RedisClients, Keys, int]) -> None:
    await check_turn_broker(RedisTurnBroker(*redis))


async def test_the_registry_on_redis_keeps_the_port(redis: tuple[RedisClients, Keys, int]) -> None:
    await check_turn_registry(RedisTurnRegistry(*redis))
