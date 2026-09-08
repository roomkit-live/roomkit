"""Redis delivery recovery contracts. Set REDIS_URL to run against Redis 6.2+."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch
from uuid import uuid4

import pytest

from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.redis import RedisDeliveryBackend

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")


@pytest.fixture
async def backends():
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    first = RedisDeliveryBackend(client=client, stream_prefix=prefix, claim_idle_seconds=0.3)
    second = RedisDeliveryBackend(client=client, stream_prefix=prefix, claim_idle_seconds=0.3)
    await client.xgroup_create(first._pending_key, first._group, id="0", mkstream=True)
    try:
        yield first, second, client
    finally:
        await first.close()
        await second.close()
        await client.delete(first._pending_key, first._dl_key)
        await client.aclose()


async def test_abandoned_batch_is_reclaimed_after_worker_restart(backends) -> None:
    first, second, client = backends
    original = DeliveryItem(room_id="r1", content="persist me")
    await first.enqueue(original)
    [item] = await first.dequeue("old")
    entry = first._entry_ids[item.id]
    await first.close()
    # Simulate elapsed idle time without a timing-sensitive long sleep.
    await client.xclaim(first._pending_key, first._group, "old", 0, [entry], idle=1000)
    [recovered] = await second.dequeue("new", timeout=0)
    assert recovered.id == original.id
    assert recovered.worker_id == "new"
    await second.ack(recovered.id)
    assert await second.get_queue_depth() == 0
    assert (await client.xpending(first._pending_key, first._group))["pending"] == 0


async def test_live_worker_renews_its_whole_batch(backends) -> None:
    first, second, client = backends
    for _ in range(2):
        await first.enqueue(DeliveryItem(room_id="r1", content="slow delivery"))
    assert len(await first.dequeue("live", batch_size=2)) == 2
    # Longer than the lease: another consumer must not steal either item.
    await asyncio.sleep(0.7)
    assert await second.dequeue("other", batch_size=2, timeout=0) == []
    pending = await client.xpending_range(first._pending_key, first._group, "-", "+", 10)
    assert len(pending) == 2
    assert all(row["consumer"] == b"live" for row in pending)


@pytest.mark.parametrize("operation", ["ack", "nack", "dead_letter"])
async def test_lost_transition_response_is_safe_to_retry(backends, operation: str) -> None:
    first, second, client = backends
    await first.enqueue(DeliveryItem(room_id="r1", content="once", max_retries=3))
    [item] = await first.dequeue("old")
    real_eval = client.eval

    async def lost_response(*args, **kwargs):
        result = await real_eval(*args, **kwargs)
        if args[1] == 2:  # Transition, not lease renewal.
            raise ConnectionError("server committed but response was lost")
        return result

    args = (item.id,) if operation == "ack" else (item.id, "retry")
    with (
        patch.object(client, "eval", side_effect=lost_response),
        pytest.raises(ConnectionError),
    ):
        await getattr(first, operation)(*args)
    assert item.id in first._entry_ids
    await getattr(first, operation)(*args)
    assert item.id not in first._entry_ids
    if operation == "nack":
        [retried] = await second.dequeue("new", timeout=0)
        assert retried.retry_count == 1
        assert await second.get_queue_depth() == 1
    elif operation == "dead_letter":
        assert len(await first.get_dead_letter_items()) == 1
        assert await first.get_queue_depth() == 0
    else:
        assert await first.get_queue_depth() == 0


async def test_old_worker_cannot_ack_reclaimed_work(backends) -> None:
    first, second, client = backends
    await first.enqueue(DeliveryItem(room_id="r1", content="new owner"))
    [item] = await first.dequeue("old")
    entry = first._entry_ids[item.id]
    await client.xclaim(first._pending_key, first._group, "new", 0, [entry])
    await first.ack(item.id)
    assert await second.get_queue_depth() == 1
    pending = await client.xpending_range(first._pending_key, first._group, "-", "+", 10)
    assert pending[0]["consumer"] == b"new"


async def test_nack_exhaustion_moves_once_to_dead_letter(backends) -> None:
    first, second, _client = backends
    await first.enqueue(DeliveryItem(room_id="r1", content="bad", max_retries=1))
    [item] = await first.dequeue("old")
    await first.nack(item.id, "failed")
    assert await second.get_queue_depth() == 0
    [dead] = await second.get_dead_letter_items()
    assert dead.id == item.id
    assert dead.retry_count == 1
    assert dead.error == "failed"
