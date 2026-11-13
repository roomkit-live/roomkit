"""Redis delivery recovery contracts. Set REDIS_URL to run against Redis 6.2+."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from roomkit import RoomKit
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.redis import RedisDeliveryBackend
from roomkit.delivery.worker import _settle_item, execute_delivery
from roomkit.models.delivery import DeliveryError, DeliveryOutcome
from roomkit.store.sqlite import SQLiteStore
from tests.test_framework import SimpleChannel
from tests.test_proactive_delivery_voice import voice_room

pytestmark = pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="REDIS_URL not set")

WORKER = Path(__file__).parents[1] / "redis_delivery_worker.py"


@pytest.fixture
async def backends():
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    # Retries go out at once here; their delay has a test of its own.
    first = RedisDeliveryBackend(
        client=client, stream_prefix=prefix, claim_idle_seconds=0.3, retry_delay=0
    )
    second = RedisDeliveryBackend(
        client=client, stream_prefix=prefix, claim_idle_seconds=0.3, retry_delay=0
    )
    await client.xgroup_create(first._pending_key, first._group, id="0", mkstream=True)
    try:
        yield first, second, client
    finally:
        await first.close()
        await second.close()
        await client.delete(first._pending_key, first._dl_key, first._delayed_key)
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


async def _line_starting(process: asyncio.subprocess.Process, prefix: str) -> str:
    """The next line *process* prints that starts with *prefix*."""
    assert process.stdout is not None
    while True:
        got = await asyncio.wait_for(process.stdout.readline(), 20)
        assert got, f"the worker process ended before printing {prefix!r}"
        line = got.decode().strip()
        if line.startswith(prefix):
            return line


async def test_a_killed_worker_s_batch_is_taken_back_and_delivered_by_another_process(
    tmp_path: Path,
) -> None:
    """Queue delivery remains at least once (RFC §22.1, §22.2). A worker of
    another process (tests/redis_delivery_worker.py) takes a batch and is
    killed before it acks any item of it. Once no worker renews the items and
    their idle time passes ``claim_idle_seconds``, a worker of this process
    takes them back on its own and delivers each: an item delivered twice is
    allowed, an item lost is not."""
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    database = tmp_path / "rooms.db"
    claim_idle = 1.0
    backend = RedisDeliveryBackend(
        client=client, stream_prefix=prefix, claim_idle_seconds=claim_idle, retry_delay=0
    )
    kit = RoomKit(store=SQLiteStore(database), delivery_backend=backend)
    kit.register_channel(SimpleChannel("out"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "out")
    sent = [f"item {n}" for n in range(3)]
    worker: asyncio.subprocess.Process | None = None
    try:
        for body in sent:
            assert (await kit.deliver("r1", body, channel_id="out")).status == "queued"
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            str(WORKER),
            os.environ["REDIS_URL"],
            prefix,
            str(claim_idle),
            str(database),
            stdout=asyncio.subprocess.PIPE,
        )
        await _line_starting(worker, "delivering ")
        # Its worker holds the whole batch: every item pending, to it alone.
        held = await client.xpending(backend._pending_key, backend._group)
        assert held["pending"] == len(sent)
        assert [consumer["pending"] for consumer in held["consumers"]] == [len(sent)]

        async with kit:  # this process's worker starts
            worker.kill()
            await worker.wait()
            killed = time.monotonic()
            while (await client.xpending(backend._pending_key, backend._group))[
                "pending"
            ] or await backend.get_queue_depth():
                assert time.monotonic() - killed < claim_idle + 10
                await asyncio.sleep(0.05)
            delivered = [
                e.content.body
                for e in await kit.store.list_events("r1")
                if e.source.channel_id == "out"
            ]
            dead = await backend.get_dead_letter_items()
    finally:
        if worker is not None and worker.returncode is None:
            worker.kill()
            await worker.wait()
        await kit.close()
        await client.delete(backend._pending_key, backend._dl_key, backend._delayed_key)
        await client.aclose()

    assert sorted(set(delivered)) == sent
    assert dead == []


@pytest.mark.parametrize("uncertain", [False, True])
async def test_reclaimed_voice_item_reuses_durable_receipt_after_restart(
    backends, tmp_path, uncertain
) -> None:
    first, second, client = backends
    database = tmp_path / "voice.db"
    async with voice_room(1, store=SQLiteStore(database)) as (kit, _, provider, sessions):
        request = DeliveryItem(
            room_id="r",
            content="done",
            channel_id="voice",
            session_id=sessions[0].id,
            idempotency_key="task:42",
        )
        await first.enqueue(request)
        [item] = await first.dequeue("old")
        entry = first._entry_ids[item.id]
        send = provider.inject_text

        async def lost_confirmation(*args, **kwargs):
            await send(*args, **kwargs)
            raise ConnectionError("confirmation lost")

        if uncertain:
            with patch.object(provider, "inject_text", side_effect=lost_confirmation):
                original = await execute_delivery(kit, item)
        else:
            original = await execute_delivery(kit, item)
        assert len(provider.injected_texts) == 1
        await first.close()
    # The queue's original entry predates the outcome. A different worker and
    # store connection recover the receipt independently of that stale payload.
    await client.xclaim(first._pending_key, first._group, "old", 0, [entry], idle=1000)
    [recovered] = await second.dequeue("new", timeout=0)
    assert recovered.outcome is None
    async with RoomKit(store=SQLiteStore(database)) as restarted:
        replay = await execute_delivery(restarted, recovered)
        assert replay.status == original.status == ("unknown" if uncertain else "sent")
        assert replay.duplicate
        await _settle_item(second, recovered, replay)
    if uncertain:
        [dead] = await second.get_dead_letter_items()
        assert dead.outcome.status == "unknown" and dead.retry_count == 0
    assert await second.get_queue_depth() == 0


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
    await first.nack(item.id, "transient")
    # max_retries counts retries after the first attempt: one more goes out.
    [retried] = await second.dequeue("new", timeout=0)
    await second.nack(retried.id, "failed")
    assert await second.get_queue_depth() == 0
    [dead] = await second.get_dead_letter_items()
    assert dead.id == item.id
    assert dead.retry_count == 2
    assert dead.error == "failed"


@pytest.mark.parametrize("target", [{"addressed_to": ["agent-a"]}, {"session_id": "session-a"}])
async def test_targets_and_outcome_survive_reclaim_and_retry(backends, target) -> None:
    first, second, client = backends
    original = DeliveryItem(
        room_id="r1",
        content="external result",
        channel_id="transport",
        idempotency_key="external:1",
        metadata={"source": "external"},
        **target,
    )
    await first.enqueue(original)
    [claimed] = await first.dequeue("old")
    entry = first._entry_ids[claimed.id]
    await first.close()
    await client.xclaim(first._pending_key, first._group, "old", 0, [entry], idle=1000)
    [recovered] = await second.dequeue("new", timeout=0)
    assert recovered.addressed_to == original.addressed_to
    assert recovered.session_id == original.session_id
    assert recovered.idempotency_key == original.idempotency_key
    recovered.outcome = DeliveryOutcome(
        status="unavailable",
        reason="target_unavailable",
        error=DeliveryError(code="target_unavailable", message="offline"),
    )
    await second.nack(recovered.id, "offline")
    [retry] = await first.dequeue("retry", timeout=0)
    assert retry.retry_count == 1
    assert retry.metadata == original.metadata
    assert retry.addressed_to == original.addressed_to
    assert retry.session_id == original.session_id
    assert retry.idempotency_key == original.idempotency_key
    assert retry.outcome.status == "unavailable"
    assert retry.outcome.error.message == "offline"
    await first.ack(retry.id)


async def test_a_retry_waits_out_its_delay_on_the_server_clock() -> None:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    first = RedisDeliveryBackend(client=client, stream_prefix=prefix, retry_delay=0.5)
    second = RedisDeliveryBackend(client=client, stream_prefix=prefix, retry_delay=0.5)
    await client.xgroup_create(first._pending_key, first._group, id="0", mkstream=True)
    try:
        await first.enqueue(DeliveryItem(room_id="r1", content="later"))
        [item] = await first.dequeue("old")
        await first.nack(item.id, "offline")

        # Pending, and no worker gets it before its delay has passed.
        assert await second.get_queue_depth() == 1
        assert await second.dequeue("new", timeout=0) == []
        await asyncio.sleep(0.6)
        [retried] = await second.dequeue("new", timeout=0)
        assert retried.id == item.id and retried.retry_count == 1
        assert await second.dequeue("other", timeout=0) == []
    finally:
        await first.close()
        await second.close()
        await client.delete(first._pending_key, first._dl_key, first._delayed_key)
        await client.aclose()


async def test_a_retry_falling_due_during_a_wait_is_returned_then() -> None:
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(os.environ["REDIS_URL"])
    prefix = f"roomkit:test:{uuid4().hex}"
    backend = RedisDeliveryBackend(client=client, stream_prefix=prefix, retry_delay=0.2)
    await client.xgroup_create(backend._pending_key, backend._group, id="0", mkstream=True)
    try:
        await backend.enqueue(DeliveryItem(room_id="r1", content="soon"))
        [item] = await backend.dequeue("w1")
        await backend.nack(item.id, "offline")

        # The wait is cut short when the retry falls due, not at its timeout.
        loop = asyncio.get_running_loop()
        started = loop.time()
        [retried] = await backend.dequeue("w1", timeout=3.0)
        assert retried.id == item.id and retried.retry_count == 1
        assert loop.time() - started < 1.0
    finally:
        await backend.close()
        await client.delete(backend._pending_key, backend._dl_key, backend._delayed_key)
        await client.aclose()
