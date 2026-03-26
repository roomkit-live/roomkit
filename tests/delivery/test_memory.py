"""Tests for InMemoryDeliveryBackend."""

from __future__ import annotations

from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend


class TestEnqueueDequeue:
    async def test_basic_flow(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello")
        await backend.enqueue(item)

        assert await backend.get_queue_depth() == 1

        items = await backend.dequeue("w1", timeout=1.0)
        assert len(items) == 1
        assert items[0].id == item.id
        assert items[0].status == DeliveryItemStatus.IN_PROGRESS
        assert items[0].worker_id == "w1"

    async def test_dequeue_timeout_returns_empty(self) -> None:
        backend = InMemoryDeliveryBackend()
        items = await backend.dequeue("w1", timeout=0.1)
        assert items == []

    async def test_batch_dequeue(self) -> None:
        backend = InMemoryDeliveryBackend()
        for i in range(5):
            await backend.enqueue(DeliveryItem(room_id="r1", content=f"msg-{i}"))

        items = await backend.dequeue("w1", batch_size=3, timeout=1.0)
        assert len(items) == 3
        assert await backend.get_queue_depth() == 2

    async def test_batch_dequeue_partial(self) -> None:
        backend = InMemoryDeliveryBackend()
        await backend.enqueue(DeliveryItem(room_id="r1", content="only-one"))

        items = await backend.dequeue("w1", batch_size=5, timeout=0.1)
        assert len(items) == 1


class TestAck:
    async def test_ack_removes_from_in_flight(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello")
        await backend.enqueue(item)
        items = await backend.dequeue("w1", timeout=1.0)

        await backend.ack(items[0].id)
        assert items[0].id not in backend._in_flight

    async def test_ack_unknown_id_is_noop(self) -> None:
        backend = InMemoryDeliveryBackend()
        await backend.ack("nonexistent")  # should not raise


class TestNack:
    async def test_nack_re_enqueues(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello", max_retries=3)
        await backend.enqueue(item)

        dequeued = await backend.dequeue("w1", timeout=1.0)
        await backend.nack(dequeued[0].id, error="transient failure")

        # Should be back in queue
        assert await backend.get_queue_depth() == 1
        re_dequeued = await backend.dequeue("w1", timeout=1.0)
        assert re_dequeued[0].retry_count == 1

    async def test_nack_dead_letters_after_max_retries(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello", max_retries=1)
        await backend.enqueue(item)

        dequeued = await backend.dequeue("w1", timeout=1.0)
        await backend.nack(dequeued[0].id, error="permanent failure")

        # Should be in dead letter, not re-enqueued
        assert await backend.get_queue_depth() == 0
        dl_items = await backend.get_dead_letter_items()
        assert len(dl_items) == 1
        assert dl_items[0].status == DeliveryItemStatus.DEAD_LETTER
        assert dl_items[0].error == "permanent failure"


class TestDeadLetter:
    async def test_dead_letter_moves_item(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello")
        await backend.enqueue(item)

        dequeued = await backend.dequeue("w1", timeout=1.0)
        await backend.dead_letter(dequeued[0].id, "fatal error")

        assert dequeued[0].id not in backend._in_flight
        dl_items = await backend.get_dead_letter_items()
        assert len(dl_items) == 1
        assert dl_items[0].error == "fatal error"

    async def test_get_dead_letter_items_limit(self) -> None:
        backend = InMemoryDeliveryBackend()
        for i in range(10):
            item = DeliveryItem(room_id="r1", content=f"msg-{i}")
            await backend.enqueue(item)
            dequeued = await backend.dequeue("w1", timeout=1.0)
            await backend.dead_letter(dequeued[0].id, "error")

        assert len(await backend.get_dead_letter_items(limit=3)) == 3


class TestLifecycle:
    async def test_close_without_start(self) -> None:
        backend = InMemoryDeliveryBackend()
        await backend.close()  # should not raise

    async def test_start_creates_worker_task(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        kit = MagicMock()
        kit.process_inbound = AsyncMock()
        kit.get_channel = MagicMock(return_value=None)
        kit.store.list_bindings = AsyncMock(return_value=[])

        backend = InMemoryDeliveryBackend()
        await backend.start(kit)
        assert backend._worker_task is not None
        assert not backend._worker_task.done()

        await backend.close()
        assert backend._worker_task is None

    async def test_double_start_is_noop(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        kit = MagicMock()
        kit.process_inbound = AsyncMock()
        kit.get_channel = MagicMock(return_value=None)
        kit.store.list_bindings = AsyncMock(return_value=[])

        backend = InMemoryDeliveryBackend()
        await backend.start(kit)
        first_task = backend._worker_task
        await backend.start(kit)  # second start
        assert backend._worker_task is first_task  # same task, not leaked
        await backend.close()

    async def test_close_re_enqueues_in_flight(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello")
        await backend.enqueue(item)

        dequeued = await backend.dequeue("w1", timeout=1.0)
        assert len(dequeued) == 1
        assert dequeued[0].id in backend._in_flight
        assert await backend.get_queue_depth() == 0

        await backend.close()

        # Item should be back in queue, not lost
        assert await backend.get_queue_depth() == 1
        assert len(backend._in_flight) == 0

    async def test_nack_queue_full_dead_letters(self) -> None:
        backend = InMemoryDeliveryBackend(max_queue_size=1)
        # Fill the queue
        await backend.enqueue(DeliveryItem(room_id="r1", content="filler"))
        # Dequeue and enqueue another to keep queue full
        item = DeliveryItem(room_id="r1", content="retry-me", max_retries=5)
        backend._in_flight[item.id] = item

        await backend.nack(item.id, error="transient")

        # Queue was full, so item should be dead-lettered
        dl = await backend.get_dead_letter_items()
        assert len(dl) == 1
        assert dl[0].error == "queue full on retry"
