"""Tests for delivery worker execution and loop."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery, run_worker_loop


def _mock_kit() -> MagicMock:
    kit = MagicMock()
    kit.process_inbound = AsyncMock()
    kit.get_channel = MagicMock(return_value=None)
    kit.store.list_bindings = AsyncMock(return_value=[])
    return kit


class TestExecuteDelivery:
    async def test_calls_strategy_deliver(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(
            room_id="r1",
            content="hello",
            strategy={"type": "immediate", "params": {}},
        )

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()
            mock_deser.return_value = mock_strategy
            await execute_delivery(kit, item)

            mock_strategy.deliver.assert_called_once()
            ctx = mock_strategy.deliver.call_args[0][0]
            assert ctx.room_id == "r1"
            assert ctx.content == "hello"

    async def test_passes_channel_id_and_metadata(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(
            room_id="r1",
            content="msg",
            channel_id="ch1",
            metadata={"key": "val"},
        )

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()
            mock_deser.return_value = mock_strategy
            await execute_delivery(kit, item)

            ctx = mock_strategy.deliver.call_args[0][0]
            assert ctx.channel_id == "ch1"
            assert ctx.metadata == {"key": "val"}


class TestRunWorkerLoop:
    async def test_acks_on_success(self) -> None:
        kit = _mock_kit()
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello")
        await backend.enqueue(item)

        task = asyncio.create_task(run_worker_loop(backend, kit, "w1", poll_timeout=0.1))
        # Give the loop time to process
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert await backend.get_queue_depth() == 0
        assert item.id not in backend._in_flight

    async def test_nacks_on_failure(self) -> None:
        kit = _mock_kit()
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(room_id="r1", content="hello", max_retries=2)
        await backend.enqueue(item)

        with patch("roomkit.delivery.worker.execute_delivery", side_effect=RuntimeError("boom")):
            task = asyncio.create_task(run_worker_loop(backend, kit, "w1", poll_timeout=0.1))
            # Give enough time for retries to exhaust
            await asyncio.sleep(0.5)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # After max_retries=2, item should be dead-lettered
        dl = await backend.get_dead_letter_items()
        assert len(dl) == 1
        assert dl[0].status == DeliveryItemStatus.DEAD_LETTER

    async def test_loop_cancellation(self) -> None:
        kit = _mock_kit()
        backend = InMemoryDeliveryBackend()

        task = asyncio.create_task(run_worker_loop(backend, kit, "w1", poll_timeout=0.1))
        await asyncio.sleep(0.1)
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled()
