"""Tests for delivery worker execution and loop."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery, fire_delivery_hooks, run_worker_loop
from roomkit.models.enums import HookTrigger


def _mock_kit() -> MagicMock:
    kit = MagicMock()
    kit.process_inbound = AsyncMock()
    kit.get_channel = MagicMock(return_value=None)
    kit.store.list_bindings = AsyncMock(return_value=[])
    kit._build_context = AsyncMock(return_value=MagicMock())
    kit.hook_engine = MagicMock()
    kit.hook_engine.run_async_hooks = AsyncMock()
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


class TestDeliveryHooks:
    async def test_before_deliver_fires_before_strategy(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(room_id="r1", content="hello")

        call_order: list[str] = []
        kit.hook_engine.run_async_hooks = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("hook")
        )

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()

            async def track_deliver(ctx: object) -> None:
                call_order.append("deliver")

            mock_strategy.deliver = track_deliver
            mock_deser.return_value = mock_strategy
            await execute_delivery(kit, item)

        # hook (BEFORE) → deliver → hook (AFTER)
        assert call_order == ["hook", "deliver", "hook"]

    async def test_after_deliver_fires_even_on_failure(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(room_id="r1", content="hello")

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()
            mock_strategy.deliver = AsyncMock(side_effect=RuntimeError("boom"))
            mock_deser.return_value = mock_strategy

            with contextlib.suppress(RuntimeError):
                await execute_delivery(kit, item)

        # Both hooks should have been called
        assert kit.hook_engine.run_async_hooks.call_count == 2
        # Second call (AFTER_DELIVER) should have error in metadata
        after_call = kit.hook_engine.run_async_hooks.call_args_list[1]
        trigger = after_call[0][1]
        assert trigger == HookTrigger.AFTER_DELIVER

    async def test_after_deliver_metadata_contains_error(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(room_id="r1", content="hello")

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()
            mock_strategy.deliver = AsyncMock(side_effect=ValueError("bad"))
            mock_deser.return_value = mock_strategy

            with contextlib.suppress(ValueError):
                await execute_delivery(kit, item)

        after_event = kit.hook_engine.run_async_hooks.call_args_list[1][0][2]
        assert after_event.metadata.get("error") == "bad"

    async def test_hook_failure_does_not_abort_delivery(self) -> None:
        kit = _mock_kit()
        kit._build_context = AsyncMock(side_effect=RuntimeError("no room"))
        item = DeliveryItem(room_id="r1", content="hello")

        with patch("roomkit.delivery.worker.deserialize_strategy") as mock_deser:
            mock_strategy = AsyncMock()
            mock_deser.return_value = mock_strategy
            # Should not raise — hook failure is swallowed
            await execute_delivery(kit, item)
            mock_strategy.deliver.assert_called_once()

    async def test_fire_hooks_directly(self) -> None:
        kit = _mock_kit()
        item = DeliveryItem(room_id="r1", content="test", channel_id="ch1")

        await fire_delivery_hooks(kit, item, HookTrigger.BEFORE_DELIVER)

        kit._build_context.assert_called_once_with("r1")
        kit.hook_engine.run_async_hooks.assert_called_once()
        call_args = kit.hook_engine.run_async_hooks.call_args[0]
        assert call_args[0] == "r1"
        assert call_args[1] == HookTrigger.BEFORE_DELIVER
        event = call_args[2]
        assert event.metadata["channel_id"] == "ch1"
        assert event.metadata["delivery_item_id"] == item.id


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
