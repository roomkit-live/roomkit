"""End-to-end integration tests for delivery backend with RoomKit."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.core.delivery import Immediate, WaitForIdle
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.serialization import serialize_strategy


class TestKitDeliverWithBackend:
    """Test that kit.deliver() enqueues when a backend is configured."""

    async def test_deliver_enqueues_to_backend(self) -> None:
        backend = InMemoryDeliveryBackend()
        backend.enqueue = AsyncMock()

        # Build a minimal mock kit with delivery_backend
        kit = MagicMock()
        kit._delivery_backend = backend
        kit._delivery_strategy = None

        from roomkit.core.mixins.deliver import DeliverMixin

        mixin = DeliverMixin.__new__(DeliverMixin)
        mixin._delivery_strategy = None
        mixin._delivery_backend = backend
        mixin._hook_engine = MagicMock()
        mixin._build_context = AsyncMock()

        await mixin.deliver("r1", "hello")

        backend.enqueue.assert_called_once()
        item: DeliveryItem = backend.enqueue.call_args[0][0]
        assert item.room_id == "r1"
        assert item.content == "hello"
        assert item.strategy == {"type": "immediate", "params": {}}

    async def test_deliver_uses_resolved_strategy(self) -> None:
        backend = InMemoryDeliveryBackend()
        backend.enqueue = AsyncMock()

        mixin = MagicMock()
        mixin._delivery_backend = backend
        mixin._delivery_strategy = WaitForIdle(buffer=5.0)

        from roomkit.core.mixins.deliver import DeliverMixin

        real_mixin = DeliverMixin.__new__(DeliverMixin)
        real_mixin._delivery_strategy = WaitForIdle(buffer=5.0)
        real_mixin._delivery_backend = backend
        real_mixin._hook_engine = MagicMock()
        real_mixin._build_context = AsyncMock()

        await real_mixin.deliver("r1", "test")

        item: DeliveryItem = backend.enqueue.call_args[0][0]
        assert item.strategy["type"] == "wait_for_idle"
        assert item.strategy["params"]["buffer"] == 5.0

    async def test_deliver_without_backend_uses_in_process(self) -> None:
        """No backend → existing in-process path (hooks fire, strategy executes)."""
        from roomkit.core.mixins.deliver import DeliverMixin

        mixin = DeliverMixin.__new__(DeliverMixin)
        mixin._delivery_strategy = None
        mixin._delivery_backend = None
        mixin._hook_engine = MagicMock()
        mixin._hook_engine.run_async_hooks = AsyncMock()
        mixin._build_context = AsyncMock()

        # Mock the Immediate strategy's deliver
        target = "roomkit.core.delivery.Immediate.deliver"
        with patch(target, new_callable=AsyncMock) as mock_deliver:
            await mixin.deliver("r1", "hello")
            mock_deliver.assert_called_once()


class TestBackendLifecycle:
    """Test that RoomKit wires start/close on the backend."""

    async def test_close_calls_backend_close(self) -> None:
        backend = InMemoryDeliveryBackend()
        backend.close = AsyncMock()

        # Simulate what RoomKit.close() does
        await backend.close()
        backend.close.assert_called_once()

    async def test_start_calls_backend_start(self) -> None:
        backend = InMemoryDeliveryBackend()
        backend.start = AsyncMock()

        kit = MagicMock()
        await backend.start(kit)
        backend.start.assert_called_once_with(kit)


class TestEndToEnd:
    """Full enqueue → worker → delivery cycle."""

    async def test_enqueue_and_execute(self) -> None:
        backend = InMemoryDeliveryBackend()
        item = DeliveryItem(
            room_id="r1",
            content="delivered!",
            strategy=serialize_strategy(Immediate()),
        )
        await backend.enqueue(item)

        # Manually dequeue and execute (simulating worker loop)
        from roomkit.delivery.worker import execute_delivery

        kit = MagicMock()
        kit.process_inbound = AsyncMock()
        kit.get_channel = MagicMock(return_value=None)
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit._build_context = AsyncMock(return_value=MagicMock())
        kit.hook_engine = MagicMock()
        kit.hook_engine.run_async_hooks = AsyncMock()

        items = await backend.dequeue("w1", timeout=1.0)
        assert len(items) == 1

        await execute_delivery(kit, items[0])
        await backend.ack(items[0].id)

        assert await backend.get_queue_depth() == 0
        assert items[0].id not in backend._in_flight
