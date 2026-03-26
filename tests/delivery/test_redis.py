"""Tests for RedisDeliveryBackend (delivery/redis.py)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus

# -- Mock Redis -----------------------------------------------------------


def _build_mock_redis_module() -> MagicMock:
    """Build a mock redis.asyncio module."""
    mod = MagicMock()
    mod.from_url = MagicMock(return_value=_build_mock_client())
    return mod


def _build_mock_client() -> AsyncMock:
    """Build a mock redis.asyncio.Redis client."""
    client = AsyncMock()
    client.xadd = AsyncMock(return_value=b"1679001234567-0")
    client.xreadgroup = AsyncMock(return_value=[])
    client.xack = AsyncMock(return_value=1)
    client.xdel = AsyncMock(return_value=1)
    client.xlen = AsyncMock(return_value=0)
    client.xrevrange = AsyncMock(return_value=[])
    client.xgroup_create = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _make_backend(client: AsyncMock | None = None):
    """Create a RedisDeliveryBackend with mocked redis."""
    mock_mod = _build_mock_redis_module()
    mock_client = client or _build_mock_client()

    with patch.dict(sys.modules, {"redis": MagicMock(), "redis.asyncio": mock_mod}):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.delivery.redis")
        importlib.reload(mod)
        backend = mod.RedisDeliveryBackend(client=mock_client)

    return backend, mock_client


def _make_item(**kwargs) -> DeliveryItem:
    return DeliveryItem(room_id="r1", content="hello", **kwargs)


# -- Tests ----------------------------------------------------------------


class TestEnqueue:
    async def test_calls_xadd(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        await backend.enqueue(item)

        client.xadd.assert_called_once()
        call_args = client.xadd.call_args
        assert call_args[0][0] == "roomkit:delivery:pending"
        data = call_args[0][1]["data"]
        restored = DeliveryItem.model_validate_json(data)
        assert restored.room_id == "r1"
        assert restored.content == "hello"

    async def test_sets_status_pending(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        item.status = DeliveryItemStatus.FAILED
        await backend.enqueue(item)
        assert item.status == DeliveryItemStatus.PENDING


class TestDequeue:
    async def test_returns_empty_on_timeout(self) -> None:
        backend, client = _make_backend()
        client.xreadgroup = AsyncMock(return_value=[])
        items = await backend.dequeue("w1", timeout=0.1)
        assert items == []

    async def test_parses_items(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(
            return_value=[[b"roomkit:delivery:pending", [(b"123-0", entry_data)]]]
        )

        items = await backend.dequeue("w1")
        assert len(items) == 1
        assert items[0].room_id == "r1"
        assert items[0].status == DeliveryItemStatus.IN_PROGRESS
        assert items[0].worker_id == "w1"

    async def test_calls_xreadgroup_with_correct_args(self) -> None:
        backend, client = _make_backend()
        client.xreadgroup = AsyncMock(return_value=[])
        await backend.dequeue("w1", batch_size=3, timeout=2.0)

        client.xreadgroup.assert_called_once_with(
            backend._group,
            "w1",
            {"roomkit:delivery:pending": ">"},
            count=3,
            block=2000,
        )

    async def test_tracks_entry_id(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"stream", [(b"456-0", entry_data)]]])

        items = await backend.dequeue("w1")
        assert items[0].id in backend._entry_ids
        assert backend._entry_ids[items[0].id] == "456-0"

    async def test_parses_string_mode_response(self) -> None:
        """redis-py with decode_responses=True returns strings, not bytes."""
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {"data": item.model_dump_json()}
        client.xreadgroup = AsyncMock(return_value=[["stream", [("123-0", entry_data)]]])

        items = await backend.dequeue("w1")
        assert len(items) == 1
        assert items[0].room_id == "r1"
        assert backend._entry_ids[items[0].id] == "123-0"


class TestAck:
    async def test_calls_xack_and_xdel(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"789-0", entry_data)]]])

        items = await backend.dequeue("w1")
        await backend.ack(items[0].id)

        client.xack.assert_called_once_with("roomkit:delivery:pending", backend._group, "789-0")
        client.xdel.assert_called_once_with("roomkit:delivery:pending", "789-0")

    async def test_cleans_up_tracking(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"100-0", entry_data)]]])

        items = await backend.dequeue("w1")
        item_id = items[0].id
        await backend.ack(item_id)
        assert item_id not in backend._entry_ids
        assert item_id not in backend._items

    async def test_unknown_id_is_noop(self) -> None:
        backend, _client = _make_backend()
        await backend.ack("nonexistent")  # should not raise


class TestNack:
    async def test_re_enqueues_when_retries_remain(self) -> None:
        backend, client = _make_backend()
        item = _make_item(max_retries=3)
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"200-0", entry_data)]]])

        items = await backend.dequeue("w1")
        await backend.nack(items[0].id, error="transient")

        # Should have xadd (re-enqueue) + xack + xdel (old entry)
        assert client.xadd.call_count == 1
        assert client.xack.call_count == 1
        assert client.xdel.call_count == 1

    async def test_dead_letters_after_max_retries(self) -> None:
        backend, client = _make_backend()
        item = _make_item(max_retries=1)
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"300-0", entry_data)]]])

        items = await backend.dequeue("w1")
        await backend.nack(items[0].id, error="permanent")

        # Should have xadd to DL stream + xack + xdel on pending
        xadd_calls = client.xadd.call_args_list
        assert len(xadd_calls) == 1
        assert xadd_calls[0][0][0] == "roomkit:delivery:dead_letter"


class TestDeadLetter:
    async def test_moves_to_dl_stream(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"400-0", entry_data)]]])

        items = await backend.dequeue("w1")
        await backend.dead_letter(items[0].id, "fatal error")

        client.xadd.assert_called_once()
        call_args = client.xadd.call_args
        assert call_args[0][0] == "roomkit:delivery:dead_letter"
        assert call_args[1]["maxlen"] == 10_000
        assert call_args[1]["approximate"] is True

    async def test_cleans_up_tracking(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"500-0", entry_data)]]])

        items = await backend.dequeue("w1")
        item_id = items[0].id
        assert item_id in backend._entry_ids
        assert item_id in backend._items

        await backend.dead_letter(item_id, "fatal")
        assert item_id not in backend._entry_ids
        assert item_id not in backend._items


class TestGetQueueDepth:
    async def test_calls_xlen(self) -> None:
        backend, client = _make_backend()
        client.xlen = AsyncMock(return_value=42)
        depth = await backend.get_queue_depth()
        assert depth == 42
        client.xlen.assert_called_once_with("roomkit:delivery:pending")


class TestGetDeadLetterItems:
    async def test_calls_xrevrange(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        client.xrevrange = AsyncMock(
            return_value=[(b"500-0", {b"data": item.model_dump_json().encode()})]
        )

        items = await backend.get_dead_letter_items(limit=10)
        assert len(items) == 1
        assert items[0].room_id == "r1"
        client.xrevrange.assert_called_once_with(
            "roomkit:delivery:dead_letter", "+", "-", count=10
        )


class TestLifecycle:
    async def test_start_creates_consumer_group(self) -> None:
        backend, client = _make_backend()
        kit = MagicMock()
        await backend.start(kit)

        client.xgroup_create.assert_called_once_with(
            "roomkit:delivery:pending",
            "roomkit-workers",
            id="0",
            mkstream=True,
        )
        assert backend._worker_task is not None
        backend._worker_task.cancel()

    async def test_start_idempotent(self) -> None:
        backend, client = _make_backend()
        client.xgroup_create = AsyncMock(side_effect=Exception("BUSYGROUP"))
        kit = MagicMock()
        await backend.start(kit)  # should not raise
        assert backend._worker_task is not None
        backend._worker_task.cancel()

    async def test_double_start_is_noop(self) -> None:
        backend, client = _make_backend()
        kit = MagicMock()
        await backend.start(kit)
        first_task = backend._worker_task
        await backend.start(kit)  # second start
        assert backend._worker_task is first_task  # same task, not leaked
        await backend.close()

    async def test_close_stops_worker(self) -> None:
        backend, client = _make_backend()
        kit = MagicMock()
        await backend.start(kit)
        assert backend._worker_task is not None

        await backend.close()
        assert backend._worker_task is None

    async def test_close_closes_owned_connection(self) -> None:
        client = _build_mock_client()
        mock_mod = _build_mock_redis_module()

        with patch.dict(
            sys.modules,
            {"redis": MagicMock(), "redis.asyncio": mock_mod},
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.delivery.redis")
            importlib.reload(mod)
            # No client injected → backend owns the connection
            backend = mod.RedisDeliveryBackend(url="redis://localhost")

        backend._client = client
        backend._owns_client = True
        await backend.close()
        client.aclose.assert_called_once()

    async def test_close_preserves_injected_client(self) -> None:
        backend, client = _make_backend()
        assert not backend._owns_client
        await backend.close()
        client.aclose.assert_not_called()


class TestImportError:
    def test_helpful_message(self) -> None:
        with (
            patch.dict(sys.modules, {"redis": None, "redis.asyncio": None}),
            pytest.raises(ImportError, match="pip install roomkit\\[redis\\]"),
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.delivery.redis")
            importlib.reload(mod)
            mod.RedisDeliveryBackend()
