"""Tests for RedisDeliveryBackend (delivery/redis.py)."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.delivery._redis_scripts import PROMOTE
from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from tests.loop_clock import HeldClock
from tests.private_module import load_private_copy

# -- Mock Redis -----------------------------------------------------------


def _build_mock_redis_module() -> MagicMock:
    """Build a mock redis.asyncio module."""
    mod = MagicMock()
    mod.from_url = MagicMock(return_value=_build_mock_client())
    return mod


async def _eval(script: str, *args: object) -> list[int] | int:
    """PROMOTE finds no retry waiting; every transition succeeds."""
    return [0] if script == PROMOTE else 1


def _build_mock_client() -> AsyncMock:
    """Build a mock redis.asyncio.Redis client."""
    client = AsyncMock()
    client.xadd = AsyncMock(return_value=b"1679001234567-0")
    client.xreadgroup = AsyncMock(return_value=[])
    client.xautoclaim = AsyncMock(return_value=[b"0-0", [], []])
    client.eval = AsyncMock(side_effect=_eval)
    client.xack = AsyncMock(return_value=1)
    client.xdel = AsyncMock(return_value=1)
    client.xlen = AsyncMock(return_value=0)
    client.zcard = AsyncMock(return_value=0)
    client.xrevrange = AsyncMock(return_value=[])
    client.xgroup_create = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _make_backend(client: AsyncMock | None = None):
    """Create a RedisDeliveryBackend with mocked redis."""
    mock_mod = _build_mock_redis_module()
    mock_client = client or _build_mock_client()

    fakes = {"redis": MagicMock(), "redis.asyncio": mock_mod}
    with patch.dict(sys.modules, fakes):
        mod = load_private_copy("roomkit.delivery.redis", fakes)
        backend = mod.RedisDeliveryBackend(client=mock_client)

    return backend, mock_client


async def _reads_nothing(*args: object, block: int | None = None, **kwargs: object) -> list:
    """An XREADGROUP that finds nothing: it waits its *block*, as Redis does."""
    await asyncio.sleep((block or 0) / 1000)
    return []


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
        items = await backend.dequeue("w1", timeout=0.01)
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
        client.xreadgroup = AsyncMock(side_effect=_reads_nothing)
        await backend.dequeue("w1", batch_size=3, timeout=0.05)

        client.xreadgroup.assert_called_once_with(
            backend._group,
            "w1",
            {"roomkit:delivery:pending": ">"},
            count=3,
            block=50,
        )

    async def test_a_wait_longer_than_a_read_slice_reads_in_slices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No read outlasts a slice, so none outlasts its client's call bound
        on a half-open connection (RMK-868); the wait still lasts its timeout."""
        backend, client = _make_backend()
        # The private copy's own globals: what its dequeue reads.
        monkeypatch.setitem(type(backend).dequeue.__globals__, "_READ_SLICE_S", 0.05)
        # The wait runs on the loop's clock, held: each read waits its block
        # by moving it, as Redis would; a pause of the worker spends none of
        # the wait (RMK-1714).
        clock = HeldClock(monkeypatch)

        async def reads_nothing(*args: object, block: int | None = None, **kwargs: object) -> list:
            await clock.past((block or 0) / 1000)
            return []

        client.xreadgroup = AsyncMock(side_effect=reads_nothing)
        clock.hold()
        wait = asyncio.ensure_future(backend.dequeue("w1", timeout=0.175))
        await clock.first(wait)
        clock.release()

        assert wait.done(), "the wait outlasted its timeout"
        assert wait.result() == []
        blocks = [call.kwargs["block"] for call in client.xreadgroup.call_args_list]
        assert len(blocks) >= 3
        assert max(blocks) <= 50
        assert sum(blocks) >= 150

    async def test_tracks_entry_id(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"stream", [(b"456-0", entry_data)]]])

        items = await backend.dequeue("w1")
        assert items[0].id in backend._entry_ids
        assert backend._entry_ids[items[0].id] == "456-0"

    async def test_a_wait_ends_when_a_retry_falls_due(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        # A retry falls due in 200 ms; promoted on the next pass.
        promotions = iter([[0, 200], [1]])

        async def promote_then_settle(script: str, *args: object) -> list[int] | int:
            return next(promotions) if script == PROMOTE else 1

        client.eval.side_effect = promote_then_settle
        client.xreadgroup.side_effect = [
            [],
            [[b"roomkit:delivery:pending", [(b"7-0", {b"data": item.model_dump_json()})]]],
        ]

        items = await backend.dequeue("w1", timeout=5.0)

        assert [i.id for i in items] == [item.id]
        blocks = [call.kwargs["block"] for call in client.xreadgroup.call_args_list]
        assert blocks[0] == 200
        assert blocks[1] <= 5000

    async def test_a_retry_due_after_the_timeout_does_not_shorten_it(self) -> None:
        backend, client = _make_backend()

        async def promote_far_off(script: str, *args: object) -> list[int] | int:
            return [0, 60_000] if script == PROMOTE else 1

        client.eval.side_effect = promote_far_off
        client.xreadgroup = AsyncMock(side_effect=_reads_nothing)

        assert await backend.dequeue("w1", timeout=0.05) == []
        client.xreadgroup.assert_called_once()
        assert client.xreadgroup.call_args.kwargs["block"] == 50

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
        client.eval.reset_mock()  # the dequeue's own call queues due retries
        await backend.ack(items[0].id)

        client.eval.assert_awaited_once()
        assert client.eval.call_args.args[2:] == (
            "roomkit:delivery:pending",
            "roomkit:delivery:pending",
            backend._group,
            "789-0",
            "w1",
            "",
            0,
            "",
        )

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
        client.eval.reset_mock()
        await backend.nack(items[0].id, error="transient")

        client.eval.assert_awaited_once()
        # The retry waits in the delayed set, 1 s for the first one.
        assert client.eval.call_args.args[3] == "roomkit:delivery:delayed"
        assert client.eval.call_args.args[9] == 1000
        replacement = DeliveryItem.model_validate_json(client.eval.call_args.args[7])
        assert replacement.retry_count == 1
        assert replacement.status == DeliveryItemStatus.PENDING

    async def test_dead_letters_after_max_retries(self) -> None:
        backend, client = _make_backend()
        item = _make_item(max_retries=0)
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"300-0", entry_data)]]])

        items = await backend.dequeue("w1")
        client.eval.reset_mock()
        await backend.nack(items[0].id, error="permanent")

        client.eval.assert_awaited_once()
        assert client.eval.call_args.args[9] == ""
        assert client.eval.call_args.args[3] == "roomkit:delivery:dead_letter"
        replacement = DeliveryItem.model_validate_json(client.eval.call_args.args[7])
        assert replacement.status == DeliveryItemStatus.DEAD_LETTER


class TestDeadLetter:
    async def test_moves_to_dl_stream(self) -> None:
        backend, client = _make_backend()
        item = _make_item()
        entry_data = {b"data": item.model_dump_json().encode()}
        client.xreadgroup = AsyncMock(return_value=[[b"s", [(b"400-0", entry_data)]]])

        items = await backend.dequeue("w1")
        client.eval.reset_mock()
        await backend.dead_letter(items[0].id, "fatal error")

        client.eval.assert_awaited_once()
        assert client.eval.call_args.args[3] == "roomkit:delivery:dead_letter"
        assert client.eval.call_args.args[8] == 10_000

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
        client.zcard = AsyncMock(return_value=3)
        depth = await backend.get_queue_depth()
        # Retries waiting out their delay are pending too.
        assert depth == 45
        client.xlen.assert_called_once_with("roomkit:delivery:pending")
        client.zcard.assert_called_once_with("roomkit:delivery:delayed")


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

        fakes = {"redis": MagicMock(), "redis.asyncio": mock_mod}
        with patch.dict(sys.modules, fakes):
            mod = load_private_copy("roomkit.delivery.redis", fakes)
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
        missing = {"redis": None, "redis.asyncio": None}
        with (
            patch.dict(sys.modules, missing),
            pytest.raises(ImportError, match="pip install roomkit\\[redis\\]"),
        ):
            mod = load_private_copy("roomkit.delivery.redis", missing)
            mod.RedisDeliveryBackend()


@pytest.mark.parametrize("operation", ["ack", "nack", "dead_letter"])
async def test_transition_failure_preserves_tracking(operation: str) -> None:
    backend, client = _make_backend()
    item = _make_item()
    client.xreadgroup.return_value = [[b"s", [(b"1-0", {b"data": item.model_dump_json()})]]]
    await backend.dequeue("w1")
    client.eval.side_effect = ConnectionError("offline")
    args = (item.id,) if operation == "ack" else (item.id, "error")
    try:
        with pytest.raises(ConnectionError):
            await getattr(backend, operation)(*args)
        assert item.id in backend._entry_ids
        assert backend._items[item.id].retry_count == 0
        client.eval.side_effect = None
        await getattr(backend, operation)(*args)
        assert item.id not in backend._entry_ids
    finally:
        await backend.close()


async def test_dequeue_recovers_pending_before_reading_new() -> None:
    backend, client = _make_backend()
    item = _make_item()
    client.xautoclaim.return_value = [b"2-0", [(b"1-0", {b"data": item.model_dump_json()})], []]
    try:
        items = await backend.dequeue("replacement")
        assert [i.id for i in items] == [item.id]
        assert items[0].worker_id == "replacement"
        assert backend._claim_cursor == b"2-0"
        client.xreadgroup.assert_not_called()
    finally:
        await backend.close()


async def test_start_does_not_hide_connection_failure() -> None:
    backend, client = _make_backend()
    client.xgroup_create.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        await backend.start(MagicMock())
    assert backend._worker_task is None
