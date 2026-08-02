"""Tests for RedisStatusBackend (orchestration/status_redis.py)."""

from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.orchestration.status_bus import StatusEntry, StatusLevel

# -- Fake Redis -----------------------------------------------------------


class FakePubSub:
    """Queue-backed PubSub fake (see tests/test_realtime_redis.py)."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self.subscribed: list[str] = []
        self.closed = False
        self.confirm_subscribes = True

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)
        if self.confirm_subscribes:
            self._queue.put_nowait(
                {"type": "subscribe", "pattern": None, "channel": channel.encode(), "data": 1}
            )

    async def unsubscribe(self, channel: str) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float = 0.0
    ) -> dict | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def inject(self, channel: str | bytes, data: str | bytes) -> None:
        self._queue.put_nowait(
            {"type": "message", "pattern": None, "channel": channel, "data": data}
        )


class FakePipeline:
    """Records queued commands; execute() resolves them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.executed = 0

    def lpush(self, key: str, value: str) -> None:
        self.calls.append(("lpush", key, value))

    def ltrim(self, key: str, start: int, end: int) -> None:
        self.calls.append(("ltrim", key, start, end))

    def publish(self, channel: str, value: str) -> None:
        self.calls.append(("publish", channel, value))

    async def execute(self) -> list:
        self.executed += 1
        return []


def _build_client(pubsub: FakePubSub, pipeline: FakePipeline) -> AsyncMock:
    client = AsyncMock()
    client.lrange = AsyncMock(return_value=[])
    client.aclose = AsyncMock()
    # pubsub() and pipeline() are called synchronously in production code.
    client.pubsub = MagicMock(return_value=pubsub)
    client.pipeline = MagicMock(return_value=pipeline)
    return client


def _make_backend(**kwargs):
    """Create a RedisStatusBackend with mocked redis."""
    pubsub = FakePubSub()
    pipeline = FakePipeline()
    client = _build_client(pubsub, pipeline)
    mock_mod = MagicMock()
    mock_mod.from_url = MagicMock(return_value=client)

    with patch.dict(sys.modules, {"redis": MagicMock(), "redis.asyncio": mock_mod}):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.orchestration.status_redis")
        importlib.reload(mod)
        backend = mod.RedisStatusBackend(client=client, **kwargs)

    return backend, client, pubsub, pipeline


def _entry(
    agent_id: str = "exec",
    action: str = "task",
    status: StatusLevel = StatusLevel.OK,
    detail: str = "",
) -> StatusEntry:
    return StatusEntry(
        ts=datetime.now(UTC).isoformat(),
        agent_id=agent_id,
        action=action,
        status=status,
        detail=detail,
    )


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# -- Tests ----------------------------------------------------------------


class TestPublish:
    async def test_lpush_ltrim_publish_in_order(self) -> None:
        backend, _client, _pubsub, pipeline = _make_backend()
        entry = _entry(detail="hello")
        await backend.publish(entry)

        assert pipeline.executed == 1
        ops = [c[0] for c in pipeline.calls]
        assert ops == ["lpush", "ltrim", "publish"]

        _, key, payload = pipeline.calls[0]
        assert key == "roomkit:status:entries"
        restored = StatusEntry.model_validate_json(payload)
        assert restored.agent_id == "exec"
        assert restored.detail == "hello"

        assert pipeline.calls[1] == ("ltrim", "roomkit:status:entries", 0, 499)
        assert pipeline.calls[2][1] == "roomkit:status:events"
        await backend.close()

    async def test_custom_prefix_and_max_entries(self) -> None:
        backend, _client, _pubsub, pipeline = _make_backend(key_prefix="myapp", max_entries=10)
        await backend.publish(_entry())
        assert pipeline.calls[0][1] == "myapp:entries"
        assert pipeline.calls[1] == ("ltrim", "myapp:entries", 0, 9)
        assert pipeline.calls[2][1] == "myapp:events"
        await backend.close()

    async def test_noop_after_close(self) -> None:
        backend, _client, _pubsub, pipeline = _make_backend()
        await backend.close()
        await backend.publish(_entry())
        assert pipeline.executed == 0


class TestRecent:
    async def test_chronological_order_and_limit(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        # LPUSH order: newest first in the list
        entries = [_entry(action=f"a{i}") for i in range(5)]
        client.lrange = AsyncMock(
            return_value=[e.model_dump_json().encode() for e in reversed(entries)]
        )

        result = await backend.recent(3)
        assert [e.action for e in result] == ["a2", "a3", "a4"]
        client.lrange.assert_awaited_once_with("roomkit:status:entries", 0, -1)
        await backend.close()

    async def test_str_mode(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        client.lrange = AsyncMock(return_value=[_entry(action="a1").model_dump_json()])
        result = await backend.recent(5)
        assert [e.action for e in result] == ["a1"]
        await backend.close()

    async def test_filters(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        entries = [
            _entry(agent_id="exec", status=StatusLevel.OK),
            _entry(agent_id="voice", status=StatusLevel.COMPLETED),
            _entry(agent_id="exec", status=StatusLevel.COMPLETED),
        ]
        client.lrange = AsyncMock(
            return_value=[e.model_dump_json().encode() for e in reversed(entries)]
        )

        by_agent = await backend.recent(10, agent_id="exec")
        assert len(by_agent) == 2
        by_status = await backend.recent(10, status="completed")
        assert len(by_status) == 2
        both = await backend.recent(10, agent_id="exec", status="completed")
        assert len(both) == 1
        await backend.close()

    async def test_skips_malformed_entries(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        client.lrange = AsyncMock(
            return_value=[b"not json", _entry(action="ok").model_dump_json().encode()]
        )
        result = await backend.recent(10)
        assert [e.action for e in result] == ["ok"]
        await backend.close()


class TestSubscribe:
    async def test_first_subscriber_starts_reader(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        assert backend._reader_task is None

        await backend.subscribe(AsyncMock())
        assert pubsub.subscribed == ["roomkit:status:events"]
        assert backend._reader_task is not None

        await backend.subscribe(AsyncMock())
        # Still a single Redis-level subscription
        assert pubsub.subscribed == ["roomkit:status:events"]
        await backend.close()

    async def test_async_callback_notified(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        received: list[StatusEntry] = []

        async def callback(entry: StatusEntry) -> None:
            received.append(entry)

        await backend.subscribe(callback)
        entry = _entry(detail="from another process")
        pubsub.inject(b"roomkit:status:events", entry.model_dump_json().encode())

        await _wait_until(lambda: received)
        assert received[0].detail == "from another process"
        await backend.close()

    async def test_sync_callback_notified(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        received: list[StatusEntry] = []

        def callback(entry: StatusEntry) -> None:
            received.append(entry)

        await backend.subscribe(callback)
        pubsub.inject(b"roomkit:status:events", _entry().model_dump_json().encode())

        await _wait_until(lambda: received)
        await backend.close()

    async def test_callback_error_does_not_stop_others(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        received: list[StatusEntry] = []

        def broken(entry: StatusEntry) -> None:
            raise ValueError("boom")

        def working(entry: StatusEntry) -> None:
            received.append(entry)

        await backend.subscribe(broken)
        await backend.subscribe(working)
        pubsub.inject(b"roomkit:status:events", _entry().model_dump_json().encode())

        await _wait_until(lambda: received)
        await backend.close()

    async def test_malformed_payload_skipped(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        received: list[StatusEntry] = []

        async def callback(entry: StatusEntry) -> None:
            received.append(entry)

        await backend.subscribe(callback)
        pubsub.inject(b"roomkit:status:events", b"not json")
        entry = _entry(action="valid")
        pubsub.inject(b"roomkit:status:events", entry.model_dump_json().encode())

        await _wait_until(lambda: received)
        assert len(received) == 1
        assert received[0].action == "valid"
        await backend.close()

    async def test_subscribe_after_close_raises(self) -> None:
        backend, _client, _pubsub, _pipeline = _make_backend()
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.subscribe(AsyncMock())


class TestUnsubscribe:
    async def test_removes_callback(self) -> None:
        backend, _client, pubsub, _pipeline = _make_backend()
        received: list[StatusEntry] = []

        def callback(entry: StatusEntry) -> None:
            received.append(entry)

        await backend.subscribe(callback)
        await backend.unsubscribe(callback)
        pubsub.inject(b"roomkit:status:events", _entry().model_dump_json().encode())
        await asyncio.sleep(0.05)
        assert received == []
        # Reader stays alive until close()
        assert backend._reader_task is not None
        await backend.close()

    async def test_unknown_callback_is_noop(self) -> None:
        backend, _client, _pubsub, _pipeline = _make_backend()
        await backend.unsubscribe(lambda e: None)  # should not raise
        await backend.close()


class TestClose:
    async def test_close_stops_reader_and_pubsub(self) -> None:
        backend, client, pubsub, _pipeline = _make_backend()
        await backend.subscribe(AsyncMock())

        await backend.close()
        assert backend._reader_task is None
        assert pubsub.closed is True
        # Injected client is not closed by the backend
        client.aclose.assert_not_awaited()

    async def test_close_closes_owned_client(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        backend._owns_client = True
        await backend.close()
        client.aclose.assert_awaited_once()

    async def test_close_idempotent(self) -> None:
        backend, client, _pubsub, _pipeline = _make_backend()
        backend._owns_client = True
        await backend.close()
        await backend.close()
        client.aclose.assert_awaited_once()


class TestImportError:
    def test_helpful_message(self) -> None:
        with (
            patch.dict(sys.modules, {"redis": None, "redis.asyncio": None}),
            pytest.raises(ImportError, match="pip install roomkit\\[redis\\]"),
        ):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.orchestration.status_redis")
            importlib.reload(mod)
            mod.RedisStatusBackend()
