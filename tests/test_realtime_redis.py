"""Tests for RedisRealtimeBackend (realtime/redis.py)."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from tests.redis_fakes import FakePubSub

# -- Fake Redis -----------------------------------------------------------


def _build_client(pubsub: FakePubSub) -> AsyncMock:
    client = AsyncMock()
    client.publish = AsyncMock(return_value=1)
    client.aclose = AsyncMock()
    # pubsub() is called synchronously in production code — an AsyncMock
    # attribute would return an un-awaited coroutine.
    client.pubsub = MagicMock(return_value=pubsub)
    return client


def _make_backend(**kwargs):
    """Create a RedisRealtimeBackend with mocked redis."""
    pubsub = FakePubSub()
    client = _build_client(pubsub)
    mock_mod = MagicMock()
    mock_mod.from_url = MagicMock(return_value=client)

    with patch.dict(sys.modules, {"redis": MagicMock(), "redis.asyncio": mock_mod}):
        importlib.invalidate_caches()
        mod = importlib.import_module("roomkit.realtime.redis")
        importlib.reload(mod)
        backend = mod.RedisRealtimeBackend(client=client, **kwargs)

    return backend, client, pubsub


def _event(**kwargs) -> EphemeralEvent:
    defaults = dict(
        room_id="room-1",
        type=EphemeralEventType.TYPING_START,
        user_id="user-1",
    )
    defaults.update(kwargs)
    return EphemeralEvent(**defaults)


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# -- Tests ----------------------------------------------------------------


class TestPublish:
    async def test_publishes_prefixed_json(self) -> None:
        backend, client, _pubsub = _make_backend()
        event = _event(data={"key": "value"})
        await backend.publish("room:r1", event)

        client.publish.assert_awaited_once()
        channel, payload = client.publish.await_args[0]
        assert channel == "roomkit:realtime:room:r1"
        restored = EphemeralEvent.from_dict(json.loads(payload))
        assert restored.id == event.id
        assert restored.room_id == "room-1"
        assert restored.data == {"key": "value"}
        await backend.close()

    async def test_custom_prefix(self) -> None:
        backend, client, _pubsub = _make_backend(channel_prefix="myapp")
        await backend.publish("room:r1", _event())
        assert client.publish.await_args[0][0] == "myapp:room:r1"
        await backend.close()

    async def test_noop_after_close(self) -> None:
        backend, client, _pubsub = _make_backend()
        await backend.close()
        await backend.publish("room:r1", _event())
        client.publish.assert_not_awaited()


class TestSubscribe:
    async def test_subscribes_prefixed_channel_and_starts_reader(self) -> None:
        backend, _client, pubsub = _make_backend()
        assert backend._reader_task is None

        await backend.subscribe("room:r1", AsyncMock())

        assert pubsub.subscribed == ["roomkit:realtime:room:r1"]
        assert backend._reader_task is not None
        assert backend.subscription_count == 1
        await backend.close()

    async def test_roundtrip_bytes_payload(self) -> None:
        """from_url() default mode: channel and data arrive as bytes."""
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        await backend.subscribe("room:r1", callback)
        event = _event()
        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(event.to_dict()).encode())

        await _wait_until(lambda: received)
        assert received[0].id == event.id
        assert received[0].type == EphemeralEventType.TYPING_START
        await backend.close()

    async def test_roundtrip_str_payload(self) -> None:
        """decode_responses=True mode: channel and data arrive as str."""
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        await backend.subscribe("room:r1", callback)
        event = _event()
        pubsub.inject("roomkit:realtime:room:r1", json.dumps(event.to_dict()))

        await _wait_until(lambda: received)
        assert received[0].id == event.id
        await backend.close()

    async def test_multiple_subscribers_same_channel(self) -> None:
        backend, _client, pubsub = _make_backend()
        received1: list[EphemeralEvent] = []
        received2: list[EphemeralEvent] = []

        async def cb1(event: EphemeralEvent) -> None:
            received1.append(event)

        async def cb2(event: EphemeralEvent) -> None:
            received2.append(event)

        await backend.subscribe("room:r1", cb1)
        await backend.subscribe("room:r1", cb2)
        # Only one Redis-level subscribe for the shared channel
        assert pubsub.subscribed == ["roomkit:realtime:room:r1"]

        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(_event().to_dict()).encode())
        await _wait_until(lambda: received1 and received2)
        await backend.close()

    async def test_other_channel_not_delivered(self) -> None:
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        await backend.subscribe("room:r1", callback)
        pubsub.inject(b"roomkit:realtime:room:r2", json.dumps(_event().to_dict()).encode())
        await asyncio.sleep(0.05)
        assert received == []
        await backend.close()

    async def test_callback_error_does_not_stop_delivery(self) -> None:
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def flaky(event: EphemeralEvent) -> None:
            received.append(event)
            if len(received) == 1:
                raise ValueError("boom")

        await backend.subscribe("room:r1", flaky)
        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(_event().to_dict()).encode())
        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(_event().to_dict()).encode())

        await _wait_until(lambda: len(received) == 2)
        await backend.close()

    async def test_malformed_payload_skipped(self) -> None:
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        await backend.subscribe("room:r1", callback)
        pubsub.inject(b"roomkit:realtime:room:r1", b"not json")
        pubsub.inject(b"roomkit:realtime:room:r1", b'{"missing": "fields"}')
        event = _event()
        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(event.to_dict()).encode())

        await _wait_until(lambda: received)
        assert len(received) == 1
        assert received[0].id == event.id
        await backend.close()

    async def test_confirmation_timeout_warns_but_returns(self, monkeypatch) -> None:
        backend, _client, pubsub = _make_backend()
        pubsub.confirm_subscribes = False
        monkeypatch.setattr(
            sys.modules["roomkit.realtime.redis"], "_SUBSCRIBE_CONFIRM_TIMEOUT", 0.05
        )

        sub_id = await backend.subscribe("room:r1", AsyncMock())
        assert sub_id
        await backend.close()

    async def test_subscribe_after_close_raises(self) -> None:
        backend, _client, _pubsub = _make_backend()
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.subscribe("room:r1", AsyncMock())


class TestUnsubscribe:
    async def test_last_subscriber_drops_redis_channel(self) -> None:
        backend, _client, pubsub = _make_backend()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await backend.subscribe("room:r1", callback)
        assert await backend.unsubscribe(sub_id) is True
        assert pubsub.unsubscribed == ["roomkit:realtime:room:r1"]
        assert backend.subscription_count == 0

        pubsub.inject(b"roomkit:realtime:room:r1", json.dumps(_event().to_dict()).encode())
        await asyncio.sleep(0.05)
        assert received == []
        await backend.close()

    async def test_remaining_subscriber_keeps_channel(self) -> None:
        backend, _client, pubsub = _make_backend()
        sub1 = await backend.subscribe("room:r1", AsyncMock())
        await backend.subscribe("room:r1", AsyncMock())

        await backend.unsubscribe(sub1)
        assert pubsub.unsubscribed == []
        assert backend.subscription_count == 1
        await backend.close()

    async def test_unknown_returns_false(self) -> None:
        backend, _client, _pubsub = _make_backend()
        assert await backend.unsubscribe("nonexistent") is False
        await backend.close()


class TestClose:
    async def test_close_stops_reader_and_pubsub(self) -> None:
        backend, client, pubsub = _make_backend()
        await backend.subscribe("room:r1", AsyncMock())

        await backend.close()
        assert backend._reader_task is None
        assert pubsub.closed is True
        assert backend.subscription_count == 0
        # Injected client is not closed by the backend
        client.aclose.assert_not_awaited()

    async def test_close_closes_owned_client(self) -> None:
        backend, client, _pubsub = _make_backend()
        backend._owns_client = True
        await backend.close()
        client.aclose.assert_awaited_once()

    async def test_close_idempotent(self) -> None:
        backend, client, _pubsub = _make_backend()
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
            mod = importlib.import_module("roomkit.realtime.redis")
            importlib.reload(mod)
            mod.RedisRealtimeBackend()
