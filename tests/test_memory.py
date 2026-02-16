"""Tests for memory providers."""

from __future__ import annotations

from roomkit.memory.base import MemoryResult
from roomkit.memory.mock import MockMemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIMessage
from tests.conftest import make_event


class TestMemoryResult:
    def test_defaults_to_empty_lists(self) -> None:
        result = MemoryResult()
        assert result.messages == []
        assert result.events == []

    def test_accepts_messages_and_events(self) -> None:
        msg = AIMessage(role="system", content="summary")
        ev = make_event(body="hello")
        result = MemoryResult(messages=[msg], events=[ev])
        assert len(result.messages) == 1
        assert len(result.events) == 1


class TestSlidingWindowMemory:
    async def test_returns_last_n_events(self) -> None:
        events = [make_event(body=f"msg{i}") for i in range(10)]
        ctx = RoomContext(room=Room(id="r1"), recent_events=events)
        mem = SlidingWindowMemory(max_events=3)

        result = await mem.retrieve("r1", events[-1], ctx)

        assert len(result.events) == 3
        assert result.events[0].content.body == "msg7"  # type: ignore[union-attr]
        assert result.events[2].content.body == "msg9"  # type: ignore[union-attr]
        assert result.messages == []

    async def test_returns_all_when_fewer_than_max(self) -> None:
        events = [make_event(body=f"msg{i}") for i in range(3)]
        ctx = RoomContext(room=Room(id="r1"), recent_events=events)
        mem = SlidingWindowMemory(max_events=50)

        result = await mem.retrieve("r1", events[-1], ctx)

        assert len(result.events) == 3

    async def test_returns_empty_when_no_events(self) -> None:
        ctx = RoomContext(room=Room(id="r1"), recent_events=[])
        event = make_event(body="current")
        mem = SlidingWindowMemory(max_events=50)

        result = await mem.retrieve("r1", event, ctx)

        assert result.events == []
        assert result.messages == []

    async def test_name(self) -> None:
        mem = SlidingWindowMemory()
        assert mem.name == "SlidingWindowMemory"

    async def test_default_max_events(self) -> None:
        mem = SlidingWindowMemory()
        assert mem._max_events == 50


class TestMockMemoryProvider:
    async def test_returns_configured_messages(self) -> None:
        msg = AIMessage(role="system", content="summary")
        mock = MockMemoryProvider(messages=[msg])
        event = make_event(body="hello")
        ctx = RoomContext(room=Room(id="r1"))

        result = await mock.retrieve("r1", event, ctx)

        assert len(result.messages) == 1
        assert result.messages[0].content == "summary"

    async def test_returns_configured_events(self) -> None:
        ev = make_event(body="past")
        mock = MockMemoryProvider(events=[ev])
        event = make_event(body="hello")
        ctx = RoomContext(room=Room(id="r1"))

        result = await mock.retrieve("r1", event, ctx)

        assert len(result.events) == 1

    async def test_tracks_retrieve_calls(self) -> None:
        mock = MockMemoryProvider()
        event = make_event(body="hello")
        ctx = RoomContext(room=Room(id="r1"))

        await mock.retrieve("r1", event, ctx, channel_id="ai1")

        assert len(mock.retrieve_calls) == 1
        assert mock.retrieve_calls[0].room_id == "r1"
        assert mock.retrieve_calls[0].channel_id == "ai1"

    async def test_tracks_ingest_calls(self) -> None:
        mock = MockMemoryProvider()
        event = make_event(body="hello")

        await mock.ingest("r1", event, channel_id="ai1")

        assert len(mock.ingest_calls) == 1
        assert mock.ingest_calls[0].room_id == "r1"
        assert mock.ingest_calls[0].channel_id == "ai1"

    async def test_tracks_clear_calls(self) -> None:
        mock = MockMemoryProvider()
        await mock.clear("r1")
        assert mock.clear_calls == ["r1"]

    async def test_close(self) -> None:
        mock = MockMemoryProvider()
        assert mock.closed is False
        await mock.close()
        assert mock.closed is True

    async def test_name(self) -> None:
        mock = MockMemoryProvider()
        assert mock.name == "MockMemoryProvider"


class TestMemoryProviderDefaults:
    """Test that base ABC concrete methods are no-ops."""

    async def test_ingest_is_noop(self) -> None:
        mem = SlidingWindowMemory()
        event = make_event(body="hello")
        await mem.ingest("r1", event)  # should not raise

    async def test_clear_is_noop(self) -> None:
        mem = SlidingWindowMemory()
        await mem.clear("r1")  # should not raise

    async def test_close_is_noop(self) -> None:
        mem = SlidingWindowMemory()
        await mem.close()  # should not raise
