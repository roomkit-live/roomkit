"""Mock memory provider for testing."""

from __future__ import annotations

from dataclasses import dataclass

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage


@dataclass
class _RetrieveCall:
    room_id: str
    current_event: RoomEvent
    context: RoomContext
    channel_id: str | None


@dataclass
class _IngestCall:
    room_id: str
    event: RoomEvent
    channel_id: str | None


class MockMemoryProvider(MemoryProvider):
    """Mock memory provider that records calls and returns configured results.

    Example::

        mock = MockMemoryProvider(
            messages=[AIMessage(role="system", content="Summary of conversation")],
        )
        result = await mock.retrieve("room1", event, context)
        assert len(mock.retrieve_calls) == 1
    """

    def __init__(
        self,
        messages: list[AIMessage] | None = None,
        events: list[RoomEvent] | None = None,
    ) -> None:
        self._messages = messages or []
        self._events = events or []
        self.retrieve_calls: list[_RetrieveCall] = []
        self.ingest_calls: list[_IngestCall] = []
        self.clear_calls: list[str] = []
        self.closed: bool = False

    @property
    def name(self) -> str:
        return "MockMemoryProvider"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        self.retrieve_calls.append(
            _RetrieveCall(
                room_id=room_id,
                current_event=current_event,
                context=context,
                channel_id=channel_id,
            )
        )
        return MemoryResult(messages=list(self._messages), events=list(self._events))

    async def ingest(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        channel_id: str | None = None,
    ) -> None:
        self.ingest_calls.append(_IngestCall(room_id=room_id, event=event, channel_id=channel_id))

    async def clear(self, room_id: str) -> None:
        self.clear_calls.append(room_id)

    async def close(self) -> None:
        self.closed = True
