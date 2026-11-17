"""Explicit presence calls preserve a backend's recoverable failure and cause."""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.realtime import EphemeralEvent
from roomkit.realtime.memory import InMemoryRealtime


class _BrokenPublication(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.failure = ConnectionError("realtime down")
        self.failure.__cause__ = OSError("transport unavailable")

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        raise self.failure


async def test_an_explicit_presence_publication_preserves_the_backend_error_and_cause() -> None:
    backend = _BrokenPublication()
    kit = RoomKit(realtime=backend)
    try:
        await kit.create_room(room_id="r1")
        with pytest.raises(ConnectionError) as raised:
            await kit.publish_presence("r1", "helper", "online")
        assert raised.value is backend.failure
        assert str(raised.value) == "realtime down"
        assert raised.value.__cause__ is backend.failure.__cause__
        assert str(raised.value.__cause__) == "transport unavailable"
    finally:
        await kit.close()
