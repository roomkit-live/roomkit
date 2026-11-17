"""Recoverable presence publications are retried while the kit is open (RFC §10.6.7)."""

from __future__ import annotations

import asyncio

from roomkit import AIChannel, RoomKit
from roomkit.models import ChannelCategory
from roomkit.providers.ai import MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime


class _Retries(InMemoryRealtime):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.attempts = 0
        self.published: list[tuple[str, str]] = []
        self.delivered = asyncio.Event()

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            self.attempts += 1
            if self.failures:
                self.failures -= 1
                raise ConnectionError("realtime down")
            self.published.append((event.type.value, event.user_id))
            self.delivered.set()
        await super().publish_to_room(room_id, event)


async def test_a_live_presence_retries_repeated_publication_errors_once_per_attempt() -> None:
    backend = _Retries(2)
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        binding = await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        assert binding.channel_id == "helper"
        await asyncio.wait_for(backend.delivered.wait(), 10)
        assert backend.attempts == 3
        assert backend.published == [("presence_online", "helper")]
    finally:
        await kit.close()


class _HeldRetry(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("realtime down")
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        await super().publish_to_room(room_id, event)


async def test_closing_the_kit_cancels_an_inflight_presence_retry() -> None:
    backend = _HeldRetry()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await asyncio.wait_for(backend.started.wait(), 10)
        await asyncio.wait_for(kit.close(), 2)
        assert backend.cancelled.is_set()
        assert backend.attempts == 2
    finally:
        await kit.close()
