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


class _HeldAttach(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        await super().publish_to_room(room_id, event)


async def test_closing_the_kit_ends_an_attach_held_in_presence_publication() -> None:
    from roomkit.core.exceptions import KitClosedError

    backend = _HeldAttach()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        attaching = asyncio.create_task(
            kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        )
        await asyncio.wait_for(backend.started.wait(), 10)
        await asyncio.wait_for(kit.close(), 10)
        result = await asyncio.wait_for(asyncio.gather(attaching, return_exceptions=True), 2)
        assert isinstance(result[0], KitClosedError)
        assert backend.cancelled.is_set()
    finally:
        await kit.close()


class _ReleasedAttach(_HeldAttach):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.completed = False

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            self.started.set()
            await self.release.wait()
            self.completed = True
        await InMemoryRealtime.publish_to_room(self, room_id, event)


async def test_an_initial_presence_publication_has_the_close_grace() -> None:
    backend = _ReleasedAttach()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        attaching = asyncio.create_task(
            kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        )
        await asyncio.wait_for(backend.started.wait(), 10)
        closing = asyncio.create_task(kit.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        assert not attaching.done()
        backend.release.set()
        binding = await asyncio.wait_for(attaching, 2)
        await asyncio.wait_for(closing, 2)
        assert binding.channel_id == "helper"
        assert backend.completed
    finally:
        backend.release.set()
        await kit.close()


class _ClosingAttach(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.kit: RoomKit | None = None
        self.after_close = False

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            assert self.kit is not None
            await self.kit.close()
            self.after_close = True
        await super().publish_to_room(room_id, event)


async def test_a_presence_backend_can_close_its_own_attach() -> None:
    backend = _ClosingAttach()
    kit = RoomKit(realtime=backend)
    backend.kit = kit
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        binding = await asyncio.wait_for(
            kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE), 2
        )
        assert binding.channel_id == "helper"
        assert backend.after_close
    finally:
        await kit.close()


class _OneHeldAttach(_HeldAttach):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            self.attempts += 1
            if self.attempts == 1:
                await super().publish_to_room(room_id, event)
        await InMemoryRealtime.publish_to_room(self, room_id, event)


async def test_an_abandoned_presence_releases_the_gate_but_keeps_its_claim() -> None:
    backend = _OneHeldAttach()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        attaching = asyncio.create_task(
            kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        )
        await asyncio.wait_for(backend.started.wait(), 10)
        attaching.cancel()
        result = await asyncio.gather(attaching, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert backend.cancelled.is_set()
        await asyncio.wait_for(kit.attach_channel("r1", "helper"), 2)
        await asyncio.sleep(0.05)
        assert backend.attempts == 1
        assert await asyncio.wait_for(kit.detach_channel("r1", "helper"), 2)
        await asyncio.wait_for(kit.attach_channel("r1", "helper"), 2)
        assert backend.attempts == 2
    finally:
        await kit.close()


class _HeldChange(_HeldAttach):
    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if self.armed:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        await InMemoryRealtime.publish_to_room(self, room_id, event)


async def test_closing_the_kit_ends_a_hook_change_held_in_presence_publication() -> None:
    from roomkit.core.exceptions import KitClosedError

    for enabling in (True, False):
        backend = _HeldChange()
        kit = RoomKit(realtime=backend)
        kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
        try:
            await kit.create_room(room_id="r1")
            if not enabling:
                await kit.enable_room_hook("r1", "unavailable")
            await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
            backend.armed = True
            changing = asyncio.create_task(
                kit.enable_room_hook("r1", "unavailable")
                if enabling
                else kit.disable_room_hook("r1", "unavailable")
            )
            await asyncio.wait_for(backend.started.wait(), 2)
            await asyncio.wait_for(kit.close(), 10)
            result = await asyncio.wait_for(asyncio.gather(changing, return_exceptions=True), 2)
            assert isinstance(result[0], KitClosedError)
            assert backend.cancelled.is_set()
        finally:
            await kit.close()


async def test_close_ends_a_detach_waiting_on_the_presence_gate() -> None:
    from roomkit.core.exceptions import KitClosedError
    from roomkit.models import FrameworkEvent

    backend = _HeldAttach()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    removed = asyncio.Event()

    @kit.on("room_channel_detached")
    async def detached(_event: FrameworkEvent) -> None:
        removed.set()

    try:
        await kit.create_room(room_id="r1")
        attaching = asyncio.create_task(kit.attach_channel("r1", "helper"))
        await asyncio.wait_for(backend.started.wait(), 10)
        detaching = asyncio.create_task(kit.detach_channel("r1", "helper"))
        await asyncio.wait_for(removed.wait(), 2)
        assert not detaching.done()
        await asyncio.wait_for(kit.close(), 10)
        results = await asyncio.wait_for(
            asyncio.gather(attaching, detaching, return_exceptions=True), 2
        )
        assert all(isinstance(result, KitClosedError) for result in results)
        assert backend.cancelled.is_set()
    finally:
        await kit.close()
