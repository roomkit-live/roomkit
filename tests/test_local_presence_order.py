"""A detach waits for the earlier presence before forgetting its claim (RFC §10.6.7)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models import ChannelCategory, FrameworkEvent, Room
from roomkit.providers.ai import MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.store.memory import InMemoryStore


class _HeldPresence(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.PRESENCE_ONLINE:
            if not self.started.is_set():
                self.started.set()
                await self.release.wait()
            self.published.append((event.type.value, event.user_id))
        await super().publish(channel, event)


@pytest.mark.parametrize("refuses", [False, True])
async def test_detach_waits_for_an_earlier_presence_then_reattach_publishes_again(
    refuses: bool,
) -> None:
    backend = _HeldPresence()
    kit = RoomKit(realtime=backend)
    agent = AIChannel("helper", provider=MockAIProvider(responses=["hi"]))
    called: list[str] = []

    async def teardown(room_id: str) -> None:
        called.append(room_id)
        if refuses:
            raise ValueError("detach callback failed")

    agent.on_room_detached = teardown
    kit.register_channel(agent)
    removed = asyncio.Event()

    @kit.on("room_channel_detached")
    async def detached(_event: FrameworkEvent) -> None:
        removed.set()

    try:
        await kit.create_room(room_id="r1")
        attaching = asyncio.create_task(
            kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        )
        await asyncio.wait_for(backend.started.wait(), 10)
        detaching = asyncio.create_task(kit.detach_channel("r1", "helper"))
        await asyncio.wait_for(removed.wait(), 2)
        done, _ = await asyncio.wait({detaching}, timeout=0.05)
        returned_before_publication = detaching in done
        backend.release.set()
        await asyncio.wait_for(attaching, 2)
        assert await asyncio.wait_for(detaching, 2)
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        assert not returned_before_publication
        assert called == ["r1"]
        assert backend.published == [("presence_online", "helper"), ("presence_online", "helper")]
    finally:
        backend.release.set()
        await kit.close()


class _FailedStateRemoval(InMemoryStore):
    async def patch_room_metadata(
        self, room_id: str, patch: dict[str, Any], *, unset: Sequence[str] = ()
    ) -> Room | None:
        if "roomkit:channel_state:helper" in unset:
            raise OSError("state cleanup failed")
        return await super().patch_room_metadata(room_id, patch, unset=unset)


async def test_a_failed_detach_state_cleanup_still_forgets_the_presence_claim() -> None:
    backend = _HeldPresence()
    backend.started.set()
    kit = RoomKit(store=_FailedStateRemoval(), realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await kit.store.patch_room_metadata(
            "r1", {"roomkit:channel_state:helper": {"model": "kept"}}
        )
        with pytest.raises(OSError, match="^state cleanup failed$"):
            await kit.detach_channel("r1", "helper")
        assert await kit.store.get_binding("r1", "helper") is None
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        assert backend.published == [("presence_online", "helper"), ("presence_online", "helper")]
    finally:
        await kit.close()


@pytest.mark.parametrize("missing_binding", [False, True])
async def test_a_detach_that_removes_nothing_preserves_the_presence_claim(
    missing_binding: bool,
) -> None:
    backend = _HeldPresence()
    backend.started.set()
    kit = RoomKit(realtime=backend)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    try:
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        if missing_binding:
            assert await kit.store.remove_binding("r1", "helper")
            assert not await kit.detach_channel("r1", "helper")
        else:
            with pytest.raises(RoomNotFoundError):
                await kit.detach_channel("r1", "helper", organization_id="other")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        assert backend.published == [("presence_online", "helper")]
    finally:
        await kit.close()
