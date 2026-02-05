"""Tests for lifecycle hooks firing on dynamic operations (Area 3.2)."""

from __future__ import annotations

import pytest

from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestRoomLifecycleHooks:
    async def test_hook_fired_on_create_room(self, kit: RoomKit) -> None:
        """ON_ROOM_CREATED fires when a room is created."""
        fired: list[RoomEvent] = []

        async def on_created(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_ROOM_CREATED, on_created))

        await kit.create_room(room_id="r1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"

    async def test_hook_fired_on_close_room(self, kit: RoomKit) -> None:
        """ON_ROOM_CLOSED fires when a room is closed."""
        fired: list[RoomEvent] = []

        async def on_closed(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_ROOM_CLOSED, on_closed))

        await kit.create_room(room_id="r1")
        await kit.close_room("r1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"


class TestChannelLifecycleHooks:
    async def test_hook_fired_on_attach(self, kit: RoomKit) -> None:
        """ON_CHANNEL_ATTACHED fires when a channel is attached."""
        fired: list[RoomEvent] = []

        async def on_attached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_ATTACHED, on_attached))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        # Filter to just ON_CHANNEL_ATTACHED events (ignoring ON_ROOM_CREATED)
        assert len(fired) == 1
        assert fired[0].room_id == "r1"

    async def test_hook_fired_on_detach(self, kit: RoomKit) -> None:
        """ON_CHANNEL_DETACHED fires when a channel is detached."""
        fired: list[RoomEvent] = []

        async def on_detached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_DETACHED, on_detached))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.detach_channel("r1", "sms1")
        assert len(fired) == 1

    async def test_hook_fired_on_mute(self, kit: RoomKit) -> None:
        """ON_CHANNEL_MUTED fires when a channel is muted."""
        fired: list[RoomEvent] = []

        async def on_muted(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_MUTED, on_muted))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.mute("r1", "sms1")
        assert len(fired) == 1

    async def test_hook_fired_on_unmute(self, kit: RoomKit) -> None:
        """ON_CHANNEL_UNMUTED fires when a channel is unmuted."""
        fired: list[RoomEvent] = []

        async def on_unmuted(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_UNMUTED, on_unmuted))

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.mute("r1", "sms1")
        await kit.unmute("r1", "sms1")
        assert len(fired) == 1

    async def test_detach_nonexistent_no_hook(self, kit: RoomKit) -> None:
        """ON_CHANNEL_DETACHED does not fire if channel was not attached."""
        fired: list[RoomEvent] = []

        async def on_detached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_DETACHED, on_detached))

        await kit.create_room(room_id="r1")
        result = await kit.detach_channel("r1", "nonexistent")
        assert result is False
        assert len(fired) == 0


def _async_hook(trigger: HookTrigger, fn: object) -> object:
    """Create an async HookRegistration."""
    from roomkit.core.hooks import HookRegistration

    return HookRegistration(
        trigger=trigger,
        execution=HookExecution.ASYNC,
        fn=fn,
        name=f"test_{trigger}",
    )
