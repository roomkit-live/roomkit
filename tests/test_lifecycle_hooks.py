"""Tests for lifecycle hooks firing on dynamic operations (Area 3.2)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    Access,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import RoomTimers
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestRoomLifecycleHooks:
    async def test_hook_fired_on_create_room(self, kit: RoomKit) -> None:
        """ON_ROOM_CREATED fires when a room is created, and the creation
        awaits its hooks (RFC §9.3): they have run when it returns."""
        fired: list[RoomEvent] = []

        async def on_created(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_ROOM_CREATED, on_created))

        await kit.create_room(room_id="r1")
        assert len(fired) == 1
        assert fired[0].room_id == "r1"


async def _change(kit: RoomKit, trigger: HookTrigger) -> Callable[[], Awaitable[object]]:
    """Set up room ``r1``, ``sms1`` attached, for the call that fires
    *trigger*, and return that call."""
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    match trigger:
        case HookTrigger.ON_ROOM_CLOSED:
            return lambda: kit.close_room("r1")
        case HookTrigger.ON_ROOM_PAUSED:
            await kit.set_room_timers("r1", RoomTimers(inactive_after_seconds=0))
            return lambda: kit.check_room_timers("r1")
        case HookTrigger.ON_CHANNEL_ATTACHED:
            kit.register_channel(SimpleChannel("sms2"))
            return lambda: kit.attach_channel("r1", "sms2")
        case HookTrigger.ON_CHANNEL_DETACHED:
            return lambda: kit.detach_channel("r1", "sms1")
        case HookTrigger.ON_CHANNEL_MUTED:
            return lambda: kit.mute("r1", "sms1")
        case HookTrigger.ON_CHANNEL_UNMUTED:
            await kit.mute("r1", "sms1")
            return lambda: kit.unmute("r1", "sms1")
        case HookTrigger.ON_EVENT_UPDATED:
            said = await kit.send_event("r1", "sms1", TextContent(body="Hi."))
            return lambda: kit.update_event("r1", said.id, content=TextContent(body="Hey."))
        case HookTrigger.ON_EVENT_DELETED:
            said = await kit.send_event("r1", "sms1", TextContent(body="Hi."))
            return lambda: kit.delete_event("r1", said.id)
        case HookTrigger.ON_PARTICIPANT_JOINED:
            return lambda: kit.add_member("r1", "sms1", "p1")
    await kit.add_member("r1", "sms1", "p1")
    if trigger is HookTrigger.ON_PARTICIPANT_LEFT:
        return lambda: kit.remove_member("r1", "p1")
    return lambda: kit.rename_member("r1", "p1", "Bob")


@pytest.mark.parametrize(
    "trigger",
    [
        HookTrigger.ON_ROOM_CLOSED,
        HookTrigger.ON_ROOM_PAUSED,
        HookTrigger.ON_CHANNEL_ATTACHED,
        HookTrigger.ON_CHANNEL_DETACHED,
        HookTrigger.ON_CHANNEL_MUTED,
        HookTrigger.ON_CHANNEL_UNMUTED,
        HookTrigger.ON_EVENT_UPDATED,
        HookTrigger.ON_EVENT_DELETED,
        HookTrigger.ON_PARTICIPANT_JOINED,
        HookTrigger.ON_PARTICIPANT_LEFT,
        HookTrigger.ON_PARTICIPANT_UPDATED,
    ],
    ids=str,
)
async def test_the_call_returns_before_its_hooks_end(kit: RoomKit, trigger: HookTrigger) -> None:
    """RFC §9.3: an ASYNC trigger's hooks run after the change that fired
    them, which does not wait for them, as roomkit-rs's do; only a room's
    creation awaits its hooks (above). Each of these calls waited for them
    (RMK-1020)."""
    call = await _change(kit, trigger)
    rooms: list[str] = []
    started, release = asyncio.Event(), asyncio.Event()

    @kit.hook(trigger, execution=HookExecution.ASYNC, name="slow")
    async def slow(event: RoomEvent, ctx: RoomContext) -> None:
        rooms.append(ctx.room.id)
        started.set()
        await release.wait()

    try:
        await asyncio.wait_for(call(), 0.5)
        await asyncio.wait_for(started.wait(), 0.5)
        assert rooms == ["r1"]
    finally:
        release.set()
        await kit.close()


class TestChannelLifecycleHooks:
    async def test_detach_nonexistent_no_hook(self, kit: RoomKit) -> None:
        """ON_CHANNEL_DETACHED does not fire if channel was not attached."""
        fired: list[RoomEvent] = []

        async def on_detached(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(event)

        kit.hook_engine.register(_async_hook(HookTrigger.ON_CHANNEL_DETACHED, on_detached))

        await kit.create_room(room_id="r1")
        result = await kit.detach_channel("r1", "nonexistent")
        await kit.hook_engine.drain_observers()
        assert result is False
        assert len(fired) == 0


class _GatedChannel(SimpleChannel):
    """A channel that keeps the binding state it was last pushed, as a
    session-based channel caches it for audio gating."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.pushed: list[ChannelBinding] = []

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        self.pushed.append(binding)


# (operation, its arguments, what the stored binding holds already, whether
# the call pushes the binding to the channel's gate)
_NO_CHANGE = [
    ("mute", (), {"muted": True}, True),
    ("unmute", (), {}, True),
    ("mute_output", (), {"output_muted": True}, True),
    ("unmute_output", (), {}, True),
    ("set_visibility", ("all",), {}, False),
    ("set_access", (Access.READ_ONLY,), {"access": Access.READ_ONLY}, True),
    ("update_binding_metadata", ({"team": "blue"},), {"metadata": {"team": "blue"}}, False),
]


@pytest.mark.parametrize(("operation", "args", "stored", "pushes"), _NO_CHANGE)
async def test_a_binding_change_that_changes_nothing_writes_and_fires_nothing(
    operation: str,
    args: tuple[Any, ...],
    stored: dict[str, Any],
    pushes: bool,
    monkeypatch: Any,
) -> None:
    """RFC §7.2: a retried call duplicates no hook and no record, and the
    binding is not written again. A gate change still pushes the binding as
    stored to the channel, whose cached state may have drifted from it (the
    store written by another process): a repeated mute must not leave audio
    flowing."""
    kit = RoomKit()
    channel = _GatedChannel("sms1")
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    binding = await kit.attach_channel("r1", "sms1")
    # Written behind the channel's back: its cached state is now stale.
    await kit.store.update_binding(binding.model_copy(update=stored))
    before = await kit.store.get_binding("r1", "sms1")
    count = (await kit.get_room("r1")).event_count
    fired: list[HookTrigger] = []
    for trigger in (HookTrigger.ON_CHANNEL_MUTED, HookTrigger.ON_CHANNEL_UNMUTED):

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def on_change(event: RoomEvent, ctx: RoomContext, trigger: Any = trigger) -> None:
            fired.append(trigger)

    writes: list[ChannelBinding] = []
    monkeypatch.setattr(kit.store, "update_binding", _recording(writes))
    channel.pushed.clear()

    result = await getattr(kit, operation)("r1", "sms1", *args)
    await kit.hook_engine.drain_observers()

    assert result == before
    assert writes == []
    assert fired == []
    assert channel.pushed == ([before] if pushes else [])
    assert (await kit.get_room("r1")).event_count == count
    await kit.close()


def _recording(writes: list[ChannelBinding]) -> Any:
    async def update_binding(binding: ChannelBinding) -> ChannelBinding:
        writes.append(binding)
        return binding

    return update_binding


def _async_hook(trigger: HookTrigger, fn: object) -> object:
    """Create an async HookRegistration."""
    from roomkit.core.hooks import HookRegistration

    return HookRegistration(
        trigger=trigger,
        execution=HookExecution.ASYNC,
        fn=fn,
        name=f"test_{trigger}",
    )
