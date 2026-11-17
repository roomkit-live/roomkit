"""The public binding callback follows gate changes, including repeated ones."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from roomkit import Access, Channel, ChannelBinding, HookExecution, HookTrigger, RoomKit
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.models import RoomContext, RoomEvent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

_CHANGES = [
    ("mute", (), {}, {"muted": True}, True),
    ("unmute", (), {"muted": True}, {"muted": False}, True),
    ("mute_output", (), {}, {"output_muted": True}, True),
    ("unmute_output", (), {"output_muted": True}, {"output_muted": False}, True),
    ("set_access", (Access.READ_ONLY,), {}, {"access": Access.READ_ONLY}, True),
    ("set_visibility", ("transport",), {}, {"visibility": "transport"}, False),
    ("update_binding_metadata", ({"team": "blue"},), {}, {"metadata": {"team": "blue"}}, False),
]


class _Locks(RoomLockManager):
    def __init__(self) -> None:
        self.inner = InMemoryLockManager()
        self.held: set[str] = set()

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        async with self.inner.locked(room_id):
            outermost = room_id not in self.held
            self.held.add(room_id)
            try:
                yield
            finally:
                if outermost:
                    self.held.remove(room_id)


class _Store(InMemoryStore):
    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self.trace = trace

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        result = await super().update_binding(binding)
        self.trace.append("stored")
        return result


class _BindingChannel(SimpleChannel):
    def __init__(self, trace: list[str], locks: _Locks) -> None:
        super().__init__("sms")
        self.trace = trace
        self.locks = locks
        self.pushed: list[tuple[str, ChannelBinding]] = []
        self.refusal: RuntimeError | None = None

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        assert room_id in self.locks.held
        self.trace.append("notified")
        self.pushed.append((room_id, binding))
        if self.refusal is not None:
            raise self.refusal


async def _setup(
    initial: dict[str, Any],
) -> tuple[RoomKit, _BindingChannel, list[str], ChannelBinding]:
    trace: list[str] = []
    locks = _Locks()
    kit = RoomKit(store=_Store(trace), lock_manager=locks)
    channel = _BindingChannel(trace, locks)
    kit.register_channel(channel)
    await kit.create_room(room_id="room")
    binding = await kit.attach_channel("room", "sms", **initial)

    async def changed(_event: RoomEvent, _context: RoomContext) -> None:
        trace.append("hooked")

    for trigger in (HookTrigger.ON_CHANNEL_MUTED, HookTrigger.ON_CHANNEL_UNMUTED):
        kit.hook(trigger, execution=HookExecution.ASYNC)(changed)
    trace.clear()
    return kit, channel, trace, binding


@pytest.mark.parametrize(("operation", "args", "initial", "change", "notifies"), _CHANGES)
async def test_a_gate_change_notifies_after_storage_before_its_hook(
    operation: str,
    args: tuple[Any, ...],
    initial: dict[str, Any],
    change: dict[str, Any],
    notifies: bool,
) -> None:
    kit, channel, trace, before = await _setup(initial)
    try:
        result = await getattr(kit, operation)("room", "sms", *args)
        await kit.hook_engine.drain_observers()
        assert result == before.model_copy(update=change)
        assert result == await kit.get_binding("room", "sms")
        assert channel.pushed == ([("room", result)] if notifies else [])
        expected = ["stored", "notified"] if notifies else ["stored"]
        if operation in {"mute", "unmute"}:
            expected.append("hooked")
        assert trace == expected
    finally:
        await kit.close()


@pytest.mark.parametrize(("operation", "args", "initial", "change", "notifies"), _CHANGES[:5])
@pytest.mark.parametrize("repeated", [False, True])
async def test_a_callback_refusal_keeps_storage_and_stops_records_and_hooks(
    operation: str,
    args: tuple[Any, ...],
    initial: dict[str, Any],
    change: dict[str, Any],
    notifies: bool,
    repeated: bool,
) -> None:
    kit, channel, trace, before = await _setup(initial)
    try:
        expected = before.model_copy(update=change)
        if repeated:
            await kit.store.update_binding(expected)
            trace.clear()
        events = await kit.store.list_events("room")
        refusal = RuntimeError("binding cache refused")
        channel.refusal = refusal
        with pytest.raises(RuntimeError, match="binding cache refused") as raised:
            await getattr(kit, operation)("room", "sms", *args)
        await kit.hook_engine.drain_observers()
        assert raised.value is refusal
        assert await kit.get_binding("room", "sms") == expected
        assert channel.pushed == [("room", expected)]
        assert trace == (["notified"] if repeated else ["stored", "notified"])
        assert await kit.store.list_events("room") == events
    finally:
        await kit.close()


async def test_the_default_binding_callback_does_nothing() -> None:
    kit, channel, trace, binding = await _setup({})
    try:
        assert Channel.update_binding(channel, "room", binding) is None
        assert trace == []
        assert channel.pushed == []
        assert await kit.get_binding("room", "sms") == binding
    finally:
        await kit.close()
