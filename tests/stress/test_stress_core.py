"""Stress lane: the core pipeline under contention (run with ``make stress``).

The functional suite verifies behaviour at 2-4 rooms; nothing there
characterises the serialization invariant (RFC §8.1) or the framework's
transient state under real concurrency. These tests are the measured
envelope: many rooms at once, many turns racing into one room, delivery
failures under load, and a long conversation's effect on framework state.

They are excluded from the default run (``-m 'not stress'`` in addopts) so
their wall-clock cost and machine sensitivity never flake CI; run them
deliberately via ``make stress``.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from tests.test_framework import SimpleChannel

pytestmark = pytest.mark.stress


def _msg(channel_id: str, sender: str, body: str) -> InboundMessage:
    return InboundMessage(channel_id=channel_id, sender_id=sender, content=TextContent(body=body))


async def _room_indices(kit: RoomKit, room_id: str) -> list[int]:
    events = await kit._store.get_conversation(room_id, limit=10_000)
    return sorted(e.index for e in events)


def _assert_contiguous(indices: list[int], room_id: str) -> None:
    """The RFC §8.1 invariant: unique, gap-free, monotonically increasing."""
    assert indices, f"room {room_id} has no events"
    expected = list(range(indices[0], indices[0] + len(indices)))
    assert indices == expected, f"room {room_id} indices not contiguous/unique: {indices}"


class TestManyRoomsUnderConcurrentLoad:
    async def test_event_indices_stay_sequential_across_100_rooms(self) -> None:
        """100 rooms × 5 concurrent turns each: every room's log stays gap-free."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        rooms = [f"r{i}" for i in range(100)]
        for room_id in rooms:
            await kit.create_room(room_id=room_id)
            await kit.attach_channel(room_id, "sms1")

        async def turn(room_id: str, n: int) -> None:
            result = await kit.process_inbound(
                _msg("sms1", f"user-{room_id}", f"m{n}"), room_id=room_id
            )
            assert not result.blocked

        await asyncio.gather(*(turn(room_id, n) for room_id in rooms for n in range(5)))

        for room_id in rooms:
            indices = await _room_indices(kit, room_id)
            _assert_contiguous(indices, room_id)
            # 5 messages + the system events of create/attach, nothing lost.
            assert len(indices) >= 5

    async def test_50_turns_racing_into_one_room_serialize(self) -> None:
        """One room, 50 concurrent turns: the per-room lock is the only thing
        between this test and duplicate indices."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        results = await asyncio.gather(
            *(kit.process_inbound(_msg("sms1", "u1", f"m{n}"), room_id="r1") for n in range(50))
        )
        assert all(not r.blocked for r in results)

        indices = await _room_indices(kit, "r1")
        _assert_contiguous(indices, "r1")
        assert len(indices) >= 50


class _FlakyChannel(SimpleChannel):
    """Delivery fails every other event — the transport, not the pipeline."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.calls = 0

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.calls += 1
        if self.calls % 2 == 0:
            raise RuntimeError("provider outage (injected)")
        return await super().deliver(event, binding, context)


class TestDeliveryFailureUnderLoad:
    async def test_flaky_delivery_never_corrupts_the_event_log(self) -> None:
        """Broadcast failures are per-target results (RFC §13.6); the committed
        log must stay gap-free while half the deliveries blow up."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(_FlakyChannel("flaky1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "flaky1")

        results = await asyncio.gather(
            *(kit.process_inbound(_msg("sms1", "u1", f"m{n}"), room_id="r1") for n in range(30))
        )
        assert all(not r.blocked for r in results)

        indices = await _room_indices(kit, "r1")
        _assert_contiguous(indices, "r1")
        assert len(indices) >= 30


class TestLongConversationTransientState:
    async def test_framework_state_does_not_grow_with_the_conversation(self) -> None:
        """1000 sequential turns: the store grows (its job); the framework's
        transient state must not — gates cleared, hook tasks drained, the
        lock manager bounded by its LRU."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        for n in range(1000):
            result = await kit.process_inbound(_msg("sms1", "u1", f"m{n}"), room_id="r1")
            assert not result.blocked

        indices = await _room_indices(kit, "r1")
        _assert_contiguous(indices, "r1")
        assert len(indices) >= 1000

        # Transient state is flat, whatever the conversation's length.
        assert not kit._greeting_gates
        if kit._pending_hook_tasks:
            await asyncio.gather(*kit._pending_hook_tasks, return_exceptions=True)
        assert not kit._pending_hook_tasks
        assert kit._lock_manager.size <= 1024  # InMemoryLockManager LRU cap
