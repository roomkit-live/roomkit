"""RMK-105 — the locked pass carries the history it was handed.

An inbound message builds a ``RoomContext`` before the room lock, for the
channel and the identity resolver (RFC §10.1 steps 3-5), then re-reads room,
bindings and participants under the lock — that is what the lock is for
(steps 6 and 12). The history is the one part it does not re-read: a context
deserialises up to 50 stored events into pydantic models, 41% of the worker CPU
a message costs on the measured bench, and a second copy of it buys nothing.

What is asserted here is the number of history reads — deterministic, where a
second copy regresses silently — together with what that number must not cost:
a hook and a broadcast MUST see the history a fresh read would give them.

Every room here has a reader: since RMK-103 the history is loaded for a hook
or a channel that declares it reads it, and for nobody else, so a room with
neither makes no read at all — and there is nothing to carry.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.store.memory import InMemoryStore


class CountingStore(InMemoryStore):
    """An in-memory store that counts the conversation reads asked of it.

    ``get_conversation`` is what turns stored rows into ``RoomEvent`` models for
    a context, and ``_build_context`` is its only caller inside the framework —
    so this counter is exactly "how many times a message deserialised the room's
    history".
    """

    def __init__(self) -> None:
        super().__init__()
        self.history_reads = 0

    async def get_conversation(
        self, room_id: str, *, limit: int = 50, after_index: int | None = None
    ) -> list[RoomEvent]:
        self.history_reads += 1
        return await super().get_conversation(room_id, limit=limit, after_index=after_index)


class PlainChannel(Channel):
    """A transport channel that reads no history and answers nothing."""

    channel_type = ChannelType.SMS

    def __init__(self, channel_id: str = "sms") -> None:
        super().__init__(channel_id)
        self.delivered_contexts: list[RoomContext] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered_contexts.append(context)
        return ChannelOutput.empty()


class WideWindowChannel(PlainChannel):
    """A channel that declares a wider history window than the floor."""

    channel_type = ChannelType.AI

    @property
    def recent_events_window(self) -> int:
        return 200


class InterleavingChannel(PlainChannel):
    """A channel that commits an event while it handles an inbound message.

    Stands in for anything that writes to the room between the pre-lock context
    and the lock — an identity hook answering a challenge, a companion service
    calling ``send_event``. The event lands after the pre-lock context was
    built, so the locked pass MUST NOT hand that stale history to the hooks.
    """

    def __init__(self, channel_id: str, kit: RoomKit) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.interleaved: RoomEvent | None = None

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        if self.interleaved is None:
            self.interleaved = await self._kit.send_event(
                context.room.id,
                self.channel_id,
                TextContent(body="interleaved"),
            )
        return await super().handle_inbound(message, context)


async def _send(kit: RoomKit, room_id: str, channel_id: str, count: int) -> None:
    for i in range(count):
        await kit.process_inbound(
            InboundMessage(
                channel_id=channel_id,
                sender_id="+15550000",
                content=TextContent(body=f"msg {i}"),
            ),
            room_id=room_id,
        )


def _with_a_reader(kit: RoomKit) -> None:
    """One no-op hook: the reason the floor of history is loaded at all."""

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def observe(event: RoomEvent, ctx: RoomContext) -> None:
        return None


async def _room_with(channel: Channel, store: InMemoryStore) -> tuple[RoomKit, str]:
    kit = RoomKit(store=store)
    _with_a_reader(kit)
    kit.register_channel(channel)
    room = await kit.create_room(room_id="carry")
    await kit.attach_channel(room.id, channel.channel_id)
    return kit, room.id


async def test_a_quiet_room_deserialises_its_history_once_per_message() -> None:
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        # Room creation and the first messages carry one-off costs; the budget
        # is what a message costs in a room already running.
        await _send(kit, room_id, "sms", 3)
        store.history_reads = 0

        await _send(kit, room_id, "sms", 5)
    finally:
        await kit.close()

    assert store.history_reads == 5, (
        f"{store.history_reads / 5:.1f} history reads per message — the locked pass "
        "is deserialising a history it was handed"
    )


async def test_a_directly_injected_event_deserialises_its_history_once() -> None:
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        await kit.send_event(room_id, "sms", TextContent(body="warm-up"))
        store.history_reads = 0

        await kit.send_event(room_id, "sms", TextContent(body="direct"))
    finally:
        await kit.close()

    assert store.history_reads == 1


async def test_the_locked_pass_sees_the_history_a_fresh_read_would_give() -> None:
    """The carry is an optimisation, not a semantic: hooks see what they saw."""
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    seen: list[tuple[list[str], list[str]]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def compare(event: RoomEvent, context: RoomContext) -> HookResult:
        fresh = await kit._build_context(room_id)
        seen.append(
            (
                [e.id for e in context.recent_events],
                [e.id for e in fresh.recent_events],
            )
        )
        return HookResult(action="allow")

    try:
        await _send(kit, room_id, "sms", 4)
    finally:
        await kit.close()

    assert len(seen) == 4
    for carried, fresh in seen:
        assert carried == fresh


async def test_an_event_committed_between_the_two_passes_is_not_missed() -> None:
    """The room's counter moved, so the carried history is refused."""
    store = CountingStore()
    kit = RoomKit(store=store)
    channel = InterleavingChannel("sms", kit)
    kit.register_channel(channel)
    room = await kit.create_room(room_id="carry")
    await kit.attach_channel(room.id, "sms")
    hook_history: list[list[str]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def record(event: RoomEvent, context: RoomContext) -> HookResult:
        if event.source.channel_id == "sms" and event.type == EventType.MESSAGE:
            hook_history.append([e.id for e in context.recent_events])
        return HookResult(action="allow")

    try:
        await _send(kit, room.id, "sms", 1)
    finally:
        await kit.close()

    interleaved = channel.interleaved
    assert interleaved is not None
    # The last recorded pass is the inbound message's own: the event committed
    # from handle_inbound landed after its pre-lock context was built, and the
    # locked pass must still have it.
    assert interleaved.id in hook_history[-1]


async def test_a_wider_window_bound_between_the_two_passes_is_read_in_full() -> None:
    """A channel bound since needs more history than the carried window holds."""
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        # Past the 50-event floor, so the carried window is saturated and a
        # 200-event window genuinely needs events it does not hold.
        await _send(kit, room_id, "sms", 60)
        narrow = await kit._build_context(room_id)
        assert len(narrow.recent_events) == 50

        kit.register_channel(WideWindowChannel("ai"))
        await kit.attach_channel(room_id, "ai")
        store.history_reads = 0
        widened = await kit._build_context(room_id, carrying=narrow)
    finally:
        await kit.close()

    assert store.history_reads == 1
    assert len(widened.recent_events) == 60


async def test_a_carried_window_narrower_than_asked_for_is_read_in_full() -> None:
    """The mirror image, asserted on the helper itself: never hand back less."""
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        await _send(kit, room_id, "sms", 3)
        carrying = await kit._build_context(room_id)
        store.history_reads = 0

        # The floor is 50; asking for 200 with a 50-wide carried window must
        # not be answered from it, even though this room has only 3 events.
        widened = await kit._build_context(room_id, recent_limit=200, carrying=carrying)
        # And asking for less than the carried window is answered from it.
        trimmed = await kit._build_context(room_id, recent_limit=2, carrying=carrying)
    finally:
        await kit.close()

    assert store.history_reads == 1
    assert len(widened.recent_events) == 3
    assert [e.id for e in trimmed.recent_events] == [e.id for e in carrying.recent_events[-2:]]


async def test_a_context_from_another_room_is_never_carried() -> None:
    """Two rooms at the same event count still have different conversations."""
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        elsewhere = await kit.create_room(room_id="elsewhere")
        await kit.attach_channel(elsewhere.id, "sms")
        await _send(kit, room_id, "sms", 2)
        await _send(kit, elsewhere.id, "sms", 2)

        # Same counter, so only the room itself can refuse this carry.
        other_context = await kit._build_context(elsewhere.id)
        here = await kit._build_context(room_id)
        assert other_context.room.event_count == here.room.event_count
        store.history_reads = 0
        context = await kit._build_context(room_id, carrying=other_context)
    finally:
        await kit.close()

    assert store.history_reads == 1
    assert [e.room_id for e in context.recent_events] == [room_id, room_id]


async def test_a_carried_context_never_outlives_its_room_state() -> None:
    """Room, bindings and participants are re-read; only history is carried."""
    store = CountingStore()
    kit, room_id = await _room_with(PlainChannel(), store)
    try:
        await _send(kit, room_id, "sms", 2)
        carrying = await kit._build_context(room_id)
        assert len(carrying.bindings) == 1

        kit.register_channel(PlainChannel("sms2"))
        await kit.attach_channel(room_id, "sms2")
        refreshed = await kit._build_context(room_id, carrying=carrying)
    finally:
        await kit.close()

    assert {b.channel_id for b in refreshed.bindings} == {"sms", "sms2"}


async def test_the_delivered_context_carries_the_committed_event() -> None:
    """RFC §10.2: the delivery set sees the trigger appended to the history."""
    store = CountingStore()
    channel: Any = PlainChannel("sms")
    other = PlainChannel("sms2")
    kit = RoomKit(store=store)
    _with_a_reader(kit)
    kit.register_channel(channel)
    kit.register_channel(other)
    room = await kit.create_room(room_id="carry")
    await kit.attach_channel(room.id, "sms")
    await kit.attach_channel(room.id, "sms2")
    try:
        await _send(kit, room.id, "sms", 3)
    finally:
        await kit.close()

    assert len(other.delivered_contexts) == 3
    for i, context in enumerate(other.delivered_contexts):
        bodies = [
            e.content.body for e in context.recent_events if isinstance(e.content, TextContent)
        ]
        assert bodies == [f"msg {n}" for n in range(i + 1)]
