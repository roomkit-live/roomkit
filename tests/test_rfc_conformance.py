"""RFC conformance tests for the inbound/direct-injection pipeline.

These tests encode normative invariants from ``roomkit-specs/roomkit-rfc.md``
that the current implementation violates. They are written to FAIL against the
pre-fix code and PASS once the Lot 1 P0 fixes land:

- §10.3 — an EDIT/DELETE blocked by a BEFORE_BROADCAST hook MUST NOT mutate
  the target event (mutation happens only after the allow decision).
- §10.5 / §10.1 — ``send_event()`` direct injection MUST traverse the normal
  pipeline, BEFORE_BROADCAST hooks included.
- §7.5 — a source whose binding cannot write (READ_ONLY / NONE / muted) MUST
  NOT inject a DELIVERED event into the timeline.
- §8.1 — every stored event MUST receive a unique, monotonically increasing
  index assigned atomically (no duplicate index 0 for injected events).
"""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import (
    DeleteContent,
    EditContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult, InjectedEvent


class RecordingTransport(Channel):
    """Transport channel that honors ``message.event_type`` and records
    everything delivered to it (so tests can assert (non-)broadcast)."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


async def _setup(
    *,
    source_access: Access = Access.READ_WRITE,
    source_muted: bool = False,
) -> tuple[RoomKit, RecordingTransport, RecordingTransport]:
    """Room with a source transport (``src``) and a target transport (``dst``)."""
    kit = RoomKit()
    src = RecordingTransport("src")
    dst = RecordingTransport("dst")
    kit.register_channel(src)
    kit.register_channel(dst)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src", access=source_access)
    await kit.attach_channel("r1", "dst")
    if source_muted:
        await kit.mute("r1", "src")
    return kit, src, dst


async def _send(kit: RoomKit, channel_id: str, body: str, **kw: object) -> RoomEvent:
    msg = InboundMessage(
        channel_id=channel_id,
        sender_id="u1",
        content=TextContent(body=body),
        **kw,  # type: ignore[arg-type]
    )
    result = await kit.process_inbound(msg)
    assert result.event is not None
    return result.event


class TestEditDeleteMutationOrdering:
    """§10.3 — a blocked EDIT/DELETE must not mutate the target event."""

    async def test_blocked_edit_does_not_mutate_target(self) -> None:
        kit, src, _dst = await _setup()
        original = await _send(kit, "src", "v1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def block_edits(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.type is EventType.EDIT:
                return HookResult.block(reason="moderated")
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=original.id,
                    new_content=TextContent(body="v2-should-not-apply"),
                    edit_source="u1",
                ),
            )
        )

        stored = await kit.store.get_event(original.id)
        assert stored is not None
        assert isinstance(stored.content, TextContent)
        # The moderation hook blocked the edit → original content unchanged.
        assert stored.content.body == "v1"
        assert stored.metadata.get("edited") is not True

    async def test_blocked_delete_does_not_mark_target(self) -> None:
        kit, src, _dst = await _setup()
        original = await _send(kit, "src", "keep me")

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def block_deletes(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.type is EventType.DELETE:
                return HookResult.block(reason="moderated")
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.DELETE,
                content=DeleteContent(target_event_id=original.id),
            )
        )

        stored = await kit.store.get_event(original.id)
        assert stored is not None
        assert stored.metadata.get("deleted") is not True


class TestSendEventTraversesHooks:
    """§10.5 / §10.1 — send_event() must run BEFORE_BROADCAST hooks."""

    async def test_send_event_blocked_is_not_broadcast(self) -> None:
        kit, _src, dst = await _setup()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def block_all(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block(reason="moderated")

        await kit.send_event("r1", "src", TextContent(body="direct injection"))

        # The hook blocked the event: it must not reach other channels...
        assert dst.delivered == []
        # ...and it must be persisted BLOCKED, never DELIVERED.
        events = await kit.store.list_events("r1")
        injected = [
            e
            for e in events
            if isinstance(e.content, TextContent) and e.content.body == "direct injection"
        ]
        assert injected, "send_event should still persist the (blocked) event"
        assert all(e.status is EventStatus.BLOCKED for e in injected)


class TestSourceWritePermission:
    """§7.5 — a non-writable source must not inject a DELIVERED event."""

    @pytest.mark.parametrize(
        ("access", "muted"),
        [(Access.READ_ONLY, False), (Access.NONE, False), (Access.READ_WRITE, True)],
    )
    async def test_non_writable_source_event_blocked(self, access: Access, muted: bool) -> None:
        kit, _src, dst = await _setup(source_access=access, source_muted=muted)

        event = await _send(kit, "src", "should not write")

        # Not broadcast to other channels.
        assert dst.delivered == []
        # Persisted as BLOCKED, not DELIVERED.
        stored = await kit.store.get_event(event.id)
        assert stored is not None
        assert stored.status is EventStatus.BLOCKED


class TestAtomicIndexing:
    """§8.1 — every stored event gets a unique, monotonic, atomic index."""

    async def test_injected_event_receives_atomic_index(self) -> None:
        kit, _src, _dst = await _setup()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def inject_on_trigger(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and event.content.body == "trigger":
                ghost = RoomEvent(
                    room_id=event.room_id,
                    source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
                    content=TextContent(body="injected-ghost"),
                )
                return HookResult.allow(injected=[InjectedEvent(event=ghost)])
            return HookResult.allow()

        await _send(kit, "src", "first")
        await _send(kit, "src", "trigger")

        events = await kit.store.list_events("r1")
        indexes = sorted(e.index for e in events)
        # RFC §8.1: indexes start at 0 and increase by exactly 1 — no duplicates.
        assert len(indexes) == len(set(indexes)), f"duplicate indexes: {indexes}"
        assert indexes == list(range(len(indexes))), f"non-contiguous indexes: {indexes}"
