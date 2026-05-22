"""Tests for BEFORE_BROADCAST side effects on the reentry path.

Reentry events (AI responses queued by ``broadcast()`` as the user's
message bounces through an intelligence channel) flow through a second
``BEFORE_BROADCAST`` pass before being re-broadcast. RFC § 9.5 mandates
the same block-handling behaviour on both passes:

  blocked? → store BLOCKED, emit event_blocked, deliver InjectedEvents

These tests lock that symmetry in place and also cover the allow/modify
reentry path's delivery of InjectedEvents.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from tests.test_framework import AILikeChannel, SimpleChannel


class RecordingChannel(Channel):
    """Transport-category channel that records every ``on_event`` call.

    Used as the delivery target for InjectedEvents so tests can assert
    that ``_deliver_injected_events`` was invoked on the reentry path.
    """

    channel_type = ChannelType.SMS
    category = ChannelCategory.TRANSPORT

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.on_event_calls: list[RoomEvent] = []
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.on_event_calls.append(event)
        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


def _ai_room_setup() -> tuple[RoomKit, SimpleChannel, AILikeChannel]:
    """Build a room with a transport channel (sms1) and an AI channel (ai1).

    The AI channel emits a reentry event on every inbound — that
    reentry is what the BEFORE_BROADCAST hook under test will see.
    """
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    ai = AILikeChannel("ai1", response="AI reply")
    kit.register_channel(sms)
    kit.register_channel(ai)
    return kit, sms, ai


def _user_msg() -> InboundMessage:
    return InboundMessage(
        channel_id="sms1",
        sender_id="user1",
        content=TextContent(body="hello"),
    )


async def _attach(kit: RoomKit, *channel_ids: str) -> None:
    await kit.create_room(room_id="r1")
    for cid in channel_ids:
        await kit.attach_channel("r1", cid)


# ── Block path ──────────────────────────────────────────────────────


class TestReentryBlockPersistsBlockedEvent:
    async def test_ai_reentry_blocked_is_stored_as_blocked(self) -> None:
        kit, _sms, _ai = _ai_room_setup()
        await _attach(kit, "sms1", "ai1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_ai_reentry")
        async def block_ai(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_type == ChannelType.AI:
                return HookResult.block("policy: ai must be reviewed")
            return HookResult.allow()

        await kit.process_inbound(_user_msg())

        events = await kit.store.list_events("r1")
        ai_events = [e for e in events if e.source.channel_type == ChannelType.AI]
        assert len(ai_events) == 1
        assert ai_events[0].status == EventStatus.BLOCKED
        # ``blocked_by`` is whichever non-None field the hook supplied;
        # the helper falls back to ``reason`` when ``blocked_by`` is None.
        assert ai_events[0].blocked_by in {"block_ai_reentry", "policy: ai must be reviewed"}


class TestReentryBlockEmitsFrameworkEvent:
    async def test_event_blocked_fires_for_reentry_block(self) -> None:
        kit, _sms, _ai = _ai_room_setup()
        await _attach(kit, "sms1", "ai1")

        received: list[FrameworkEvent] = []

        @kit.on("event_blocked")
        async def on_blocked(fe: FrameworkEvent) -> None:
            received.append(fe)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_ai_reentry")
        async def block_ai(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_type == ChannelType.AI:
                return HookResult.block("policy violation")
            return HookResult.allow()

        await kit.process_inbound(_user_msg())

        ai_blocks = [fe for fe in received if fe.data.get("reason") == "policy violation"]
        assert len(ai_blocks) == 1


class TestReentryBlockDeliversInjectedEvents:
    async def test_injected_events_reach_target_on_reentry_block(self) -> None:
        kit, _sms, _ai = _ai_room_setup()
        recorder = RecordingChannel("rec1")
        kit.register_channel(recorder)
        await _attach(kit, "sms1", "ai1", "rec1")

        followup = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="hook", channel_type=ChannelType.SMS),
            content=TextContent(body="injected from block"),
        )

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_with_inject")
        async def block_and_inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_type == ChannelType.AI:
                return HookResult.block(
                    "rejected — see followup",
                    injected=[InjectedEvent(event=followup, target_channel_ids=["rec1"])],
                )
            return HookResult.allow()

        await kit.process_inbound(_user_msg())

        assert any(e.id == followup.id for e in recorder.on_event_calls), (
            "RecordingChannel.on_event must be invoked for the injected event "
            "delivered by _handle_block on the reentry block path."
        )


# ── Allow / modify path ─────────────────────────────────────────────


class TestReentryAllowDeliversInjectedEvents:
    async def test_injected_events_reach_target_on_reentry_allow(self) -> None:
        kit, _sms, _ai = _ai_room_setup()
        recorder = RecordingChannel("rec1")
        kit.register_channel(recorder)
        await _attach(kit, "sms1", "ai1", "rec1")

        followup = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="hook", channel_type=ChannelType.SMS),
            content=TextContent(body="injected on allow"),
        )

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="allow_with_inject")
        async def allow_and_inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_type == ChannelType.AI:
                return HookResult.allow(
                    injected=[InjectedEvent(event=followup, target_channel_ids=["rec1"])],
                )
            return HookResult.allow()

        await kit.process_inbound(_user_msg())

        assert any(e.id == followup.id for e in recorder.on_event_calls)


class TestReentryModifyDeliversInjectedEvents:
    async def test_injected_events_reach_target_on_reentry_modify(self) -> None:
        kit, _sms, _ai = _ai_room_setup()
        recorder = RecordingChannel("rec1")
        kit.register_channel(recorder)
        await _attach(kit, "sms1", "ai1", "rec1")

        followup = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="hook", channel_type=ChannelType.SMS),
            content=TextContent(body="injected on modify"),
        )

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="modify_with_inject")
        async def modify_and_inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_type == ChannelType.AI:
                modified = event.model_copy(
                    update={"content": TextContent(body="AI reply (reviewed)")}
                )
                return HookResult.modify(
                    event=modified,
                    injected=[InjectedEvent(event=followup, target_channel_ids=["rec1"])],
                )
            return HookResult.allow()

        await kit.process_inbound(_user_msg())

        assert any(e.id == followup.id for e in recorder.on_event_calls)
