"""Every point where the timeline can grow meets the same gates (RFC §5.1, §7.5).

RFC §5.1 requires the room-status refusal "at **every** point where the
timeline can grow… a hook's injected event, and the framework's own
re-injection alike", and §7.5 rule 2 requires an event whose source binding
cannot write to be stored BLOCKED rather than DELIVERED. These cover the
growth points outside the locked inbound path: the reentry pass, the
lifecycle system events and the responses of a muted source.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from tests.test_framework import AILikeChannel, SimpleChannel


class ClosingAIChannel(Channel):
    """An intelligence channel that closes the room while answering.

    ``on_event`` runs off the room lock, during the trigger's delivery set —
    so the close lands after the trigger committed and before the response's
    own commit pass takes the lock. That is exactly the race RFC §10.1 step 6
    covers for reentries.
    """

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, kit: RoomKit) -> None:
        super().__init__(channel_id)
        self._kit = kit

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        await self._kit.close_room(event.room_id)
        resp = RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body="answer after close"),
            chain_depth=event.chain_depth + 1,
        )
        return ChannelOutput(responded=True, response_events=[resp])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


def _user_msg() -> InboundMessage:
    return InboundMessage(
        channel_id="sms1",
        sender_id="user1",
        content=TextContent(body="hello"),
    )


class TestReentryMeetsTheStatusGate:
    async def test_response_landing_after_close_is_refused(self) -> None:
        """RFC §5.1 / §10.1 step 6 — a reentry re-enters the locked section
        and meets the status gate there; nothing is stored."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ClosingAIChannel("ai1", kit))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1")

        await kit.process_inbound(_user_msg())

        events = await kit.store.list_events("r1")
        assert [e for e in events if e.source.channel_type == ChannelType.AI] == []
        room = await kit.get_room("r1")
        assert room.event_count == len(events)


class TestNonWritableSourceResponses:
    async def test_read_only_source_response_is_blocked_not_delivered(self) -> None:
        """RFC §7.5 rule 2 — a READ_ONLY observer's answer is stored BLOCKED
        with ``source_read_only``, not DELIVERED and visible to every
        channel."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(AILikeChannel("ai1", response="observer reply"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", access=Access.READ_ONLY)

        await kit.process_inbound(_user_msg())

        events = await kit.store.list_events("r1")
        ai_events = [e for e in events if e.source.channel_type == ChannelType.AI]
        assert len(ai_events) == 1
        assert ai_events[0].status == EventStatus.BLOCKED
        assert ai_events[0].blocked_by == "source_read_only"

    async def test_muted_source_response_is_recorded_blocked(self) -> None:
        """RFC §7.5 rule 2 — muting silences the voice, it does not erase the
        record: the suppressed response is stored BLOCKED with
        ``source_muted``."""
        kit = RoomKit()
        sms = SimpleChannel("sms1")
        kit.register_channel(sms)
        kit.register_channel(AILikeChannel("ai1", response="muted reply"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", muted=True)

        await kit.process_inbound(_user_msg())

        events = await kit.store.list_events("r1")
        ai_events = [e for e in events if e.source.channel_type == ChannelType.AI]
        assert len(ai_events) == 1
        assert ai_events[0].status == EventStatus.BLOCKED
        assert ai_events[0].blocked_by == "source_muted"
        # Still not broadcast — the transport saw nothing.
        assert [e.content for e in sms.delivered if e.source.channel_id == "ai1"] == []


class TestLifecycleSystemEventsMeetTheStatusGate:
    async def test_member_change_on_closed_room_writes_nothing(self) -> None:
        """RFC §5.1 — lifecycle system events are timeline growth too."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.close_room("r1")

        before = await kit.store.list_events("r1")
        await kit.add_member("r1", "sms1", "user9", display_name="Late")
        after = await kit.store.list_events("r1")

        assert [e.id for e in after] == [e.id for e in before]
        room = await kit.get_room("r1")
        assert room.event_count == len(after)

    async def test_timer_close_still_records_its_own_transition(self) -> None:
        """The record OF the closing is the one write a closing room owes its
        timeline — it must survive the gate that it itself installs."""
        from datetime import UTC, datetime, timedelta

        from roomkit.models.room import RoomTimers

        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.set_room_timers(
            "r1",
            RoomTimers(
                closed_after_seconds=1,
                last_activity_at=datetime.now(UTC) - timedelta(seconds=60),
            ),
        )

        await kit.check_room_timers("r1")

        room = await kit.get_room("r1")
        assert room.status.value == "closed"
        events = await kit.store.list_events("r1")
        codes = [e.content.code for e in events if e.content.type == "system"]
        assert "room_closed_by_timer" in codes
