"""A closed room refuses new events, at every entry (RFC §5.1, §10.1 step 6).

The inbound router already skipped non-ACTIVE rooms, so implicit routing was
never the hole. The holes were the paths that name the room: the explicit
``room_id`` argument, ``send_event()``, and the framework's own re-injection.
"""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.exceptions import RoomClosedError
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger, RoomStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent


class _Transport(Channel):
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
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


async def _kit() -> tuple[RoomKit, _Transport]:
    kit = RoomKit()
    src = _Transport("ws")
    kit.register_channel(src)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ws")
    return kit, src


async def _set_status(kit: RoomKit, room_id: str, status: RoomStatus) -> None:
    room = await kit._store.get_room(room_id)
    assert room is not None
    await kit._store.update_room(room.model_copy(update={"status": status}))


class TestClosedRoomRefuses:
    async def test_inbound_with_an_explicit_room_id_is_refused(self) -> None:
        kit, src = await _kit()
        await kit.close_room("r1")
        src.delivered.clear()

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is True
        assert result.reason == "room_closed"
        assert src.delivered == []

    async def test_send_event_raises(self) -> None:
        """Direct injection returns the committed event, so it cannot return a
        refusal — handing back one marked DELIVERED would be a lie."""
        kit, _src = await _kit()
        await kit.close_room("r1")

        with pytest.raises(RoomClosedError):
            await kit.send_event("r1", "ws", TextContent(body="hi"))

        events = await kit._store.list_events("r1")
        assert not any(getattr(e.content, "body", None) == "hi" for e in events)

    async def test_an_archived_room_refuses_too(self) -> None:
        kit, _src = await _kit()
        await _set_status(kit, "r1", RoomStatus.ARCHIVED)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is True
        assert result.reason == "room_closed"

    async def test_nothing_is_written_not_even_a_blocked_record(self) -> None:
        """§5.1: an audit record in a closed room is what the status forbids."""
        kit, _src = await _kit()
        await kit.close_room("r1")
        before = await kit._store.list_events("r1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        after = await kit._store.list_events("r1")
        assert [e.id for e in after] == [e.id for e in before]

    async def test_the_refusal_is_observable_with_the_one_data_contract(self) -> None:
        """RFC §8.2: ``room_refused_event`` carries ``status``, ``operation``
        and ``event_type`` whichever path refused; this is the inbound one."""
        kit, _src = await _kit()
        refused: list[FrameworkEvent] = []

        @kit.on("room_refused_event")
        async def on_refused(fe: FrameworkEvent) -> None:
            refused.append(fe)

        await kit.close_room("r1")
        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is True
        assert len(refused) == 1
        assert refused[0].room_id == "r1"
        assert refused[0].event_id is not None
        assert refused[0].data == {
            "status": str(RoomStatus.CLOSED),
            "operation": "inbound",
            "event_type": str(EventType.MESSAGE),
        }

    async def test_the_closure_itself_stays_observable(self) -> None:
        """The one exception: the transition reports itself, refusal or not."""
        kit, _src = await _kit()
        fired: list[str] = []

        @kit.hook(HookTrigger.ON_ROOM_CLOSED, execution=HookExecution.ASYNC)
        async def on_closed(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append(ctx.room.id)

        await kit.close_room("r1")

        assert fired == ["r1"]


class TestOpenRoomsAreUnaffected:
    async def test_an_active_room_still_accepts(self) -> None:
        kit, _src = await _kit()

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is False
        # The sender does not receive its own event (§7.5-5); the timeline is
        # what says the event was accepted.
        events = await kit._store.list_events("r1")
        assert any(getattr(e.content, "body", None) == "hi" for e in events)

    async def test_a_paused_room_still_accepts(self) -> None:
        """PAUSED is resumable, not terminal — it must keep taking events."""
        kit, _src = await _kit()
        await _set_status(kit, "r1", RoomStatus.PAUSED)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is False

    async def test_history_stays_readable_after_closing(self) -> None:
        kit, _src = await _kit()
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        await kit.close_room("r1")

        events = await kit._store.list_events("r1")
        assert any(getattr(e.content, "body", None) == "hi" for e in events)

    async def test_reopening_a_room_makes_it_accept_again(self) -> None:
        kit, _src = await _kit()
        await kit.close_room("r1")
        await _set_status(kit, "r1", RoomStatus.ACTIVE)

        result = await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert result.blocked is False
