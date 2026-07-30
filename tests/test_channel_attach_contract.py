"""What a channel gets to say about being attached to a room.

``attach_channel()`` used to be a store write announced as a success. A channel
with outside-world work to do on attachment — a conference channel creating its
SFU room, RFC §12.10.4 step 1 — could only hang that work off the
``ON_CHANNEL_ATTACHED`` hook, where the hook engine logs failures and never
raises them. The binding was already written, so a refused attachment came back
looking exactly like one that had worked.

The contract these tests describe replaces that: ``on_room_attached`` is awaited
by the attach itself, at the one point where undoing it costs nothing, and a
channel that raises there is a channel that has not been attached.
"""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent

ROOM = "room-1"


class RefusingBackendError(RuntimeError):
    """What an unreachable outside world raises through a channel."""


class _PlainChannel(Channel):
    """A channel that overrides neither half of the contract."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _RecordingChannel(_PlainChannel):
    """A channel that writes down when the framework talked to it.

    ``seen`` is the order of events as the channel observed them, which is what
    most of these tests are actually about: the point of the contract is that
    ``on_room_attached`` lands before anything else has heard of the
    attachment.
    """

    def __init__(self, channel_id: str, *, fail_attach: bool = False) -> None:
        super().__init__(channel_id)
        self.fail_attach = fail_attach
        self.fail_detach = False
        self.attached: list[ChannelBinding] = []
        self.detached: list[str] = []
        self.seen: list[str] = []

    async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
        self.seen.append("on_room_attached")
        if self.fail_attach:
            raise RefusingBackendError("the outside world said no")
        self.attached.append(binding)

    async def on_room_detached(self, room_id: str) -> None:
        self.seen.append("on_room_detached")
        if self.fail_detach:
            raise RefusingBackendError("the outside world said no")
        self.detached.append(room_id)


async def _kit_with(channel: Channel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    return kit


async def _stored_event_types(kit: RoomKit, room_id: str) -> list[EventType]:
    events = await kit.store.list_events(room_id, limit=100, visibility_filter=None)
    return [event.type for event in events]


class TestAttachOrder:
    async def test_the_channel_is_told_before_anything_else_is(self) -> None:
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)

        @kit.hook(
            HookTrigger.ON_CHANNEL_ATTACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            channel.seen.append("hook")

        await kit.attach_channel(ROOM, "sms")

        assert channel.seen == ["on_room_attached", "hook"]

    async def test_the_binding_the_channel_receives_is_the_stored_one(self) -> None:
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)

        binding = await kit.attach_channel(ROOM, "sms")

        assert [b.channel_id for b in channel.attached] == ["sms"]
        assert channel.attached[0].room_id == ROOM
        assert channel.attached[0].access == binding.access

    async def test_a_channel_with_nothing_to_establish_attaches_as_before(self) -> None:
        """The ABC's default is a no-op, so an ordinary channel is unaffected."""
        channel = _PlainChannel("sms")
        kit = await _kit_with(channel)

        binding = await kit.attach_channel(ROOM, "sms")

        assert binding.channel_id == "sms"
        assert [b.channel_id for b in await kit.list_bindings(ROOM)] == ["sms"]
        assert await kit.detach_channel(ROOM, "sms") is True


class TestRefusedAttach:
    async def test_the_failure_reaches_the_caller(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

    async def test_no_binding_survives(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        assert await kit.list_bindings(ROOM) == []

    async def test_nothing_was_announced(self) -> None:
        """No indexed event, no hook — so there is nothing to compensate.

        This is why the await sits between the binding write and the system
        event rather than where the hook used to fire: an event is indexed and
        does not come back, so a rollback from any later point would have to
        announce the detach of an attachment that never happened.
        """
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)
        fired: list[str] = []

        @kit.hook(
            HookTrigger.ON_CHANNEL_ATTACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            fired.append(event.room_id)

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        assert fired == []
        assert EventType.CHANNEL_ATTACHED not in await _stored_event_types(kit, ROOM)

    async def test_a_failed_rollback_does_not_hide_why(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The channel's refusal is the reason; a store that also fails is news."""
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        async def _broken_remove(room_id: str, channel_id: str) -> bool:
            raise RuntimeError("store is down too")

        kit.store.remove_binding = _broken_remove  # type: ignore[method-assign]

        with (
            caplog.at_level("ERROR", logger="roomkit.framework"),
            pytest.raises(RefusingBackendError),
        ):
            await kit.attach_channel(ROOM, "sms")

        assert "store is down too" in caplog.text

    async def test_the_room_can_be_attached_once_the_backend_recovers(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        channel.fail_attach = False
        binding = await kit.attach_channel(ROOM, "sms")

        assert binding.channel_id == "sms"
        assert [b.channel_id for b in await kit.list_bindings(ROOM)] == ["sms"]


class TestRefusedReattach:
    """Attaching over a live attachment, and being refused.

    Undoing the write is only the whole answer when there was nothing there
    before. A second attach *replaces* a binding, and a channel that refuses the
    new one has said nothing about the old: it is still attached, still holding
    whatever it joined. Rolling back by deleting the row took the room's only
    handle on that away — ``detach_channel()`` found nothing to remove and
    returned false, and the attachment it would have torn down ran on with
    nothing able to reach it.
    """

    async def test_the_binding_it_replaced_is_put_back(self) -> None:
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        first = await kit.attach_channel(ROOM, "sms")
        channel.fail_attach = True

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        bindings = await kit.list_bindings(ROOM)
        assert [b.channel_id for b in bindings] == ["sms"]
        assert bindings[0].access == first.access

    async def test_the_attachment_it_could_not_replace_can_still_be_detached(self) -> None:
        """What the restored binding is for: the room can still let go."""
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        channel.fail_attach = True

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        assert await kit.detach_channel(ROOM, "sms") is True
        assert channel.detached == [ROOM]

    async def test_a_first_attach_still_leaves_nothing_behind(self) -> None:
        """The other half of the same rule: there was no binding to restore."""
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        with pytest.raises(RefusingBackendError):
            await kit.attach_channel(ROOM, "sms")

        assert await kit.list_bindings(ROOM) == []
        assert await kit.detach_channel(ROOM, "sms") is False


class TestDetach:
    async def test_the_channel_lets_go_before_the_hooks_run(self) -> None:
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")

        @kit.hook(
            HookTrigger.ON_CHANNEL_DETACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            channel.seen.append("hook")

        channel.seen.clear()
        await kit.detach_channel(ROOM, "sms")

        assert channel.seen == ["on_room_detached", "hook"]

    async def test_a_failure_reaches_the_caller(self) -> None:
        """Nothing is rolled back — the binding is gone and the detach
        announced — but the error is not swallowed either."""
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        channel.fail_detach = True

        with pytest.raises(RefusingBackendError):
            await kit.detach_channel(ROOM, "sms")

        assert await kit.list_bindings(ROOM) == []
        assert EventType.CHANNEL_DETACHED in await _stored_event_types(kit, ROOM)

    async def test_a_failure_still_tells_the_rooms_observers(self) -> None:
        """The detach has happened as far as the room is concerned — the
        binding is gone and CHANNEL_DETACHED is indexed — so a channel that
        raised on its way out changes how *well* it let go, not whether. An
        observer told nothing goes on believing the channel is attached, which
        is exactly the state the detach existed to leave behind.
        """
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        channel.fail_detach = True
        fired: list[str] = []

        @kit.hook(
            HookTrigger.ON_CHANNEL_DETACHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            fired.append("hook")

        @kit.on("room_channel_detached")
        async def _framework_event(event: object) -> None:
            fired.append("framework_event")

        with pytest.raises(RefusingBackendError):
            await kit.detach_channel(ROOM, "sms")

        assert fired == ["hook", "framework_event"]

    async def test_an_announcement_that_fails_too_does_not_hide_why(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The channel's refusal is the reason the caller is owed; an
        announcement that fails as well is a second problem, not a replacement
        for the first.
        """
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        channel.fail_detach = True

        async def _broken_emit(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the status bus is down too")

        kit._emit_framework_event = _broken_emit  # type: ignore[method-assign]

        with (
            caplog.at_level("ERROR", logger="roomkit.framework"),
            pytest.raises(RefusingBackendError),
        ):
            await kit.detach_channel(ROOM, "sms")

        assert "the status bus is down too" in caplog.text

    async def test_an_unregistered_channel_is_not_told(self) -> None:
        """Its binding still goes: the room is the thing being detached from."""
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        kit.unregister_channel("sms")

        assert await kit.detach_channel(ROOM, "sms") is True
        assert channel.detached == []

    async def test_detaching_nothing_tells_no_one(self) -> None:
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)

        assert await kit.detach_channel(ROOM, "sms") is False
        assert channel.seen == []


class TestInboundAutoAttach:
    """A refused attach fails the inbound message that triggered it.

    ``process_inbound`` attaches a channel on three paths, and it is the one
    surface where this exception surfaces in code the integrator did not write.
    Failing is the decision: a message routed to a room whose conference does
    not exist has nowhere to go, and degrading would deliver it into a room that
    only looks attached.
    """

    async def test_on_the_auto_created_room_path(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = RoomKit()
        kit.register_channel(channel)

        with pytest.raises(RefusingBackendError):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms",
                    sender_id="user-1",
                    content=TextContent(body="hello"),
                )
            )

    async def test_on_the_named_but_unknown_room_path(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = RoomKit()
        kit.register_channel(channel)

        with pytest.raises(RefusingBackendError):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms",
                    sender_id="user-1",
                    content=TextContent(body="hello"),
                ),
                room_id="room-from-sip",
            )

        assert await kit.store.get_room("room-from-sip") is not None
        assert await kit.list_bindings("room-from-sip") == []

    async def test_on_the_existing_room_path(self) -> None:
        channel = _RecordingChannel("sms", fail_attach=True)
        kit = await _kit_with(channel)

        with pytest.raises(RefusingBackendError):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms",
                    sender_id="user-1",
                    content=TextContent(body="hello"),
                ),
                room_id=ROOM,
            )

        assert await kit.list_bindings(ROOM) == []

    async def test_a_room_already_attached_is_not_asked_again(self) -> None:
        """The refusal only bites on the auto-attach path."""
        channel = _RecordingChannel("sms")
        kit = await _kit_with(channel)
        await kit.attach_channel(ROOM, "sms")
        channel.fail_attach = True

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id="user-1",
                content=TextContent(body="hello"),
            ),
            room_id=ROOM,
        )

        assert not result.blocked
