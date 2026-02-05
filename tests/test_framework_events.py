"""Tests for framework events expansion (Phase C, Area 9)."""

from __future__ import annotations

from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult
from tests.test_framework import SimpleChannel


def _msg(channel_id: str = "sms1") -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        sender_id="user1",
        content=TextContent(body="hello"),
    )


# ---- Tests: on() decorator type filtering ----


class TestOnDecoratorFiltering:
    async def test_handler_receives_matching_events_only(self) -> None:
        """Handler registered for 'room_created' does not fire for 'room_closed'."""
        kit = RoomKit()
        created: list[FrameworkEvent] = []
        closed: list[FrameworkEvent] = []

        @kit.on("room_created")
        async def on_created(fe: FrameworkEvent) -> None:
            created.append(fe)

        @kit.on("room_closed")
        async def on_closed(fe: FrameworkEvent) -> None:
            closed.append(fe)

        await kit.create_room(room_id="r1")
        assert len(created) == 1
        assert len(closed) == 0

        await kit.close_room("r1")
        assert len(created) == 1
        assert len(closed) == 1

    async def test_multiple_handlers_same_type(self) -> None:
        """Multiple handlers for the same event type all fire."""
        kit = RoomKit()
        calls: list[str] = []

        @kit.on("room_created")
        async def handler_a(fe: FrameworkEvent) -> None:
            calls.append("a")

        @kit.on("room_created")
        async def handler_b(fe: FrameworkEvent) -> None:
            calls.append("b")

        await kit.create_room(room_id="r1")
        assert calls == ["a", "b"]


# ---- Tests: room lifecycle framework events ----


class TestRoomLifecycleEvents:
    async def test_room_created_event_emitted(self) -> None:
        """create_room emits 'room_created' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("room_created")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        await kit.create_room(room_id="r1")
        assert len(received) == 1
        assert received[0].type == "room_created"
        assert received[0].room_id == "r1"

    async def test_room_closed_event_emitted(self) -> None:
        """close_room emits 'room_closed' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("room_closed")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        await kit.create_room(room_id="r1")
        await kit.close_room("r1")
        assert len(received) == 1
        assert received[0].type == "room_closed"
        assert received[0].room_id == "r1"


# ---- Tests: channel lifecycle framework events ----


class TestChannelLifecycleEvents:
    async def test_attach_event_emitted(self) -> None:
        """attach_channel emits 'room_channel_attached' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("room_channel_attached")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        assert len(received) == 1
        assert received[0].channel_id == "sms1"
        assert received[0].room_id == "r1"

    async def test_detach_event_emitted(self) -> None:
        """detach_channel emits 'room_channel_detached' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("room_channel_detached")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.detach_channel("r1", "sms1")

        assert len(received) == 1
        assert received[0].channel_id == "sms1"
        assert received[0].room_id == "r1"

    async def test_detach_no_event_on_noop(self) -> None:
        """detach_channel does not emit event if nothing removed."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("room_channel_detached")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        await kit.create_room(room_id="r1")
        await kit.detach_channel("r1", "nonexistent")

        assert len(received) == 0


# ---- Tests: delivery tracking events ----


class TestDeliveryTrackingEvents:
    async def test_delivery_succeeded_event(self) -> None:
        """Successful delivery emits 'delivery_succeeded' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("delivery_succeeded")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        await kit.process_inbound(_msg())

        # sms2 should have received the broadcast → delivery_succeeded
        delivered_channels = [fe.channel_id for fe in received]
        assert "sms2" in delivered_channels

    async def test_delivery_failed_event(self) -> None:
        """Failed delivery emits 'delivery_failed' framework event."""
        from roomkit.channels.base import Channel

        class FailingChannel(Channel):
            channel_type = ChannelType.EMAIL

            def __init__(self, channel_id: str) -> None:
                super().__init__(channel_id)

            async def handle_inbound(
                self, message: InboundMessage, context: RoomContext
            ) -> RoomEvent:
                return RoomEvent(
                    room_id=context.room.id,
                    source=EventSource(
                        channel_id=self.channel_id,
                        channel_type=self.channel_type,
                    ),
                    content=message.content,
                )

            async def deliver(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                raise RuntimeError("Delivery failed!")

        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("delivery_failed")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        ch1 = SimpleChannel("sms1")
        ch2 = FailingChannel("email1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "email1")

        await kit.process_inbound(_msg())

        failed_channels = [fe.channel_id for fe in received]
        assert "email1" in failed_channels


# ---- Tests: hook error framework events ----


class TestHookErrorEvents:
    async def test_hook_error_event_emitted(self) -> None:
        """A failing sync hook emits a 'hook_error' framework event."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("hook_error")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        # Register a sync hook that raises
        async def failing_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            raise ValueError("Hook exploded!")

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=failing_hook,
                name="failing_hook",
            )
        )

        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        await kit.process_inbound(_msg())

        assert len(received) >= 1
        assert received[0].data["hook"] == "failing_hook"
        assert "exploded" in received[0].data["error"].lower()


# ---- Tests: event_processed framework event ----


class TestEventProcessedEvent:
    async def test_event_processed_fires_on_success(self) -> None:
        """event_processed fires after successful inbound processing."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("event_processed")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        result = await kit.process_inbound(_msg())
        assert len(received) == 1
        assert received[0].event_id == result.event.id

    async def test_event_blocked_fires_on_block(self) -> None:
        """event_blocked fires when a sync hook blocks."""
        kit = RoomKit()
        received: list[FrameworkEvent] = []

        @kit.on("event_blocked")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("compliance")

        kit.hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=blocker,
                name="blocker",
            )
        )

        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("sms2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "sms2")

        result = await kit.process_inbound(_msg())
        assert result.blocked is True
        assert len(received) == 1
        assert received[0].data["reason"] == "compliance"
