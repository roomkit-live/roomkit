"""Tests for EventRouter."""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.event_router import EventRouter
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from tests.conftest import make_binding, make_event


class StubChannel(Channel):
    """Simple channel that records deliveries."""

    channel_type = ChannelType.WEBSOCKET
    category = ChannelCategory.TRANSPORT

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


class RespondingChannel(Channel):
    """Channel that produces a response event via on_event (intelligence)."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, chain_depth: int = 0) -> None:
        super().__init__(channel_id)
        self._chain_depth = chain_depth

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        resp = make_event(
            channel_id=self.channel_id,
            channel_type=ChannelType.AI,
            body="AI response",
            chain_depth=self._chain_depth,
        )
        return ChannelOutput(responded=True, response_events=[resp])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


def _make_router(
    channels: dict[str, Channel],
    max_chain_depth: int = 5,
) -> EventRouter:
    return EventRouter(channels=channels, max_chain_depth=max_chain_depth)


def _make_context(bindings: list[ChannelBinding]) -> RoomContext:
    return RoomContext(room=Room(id="test-room"), bindings=bindings)


class TestAccessFiltering:
    async def test_read_only_source_blocked(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            access=Access.READ_ONLY,
        )
        b2 = make_binding("ch2")

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.outputs) == 0

    async def test_write_only_target_excluded(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            access=Access.WRITE_ONLY,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" not in result.outputs


class TestMuteBlocking:
    async def test_muted_source_blocked(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            muted=True,
        )
        b2 = make_binding("ch2")

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.outputs) == 0

    async def test_muted_target_receives_event_but_responses_suppressed(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            muted=True,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        # Muted channels still receive events via on_event (RFC: reading continues)
        assert "ch2" in result.outputs
        # But response_events are suppressed (no reentry)
        assert len(result.reentry_events) == 0


class TestVisibility:
    async def test_all_visibility(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        ch3 = StubChannel("ch3")
        router = _make_router({"ch1": ch1, "ch2": ch2, "ch3": ch3})

        b1 = make_binding("ch1")
        b2 = make_binding("ch2")
        b3 = make_binding("ch3")

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2, b3]))
        assert len(result.outputs) == 2

    async def test_transport_visibility(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        ch3 = StubChannel("ch3")
        router = _make_router({"ch1": ch1, "ch2": ch2, "ch3": ch3})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            visibility="transport",
        )
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )
        b3 = ChannelBinding(
            channel_id="ch3",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2, b3]))
        assert "ch2" in result.outputs
        assert "ch3" not in result.outputs

    async def test_none_visibility(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            visibility="none",
        )
        b2 = make_binding("ch2")

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.outputs) == 0


class TestTranscoding:
    async def test_rich_to_text_transcoding(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.EMAIL,
            capabilities=ChannelCapabilities(
                media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH]
            ),
        )
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        )

        from roomkit.models.event import RichContent

        event = RoomEvent(
            room_id="test-room",
            source=make_event().source,
            content=RichContent(body="**hello**", plain_text="hello"),
        )
        await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(ch2.delivered) == 1
        delivered_content = ch2.delivered[0].content
        assert isinstance(delivered_content, TextContent)
        assert delivered_content.body == "hello"


class TestChainDepth:
    async def test_response_within_limit(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = RespondingChannel("ch2", chain_depth=1)
        router = _make_router({"ch1": ch1, "ch2": ch2}, max_chain_depth=5)

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.reentry_events) == 1

    async def test_response_exceeding_limit_dropped(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = RespondingChannel("ch2", chain_depth=5)
        router = _make_router({"ch1": ch1, "ch2": ch2}, max_chain_depth=5)

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.reentry_events) == 0


class TestSideEffects:
    async def test_side_effects_collected(self) -> None:
        from roomkit.models.task import Observation

        class SideEffectChannel(Channel):
            channel_type = ChannelType.AI
            category = ChannelCategory.INTELLIGENCE

            async def handle_inbound(
                self, message: InboundMessage, context: RoomContext
            ) -> RoomEvent:
                raise NotImplementedError

            async def on_event(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                obs = Observation(
                    id="obs1",
                    room_id=event.room_id,
                    channel_id=self.channel_id,
                    content="observation text",
                    category="test",
                )
                return ChannelOutput(observations=[obs])

            async def deliver(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                return ChannelOutput.empty()

        ch1 = StubChannel("ch1")
        ch2 = SideEffectChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(result.observations) == 1


class TestOnEventVsDeliver:
    """Test that on_event is called for all channels, deliver only for transport."""

    async def test_transport_gets_on_event_and_deliver(self) -> None:
        """Transport channels get both on_event() and deliver() called."""

        class TrackingTransport(Channel):
            channel_type = ChannelType.WEBSOCKET
            category = ChannelCategory.TRANSPORT

            def __init__(self, channel_id: str) -> None:
                super().__init__(channel_id)
                self.on_event_called = False
                self.deliver_called = False

            async def handle_inbound(
                self, message: InboundMessage, context: RoomContext
            ) -> RoomEvent:
                raise NotImplementedError

            async def on_event(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                self.on_event_called = True
                return ChannelOutput.empty()

            async def deliver(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                self.deliver_called = True
                return ChannelOutput.empty()

        ch1 = StubChannel("ch1")
        ch2 = TrackingTransport("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        await router.broadcast(event, b1, _make_context([b1, b2]))
        assert ch2.on_event_called
        assert ch2.deliver_called

    async def test_intelligence_gets_only_on_event(self) -> None:
        """Intelligence channels get on_event() but NOT deliver()."""

        class TrackingIntelligence(Channel):
            channel_type = ChannelType.AI
            category = ChannelCategory.INTELLIGENCE

            def __init__(self, channel_id: str) -> None:
                super().__init__(channel_id)
                self.on_event_called = False
                self.deliver_called = False

            async def handle_inbound(
                self, message: InboundMessage, context: RoomContext
            ) -> RoomEvent:
                raise NotImplementedError

            async def on_event(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                self.on_event_called = True
                return ChannelOutput.empty()

            async def deliver(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                self.deliver_called = True
                return ChannelOutput.empty()

        ch1 = StubChannel("ch1")
        ch2 = TrackingIntelligence("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        await router.broadcast(event, b1, _make_context([b1, b2]))
        assert ch2.on_event_called
        assert not ch2.deliver_called


class TestDeliveryResultTracking:
    async def test_delivery_outputs_tracked(self) -> None:
        """Delivery outputs from transport channels are tracked separately."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" in result.delivery_outputs

    async def test_intelligence_no_delivery_output(self) -> None:
        """Intelligence channels do not appear in delivery_outputs."""
        ch1 = StubChannel("ch1")
        ch2 = RespondingChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" not in result.delivery_outputs


class TestVisibilityStamping:
    async def test_event_visibility_stamped_from_source(self) -> None:
        """Event visibility is stamped from source binding before broadcast."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            visibility="transport",
        )
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        # Event starts with default "all" visibility
        event = make_event(channel_id="ch1")
        assert event.visibility == "all"

        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        # ch2 should be in outputs because it's transport and source visibility="transport"
        assert "ch2" in result.outputs

    async def test_visibility_not_overridden_if_already_set(self) -> None:
        """Event with non-'all' visibility is not overridden by source binding.

        The stamping logic only applies when event.visibility == "all".
        If the event already has a specific visibility, it's preserved.
        Routing uses source_binding.visibility for target filtering.
        """
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        # Source binding has "transport" visibility
        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            visibility="transport",
        )
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        # Event already has "intelligence" visibility (not "all")
        # The stamping should NOT change it from "intelligence" to "transport"
        event = make_event(channel_id="ch1", visibility="intelligence")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        # Routing uses source_binding.visibility="transport" → ch2 (transport) matches
        assert "ch2" in result.outputs


class TestBroadcastConcurrency:
    async def test_many_channels_no_data_loss(self) -> None:
        """Broadcast to 10+ channels should not lose any outputs."""
        n = 12
        channels: dict[str, StubChannel] = {}
        bindings: list[ChannelBinding] = []
        for i in range(n + 1):
            cid = f"ch{i}"
            channels[cid] = StubChannel(cid)
            bindings.append(make_binding(cid, channel_type=ChannelType.WEBSOCKET))

        router = _make_router(channels)
        source = bindings[0]
        event = make_event(channel_id="ch0")
        ctx = _make_context(bindings)

        result = await router.broadcast(event, source, ctx)
        # All targets except the source should appear in outputs
        assert len(result.outputs) == n


class TestContentTruncation:
    async def test_text_truncated_to_max_length(self) -> None:
        """Text content exceeding max_length is truncated with '...'."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
            capabilities=ChannelCapabilities(
                media_types=[ChannelMediaType.TEXT],
                max_length=20,
            ),
        )

        long_text = "A" * 50
        event = make_event(channel_id="ch1", body=long_text)
        result = await router.broadcast(event, b1, _make_context([b1, b2]))

        assert "ch2" in result.outputs
        assert len(ch2.delivered) == 1
        delivered_content = ch2.delivered[0].content
        assert isinstance(delivered_content, TextContent)
        assert len(delivered_content.body) == 20
        assert delivered_content.body.endswith("...")

    async def test_short_text_not_truncated(self) -> None:
        """Text within max_length is not modified."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = _make_router({"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
            capabilities=ChannelCapabilities(
                media_types=[ChannelMediaType.TEXT],
                max_length=100,
            ),
        )

        event = make_event(channel_id="ch1", body="short")
        await router.broadcast(event, b1, _make_context([b1, b2]))
        assert len(ch2.delivered) == 1
        content = ch2.delivered[0].content
        assert isinstance(content, TextContent)
        assert content.body == "short"
