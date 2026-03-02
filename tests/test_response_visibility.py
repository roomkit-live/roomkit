"""Tests for response_visibility — controlling AI response delivery scope."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# Mock providers & channels
# ---------------------------------------------------------------------------


class _StreamingAIProvider(AIProvider):
    """Mock AI that returns a streaming response."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or ["Hello ", "world!"]

    @property
    def model_name(self) -> str:
        return "mock-streaming"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


class _SourceChannel(Channel):
    """Source channel that stamps response_visibility on inbound events."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str, *, response_visibility: str | None = None) -> None:
        super().__init__(channel_id)
        self._response_visibility = response_visibility

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
            ),
            content=message.content,
            response_visibility=self._response_visibility,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _StreamingTarget(Channel):
    """Transport channel that supports streaming delivery."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []
        self.stream_texts: list[str] = []

    @property
    def ai_delivered(self) -> list[RoomEvent]:
        """Events delivered from the AI channel only."""
        return [e for e in self.delivered if e.source.channel_id == "ai"]

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
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
        self.delivered.append(event)
        return ChannelOutput.empty()

    async def deliver_stream(
        self,
        stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> None:
        chunks: list[str] = []
        async for delta in stream:
            chunks.append(delta)
        self.stream_texts.append("".join(chunks))


class _SimpleTarget(Channel):
    """Simple transport channel without streaming support."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    @property
    def ai_delivered(self) -> list[RoomEvent]:
        """Events delivered from the AI channel only."""
        return [e for e in self.delivered if e.source.channel_id == "ai"]

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
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
        self.delivered.append(event)
        return ChannelOutput.empty()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _build_kit(
    response_visibility: str | None,
    streaming_ids: list[str] | None = None,
    simple_ids: list[str] | None = None,
) -> tuple[
    RoomKit,
    _SourceChannel,
    dict[str, _StreamingTarget],
    dict[str, _SimpleTarget],
]:
    """Build a kit with source + AI + streaming/simple transport targets."""
    streaming_ids = streaming_ids or []
    simple_ids = simple_ids or []

    source = _SourceChannel("src", response_visibility=response_visibility)
    ai = AIChannel("ai", provider=_StreamingAIProvider())

    streaming = {sid: _StreamingTarget(sid) for sid in streaming_ids}
    simple = {sid: _SimpleTarget(sid) for sid in simple_ids}

    kit = RoomKit()
    kit.register_channel(source)
    kit.register_channel(ai)
    for ch in streaming.values():
        kit.register_channel(ch)
    for ch in simple.values():
        kit.register_channel(ch)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    for sid in streaming_ids:
        await kit.attach_channel("r1", sid)
    for sid in simple_ids:
        await kit.attach_channel("r1", sid)

    return kit, source, streaming, simple


async def _send_and_wait(kit: RoomKit) -> None:
    """Send a message through the source channel and wait for delivery."""
    msg = InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body="Hi"))
    await kit.process_inbound(msg)
    await asyncio.sleep(0.3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponseVisibilityDefault:
    """No response_visibility → current behavior preserved."""

    async def test_default_streams_to_first_target(self) -> None:
        """Without response_visibility, streaming goes to first target."""
        kit, _, streaming, _ = await _build_kit(
            response_visibility=None, streaming_ids=["st1", "st2"]
        )
        await _send_and_wait(kit)

        # V1: single streaming target gets the stream
        assert len(streaming["st1"].stream_texts) == 1
        assert streaming["st1"].stream_texts[0] == "Hello world!"
        # Second target gets reentry broadcast (non-streaming delivery)
        assert len(streaming["st2"].ai_delivered) == 1
        await kit.close()

    async def test_default_delivers_to_simple_targets(self) -> None:
        """Without response_visibility, simple targets get reentry delivery."""
        kit, _, _, simple = await _build_kit(response_visibility=None, simple_ids=["t1", "t2"])
        await _send_and_wait(kit)

        assert len(simple["t1"].ai_delivered) == 1
        assert len(simple["t2"].ai_delivered) == 1
        await kit.close()


class TestResponseVisibilitySingleChannel:
    """response_visibility targets a single channel ID."""

    async def test_streaming_only_to_targeted_channel(self) -> None:
        """response_visibility='st2' → only st2 receives streaming."""
        kit, _, streaming, _ = await _build_kit(
            response_visibility="st2", streaming_ids=["st1", "st2"]
        )
        await _send_and_wait(kit)

        # st2 should get the stream
        assert len(streaming["st2"].stream_texts) == 1
        assert streaming["st2"].stream_texts[0] == "Hello world!"
        # st1 should NOT get streaming or reentry (visibility blocks it)
        assert len(streaming["st1"].stream_texts) == 0
        assert len(streaming["st1"].ai_delivered) == 0
        await kit.close()

    async def test_reentry_only_to_targeted_simple_channel(self) -> None:
        """response_visibility='t1' → only t1 gets reentry delivery."""
        kit, _, _, simple = await _build_kit(response_visibility="t1", simple_ids=["t1", "t2"])
        await _send_and_wait(kit)

        assert len(simple["t1"].ai_delivered) == 1
        assert len(simple["t2"].ai_delivered) == 0
        await kit.close()


class TestResponseVisibilityNone:
    """response_visibility='none' → no delivery at all."""

    async def test_none_skips_all_streaming(self) -> None:
        """response_visibility='none' → no streaming targets."""
        kit, _, streaming, _ = await _build_kit(
            response_visibility="none", streaming_ids=["st1", "st2"]
        )
        await _send_and_wait(kit)

        assert len(streaming["st1"].stream_texts) == 0
        assert len(streaming["st2"].stream_texts) == 0
        # Reentry broadcast also blocked (visibility="none")
        assert len(streaming["st1"].ai_delivered) == 0
        assert len(streaming["st2"].ai_delivered) == 0
        await kit.close()

    async def test_none_skips_simple_reentry(self) -> None:
        """response_visibility='none' → simple targets get nothing."""
        kit, _, _, simple = await _build_kit(response_visibility="none", simple_ids=["t1"])
        await _send_and_wait(kit)

        assert len(simple["t1"].ai_delivered) == 0
        await kit.close()


class TestResponseVisibilityCommaSeparated:
    """response_visibility with comma-separated channel IDs."""

    async def test_comma_separated_streaming_targets(self) -> None:
        """Only listed channels receive streaming."""
        kit, _, streaming, _ = await _build_kit(
            response_visibility="st1,st3",
            streaming_ids=["st1", "st2", "st3"],
        )
        await _send_and_wait(kit)

        # st1 is first match → gets stream
        assert len(streaming["st1"].stream_texts) == 1
        # st3 is in the list → gets reentry delivery
        assert len(streaming["st3"].ai_delivered) == 1
        # st2 not in list → nothing
        assert len(streaming["st2"].stream_texts) == 0
        assert len(streaming["st2"].ai_delivered) == 0
        await kit.close()

    async def test_comma_separated_simple_targets(self) -> None:
        """Only listed simple channels receive reentry delivery."""
        kit, _, _, simple = await _build_kit(
            response_visibility="t1,t3",
            simple_ids=["t1", "t2", "t3"],
        )
        await _send_and_wait(kit)

        assert len(simple["t1"].ai_delivered) == 1
        assert len(simple["t3"].ai_delivered) == 1
        assert len(simple["t2"].ai_delivered) == 0
        await kit.close()


class TestResponseVisibilityStoredEvent:
    """Verify the stored event carries the correct visibility."""

    async def test_stored_event_has_visibility(self) -> None:
        """Stored AI response event has visibility from response_visibility."""
        kit, _, _, _ = await _build_kit(response_visibility="st1", streaming_ids=["st1"])
        await _send_and_wait(kit)

        events = await kit.get_timeline("r1")
        ai_events = [e for e in events if e.source.channel_id == "ai"]
        assert len(ai_events) == 1
        assert ai_events[0].visibility == "st1"
        await kit.close()

    async def test_stored_event_default_visibility(self) -> None:
        """Without response_visibility, stored event has visibility='all'."""
        kit, _, _, _ = await _build_kit(response_visibility=None, streaming_ids=["st1"])
        await _send_and_wait(kit)

        events = await kit.get_timeline("r1")
        ai_events = [e for e in events if e.source.channel_id == "ai"]
        assert len(ai_events) == 1
        assert ai_events[0].visibility == "all"
        await kit.close()


class TestResponseVisibilityNonStreamingReentry:
    """Non-streaming AI response path (reentry events in _process_locked)."""

    async def test_reentry_visibility_stamped_from_hook(self) -> None:
        """BEFORE_BROADCAST hook stamps response_visibility; AI reentry respects it."""
        kit = RoomKit()
        source = _SimpleTarget("src")
        t1 = _SimpleTarget("t1")
        t2 = _SimpleTarget("t2")
        ai_provider = MockAIProvider(responses=["AI says hello"])
        ai = AIChannel("ai", provider=ai_provider)

        kit.register_channel(source)
        kit.register_channel(t1)
        kit.register_channel(t2)
        kit.register_channel(ai)

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")
        await kit.attach_channel("r1", "t1")
        await kit.attach_channel("r1", "t2")
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def stamp(event: RoomEvent, context: RoomContext) -> HookResult:
            if event.source.channel_id == "src":
                return HookResult(
                    action="modify",
                    event=event.model_copy(update={"response_visibility": "t1"}),
                )
            return HookResult(action="allow")

        msg = InboundMessage(channel_id="src", sender_id="u1", content=TextContent(body="Hi"))
        await kit.process_inbound(msg)

        # t1 should get the AI response, t2 should not
        assert len(t1.ai_delivered) == 1
        assert len(t2.ai_delivered) == 0
        await kit.close()


class TestResponseVisibilityOnSendEvent:
    """Verify send_event stamps response_visibility on the event."""

    async def test_send_event_stamps_response_visibility(self) -> None:
        """send_event passes response_visibility to the RoomEvent."""
        kit = RoomKit()
        source = _SimpleTarget("src")
        kit.register_channel(source)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")

        event = await kit.send_event(
            room_id="r1",
            channel_id="src",
            content=TextContent(body="test"),
            response_visibility="ws1",
        )
        assert event.response_visibility == "ws1"
        await kit.close()

    async def test_send_event_default_response_visibility(self) -> None:
        """send_event defaults response_visibility to None."""
        kit = RoomKit()
        source = _SimpleTarget("src")
        kit.register_channel(source)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")

        event = await kit.send_event(
            room_id="r1",
            channel_id="src",
            content=TextContent(body="test"),
        )
        assert event.response_visibility is None
        await kit.close()
