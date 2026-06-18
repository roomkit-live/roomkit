"""Streaming-no-tools path publishes thinking events on the realtime bus.

Regression test for the bug where ``_start_streaming_response`` used
``provider.generate_stream`` (text-only) and silently dropped thinking
deltas, leaving subscribers blind to the model's reasoning when
streaming was on and no tools were attached.

Also covers inline delivery of :class:`ThinkingDeltaMarker` to streaming
channels — the in-band mechanism that lets channels render reasoning in
arrival order with text deltas (no race with the realtime bus).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.channels.cli import CLIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.streaming import ThinkingDeltaMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from tests.test_framework import SimpleChannel


class _CollectingStreamChannel(Channel):
    """Streaming-capable channel that records every chunk it receives."""

    channel_type = ChannelType.SMS

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.received: list[object] = []

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

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[object],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            self.received.append(chunk)
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])


class _ChunkedThinkingProvider(AIProvider):
    """Provider that emits multiple thinking deltas then multiple text deltas."""

    def __init__(self, thinking_chunks: list[str], text_chunks: list[str]) -> None:
        self._thinking_chunks = thinking_chunks
        self._text_chunks = text_chunks

    @property
    def model_name(self) -> str:
        return "chunked-mock"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(
            content="".join(self._text_chunks),
            thinking="".join(self._thinking_chunks),
            finish_reason="stop",
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for chunk in self._text_chunks:
            yield chunk

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for chunk in self._thinking_chunks:
            yield StreamThinkingDelta(thinking=chunk)
        for chunk in self._text_chunks:
            yield StreamTextDelta(text=chunk)
        yield StreamDone(finish_reason="stop", usage={})


async def test_no_tools_streaming_publishes_thinking_events() -> None:
    """Thinking events fire on the streaming-no-tools path, matching the other paths."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(content="The answer is 42.", thinking="user asked the meaning of life"),
        ],
    )
    kit = RoomKit()

    sms = SimpleChannel("sms1")
    ai = AIChannel("ai1", provider=provider, thinking_budget=4096)
    kit.register_channel(sms)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    received: list[EphemeralEvent] = []

    async def on_event(ev: EphemeralEvent) -> None:
        received.append(ev)

    await kit.realtime.subscribe_to_room("r1", on_event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))
    )

    # InMemoryRealtime dispatches subscriber callbacks via background tasks;
    # yield once so they run before we inspect the list.
    await asyncio.sleep(0.05)

    starts = [e for e in received if e.type == EphemeralEventType.THINKING_START]
    ends = [e for e in received if e.type == EphemeralEventType.THINKING_END]

    assert len(starts) == 1
    assert len(ends) == 1
    assert ends[0].data["thinking"] == "user asked the meaning of life"
    assert ends[0].channel_id == "ai1"

    await kit.close()


async def test_no_tools_streaming_without_thinking_emits_no_events() -> None:
    """No thinking in the response means no THINKING_* events — boundary check."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[AIResponse(content="just text", thinking="")],
    )
    kit = RoomKit()

    sms = SimpleChannel("sms1")
    ai = AIChannel("ai1", provider=provider)
    kit.register_channel(sms)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    received: list[EphemeralEvent] = []

    async def on_event(ev: EphemeralEvent) -> None:
        received.append(ev)

    await kit.realtime.subscribe_to_room("r1", on_event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))
    )

    await asyncio.sleep(0.05)

    thinking = [
        e
        for e in received
        if e.type in (EphemeralEventType.THINKING_START, EphemeralEventType.THINKING_END)
    ]
    assert thinking == []

    await kit.close()


async def test_thinking_markers_stream_inline_before_text() -> None:
    """ThinkingDeltaMarker chunks reach the channel in order, ahead of text."""
    provider = _ChunkedThinkingProvider(
        thinking_chunks=["Let me ", "think.", " 1+1=2."],
        text_chunks=["The ", "answer ", "is 2."],
    )
    kit = RoomKit()

    sink = _CollectingStreamChannel("sink1")
    ai = AIChannel("ai1", provider=provider, thinking_budget=4096)
    kit.register_channel(sink)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sink1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="sink1", sender_id="u1", content=TextContent(body="hi"))
    )

    # Filter out persisted RoomEvents (yielded by segment_stream between
    # segments) — we only care about the inline display deltas.
    display = [c for c in sink.received if isinstance(c, str | ThinkingDeltaMarker)]

    thinking_indices = [i for i, c in enumerate(display) if isinstance(c, ThinkingDeltaMarker)]
    text_indices = [i for i, c in enumerate(display) if isinstance(c, str)]

    assert len(thinking_indices) == 3, f"expected 3 thinking markers, got {display!r}"
    assert len(text_indices) == 3, f"expected 3 text chunks, got {display!r}"
    # Every thinking marker arrives before every text chunk.
    assert max(thinking_indices) < min(text_indices)
    assert [c.thinking for c in display if isinstance(c, ThinkingDeltaMarker)] == [
        "Let me ",
        "think.",
        " 1+1=2.",
    ]
    assert [c for c in display if isinstance(c, str)] == ["The ", "answer ", "is 2."]

    await kit.close()


async def test_thinking_coalescer_batches_by_size_without_loss() -> None:
    """Deltas batch into size-bounded publishes; concatenation stays lossless."""
    from roomkit.channels._ai_streaming import _ThinkingCoalescer

    published: list[str] = []

    async def publish(event_type: object, room_id: object, text: str, round_idx: int) -> None:
        assert event_type == EphemeralEventType.THINKING_DELTA
        assert room_id == "r1"
        assert round_idx == 0
        published.append(text)

    # flush_ms huge so only the 16-char size threshold fires (deterministic).
    coalescer = _ThinkingCoalescer(publish, "r1", 0, flush_ms=1e9, flush_chars=16)
    for _ in range(10):
        await coalescer.add("aaaa")  # 4 chars each, 40 total
    await coalescer.flush()  # residual

    # 40 chars at a 16-char window: 16 + 16 + 8 = three publishes, far fewer
    # than the ten input deltas, and nothing dropped.
    assert len(published) == 3
    assert "".join(published) == "a" * 40


async def test_thinking_coalescer_disabled_publishes_each_delta() -> None:
    """flush_ms=0 disables batching: one publish per delta (pre-coalescing path)."""
    from roomkit.channels._ai_streaming import _ThinkingCoalescer

    published: list[str] = []

    async def publish(event_type: object, room_id: object, text: str, round_idx: int) -> None:
        published.append(text)

    coalescer = _ThinkingCoalescer(publish, "r1", 0, flush_ms=0, flush_chars=256)
    for chunk in ["a", "b", "c"]:
        await coalescer.add(chunk)

    assert published == ["a", "b", "c"]


async def test_thinking_coalescer_flush_splits_at_preview_limit() -> None:
    """A flush larger than the publish preview cap splits losslessly, never truncates."""
    from roomkit.channels._ai_events import THINKING_PREVIEW_LIMIT
    from roomkit.channels._ai_streaming import _ThinkingCoalescer

    published: list[str] = []

    async def publish(event_type: object, room_id: object, text: str, round_idx: int) -> None:
        published.append(text)

    # Threshold above the preview cap: one buffered batch exceeds what a
    # single publish may carry.
    coalescer = _ThinkingCoalescer(
        publish, "r1", 0, flush_ms=1e9, flush_chars=THINKING_PREVIEW_LIMIT * 3
    )
    text = "x" * (THINKING_PREVIEW_LIMIT * 2 + 500)
    await coalescer.add(text)
    await coalescer.flush()

    assert len(published) == 3
    assert all(len(p) <= THINKING_PREVIEW_LIMIT for p in published)
    assert "".join(published) == text


async def test_thinking_coalescer_flush_empty_is_noop() -> None:
    """Flushing an empty buffer publishes nothing — no stray events at boundaries."""
    from roomkit.channels._ai_streaming import _ThinkingCoalescer

    published: list[object] = []

    async def publish(*args: object) -> None:
        published.append(args)

    coalescer = _ThinkingCoalescer(publish, "r1", 0, flush_ms=1e9, flush_chars=16)
    await coalescer.flush()
    assert published == []


async def test_streaming_coalesces_thinking_deltas_on_the_bus() -> None:
    """Simple streaming path batches THINKING_DELTA on the bus, lossless, END intact."""
    provider = _ChunkedThinkingProvider(
        thinking_chunks=["aaaa"] * 10,  # 40 chars of reasoning across 10 deltas
        text_chunks=["done"],
    )
    kit = RoomKit()

    sink = _CollectingStreamChannel("sink1")
    # flush_ms huge → only the 16-char size threshold fires (deterministic).
    ai = AIChannel(
        "ai1",
        provider=provider,
        thinking_budget=4096,
        thinking_coalesce_ms=1e9,
        thinking_coalesce_chars=16,
    )
    kit.register_channel(sink)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sink1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    received: list[EphemeralEvent] = []

    async def on_event(ev: EphemeralEvent) -> None:
        received.append(ev)

    await kit.realtime.subscribe_to_room("r1", on_event)

    await kit.process_inbound(
        InboundMessage(channel_id="sink1", sender_id="u1", content=TextContent(body="hi"))
    )
    await asyncio.sleep(0.05)

    deltas = [e for e in received if e.type == EphemeralEventType.THINKING_DELTA]
    ends = [e for e in received if e.type == EphemeralEventType.THINKING_END]

    # Ten input deltas coalesce into fewer bus publishes, with no reasoning lost.
    assert 0 < len(deltas) < 10
    assert "".join(e.data["thinking"] for e in deltas) == "a" * 40
    assert len(ends) == 1
    assert ends[0].data["thinking"] == "a" * 40

    await kit.close()


@pytest.mark.asyncio
async def test_cli_thinking_icon_shares_line_with_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The 💭 sits on the same line as the reasoning text.

    Reasoning models (qwen, etc.) open their ``<think>`` block with a
    newline, so the first thinking delta is ``"\nLet me..."``. The CLI
    trims that leading whitespace so the icon and text share a line
    instead of the icon dangling alone.
    """
    cli = CLIChannel("cli", show_thinking=True, use_color=False)

    async def stream() -> AsyncIterator[object]:
        yield ThinkingDeltaMarker(thinking="\nLet me reason.")
        yield ThinkingDeltaMarker(thinking=" Step two.")
        yield "The answer is 42."

    event = SimpleNamespace(source=SimpleNamespace(channel_id="ai1"))
    await cli.deliver_stream(stream(), event, None, None)  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert "💭 Let me reason. Step two." in out
    assert "💭 \n" not in out  # icon is not left alone on its line
