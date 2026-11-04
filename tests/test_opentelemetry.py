"""Tests for OpenTelemetryProvider (telemetry/opentelemetry.py)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import roomkit.telemetry.opentelemetry as otel_module
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.context import restored_span
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.telemetry.opentelemetry import OpenTelemetryProvider
from roomkit.turns.owner import turn_request
from tests.test_framework import SimpleChannel


def _make_tracer_provider() -> MagicMock:
    """Build a mock tracer provider that produces usable spans."""

    # Use the real SDK's in-memory tracer provider for predictable behavior
    from opentelemetry.sdk.trace import TracerProvider

    return TracerProvider()


class TestOpenTelemetryProvider:
    def test_constructor_and_name(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        assert provider.name == "opentelemetry"

    def test_start_span_returns_string(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        span_id = provider.start_span(SpanKind.CUSTOM, "test.span")
        assert isinstance(span_id, str)
        assert len(span_id) == 16  # 8-byte hex

    def test_end_span_ok(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        span_id = provider.start_span(SpanKind.CUSTOM, "test.ok")
        provider.end_span(span_id)
        assert span_id not in provider._active_spans
        assert span_id in provider._ended_spans

    def test_end_span_error(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        span_id = provider.start_span(SpanKind.CUSTOM, "test.err")
        provider.end_span(span_id, status="error", error_message="boom")
        assert span_id not in provider._active_spans
        assert span_id in provider._ended_spans

    def test_end_span_nonexistent_is_noop(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        # Should not raise
        provider.end_span("nonexistent-span-id")

    def test_record_metric_counter(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        # Should not raise
        provider.record_metric("messages.count", 1.0)

    def test_record_metric_histogram(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        # "duration" in name triggers histogram
        provider.record_metric("stt.duration", 42.5, unit="ms")

    def test_close_ends_active_spans(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        sid1 = provider.start_span(SpanKind.CUSTOM, "active.1")
        sid2 = provider.start_span(SpanKind.CUSTOM, "active.2")
        assert sid1 in provider._active_spans
        assert sid2 in provider._active_spans
        provider.close()
        assert len(provider._active_spans) == 0

    def test_reset_clears_all(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        sid = provider.start_span(SpanKind.CUSTOM, "test")
        provider.end_span(sid)
        assert len(provider._ended_spans) > 0
        provider.reset()
        assert len(provider._active_spans) == 0
        assert len(provider._ended_spans) == 0

    def test_a_late_child_nests_under_its_ended_parent(self) -> None:
        """A span started after its parent ended still has it as parent: the
        provider keeps the ended parent's context (RMK-877)."""
        provider, exporter = _exporting_provider()
        parent = provider.start_span(SpanKind.CUSTOM, "voice.session")
        provider.end_span(parent)

        child = provider.start_span(SpanKind.CUSTOM, "after.tts", parent_id=parent)
        provider.end_span(child)

        spans = {span.name: span for span in exporter.get_finished_spans()}
        late = spans["roomkit.after.tts"]
        assert late.parent is not None
        assert format(late.parent.span_id, "016x") == parent
        assert late.context.trace_id == spans["roomkit.voice.session"].context.trace_id

    def test_ended_spans_kept_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Past the bound the oldest ended span is forgotten: memory stays
        bounded for a provider that lives for days (RMK-877)."""
        monkeypatch.setattr(otel_module, "_ENDED_SPANS_KEPT", 3)
        provider, _ = _exporting_provider()
        ended = []
        for n in range(4):
            span_id = provider.start_span(SpanKind.CUSTOM, f"span.{n}")
            provider.end_span(span_id)
            ended.append(span_id)

        assert list(provider._ended_spans) == ended[1:]
        assert provider.get_span_context(ended[0]) is None
        assert provider.get_span_context(ended[3]) is not None

    def test_an_ended_span_is_kept_as_its_context_alone(self) -> None:
        """What is kept of an ended span records nothing and carries none of
        its attributes, only its context (RMK-877)."""
        provider, _ = _exporting_provider()
        span_id = provider.start_span(SpanKind.CUSTOM, "big", attributes={"blob": "x" * 10_000})
        provider.end_span(span_id)

        kept = provider._ended_spans[span_id]
        assert not kept.is_recording()
        assert format(kept.get_span_context().span_id, "016x") == span_id
        assert not hasattr(kept, "attributes")

    def test_set_attribute(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(tracer_provider=tp)
        sid = provider.start_span(SpanKind.CUSTOM, "test.attr")
        # Should not raise
        provider.set_attribute(sid, "custom_key", "custom_value")
        provider.end_span(sid)

    def test_metadata_in_constructor(self) -> None:
        tp = _make_tracer_provider()
        provider = OpenTelemetryProvider(
            tracer_provider=tp,
            metadata={"env": "test", "version": "1.0"},
        )
        # Metadata should be included in span attributes
        sid = provider.start_span(SpanKind.CUSTOM, "test.meta")
        assert sid in provider._active_spans
        provider.end_span(sid)


def _exporting_provider() -> tuple[OpenTelemetryProvider, Any]:
    """A provider whose spans land in an in-memory exporter, parent links intact."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryProvider(tracer_provider=tracer_provider), exporter


class TestTraceContinuity:
    """Real OTel parent links across the lane executor and the detached consumer.

    ``MockTelemetryProvider`` tracks parents by roomkit span id and ignores the
    backend context, so a stale or missing ``telemetry_ctx`` — this provider's
    primary parent source — is invisible to it. Only the exported spans say
    what a tracing backend would actually show.
    """

    @pytest.mark.parametrize("defer", [False, True], ids=["waiting", "deferred"])
    @pytest.mark.parametrize("streaming", [False, True], ids=["reentry", "streamed"])
    async def test_turn_spans_share_the_inbound_trace(self, streaming: bool, defer: bool) -> None:
        provider, exporter = _exporting_provider()
        kit = RoomKit(telemetry=provider)
        kit.register_channel(SimpleChannel("ch1"))
        kit.register_channel(SimpleChannel("ch2"))
        kit.register_channel(
            AIChannel("ai1", provider=MockAIProvider(responses=["reply"], streaming=streaming))
        )

        @kit.hook(HookTrigger.AFTER_BROADCAST, name="after")
        async def after(event: Any, context: RoomContext) -> None:
            pass

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ch1")
        await kit.attach_channel("r1", "ch2")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        msg = InboundMessage(channel_id="ch1", sender_id="user1", content=TextContent(body="hi"))
        result = await kit.process_inbound(msg, room_id="r1", defer_delivery=defer)
        if result.delivery is not None:
            await asyncio.wait_for(result.delivery.wait(), timeout=5.0)
        await kit.close()

        spans = exporter.get_finished_spans()
        by_name: dict[str, list[Any]] = {}
        for span in spans:
            by_name.setdefault(span.name, []).append(span)
        (inbound,) = by_name["roomkit.framework.inbound"]

        assert [s.name for s in spans if s.parent is None] == ["roomkit.framework.inbound"]
        assert {s.context.trace_id for s in spans} == {inbound.context.trace_id}
        broadcasts = by_name["roomkit.framework.broadcast"]
        assert len(broadcasts) == 2
        assert {s.parent.span_id for s in broadcasts} == {inbound.context.span_id}
        after_spans = by_name["roomkit.hook.async.after"]
        assert {s.parent.span_id for s in after_spans} == {inbound.context.span_id}
        if defer:
            (tail,) = by_name["roomkit.framework.detached"]
            assert tail.parent.span_id == inbound.context.span_id
        else:
            assert "roomkit.framework.detached" not in by_name


def _second_provider(exporter: Any) -> OpenTelemetryProvider:
    """Another process's provider: its own tracer, the same exporter."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryProvider(tracer_provider=tracer_provider, service_name="executor")


def _turn_request(telemetry: Any) -> Any:
    return turn_request(
        RoomEvent(
            room_id="room",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="hello"),
        ),
        ChannelBinding(
            channel_id="agent",
            room_id="room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="room")),
        telemetry=telemetry,
    )


class TestAcrossProcesses:
    """RMK-836: a span named by its trace context continues in another
    provider's trace, as a turn's executor in another process continues the
    owner's broadcast (RFC §10.6.9)."""

    def test_a_span_another_provider_names_is_the_parent_of_its_spans(self) -> None:
        owner, exporter = _exporting_provider()
        executor = _second_provider(exporter)
        parent = owner.start_span(SpanKind.BROADCAST, "broadcast")
        carrier = owner.inject_context(parent)

        with restored_span(parent, telemetry_ctx=executor.extract_context(carrier)):
            child = executor.start_span(SpanKind.LLM_GENERATE, "generate", parent_id=parent)
        executor.end_span(child)
        owner.end_span(parent)

        spans = {s.name: s for s in exporter.get_finished_spans()}
        broadcast, generate = spans["roomkit.broadcast"], spans["roomkit.generate"]
        assert "traceparent" in carrier
        assert generate.parent is not None
        assert generate.parent.span_id == broadcast.context.span_id
        assert generate.context.trace_id == broadcast.context.trace_id

    def test_a_carrier_that_names_no_span_gives_no_context(self) -> None:
        provider, _ = _exporting_provider()

        assert provider.inject_context("0123456789abcdef") == {}
        assert provider.extract_context({"span_id": "0123456789abcdef"}) is None

    def test_a_turn_request_names_its_span_by_trace_context(self) -> None:
        provider, _ = _exporting_provider()
        span = provider.start_span(SpanKind.BROADCAST, "broadcast")

        with restored_span(span, telemetry_ctx=provider.get_span_context(span)):
            named = _turn_request(provider).trace
            bare = _turn_request(NoopTelemetryProvider()).trace
        provider.end_span(span)

        assert named is not None and named["span_id"] == span
        assert named["traceparent"].split("-")[2] == span
        assert bare == {"span_id": span}


async def test_a_hook_s_injected_event_stays_in_its_message_s_trace() -> None:
    """RMK-1301: a hook's injected event's broadcast is no root of its own."""
    provider, exporter = _exporting_provider()
    kit = RoomKit(telemetry=provider)
    kit.register_channel(SimpleChannel("ch1"))
    kit.register_channel(SimpleChannel("ch2"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ch1")
    await kit.attach_channel("r1", "ch2")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.content.body != "Hi.":  # type: ignore[union-attr]
            return HookResult.allow()
        noted = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
            content=TextContent(body="noted"),
        )
        return HookResult.allow(injected=[InjectedEvent(event=noted, target_channel_ids=["ch2"])])

    await kit.process_inbound(
        InboundMessage(channel_id="ch1", sender_id="u", content=TextContent(body="Hi.")),
        room_id="r1",
    )
    await kit.close()

    spans = exporter.get_finished_spans()
    assert len({s.context.trace_id for s in spans}) == 1
    assert [s.name for s in spans if s.parent is None] == ["roomkit.framework.inbound"]


async def test_a_turn_s_generation_is_a_child_of_the_broadcast_it_runs_under() -> None:
    """RMK-836: a span started with an explicit parent nests under it, not
    under the context an ancestor propagated: the turn's generation is the
    broadcast's child (RFC §10.6.9), the inbound pipeline its grandparent."""
    provider, exporter = _exporting_provider()
    kit = RoomKit(telemetry=provider)
    kit.register_channel(SimpleChannel("ch1"))
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["Hello."])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ch1")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="ch1", sender_id="u", content=TextContent(body="Hi.")),
        room_id="r1",
    )
    await kit.close()

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    (generate,) = [s for s in spans if s.name == "roomkit.llm.generate"]
    assert generate.parent is not None
    assert by_id[generate.parent.span_id].name == "roomkit.framework.broadcast"
