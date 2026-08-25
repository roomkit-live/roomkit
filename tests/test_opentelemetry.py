"""Tests for OpenTelemetryProvider (telemetry/opentelemetry.py)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookTrigger
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.opentelemetry import OpenTelemetryProvider
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
