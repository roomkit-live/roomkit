"""Tests for OpenTelemetryProvider (telemetry/opentelemetry.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.opentelemetry import OpenTelemetryProvider


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
