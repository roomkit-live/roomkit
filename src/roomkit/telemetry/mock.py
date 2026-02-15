"""Mock telemetry provider — records spans and metrics for test assertions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from roomkit.telemetry.base import Span, SpanKind, TelemetryProvider


class MockTelemetryProvider(TelemetryProvider):
    """Records all spans and metrics in lists for test assertions.

    Example::

        telemetry = MockTelemetryProvider()
        kit = RoomKit(telemetry=telemetry)
        # ... run operations ...
        assert len(telemetry.spans) > 0
        stt_spans = telemetry.get_spans(SpanKind.STT_TRANSCRIBE)
        assert stt_spans[0].attributes["stt.text_length"] > 0
    """

    def __init__(self) -> None:
        self._spans: dict[str, Span] = {}
        self.completed_spans: list[Span] = []
        self.metrics: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def spans(self) -> list[Span]:
        """All completed spans."""
        return self.completed_spans

    def get_spans(self, kind: SpanKind) -> list[Span]:
        """Get completed spans of a specific kind."""
        return [s for s in self.completed_spans if s.kind == kind]

    def get_active_spans(self) -> list[Span]:
        """Get spans that have been started but not ended."""
        return list(self._spans.values())

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        *,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        room_id: str | None = None,
        session_id: str | None = None,
        channel_id: str | None = None,
    ) -> str:
        span = Span(
            kind=kind,
            name=name,
            parent_id=parent_id,
            attributes=dict(attributes) if attributes else {},
            room_id=room_id,
            session_id=session_id,
            channel_id=channel_id,
        )
        self._spans[span.id] = span
        return span.id

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        span = self._spans.pop(span_id, None)
        if span is None:
            return
        span.end_time = datetime.now(UTC)
        span.status = status
        span.error_message = error_message
        if attributes:
            span.attributes.update(attributes)
        self.completed_spans.append(span)

    def set_attribute(self, span_id: str, key: str, value: Any) -> None:
        span = self._spans.get(span_id)
        if span is not None:
            span.attributes[key] = value

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.metrics.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "attributes": dict(attributes) if attributes else {},
            }
        )

    def reset(self) -> None:
        """Clear all recorded spans and metrics."""
        self._spans.clear()
        self.completed_spans.clear()
        self.metrics.clear()
