"""Console telemetry provider — logs span summaries via Python logging."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from roomkit.telemetry.base import Span, SpanKind, TelemetryProvider

logger = logging.getLogger("roomkit.telemetry")


class ConsoleTelemetryProvider(TelemetryProvider):
    """Logs span start/end and metrics to ``roomkit.telemetry`` logger.

    Zero external dependencies — uses only Python's built-in logging.
    Useful for development and debugging.

    Example::

        import logging
        logging.basicConfig(level=logging.INFO)

        from roomkit import RoomKit
        from roomkit.telemetry import ConsoleTelemetryProvider

        kit = RoomKit(telemetry=ConsoleTelemetryProvider())
    """

    def __init__(self, *, level: int = logging.INFO) -> None:
        self._level = level
        self._spans: dict[str, Span] = {}

    @property
    def name(self) -> str:
        return "console"

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
        logger.log(
            self._level,
            "[SPAN START] %s %s (id=%s%s)",
            kind,
            name,
            span.id,
            f", parent={parent_id}" if parent_id else "",
        )
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

        duration = span.duration_ms
        dur_str = f" {duration:.1f}ms" if duration is not None else ""
        attr_str = ""
        if span.attributes:
            parts = [f"{k}={v}" for k, v in span.attributes.items()]
            attr_str = f" [{', '.join(parts)}]"

        if status == "error":
            logger.log(
                self._level,
                "[SPAN ERROR] %s %s%s%s error=%s",
                span.kind,
                span.name,
                dur_str,
                attr_str,
                error_message or "unknown",
            )
        else:
            logger.log(
                self._level,
                "[SPAN END] %s %s%s%s",
                span.kind,
                span.name,
                dur_str,
                attr_str,
            )

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
        unit_str = f" {unit}" if unit else ""
        attr_str = ""
        if attributes:
            parts = [f"{k}={v}" for k, v in attributes.items()]
            attr_str = f" [{', '.join(parts)}]"
        logger.log(self._level, "[METRIC] %s = %.2f%s%s", name, value, unit_str, attr_str)

    def close(self) -> None:
        active = len(self._spans)
        if active:
            logger.warning("ConsoleTelemetryProvider closed with %d active spans", active)
        self._spans.clear()

    def reset(self) -> None:
        self._spans.clear()
