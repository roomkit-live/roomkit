"""No-op telemetry provider — zero overhead default."""

from __future__ import annotations

from typing import Any

from roomkit.telemetry.base import SpanKind, TelemetryProvider

# Singleton empty span ID to avoid allocations
_NOOP_SPAN_ID = ""


class NoopTelemetryProvider(TelemetryProvider):
    """Default telemetry provider that does nothing.

    All methods are no-ops with zero overhead. This is the default
    when no telemetry provider is configured.
    """

    @property
    def name(self) -> str:
        return "noop"

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
        return _NOOP_SPAN_ID

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass

    def set_attribute(self, span_id: str, key: str, value: Any) -> None:
        pass

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass
