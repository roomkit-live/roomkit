"""OpenTelemetry telemetry provider — bridges to the OTel SDK."""

from __future__ import annotations

import logging
from typing import Any

from roomkit.telemetry.base import SpanKind, TelemetryProvider

# opentelemetry is an optional dependency — all imports are deferred
# to method bodies so the module can be imported for inspection even
# when the SDK is not installed.

logger = logging.getLogger("roomkit.telemetry.otel")


class OpenTelemetryProvider(TelemetryProvider):
    """Bridges RoomKit telemetry to OpenTelemetry SDK.

    Requires ``opentelemetry-api`` and ``opentelemetry-sdk``::

        pip install opentelemetry-api opentelemetry-sdk

    Example::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        from roomkit import RoomKit
        from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

        kit = RoomKit(telemetry=OpenTelemetryProvider(tracer_provider=provider))
    """

    def __init__(
        self,
        *,
        tracer_provider: Any = None,
        meter_provider: Any = None,
        service_name: str = "roomkit",
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryProvider requires opentelemetry-api and opentelemetry-sdk. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            ) from exc

        self._StatusCode = StatusCode

        if tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(service_name)
        else:
            self._tracer = trace.get_tracer(service_name)

        self._meter: Any = None
        if meter_provider is not None:
            self._meter = meter_provider.get_meter(service_name)
        else:
            try:
                from opentelemetry import metrics

                self._meter = metrics.get_meter(service_name)
            except Exception:  # nosec B110
                pass

        self._active_spans: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "opentelemetry"

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
        from opentelemetry import context, trace

        # Build context with parent if provided
        ctx = None
        if parent_id and parent_id in self._active_spans:
            parent_span = self._active_spans[parent_id]
            ctx = trace.set_span_in_context(parent_span)

        otel_span = self._tracer.start_span(
            name=f"roomkit.{name}",
            context=ctx,
            attributes=self._build_attributes(kind, attributes, room_id, session_id, channel_id),
        )

        # Use the span's hex ID as our span_id
        span_ctx = otel_span.get_span_context()
        span_id = format(span_ctx.span_id, "016x")
        self._active_spans[span_id] = otel_span

        # Activate the span in OTel context
        token = context.attach(trace.set_span_in_context(otel_span))
        otel_span._roomkit_token = token  # noqa: SLF001

        return span_id

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        from opentelemetry import context

        otel_span = self._active_spans.pop(span_id, None)
        if otel_span is None:
            return

        if attributes:
            for k, v in attributes.items():
                self._set_otel_attribute(otel_span, k, v)

        if status == "error":
            otel_span.set_status(self._StatusCode.ERROR, error_message or "")
            if error_message:
                otel_span.record_exception(Exception(error_message))
        else:
            otel_span.set_status(self._StatusCode.OK)

        # Detach context token
        token = getattr(otel_span, "_roomkit_token", None)
        if token is not None:
            context.detach(token)

        otel_span.end()

    def set_attribute(self, span_id: str, key: str, value: Any) -> None:
        otel_span = self._active_spans.get(span_id)
        if otel_span is not None:
            self._set_otel_attribute(otel_span, key, value)

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self._meter is None:
            return
        try:
            # Use a histogram for timing/duration metrics, gauge for counts
            if "duration" in name or "ttfb" in name:
                histogram = self._meter.create_histogram(
                    name=name,
                    unit=unit or "ms",
                    description=f"RoomKit metric: {name}",
                )
                histogram.record(value, attributes=attributes)
            else:
                counter = self._meter.create_counter(
                    name=name,
                    unit=unit,
                    description=f"RoomKit metric: {name}",
                )
                counter.add(value, attributes=attributes)
        except Exception:
            logger.debug("Failed to record OTel metric %s", name, exc_info=True)

    def close(self) -> None:
        # End any remaining active spans
        for span_id in list(self._active_spans):
            self.end_span(span_id, status="error", error_message="provider closed")
        self._active_spans.clear()

    def reset(self) -> None:
        self._active_spans.clear()

    def _build_attributes(
        self,
        kind: SpanKind,
        attributes: dict[str, Any] | None,
        room_id: str | None,
        session_id: str | None,
        channel_id: str | None,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {"roomkit.span_kind": str(kind)}
        if room_id:
            attrs["roomkit.room_id"] = room_id
        if session_id:
            attrs["roomkit.session_id"] = session_id
        if channel_id:
            attrs["roomkit.channel_id"] = channel_id
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    attrs[f"roomkit.{k}"] = v
        return attrs

    @staticmethod
    def _set_otel_attribute(otel_span: Any, key: str, value: Any) -> None:
        """Set an attribute on an OTel span, coercing to allowed types."""
        if isinstance(value, (str, int, float, bool)):
            otel_span.set_attribute(f"roomkit.{key}", value)
        elif value is not None:
            otel_span.set_attribute(f"roomkit.{key}", str(value))
