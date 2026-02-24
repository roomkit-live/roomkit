"""Telemetry provider ABC, Span dataclass, SpanKind enum, and Attr constants."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    """Span classifications for telemetry."""

    PIPELINE_INBOUND = "pipeline.inbound"
    PIPELINE_OUTBOUND = "pipeline.outbound"
    STT_TRANSCRIBE = "stt.transcribe"
    STT_STREAM = "stt.stream"
    TTS_SYNTHESIZE = "tts.synthesize"
    LLM_GENERATE = "llm.generate"
    LLM_TOOL_CALL = "llm.tool_call"
    REALTIME_SESSION = "realtime.session"
    REALTIME_TURN = "realtime.turn"
    REALTIME_TOOL_CALL = "realtime.tool_call"
    HOOK_SYNC = "hook.sync"
    HOOK_ASYNC = "hook.async"
    INBOUND_PIPELINE = "framework.inbound"
    BROADCAST = "framework.broadcast"
    DELIVERY = "framework.delivery"
    VOICE_SESSION = "voice.session"
    PIPELINE_SPEECH_SEGMENT = "pipeline.speech_segment"
    STORE_QUERY = "store.query"
    BACKEND_CONNECT = "backend.connect"
    CUSTOM = "custom"


class Attr:
    """Well-known attribute key constants for telemetry spans and metrics."""

    # Common
    PROVIDER = "provider"
    MODEL = "model"
    ROOM_ID = "room_id"
    SESSION_ID = "session_id"
    CHANNEL_ID = "channel_id"

    # Timing
    TTFB_MS = "ttfb_ms"
    DURATION_MS = "duration_ms"

    # STT
    STT_TEXT = "stt.text"
    STT_TEXT_LENGTH = "stt.text_length"
    STT_CONFIDENCE = "stt.confidence"
    STT_IS_FINAL = "stt.is_final"
    STT_MODE = "stt.mode"

    # TTS
    TTS_VOICE = "tts.voice"
    TTS_CHAR_COUNT = "tts.char_count"
    TTS_TEXT_LENGTH = "tts.text_length"

    # LLM
    LLM_INPUT_TOKENS = "llm.input_tokens"
    LLM_OUTPUT_TOKENS = "llm.output_tokens"
    LLM_TOOL_COUNT = "llm.tool_count"
    LLM_STREAMING = "llm.streaming"

    # Hooks
    HOOK_NAME = "hook.name"
    HOOK_TRIGGER = "hook.trigger"
    HOOK_RESULT = "hook.result"

    # Pipeline
    PIPELINE_FRAME_COUNT = "pipeline.frame_count"
    PIPELINE_BYTES_PROCESSED = "pipeline.bytes_processed"
    PIPELINE_STAGES = "pipeline.stages"
    PIPELINE_FRAMES = "pipeline.frames"

    # Realtime
    REALTIME_PROVIDER = "realtime.provider"
    REALTIME_TOOL_NAME = "realtime.tool_name"

    # Delivery
    DELIVERY_CHANNEL_TYPE = "delivery.channel_type"
    DELIVERY_RECIPIENT = "delivery.recipient"
    DELIVERY_SUCCESS = "delivery.success"
    DELIVERY_ERROR = "delivery.error"
    DELIVERY_MESSAGE_ID = "delivery.message_id"

    # Store
    STORE_OPERATION = "store.operation"
    STORE_TABLE = "store.table"

    # Backend
    BACKEND_TYPE = "backend.type"


@dataclass
class Span:
    """Represents a telemetry span."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    kind: SpanKind = SpanKind.CUSTOM
    name: str = ""
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: str = "ok"
    error_message: str | None = None
    room_id: str | None = None
    session_id: str | None = None
    channel_id: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if not yet ended."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class TelemetryProvider(ABC):
    """Abstract base class for telemetry providers.

    Providers collect span and metric data from RoomKit operations.
    The default ``NoopTelemetryProvider`` has zero overhead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...

    @abstractmethod
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
        """Start a new telemetry span.

        Returns:
            A unique span ID string.
        """
        ...

    @abstractmethod
    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error_message: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End a previously started span."""
        ...

    @abstractmethod
    def set_attribute(self, span_id: str, key: str, value: Any) -> None:
        """Set an attribute on an active span."""
        ...

    @abstractmethod
    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric value."""
        ...

    def get_span_context(self, span_id: str) -> Any:
        """Return an opaque context object for the given span.

        Used to propagate backend-specific parent context (e.g. OTel Context)
        through :func:`set_current_span` for robust parent linking across
        async boundaries.

        Returns ``None`` by default.  Override in providers that carry
        backend-specific context (e.g. ``OpenTelemetryProvider``).
        """
        return None

    def flush(self) -> None:  # noqa: B027
        """Flush pending spans/metrics without closing the provider.

        Called after ending long-lived spans (e.g. VOICE_SESSION) to ensure
        they are exported promptly rather than waiting for shutdown.
        """

    def close(self) -> None:  # noqa: B027
        """Close the provider and flush any pending data."""

    def reset(self) -> None:  # noqa: B027
        """Reset internal state (useful for testing)."""

    @contextmanager
    def span(
        self,
        kind: SpanKind,
        name: str,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Context manager for span lifecycle.

        Yields the span ID. Automatically ends the span on exit,
        recording error status if an exception occurs.
        """
        span_id = self.start_span(kind, name, **kwargs)
        try:
            yield span_id
            self.end_span(span_id)
        except Exception as exc:
            self.end_span(span_id, status="error", error_message=str(exc))
            raise
