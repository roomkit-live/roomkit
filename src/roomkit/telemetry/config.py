"""Telemetry configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.telemetry.base import SpanKind, TelemetryProvider


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection.

    Attributes:
        provider: The telemetry provider to use. Defaults to
            ``NoopTelemetryProvider`` if not set.
        sample_rate: Fraction of spans to record (0.0 to 1.0).
            Default 1.0 records all spans.
        enabled_spans: If set, only these span kinds are recorded.
            ``None`` means all span kinds are enabled.
    """

    provider: TelemetryProvider | None = None
    sample_rate: float = 1.0
    enabled_spans: set[SpanKind] | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    suppressed_hook_triggers: set[str] = field(
        default_factory=lambda: {
            "on_input_audio_level",
            "on_output_audio_level",
            "on_vad_audio_level",
        }
    )
