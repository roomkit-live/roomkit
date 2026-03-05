"""Telemetry provider system for RoomKit."""

import contextlib

from roomkit.telemetry.base import Attr, Span, SpanKind, TelemetryProvider
from roomkit.telemetry.config import TelemetryConfig
from roomkit.telemetry.console import ConsoleTelemetryProvider
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider

with contextlib.suppress(ImportError):
    from roomkit.telemetry.pyroscope import PyroscopeProfiler

__all__ = [
    "Attr",
    "ConsoleTelemetryProvider",
    "MockTelemetryProvider",
    "NoopTelemetryProvider",
    "PyroscopeProfiler",
    "Span",
    "SpanKind",
    "TelemetryConfig",
    "TelemetryProvider",
]
