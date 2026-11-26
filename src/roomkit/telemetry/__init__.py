"""Telemetry provider system for RoomKit."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.telemetry.base import Attr, Span, SpanKind, TelemetryProvider
from roomkit.telemetry.config import TelemetryConfig
from roomkit.telemetry.console import ConsoleTelemetryProvider
from roomkit.telemetry.context import get_current_span, get_current_telemetry_ctx
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider

with contextlib.suppress(ImportError):
    from roomkit.telemetry.pyroscope import PyroscopeProfiler

if TYPE_CHECKING:
    from roomkit.telemetry.opentelemetry import OpenTelemetryProvider
else:
    __getattr__ = lazy_exports(
        __name__, {"OpenTelemetryProvider": "roomkit.telemetry.opentelemetry"}
    )

__all__ = [
    "Attr",
    "ConsoleTelemetryProvider",
    "MockTelemetryProvider",
    "NoopTelemetryProvider",
    "OpenTelemetryProvider",
    "PyroscopeProfiler",
    "Span",
    "SpanKind",
    "TelemetryConfig",
    "TelemetryProvider",
    "get_current_span",
    "get_current_telemetry_ctx",
]
