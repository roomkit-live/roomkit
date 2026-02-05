"""Event-driven message sources for RoomKit."""

from typing import Any

from roomkit.sources.base import (
    BaseSourceProvider,
    EmitCallback,
    SourceHealth,
    SourceProvider,
    SourceStatus,
)

__all__ = [
    "BaseSourceProvider",
    "EmitCallback",
    "SourceHealth",
    "SourceProvider",
    "SourceStatus",
    # Lazy imports for optional sources
    "WebSocketSource",
    "SSESource",
    "WhatsAppPersonalSourceProvider",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional source providers."""
    if name == "WebSocketSource":
        from roomkit.sources.websocket import WebSocketSource

        return WebSocketSource
    if name == "SSESource":
        from roomkit.sources.sse import SSESource

        return SSESource
    if name == "WhatsAppPersonalSourceProvider":
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        return WhatsAppPersonalSourceProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
