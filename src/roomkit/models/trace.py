"""Protocol trace model for channel observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass(frozen=True)
class ProtocolTrace:
    """A single protocol-level trace event emitted by a channel.

    Captures raw protocol data (SIP, RTP, WebSocket, HTTP, etc.) flowing
    through a channel.  Disabled by default; zero overhead when no
    observers are registered.

    Attributes:
        channel_id: Which channel emitted this trace.
        direction: Whether the message was inbound or outbound.
        protocol: Protocol identifier (e.g. "sip", "rtp", "ws", "http").
        summary: Human-readable one-liner (e.g. "INVITE sip:+1555@pbx").
        raw: Full payload bytes or string (optional for high-freq traces).
        metadata: Protocol-specific extras (codec info, headers, etc.).
        timestamp: When the trace was captured.
        session_id: For session-scoped traces (voice sessions, etc.).
        room_id: Set by emitter if known at emit time.
    """

    channel_id: str
    direction: Literal["inbound", "outbound"]
    protocol: str
    summary: str
    raw: bytes | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    session_id: str | None = None
    room_id: str | None = None
