"""Streaming protocol markers for structured AI response segments.

These markers are yielded by AI streaming generators alongside ``str`` text
deltas. The framework's streaming consumer uses them to persist text segments
and tool call events at each boundary, rather than concatenating everything
into a single event.

Transport channels never see these markers — only ``str`` deltas are forwarded
for real-time display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ToolCallStartMarker:
    """Yielded when a tool call begins execution.

    One marker per individual tool call. Multiple markers may be yielded
    in sequence when tools execute in parallel within the same round.
    """

    tool_name: str
    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallEndMarker:
    """Yielded when a tool call completes.

    One marker per individual tool call, matching a prior
    :class:`ToolCallStartMarker` by ``tool_id``.
    """

    tool_name: str
    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    status: Literal["completed", "failed"] = "completed"
    duration_ms: int = 0
    error: str | None = None


#: Union of all marker types that may appear in a streaming response.
StreamMarker = ToolCallStartMarker | ToolCallEndMarker

#: A single item in the streaming response: either a text delta or a marker.
StreamDelta = str | StreamMarker
