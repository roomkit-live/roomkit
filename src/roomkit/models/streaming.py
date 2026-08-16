"""Streaming protocol markers for structured AI response segments.

These markers are yielded by AI streaming generators alongside ``str`` text
deltas. The framework's streaming consumer uses them to persist text segments
and tool call events at each boundary, rather than concatenating everything
into a single event.

Channels see the full mixed stream and choose what to render. Text-only
channels filter on ``isinstance(chunk, str)`` and skip the markers; richer
channels (CLI, web) can render tool calls and thinking inline.
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
    # MCP structuredContent captured before result eviction (see AIToolResultPart).
    structured_content: dict[str, Any] | None = None


@dataclass(slots=True)
class ThinkingDeltaMarker:
    """Yielded for each chunk of the model's reasoning text.

    One marker per provider ``StreamThinkingDelta`` event, so reasoning
    arrives token-by-token in arrival order with the text deltas — no
    buffering, no race against an out-of-band channel. Channels that
    want to render reasoning inline handle this marker; others ignore it.
    """

    thinking: str


#: Why a tool loop stopped. ``completed`` is the model having answered; every
#: other value is the loop ending on a rule of its own.
LoopEndReason = Literal[
    "completed",
    "max_rounds",
    "timeout",
    "truncated",
    "empty_response",
    "cancelled",
]


@dataclass(slots=True)
class LoopEndMarker:
    """Yielded once, last, saying why the tool loop stopped.

    The loop knows exactly which of its rules fired — the round cap, the
    wall-clock deadline, a round truncated at the output cap, a model that
    answered nothing after its tools, a cancellation. Before this marker it
    logged that reason and returned, so the stream simply ended: a consumer
    could not tell a finished answer from a loop cut mid-work, and had to
    re-derive it by counting tool calls and reading a clock. Every consumer
    that cared reimplemented the same guess, and a guess is what reports a
    stopped agent as a model that returned nothing.

    Emitted on **every** exit, ``completed`` included, so "the stream ended"
    is never itself the signal. A consumer that only renders text keeps
    filtering on ``isinstance(chunk, str)`` and is unaffected.

    ``rounds`` is how many tool rounds ran before the stop. The limits the
    reason refers to are the caller's own configuration, so they are not
    repeated here.

    Streaming only: the non-streaming loop hands back an ``AIResponse`` the
    caller already holds, rather than a stream whose end is silent.
    """

    reason: LoopEndReason
    rounds: int = 0


#: Union of all marker types that may appear in a streaming response.
StreamMarker = ToolCallStartMarker | ToolCallEndMarker | ThinkingDeltaMarker | LoopEndMarker

#: A single item in the streaming response: either a text delta or a marker.
StreamDelta = str | StreamMarker
