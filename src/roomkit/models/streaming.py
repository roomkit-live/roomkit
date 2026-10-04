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

from roomkit.models.enums import ToolCallOutcome


@dataclass(slots=True)
class ToolCallStartMarker:
    """Yielded when a tool call begins execution.

    One marker per individual tool call. Multiple markers may be yielded
    in sequence when tools execute in parallel within the same round.
    """

    tool_name: str
    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ran_with: dict[str, Any] | None = None
    """The arguments the call runs with once its gate fixed them (a
    ``BEFORE_TOOL_USE`` rewrite), set while it runs: a call the turn cuts is
    closed with them, as its report is (RFC §9.3)."""


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
    # How the call ended (see ToolCallContent.outcome).
    outcome: ToolCallOutcome | None = None


@dataclass(slots=True)
class ThinkingDeltaMarker:
    """Yielded for each chunk of the model's reasoning text.

    One marker per provider ``StreamThinkingDelta`` event, so reasoning
    arrives token-by-token in arrival order with the text deltas — no
    buffering, no race against an out-of-band channel. Channels that
    want to render reasoning inline handle this marker; others ignore it.
    """

    thinking: str


@dataclass(slots=True)
class SegmentBreakMarker:
    """Yielded when the loop goes on after a round that ended without a call.

    The round's text, if any, is a segment of its own, as text before a call
    is: a continuation the channel's policy asked for, or another try at a
    round with nothing to deliver. A consumer ends the segment it holds there,
    so the next round's text never runs on from it (RFC §6.4). One that only
    renders text ignores it.
    """


#: Why a tool loop stopped. ``completed`` is the model having answered; every
#: other value is the loop ending on a rule of its own.
#:
#: ``force_stopped`` is the anti-loop ripcord: the model kept re-issuing a call
#: the guard had already blocked, so one last generation was told to answer
#: from what it had, and none of its calls runs. It usually produces text,
#: which is exactly why it needs its own name — that text is a summary of a
#: turn the platform cut short, not an answer, and a caller that reads
#: ``completed`` delivers it as one.
#:
#: ``budget_exceeded`` is a turn that reached its token or cost budget at a
#: round boundary: the calls its last generation asked for do not run, and no
#: further generation is asked for (RFC §6.4).
#:
#: ``unfinished`` is an answer the channel's continuation policy still asked
#: to go on once the tries it shares with an empty round had run out: its text
#: announced an action the model never took (RFC §6.4).
#:
#: ``error`` is a turn the provider interrupted after a tool round: the rounds
#: are kept, each round's text as its own message, and it is an error too
#: (ON_ERROR fires, the caller reads it on ``InboundResult.error``). The loop
#: yields its ``LoopEndMarker``, then the exception reaches the consumer.
#: RFC §6.4.
LoopEndReason = Literal[
    "completed",
    "max_rounds",
    "timeout",
    "budget_exceeded",
    "truncated",
    "empty_response",
    "unfinished",
    "force_stopped",
    "cancelled",
    "error",
]


@dataclass(slots=True)
class LoopEndMarker:
    """Yielded once as an AI channel's streamed response ends, saying why.

    The tool loop knows exactly which of its rules fired: the round cap, the
    wall-clock deadline, a round truncated at the output cap or the context
    window, a model that answered nothing after its tools or whose call its
    provider would not hand over, an
    answer the channel's continuation policy still found unfinished
    (``unfinished``), a cancellation. Without the marker a
    consumer could not tell a finished answer from a loop cut mid-work, and
    would re-derive it by counting tool calls and reading a clock.

    Emitted on **every** exit, ``completed`` included, and by a response
    without tools too (``rounds`` 0), so "the stream ended" is never itself
    the signal. The one exception is a provider that streams text only (no
    structured streaming): its response without tools stays a stream of
    ``str``, and carries no record. It is the stream's last item, except on ``error``: the
    provider's exception follows it, and a consumer that stops reading at
    the marker closes the stream before the error reaches it. A consumer that
    only renders text keeps filtering on ``isinstance(chunk, str)`` and is
    unaffected.

    ``rounds`` is how many tool rounds ran before the stop, as
    ``ON_AI_RESPONSE`` counts them (``round_count``): a round the loop tried
    again without a call is none. ``usage`` is what the turn's generations
    used, summed over every round: with the reason, it is the turn's record,
    which the stream's consumer writes on the turn's last message (RFC §6.4).

    ``max_rounds``, ``timeout_seconds``, ``budget_tokens`` and ``budget_usd``
    are the limits the turn ran under, so a consumer names the one a
    ``max_rounds``, ``timeout`` or ``budget_exceeded`` end hit without reading
    the channel: the budget is resolved per turn (the binding, then the turn's
    config, then the channel), which the channel cannot say. ``None`` is no
    such limit; a marker the channel did not build may leave them all unset.
    They ride the marker only: ``ON_AI_RESPONSE`` reports the reason and
    ``round_count``, not the limits.
    """

    reason: LoopEndReason
    rounds: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    max_rounds: int | None = None
    timeout_seconds: float | None = None
    budget_tokens: int | None = None
    budget_usd: float | None = None


#: Union of all marker types that may appear in a streaming response.
StreamMarker = (
    ToolCallStartMarker
    | ToolCallEndMarker
    | ThinkingDeltaMarker
    | SegmentBreakMarker
    | LoopEndMarker
)

#: A single item in the streaming response: either a text delta or a marker.
StreamDelta = str | StreamMarker
