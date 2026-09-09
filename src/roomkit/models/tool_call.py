"""Unified tool call event for all channel types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import ChannelType
from roomkit.models.streaming import LoopEndReason

if TYPE_CHECKING:
    from roomkit.providers.ai.base import AIContext
    from roomkit.voice.base import VoiceSession


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class ToolCallEvent:
    """Channel-agnostic tool call event.

    Fired through ON_TOOL_CALL hooks from both AIChannel and
    RealtimeVoiceChannel.  When ``result`` is None the hook is
    expected to provide a result; when set, the hook observes
    (and may override) the handler's result.
    """

    channel_id: str
    """ID of the channel that triggered the tool call."""

    channel_type: ChannelType
    """Type of the originating channel."""

    tool_call_id: str
    """Provider-assigned ID for this tool call."""

    name: str
    """The function name being called."""

    arguments: dict[str, Any]
    """Parsed arguments for the function call."""

    result: str | list[Any] | None = None
    """Handler result (None = hook must provide).

    Usually a string; a list of content parts when a tool returns multimodal
    output (e.g. an image). Typed ``list[Any]`` to avoid a models→providers
    import cycle — the concrete part types live in ``providers.ai.base``.
    """

    room_id: str | None = None
    """Room where the tool call originated."""

    session: VoiceSession | None = None
    """Voice session (realtime channels only)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the tool call was received."""


# Callback type injected into AIChannel by the framework.
# Returns a result (str or content parts) to override, None to keep the original.
ToolCallCallback = Callable[[ToolCallEvent], Awaitable[str | list[Any] | None]]


RESPONSE_SEGMENT_SEPARATOR = "\n\n"
"""What separates two segments of a turn in :attr:`AIResponseEvent.response_content`.

A tool call cuts the model's text: what it said before the call and what it
said after are two segments, persisted as two MESSAGE events. Joined with
nothing between them they read as one run-on sentence (``first.Working``);
this is the paragraph break that keeps them apart. It only ever sits between
two segments — never inside one, never at either end.
"""


def response_transcript(segments: Iterable[str]) -> tuple[list[str], str]:
    """The turn's text as ``ON_AI_RESPONSE`` reports it.

    Drops the empty stretches (a tool round in which the model said nothing)
    and returns the segments kept, and their join. The one place the contract
    lives: the streaming, non-streaming and ACP paths all report through it.
    """
    kept = [segment for segment in segments if segment]
    return kept, RESPONSE_SEGMENT_SEPARATOR.join(kept)


@dataclass(frozen=True)
class AIResponseEvent:
    """Emitted through ON_AI_RESPONSE hooks after AI generation completes.

    Provides response content, usage metrics, and timing for evaluation
    and scoring integrations.
    """

    channel_id: str
    """ID of the AI channel that generated the response."""

    response_content: str
    """Everything the model said in the turn, as one readable transcript.

    The turn's :attr:`segments` joined with :data:`RESPONSE_SEGMENT_SEPARATOR`
    — a blank line at every tool-call boundary, nothing inside a segment. A
    turn without a tool call is its single segment, verbatim.
    """

    room_id: str | None = None
    """Room where the response was generated."""

    tool_calls_count: int = 0
    """Number of tool calls executed during generation."""

    usage: dict[str, Any] = field(default_factory=dict)
    """Token usage from the provider (input_tokens, output_tokens)."""

    thinking: str = ""
    """Extended thinking/reasoning text (if supported by provider)."""

    round_count: int = 0
    """Number of tool execution rounds."""

    loop_end_reason: LoopEndReason | None = None
    """Which of the tool loop's rules ended the turn, or None if unreported.

    The streaming loop already named this on its :class:`LoopEndMarker` and the
    buffered loop on ``ToolLoopResult.reason``, but neither reached this event:
    a hook could see *that* a turn ended and how much work it did, never
    whether it finished or was cut off. Counting tool calls does not answer it
    — :attr:`tool_calls_count` reports the calls the turn *ran*, so a healthy
    multi-round answer and one guillotined by the round cap both report a
    positive count. Read this instead: ``"completed"`` is a turn that ended on
    its own terms, and ``"max_rounds"``, ``"timeout"``, ``"cancelled"``,
    ``"force_stopped"``, ``"truncated"`` and ``"empty_response"`` each name the
    rule that stopped it.

    None means the path that fired the hook reported no reason, not that the
    turn completed.
    """

    latency_ms: int = 0
    """Total generation time in milliseconds."""

    streaming: bool = False
    """Whether the response was streamed."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the response was generated."""

    segments: list[str] = field(default_factory=list)
    """The turn's text, one entry per stretch between tool calls, in order.

    Empty stretches are dropped, so :attr:`response_content` is exactly these
    joined with :data:`RESPONSE_SEGMENT_SEPARATOR`, and ``segments[-1]`` is
    the text that followed the last tool call — the answer, for a consumer
    that wants it without the narration before it. Empty when the turn
    produced no text.
    """

    usage_metadata: dict[str, Any] = field(default_factory=dict)
    """Optional provenance and scope of the provider's usage observations.

    ACP includes the native session, prompt source and session usage report.
    These are observations, not a price or a claim that generation succeeded.
    Existing providers and consumers can leave this empty.
    """


# Callback type for AI response observation (fire-and-forget).
AfterResponseCallback = Callable[["AIResponseEvent"], Awaitable[None]]


@dataclass
class AIGenerationEvent:
    """Emitted through BEFORE_AI_GENERATION hooks before AI provider invocation.

    Provides the full AI context for inspection and modification.
    Hooks can mutate ``ai_context`` in-place (e.g. append messages,
    modify system_prompt, adjust tools) and return ``HookResult.allow()``,
    or return ``HookResult.block(reason)`` to prevent generation.
    """

    ai_context: AIContext
    """The full built context that will be sent to the AI provider."""

    channel_id: str
    """ID of the AI channel about to generate."""

    room_id: str | None = None
    """Room where generation is happening."""

    provider_name: str | None = None
    """Name of the AI provider that will be invoked."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the generation was initiated."""


# Callback type for BEFORE_AI_GENERATION hook (sync, can block/modify).
# Returns SyncPipelineResult (from roomkit.core.hooks) — typed as Any to
# avoid circular import from models into core.  Only the framework's
# _build_before_generation_hook closure creates instances of this type.
BeforeGenerationCallback = Callable[["AIGenerationEvent"], Awaitable[Any]]
