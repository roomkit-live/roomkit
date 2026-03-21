"""Unified tool call event for all channel types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import ChannelType

if TYPE_CHECKING:
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

    result: str | None = None
    """Handler result (None = hook must provide)."""

    room_id: str | None = None
    """Room where the tool call originated."""

    session: VoiceSession | None = None
    """Voice session (realtime channels only)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the tool call was received."""


# Callback type injected into AIChannel by the framework.
# Returns str to override the result, None to keep the original.
ToolCallCallback = Callable[[ToolCallEvent], Awaitable[str | None]]


@dataclass(frozen=True)
class AIResponseEvent:
    """Emitted through ON_AI_RESPONSE hooks after AI generation completes.

    Provides response content, usage metrics, and timing for evaluation
    and scoring integrations.
    """

    channel_id: str
    """ID of the AI channel that generated the response."""

    response_content: str
    """The generated text response."""

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

    latency_ms: int = 0
    """Total generation time in milliseconds."""

    streaming: bool = False
    """Whether the response was streamed."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the response was generated."""


# Callback type for AI response observation (fire-and-forget).
AfterResponseCallback = Callable[["AIResponseEvent"], Awaitable[None]]
