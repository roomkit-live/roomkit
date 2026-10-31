"""Channel-related data models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from roomkit.models._keys import Id
from roomkit.models._time import UtcDatetime
from roomkit.models.delivery import ProviderResult
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    Visibility,
)
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.task import Observation, Task


class RateLimit(BaseModel):
    """Rate limiting configuration for a channel."""

    max_per_second: float | None = Field(default=None, gt=0.0)
    max_per_minute: float | None = Field(default=None, gt=0.0)
    max_per_hour: float | None = Field(default=None, gt=0.0)


class RetryPolicy(BaseModel):
    """Configures retry behaviour for channel delivery."""

    max_retries: int = Field(default=3, ge=0)
    base_delay_seconds: float = Field(default=1.0, gt=0.0)
    max_delay_seconds: float = Field(default=60.0, gt=0.0)
    exponential_base: float = Field(default=2.0, gt=0.0)

    retryable_errors: list[str] | None = None
    """Exception type names that may be retried (RFC §13.2).

    ``None`` retries anything the error itself does not refuse — a provider
    that marks its exception ``retryable=False`` is believed either way, which
    is how a permanent 4xx stops being replayed with full backoff. Naming types
    here narrows it further: only those are retried, everything else fails on
    its first attempt.
    """


class ChannelCapabilities(BaseModel):
    """What a channel can do."""

    media_types: list[ChannelMediaType] = Field(default_factory=lambda: [ChannelMediaType.TEXT])
    max_length: int | None = Field(default=None, gt=0)
    supports_threading: bool = False
    supports_reactions: bool = False
    supports_edit: bool = False
    supports_delete: bool = False
    supports_read_receipts: bool = False
    supports_typing: bool = False
    supports_templates: bool = False
    supports_rich_text: bool = False
    supports_buttons: bool = False
    max_buttons: int | None = Field(default=None, gt=0)
    supports_cards: bool = False
    supports_quick_replies: bool = False
    supports_media: bool = False
    supported_media_types: list[str] = Field(default_factory=list)
    """The MIME types of the files the channel takes (RFC §5.8); empty takes
    every type. A file of another type is delivered as its text form (§6.6)."""
    max_media_size_bytes: int | None = Field(default=None, gt=0)
    """The largest file the channel takes; a file of known size over it is
    delivered as its text form (RFC §5.8, §6.6). ``None``: any size."""
    supports_audio: bool = False
    max_audio_duration_seconds: int | None = Field(default=None, gt=0)
    supported_audio_formats: list[str] = Field(default_factory=list)
    supports_video: bool = False
    max_video_duration_seconds: int | None = Field(default=None, gt=0)
    supported_video_formats: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


AI_CHANNEL_CONTEXT_EVENTS = 50
"""The history an AI channel reads per turn unless told otherwise."""


def ai_channel_capabilities() -> ChannelCapabilities:
    """What an AI channel can do: text, rich content and media whatever the
    model sees, each media read by the channel itself (RFC §6.4)."""
    return ChannelCapabilities(
        media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH, ChannelMediaType.MEDIA],
        supports_rich_text=True,
        supports_media=True,
    )


class AgentIdentity(BaseModel):
    """What the room's owner reads of an agent (RFC §10.6.2, §19.1)."""

    name: str | None = None
    role: str | None = None
    description: str | None = None
    scope: str | None = None
    greeting: str | None = None
    language: str | None = None
    purpose: str | None = None
    """What a supervisor's roster says of it as a worker (RFC §19.7.3)."""
    voice: str | None = None
    """Its TTS voice (RFC §19.1)."""
    auto_greet: bool = True
    """Whether the owner sends :attr:`greeting` when a session starts."""


class ChannelDeclaration(BaseModel):
    """What the room's owner knows of a channel without its live object
    (RFC §10.6.2): enough to bind it, plan its turns and read it in a
    strategy. An executor builds the channel itself.

    A field left out reads as an AI channel built with its defaults:
    ``channel.declaration()`` gives a live channel's own.
    """

    channel_id: str = Field(min_length=1)
    channel_type: ChannelType = ChannelType.AI
    category: ChannelCategory = ChannelCategory.INTELLIGENCE
    capabilities: ChannelCapabilities = Field(default_factory=ai_channel_capabilities)
    identity: AgentIdentity | None = None
    """An agent's identity; None for a channel that is no agent."""
    thinks_while_listening: bool = False
    """It carries a thinker (RFC §6.4), which a discussion refuses."""
    takes_owner_tools: bool = True
    """Its turns declare the tools its owner serves (RFC §10.6.5): an AI
    channel's do, an external agent's over ACP do not."""
    recent_events_window: int = Field(default=AI_CHANNEL_CONTEXT_EVENTS, ge=0)
    """How many recent room events the owner loads for its turns."""
    serializes_room_turns: bool = False
    """Its turns in a room run one after the other, in the order they were
    asked, which the room's owner keeps (RFC §6.4)."""


class ChannelBinding(BaseModel):
    """A channel's attachment to a room."""

    channel_id: Id
    room_id: Id
    channel_type: ChannelType
    category: ChannelCategory = ChannelCategory.TRANSPORT
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL
    access: Access = Access.READ_WRITE
    muted: bool = False
    output_muted: bool = False
    visibility: str = Visibility.ALL
    # The binding's correspondent (RFC §10.4): the sender a room was created
    # for, or the first one routed to it. The default router admits no one
    # else through a binding that names someone, unless it is ``group``.
    participant_id: str | None = None
    # Several senders share this binding's conversation (a group chat on a
    # channel dedicated to its room), so every sender on it belongs here. The
    # integrator declares it: nothing stored tells a group channel from a
    # number shared by many correspondents (RFC §5.7).
    group: bool = False
    # Not the read-tracking source of truth: the read position lives in the
    # ``read_markers`` table (see ``mark_read`` / ``list_read_markers``). This
    # field is an optional per-binding hint only and is not advanced by the
    # read API — read receipts and unread counts derive from ``read_markers``.
    last_read_index: int | None = Field(default=None, ge=0)
    attached_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    capabilities: ChannelCapabilities = Field(default_factory=ChannelCapabilities)
    rate_limit: RateLimit | None = None
    retry_policy: RetryPolicy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def sends(self) -> bool:
        """Whether the binding's direction delivers to the outside (RFC
        §4.2): a transport is delivered to only when it sends. Every binding
        that reads still receives through ``on_event()``."""
        return self.direction in (ChannelDirection.OUTBOUND, ChannelDirection.BIDIRECTIONAL)

    @property
    def can_write(self) -> bool:
        """RFC §7.5 — whether this binding may write events into the room.

        True iff the binding has write access (``READ_WRITE`` or
        ``WRITE_ONLY``) and is not muted. Gates both direct injection into
        the timeline (inbound pipeline) and source broadcast (event router).
        """
        return self.access in (Access.READ_WRITE, Access.WRITE_ONLY) and not self.muted


class ChannelOutput(BaseModel):
    """Output produced by a channel after receiving an event."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responded: bool = False
    response_events: list[RoomEvent] = Field(default_factory=list)
    response_stream: Any = Field(default=None, exclude=True)
    response_metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description=(
            "The turn's response-metadata record (AIContext.response_metadata, "
            "the same instance), merged into each MESSAGE event persisted from "
            "response_stream as it stands when that segment is persisted — a "
            "tool handler's writes mid-loop reach the segments that follow. The "
            "non-streaming path bakes its final state into response_events."
        ),
    )
    provider_result: ProviderResult | None = None
    """Native result of a transport delivery attempt.

    Transport channels preserve it here so the framework can report provider
    message ids and metadata without overloading AI response metadata.
    """
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata_updates: dict[str, Any] = Field(default_factory=dict)
    """Keys the framework writes on the room's metadata once the delivery set
    has run, merged with the other targets' in the order they answered,
    ``on_event()``'s and ``deliver()``'s alike (RFC §10.1 step 15). A key only
    the framework writes, under ``roomkit:`` or of the room's private state
    (``author_register``), is dropped with a warning, the rest applied."""
    error: Exception | None = Field(default=None, exclude=True)
    """An error the channel met while producing this output.

    The output is still delivered (an AI turn the provider interrupted after a
    tool round keeps its rounds, RFC §6.4), and the error surfaces as for a
    channel that raised: ON_ERROR fires and the caller's
    ``InboundResult.error`` carries it.
    """

    @classmethod
    def empty(cls) -> ChannelOutput:
        """Create an empty output with no responses or side effects."""
        return cls()
