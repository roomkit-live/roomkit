"""Event and content models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    DeleteType,
    EventStatus,
    EventType,
    ToolCallOutcome,
    Visibility,
)


class TextContent(BaseModel):
    """Plain text message content."""

    type: Literal["text"] = "text"
    body: str
    language: str | None = None


class RichContent(BaseModel):
    """Rich formatted content (HTML/Markdown)."""

    type: Literal["rich"] = "rich"
    body: str
    format: Literal["html", "markdown"] = "markdown"
    plain_text: str | None = None
    buttons: list[dict[str, Any]] = Field(default_factory=list)
    cards: list[dict[str, Any]] = Field(default_factory=list)
    quick_replies: list[str] = Field(default_factory=list)


class MediaContent(BaseModel):
    """Media attachment content."""

    type: Literal["media"] = "media"
    url: str
    mime_type: str
    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    caption: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "data:")):
            raise ValueError("URL must start with http://, https://, or data:")
        return v


class LocationContent(BaseModel):
    """Geographic location content."""

    type: Literal["location"] = "location"
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    label: str | None = None
    address: str | None = None


class AudioContent(BaseModel):
    """Audio message content."""

    type: Literal["audio"] = "audio"
    url: str
    mime_type: str = "audio/ogg"
    duration_seconds: float | None = Field(default=None, ge=0.0)
    transcript: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "data:")):
            raise ValueError("URL must start with http://, https://, or data:")
        return v


class VideoContent(BaseModel):
    """Video message content."""

    type: Literal["video"] = "video"
    url: str
    mime_type: str = "video/mp4"
    duration_seconds: float | None = Field(default=None, ge=0.0)
    thumbnail_url: str | None = None

    @field_validator("url", "thumbnail_url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://", "data:")):
            raise ValueError("URL must start with http://, https://, or data:")
        return v


class CompositeContent(BaseModel):
    """Multi-part content combining multiple content types."""

    type: Literal["composite"] = "composite"
    parts: list[EventContent]

    @model_validator(mode="after")
    def _validate_parts(self) -> CompositeContent:
        if not self.parts:
            raise ValueError("CompositeContent must have at least one part")
        depth = self._nesting_depth(self)
        if depth > 5:
            raise ValueError(f"CompositeContent nesting depth {depth} exceeds maximum of 5")
        return self

    @staticmethod
    def _nesting_depth(content: object, current: int = 1) -> int:
        """Recursively compute nesting depth of CompositeContent."""
        if not isinstance(content, CompositeContent):
            return 0
        max_child = 0
        for part in content.parts:
            child_depth = CompositeContent._nesting_depth(part, current + 1)
            if child_depth > max_child:
                max_child = child_depth
        return 1 + max_child


class SystemContent(BaseModel):
    """System-generated content."""

    type: Literal["system"] = "system"
    body: str
    code: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TemplateContent(BaseModel):
    """Pre-approved template content (WhatsApp Business, etc.)."""

    type: Literal["template"] = "template"
    template_id: str
    language: str = "en"
    parameters: dict[str, str] = Field(default_factory=dict)
    body: str | None = None


class EditContent(BaseModel):
    """Edit of a previously sent message."""

    type: Literal["edit"] = "edit"
    target_event_id: str
    new_content: EventContent
    edit_source: str | None = None


class DeleteContent(BaseModel):
    """Deletion of a previously sent message."""

    type: Literal["delete"] = "delete"
    target_event_id: str
    delete_type: DeleteType = DeleteType.SENDER
    reason: str | None = None


class ToolCallContent(BaseModel):
    """Tool call content — used for both TOOL_CALL_START and TOOL_CALL_END events.

    At start: tool_name + arguments populated, status="pending".
    At end: result + duration_ms populated, status="completed" or "failed",
    and ``outcome``.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    status: Literal["pending", "completed", "failed"] = "pending"
    duration_ms: int | None = None
    error: str | None = None
    # MCP CallToolResult.structuredContent, captured before large-result
    # eviction rewrote ``result`` — UI surfaces read their data from it.
    structured_content: dict[str, Any] | None = None
    # How the call ended, on an end row: what ``status`` folds into
    # completed/failed, stated (RFC §6.4). ``None`` on a row written without
    # it, which a reader takes by its ``status``.
    outcome: ToolCallOutcome | None = None
    # A call that ran although RoomKit refused it (an ACP agent past a
    # rejected permission): its end row says so, as its report does (RFC §9.3).
    refused_but_ran: bool = False


EventContent = Annotated[
    TextContent
    | RichContent
    | MediaContent
    | LocationContent
    | AudioContent
    | VideoContent
    | CompositeContent
    | SystemContent
    | TemplateContent
    | EditContent
    | DeleteContent
    | ToolCallContent,
    Field(discriminator="type"),
]


class ChannelData(BaseModel):
    """Provider-specific channel metadata."""

    provider: str | None = None
    external_id: str | None = None
    thread_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EventSource(BaseModel):
    """Origin information for an event."""

    channel_id: str
    channel_type: ChannelType
    direction: ChannelDirection = ChannelDirection.INBOUND
    participant_id: str | None = None
    external_id: str | None = None
    provider: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    provider_message_id: str | None = None


class RoomEvent(BaseModel):
    """A single event in a room conversation."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    room_id: str
    type: EventType = EventType.MESSAGE
    source: EventSource
    content: EventContent
    status: EventStatus = EventStatus.PENDING
    blocked_by: str | None = None
    visibility: str = Visibility.ALL
    # Which intelligence channels are ASKED TO ACT on this event (RFC §19.3).
    # Not visibility: ``visibility`` says who may see, this says who is
    # solicited, and the two are independent — addressing one agent hides
    # nothing from another, and transport delivery is never affected.
    # ``None`` means unaddressed: the router decides, as before.
    addressed_to: list[str] | None = None
    response_visibility: str | None = None
    index: int = Field(default=0, ge=0)
    chain_depth: int = Field(default=0, ge=0)
    # In-app thread root (flat two-level model): a reply points at its thread
    # root; a root or non-threaded message is ``None``. The locked pipeline
    # normalises any parent reference to the root, so this is always a root id.
    # Distinct from ``channel_data.thread_id`` (provider-native reference).
    parent_event_id: str | None = None
    # The event this one answers (RFC §8.5): every event an intelligence
    # channel produces for a turn names the event that triggered the turn, so
    # an answer is tied to its request. Unrelated to the thread above.
    responds_to: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel_data: ChannelData = Field(default_factory=ChannelData)
    delivery_results: dict[str, Any] = Field(default_factory=dict)


#: Metadata key marking the terminal message of a turn the provider
#: interrupted after a round (RFC §6.4): the interruption marker, not an answer.
#: Distinct from ``interrupted``, which marks a spoken reply a barge-in cut
#: (RFC §12.3.13).
INTERRUPTION_MARKER_KEY = "interruption_marker"


def is_interruption_marker(event: RoomEvent) -> bool:
    """Whether *event* is an interruption marker (RFC §6.4).

    It says an agent's turn was cut: it solicits no agent (RFC §19.3), and
    nothing that reads an agent's answer takes it for one.
    """
    return event.metadata.get(INTERRUPTION_MARKER_KEY) is True


def is_tool_call_record(event: RoomEvent) -> bool:
    """Whether *event* is a tool call's start or end row.

    An activity record, not a message: no agent answers one, so a room of
    agents does not answer another agent's tool calls, and one not asked past
    the chain-depth limit leaves no record for it (RFC §8.3).
    """
    return event.type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END)


def answer_text(event: RoomEvent) -> str | None:
    """The text *event* carries as an agent's answer, or ``None``.

    ``None`` for anything but a non-empty text, and for the interruption
    marker, which is no answer (RFC §6.4).
    """
    if isinstance(event.content, TextContent) and event.content.body:
        return None if is_interruption_marker(event) else event.content.body
    return None


class ThreadSummary(BaseModel):
    """Aggregate view of a thread, keyed by its root event.

    Returned by :meth:`ConversationStore.get_thread_summaries` so a client can
    render a "N replies · last reply at" affordance on a root message without
    fetching every reply.
    """

    root_event_id: str
    reply_count: int = 0
    last_reply_at: datetime | None = None
