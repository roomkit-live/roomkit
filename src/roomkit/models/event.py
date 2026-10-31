"""Event and content models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from roomkit._media_url import is_data_uri, is_link, is_safe_reference, lowercase_scheme
from roomkit.models._keys import Id, IdempotencyKey
from roomkit.models._time import UtcDatetime
from roomkit.models.enums import (
    ChannelType,
    DeleteType,
    EditSource,
    EventDirection,
    EventStatus,
    EventType,
    ToolCallOutcome,
    VerificationStatus,
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
    """Media attachment content.

    ``url`` is a link (``http``, ``https``), a ``data:`` URI, or a reference:
    ``<scheme>://…`` in a scheme the application chooses, naming a media it
    holds and resolves for the AI channels (RFC §6.4). A scheme a reader could
    fetch or open itself (``file``, ``ftp``) or run (``javascript``,
    ``vbscript``) is refused, and so is a reference a path reader could follow
    (a one-letter scheme, a ``..`` segment, a backslash): some transports read
    a URL they are handed as a local path.
    """

    type: Literal["media"] = "media"
    url: str
    mime_type: str
    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    caption: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if is_link(v) or is_data_uri(v):
            return lowercase_scheme(v)
        if is_safe_reference(v):
            return v
        raise ValueError(
            "URL must be a link (http://, https://), a data: URI, or a reference "
            "<scheme>://... (two letters of scheme at least, not file, ftp, "
            "javascript or vbscript, no '..' segment and no backslash)"
        )


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
        refuse_too_deep(self)
        return self


MAX_COMPOSITE_DEPTH = 5
"""The composite levels an event's content may hold (RFC §5.3)."""


def holds_too_deep(content: object, level: int = 0) -> bool:
    """Whether *content*, standing at composite *level*, holds a level past
    :data:`MAX_COMPOSITE_DEPTH` (RFC §5.3): the outermost composite is level
    1, a composite among the parts of a level-n one is level n + 1, and a
    part that is not a composite adds none. An edit's new content stands in
    the edit's place. The walk stops at the first level past the limit."""
    if isinstance(content, EditContent):
        return holds_too_deep(content.new_content, level)
    if not isinstance(content, CompositeContent):
        return False
    if level >= MAX_COMPOSITE_DEPTH:
        return True
    return any(holds_too_deep(part, level + 1) for part in content.parts)


def refuse_too_deep(content: object) -> None:
    """Refuse *content* holding a sixth composite level where it is built
    (RFC §5.3): ``model_copy(update=...)`` skips the models' validators, so
    each boundary where content enters the kit asks again.

    Raises:
        ValueError: *content* holds more than :data:`MAX_COMPOSITE_DEPTH`
            composite levels.
    """
    if holds_too_deep(content):
        raise ValueError(
            f"CompositeContent nesting depth exceeds the maximum of {MAX_COMPOSITE_DEPTH} levels"
        )


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
    edit_source: EditSource | None = None


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
    # What the call does, in the sentence its model wrote for the person (an
    # agent's ``tool_call_intents``, RFC §6.4): display text, apart from the
    # arguments. Written only when set: a call without one has no ``intent``
    # key, on a row as on the wire.
    intent: str | None = None

    # No return annotation: one would replace the model's own fields in its
    # serialization JSON Schema (an integrator's OpenAPI reads them).
    @model_serializer(mode="wrap")
    def _intent_only_when_set(self, handler: SerializerFunctionWrapHandler):
        fields: dict[str, Any] = handler(self)
        if self.intent is None:
            fields.pop("intent", None)
        return fields


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
    """Provider-specific channel metadata, typed per channel (RFC §5.4):
    the keys a channel writes beside these (an SMS's ``segments``, an
    email's ``subject``) are kept."""

    model_config = ConfigDict(extra="allow")

    provider: str | None = None
    external_id: str | None = None
    thread_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EventSource(BaseModel):
    """Origin information for an event."""

    channel_id: Id
    channel_type: ChannelType
    direction: EventDirection = EventDirection.INBOUND
    participant_id: str | None = None
    external_id: str | None = None
    provider: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    provider_message_id: str | None = None
    verification: VerificationStatus | None = None
    """The sender's verification when the event was received (RFC §11.7.3);
    None where none applies (an unidentified sender, an agent, the framework)."""


class RoomEvent(BaseModel):
    """A single event in a room conversation."""

    id: Id = Field(default_factory=lambda: uuid4().hex)
    room_id: Id
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
    idempotency_key: IdempotencyKey = None
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel_data: ChannelData = Field(default_factory=ChannelData)
    delivery_results: dict[str, Any] = Field(default_factory=dict)


#: Metadata key marking the terminal message of a turn the provider
#: interrupted after a round (RFC §6.4): the interruption marker, not an answer.
#: Distinct from ``interrupted``, which marks a spoken reply a barge-in cut
#: (RFC §12.3.13).
INTERRUPTION_MARKER_KEY = "interruption_marker"


def with_pipeline_id(event: RoomEvent) -> RoomEvent:
    """*event* with a fresh id: the pipeline gives every event it commits its
    id, whatever id a channel or a hook built it with (RFC §8.1)."""
    return event.model_copy(update={"id": uuid4().hex})


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
    last_reply_at: UtcDatetime | None = None
