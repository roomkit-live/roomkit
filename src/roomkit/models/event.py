"""Event and content models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    EventStatus,
    EventType,
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
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
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


EventContent = Annotated[
    TextContent
    | RichContent
    | MediaContent
    | LocationContent
    | AudioContent
    | VideoContent
    | CompositeContent
    | SystemContent
    | TemplateContent,
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
    visibility: str = "all"
    index: int = Field(default=0, ge=0)
    chain_depth: int = Field(default=0, ge=0)
    parent_event_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel_data: ChannelData = Field(default_factory=ChannelData)
    delivery_results: dict[str, Any] = Field(default_factory=dict)
