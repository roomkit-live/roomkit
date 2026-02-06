"""Channel-related data models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    DeliveryMode,
)
from roomkit.models.event import RoomEvent
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
    max_media_size_bytes: int | None = Field(default=None, gt=0)
    supports_audio: bool = False
    max_audio_duration_seconds: int | None = Field(default=None, gt=0)
    supported_audio_formats: list[str] = Field(default_factory=list)
    supports_video: bool = False
    max_video_duration_seconds: int | None = Field(default=None, gt=0)
    supported_video_formats: list[str] = Field(default_factory=list)
    delivery_mode: DeliveryMode = DeliveryMode.BROADCAST
    custom: dict[str, Any] = Field(default_factory=dict)


class ChannelBinding(BaseModel):
    """A channel's attachment to a room."""

    channel_id: str
    room_id: str
    channel_type: ChannelType
    category: ChannelCategory = ChannelCategory.TRANSPORT
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL
    access: Access = Access.READ_WRITE
    muted: bool = False
    visibility: str = "all"
    participant_id: str | None = None
    last_read_index: int | None = Field(default=None, ge=0)
    attached_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capabilities: ChannelCapabilities = Field(default_factory=ChannelCapabilities)
    rate_limit: RateLimit | None = None
    retry_policy: RetryPolicy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChannelOutput(BaseModel):
    """Output produced by a channel after receiving an event."""

    responded: bool = False
    response_events: list[RoomEvent] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    metadata_updates: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> ChannelOutput:
        """Create an empty output with no responses or side effects."""
        return cls()
