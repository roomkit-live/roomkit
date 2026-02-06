"""Delivery and provider result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import EventType
from roomkit.models.event import EventContent, RoomEvent


class ProviderResult(BaseModel):
    """Result from a provider delivery attempt."""

    success: bool
    provider_message_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboundMessage(BaseModel):
    """A message received from an external provider."""

    channel_id: str
    sender_id: str
    content: EventContent
    event_type: EventType = EventType.MESSAGE
    external_id: str | None = None
    thread_id: str | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboundResult(BaseModel):
    """Result of processing an inbound message."""

    event: RoomEvent | None = None
    blocked: bool = False
    reason: str | None = None


class DeliveryResult(BaseModel):
    """Result of delivering an event to a channel."""

    channel_id: str
    success: bool
    provider_result: ProviderResult | None = None
    error: str | None = None


class DeliveryStatus(BaseModel):
    """Status update for an outbound message from a provider webhook.

    Providers send status webhooks when messages are sent, delivered, failed, etc.
    Use this with the ON_DELIVERY_STATUS hook to track outbound message delivery.

    Attributes:
        provider: Provider name (e.g., "telnyx", "twilio").
        message_id: Provider's unique message identifier.
        status: Status string (e.g., "sent", "delivered", "failed").
        recipient: Phone number/address the message was sent to.
        sender: Phone number/address the message was sent from.
        error_code: Provider-specific error code (if failed).
        error_message: Human-readable error message (if failed).
        timestamp: When the status was reported.
        raw: Original webhook payload for debugging.
    """

    provider: str
    message_id: str
    status: str
    recipient: str = ""
    sender: str = ""
    error_code: str | None = None
    error_message: str | None = None
    timestamp: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
