"""Abstract base class and types for realtime backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class EphemeralEventType(StrEnum):
    """Types of ephemeral events."""

    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    PRESENCE_ONLINE = "presence_online"
    PRESENCE_AWAY = "presence_away"
    PRESENCE_OFFLINE = "presence_offline"
    READ_RECEIPT = "read_receipt"
    REACTION = "reaction"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    THINKING_START = "thinking_start"
    THINKING_END = "thinking_end"
    CUSTOM = "custom"


@dataclass
class EphemeralEvent:
    """An ephemeral event that doesn't require persistence."""

    room_id: str
    type: EphemeralEventType
    user_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    channel_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "room_id": self.room_id,
            "type": self.type.value,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EphemeralEvent:
        """Create an EphemeralEvent from a dictionary."""
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            type=EphemeralEventType(data["type"]),
            user_id=data["user_id"],
            channel_id=data.get("channel_id"),
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


EphemeralCallback = Callable[[EphemeralEvent], Coroutine[Any, Any, None]]


class RealtimeBackend(ABC):
    """Abstract base for realtime pub/sub backends.

    Implement this to plug in any realtime backend (Redis pub/sub, NATS, etc.).
    The library ships with ``InMemoryRealtime`` for single-process deployments.
    """

    @abstractmethod
    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        """Publish an event to a channel."""
        ...

    @abstractmethod
    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        """Subscribe to a channel.

        Returns:
            A subscription ID that can be used to unsubscribe.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a channel.

        Returns:
            True if the subscription existed and was removed.
        """
        ...

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        """Convenience method to publish an event to a room channel."""
        await self.publish(f"room:{room_id}", event)

    async def subscribe_to_room(self, room_id: str, callback: EphemeralCallback) -> str:
        """Convenience method to subscribe to a room channel."""
        return await self.subscribe(f"room:{room_id}", callback)

    async def close(self) -> None:
        """Clean up resources.

        Override this method in subclasses that need cleanup.
        The default implementation does nothing.
        """
        return None
