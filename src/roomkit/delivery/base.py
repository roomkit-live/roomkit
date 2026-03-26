"""Delivery backend ABC and models.

A ``DeliveryBackend`` decouples *enqueue* from *execution* so that
delivery requests survive process restarts and can be routed across
workers in multi-process deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.delivery.backend")


class DeliveryItemStatus(StrEnum):
    """Lifecycle status of a delivery item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class DeliveryItem(BaseModel):
    """Serializable delivery request — the unit of work in the queue."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    room_id: str
    content: str
    channel_id: str | None = None
    strategy: dict[str, Any] = Field(default_factory=lambda: {"type": "immediate", "params": {}})
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DeliveryItemStatus = DeliveryItemStatus.PENDING
    worker_id: str | None = None
    error: str | None = None


class DeliveryBackend(ABC):
    """ABC for persistent delivery queue backends.

    Implementations provide a durable queue with at-least-once
    semantics via the ``enqueue`` / ``dequeue`` / ``ack`` / ``nack``
    lifecycle.
    """

    @abstractmethod
    async def enqueue(self, item: DeliveryItem) -> None:
        """Add a delivery item to the queue."""

    @abstractmethod
    async def dequeue(
        self,
        worker_id: str,
        batch_size: int = 1,
        timeout: float = 5.0,
    ) -> list[DeliveryItem]:
        """Claim up to *batch_size* items.  Blocks up to *timeout* seconds."""

    @abstractmethod
    async def ack(self, item_id: str) -> None:
        """Acknowledge successful delivery — removes the item."""

    @abstractmethod
    async def nack(self, item_id: str, error: str | None = None) -> None:
        """Negative-acknowledge.  Re-enqueues or dead-letters the item."""

    @abstractmethod
    async def dead_letter(self, item_id: str, error: str) -> None:
        """Move an item to the dead-letter queue."""

    @abstractmethod
    async def get_queue_depth(self) -> int:
        """Return the number of pending items (observability)."""

    @abstractmethod
    async def get_dead_letter_items(self, limit: int = 50) -> list[DeliveryItem]:
        """Return items in the dead-letter queue."""

    # -- Lifecycle (optional overrides) -----------------------------------

    async def start(self, kit: RoomKit) -> None:  # noqa: B027
        """Called by ``RoomKit`` on startup.  Override to start a worker loop."""

    async def close(self) -> None:  # noqa: B027
        """Called by ``RoomKit.close()``.  Override for graceful shutdown."""

    # -- Shared worker helpers --------------------------------------------

    _worker_task: asyncio.Task[None] | None = None

    async def _cancel_worker_task(self) -> None:
        """Cancel the background worker task if running."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
