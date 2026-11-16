"""Delivery backend ABC and models.

A ``DeliveryBackend`` decouples *enqueue* from *execution* so that
delivery requests survive process restarts and can be routed across
workers in multi-process deployments.
"""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.delivery import DeliveryOutcome

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
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class DeliveryItem(BaseModel):
    """Serializable delivery request — the unit of work in the queue."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    room_id: str
    content: str
    channel_id: str | None = None
    addressed_to: list[str] | None = None
    idempotency_key: str | None = None
    session_id: str | None = None
    chain_depth: int = Field(default=0, ge=0)
    instruction: bool = False
    strategy: dict[str, Any] = Field(default_factory=lambda: {"type": "immediate", "params": {}})
    metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_record: str | None = None
    """What the runtime wrote, when it delivers its own text (a background
    result handed back, ``"handback"``): the delivered event carries it as
    its provenance (RFC §6.4), which ``deliver()`` never sets and the inbound
    pipeline removes from what a caller's metadata brings."""
    retry_count: int = 0
    max_retries: int = 3
    """Retries after the first attempt, as a binding's ``RetryPolicy`` counts
    them (RFC §13.2): 3 means four attempts at most."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DeliveryItemStatus = DeliveryItemStatus.PENDING
    worker_id: str | None = None
    error: str | None = None
    outcome: DeliveryOutcome | None = None


def retry_delay(retry_count: int, base: float, maximum: float) -> float:
    """Seconds before retry number *retry_count* (from 1), doubling from
    *base* up to *maximum*: the formula of a binding's ``RetryPolicy`` (RFC
    §13.2) with its exponential base of 2. The exponent stops at 64, past
    any cap, so no retry count overflows it."""
    return min(base * 2.0 ** min(retry_count - 1, 64), maximum)


def check_retry_delays(base: float, maximum: float) -> None:
    """Refuse retry delays that are negative, infinite or out of order."""
    if not (math.isfinite(base) and math.isfinite(maximum)) or base < 0 or maximum < base:
        raise ValueError("retry delays must be finite, with 0 <= retry_delay <= max_retry_delay")


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
        """Consume a terminal item (sent, blocked or unknown); removes it."""

    @abstractmethod
    async def nack(self, item_id: str, error: str | None = None) -> None:
        """Negative-acknowledge.  Re-enqueues or dead-letters the item.

        An item with retries left is attempted again after
        :func:`retry_delay`, not at once: a destination that is briefly
        unreachable (a voice session reconnecting) would otherwise spend
        every retry within milliseconds.
        """

    @abstractmethod
    async def dead_letter(self, item_id: str, error: str) -> None:
        """Move an item to the dead-letter queue."""

    @abstractmethod
    async def get_queue_depth(self) -> int:
        """Return the number of pending items (observability)."""

    @abstractmethod
    async def get_dead_letter_items(self, limit: int = 50) -> list[DeliveryItem]:
        """Return up to *limit* items of the dead-letter queue, the most
        recently dead-lettered first."""

    # -- Lifecycle (optional overrides) -----------------------------------

    async def start(self, kit: RoomKit) -> None:  # noqa: B027
        """Called by ``RoomKit`` on startup.  Override to start a worker loop."""

    async def close(self) -> None:  # noqa: B027
        """Called by ``RoomKit.close()``.  Override for graceful shutdown."""

    # -- Shared worker helpers --------------------------------------------

    _worker_task: asyncio.Task[None] | None = None

    async def _cancel_worker_task(self) -> None:
        """Cancel the background worker task if running."""
        await cancel_and_wait(self._worker_task)
        self._worker_task = None
