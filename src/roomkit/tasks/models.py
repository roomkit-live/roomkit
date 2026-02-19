"""Data models for background task delegation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.enums import TaskStatus

logger = logging.getLogger("roomkit.tasks")


class DelegatedTaskResult(BaseModel):
    """Result of a completed delegated task."""

    task_id: str
    child_room_id: str
    parent_room_id: str
    agent_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    output: str | None = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DelegatedTask:
    """Handle for a running delegated task.

    Not a Pydantic model — holds mutable state and an ``asyncio.Event``
    for callers that want to block until the task completes.
    """

    def __init__(
        self,
        *,
        id: str,
        child_room_id: str,
        parent_room_id: str,
        agent_id: str,
        task: str,
    ) -> None:
        self.id = id
        self.child_room_id = child_room_id
        self.parent_room_id = parent_room_id
        self.agent_id = agent_id
        self.task = task
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: DelegatedTaskResult | None = None
        self._done = asyncio.Event()
        self._start_time = time.monotonic()

    async def wait(self, timeout: float | None = None) -> DelegatedTaskResult:
        """Block until the task completes or *timeout* seconds elapse.

        Raises:
            asyncio.TimeoutError: If *timeout* is exceeded.
            RuntimeError: If the task finished without a result.
        """
        await asyncio.wait_for(self._done.wait(), timeout=timeout)
        if self.result is None:
            msg = f"Task {self.id} finished without a result"
            raise RuntimeError(msg)
        return self.result

    def cancel(self) -> None:
        """Mark the task as cancelled and unblock waiters."""
        if self._done.is_set():
            return
        self.status = TaskStatus.CANCELLED
        elapsed = (time.monotonic() - self._start_time) * 1000
        self.result = DelegatedTaskResult(
            task_id=self.id,
            child_room_id=self.child_room_id,
            parent_room_id=self.parent_room_id,
            agent_id=self.agent_id,
            status=TaskStatus.CANCELLED,
            duration_ms=elapsed,
        )
        self._done.set()

    def _set_result(self, result: DelegatedTaskResult) -> None:
        """Set the task result and unblock waiters (called by TaskRunner)."""
        self.status = result.status
        self.result = result
        self._done.set()
