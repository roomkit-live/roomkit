"""Data models for background task delegation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from roomkit.core._failure_log import was_reported
from roomkit.core.exceptions import TaskCutShortError, TaskTurnFailedError
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
        self._done: asyncio.Event | None = None
        self._start_time = time.monotonic()
        # Whether the delegation posts the task on the StatusBus (RFC §23.3):
        # off for a caller that follows its tasks there itself.
        self._post_status = True

    def _get_done_event(self) -> asyncio.Event:
        """Lazily create the Event on first use (inside a running loop)."""
        if self._done is None:
            self._done = asyncio.Event()
        return self._done

    async def wait(self, timeout: float | None = None) -> DelegatedTaskResult:
        """Block until the task completes or *timeout* seconds elapse.

        Raises:
            asyncio.TimeoutError: If *timeout* is exceeded.
            RuntimeError: If the task finished without a result.
        """
        await asyncio.wait_for(self._get_done_event().wait(), timeout=timeout)
        if self.result is None:
            msg = f"Task {self.id} finished without a result"
            raise RuntimeError(msg)
        return self.result

    def cancel(self) -> None:
        """Mark the task as cancelled and unblock its waiters, and nothing
        else: the work goes on, and no hook or callback hears of it. To
        cancel the work, as every task ends (RFC §23.3), use the task
        runner's ``cancel``."""
        done = self._get_done_event()
        if done.is_set():
            return
        self.status = TaskStatus.CANCELLED
        self.result = DelegatedTaskResult(
            task_id=self.id,
            child_room_id=self.child_room_id,
            parent_room_id=self.parent_room_id,
            agent_id=self.agent_id,
            duration_ms=(time.monotonic() - self._start_time) * 1000,
            **cancelled_task_fields(None),
        )
        done.set()

    def _set_result(self, result: DelegatedTaskResult) -> None:
        """Set the task result and unblock waiters (called by TaskRunner)."""
        self.status = result.status
        self.result = result
        self._get_done_event().set()


def task_work(result: Any) -> str:
    """The work a delegated task hands back: its output, unless it failed. A
    failed task hands none, whatever its output keeps (a worker cut short
    keeps its narration there, RFC §23.3). Accepts a real result, a
    duck-typed one, or ``None``."""
    if result is None or getattr(result, "status", None) == TaskStatus.FAILED:
        return ""
    return getattr(result, "output", None) or ""


def cancelled_task_fields(context: dict[str, Any] | None) -> dict[str, Any]:
    """The outcome of a delegated task cancelled from outside (its caller's
    timeout, the runner's ``cancel`` or ``close``): cancelled, with no output
    (RFC §23.3)."""
    return {
        "status": TaskStatus.CANCELLED,
        "output": None,
        "error": "cancelled",
        "metadata": dict(context or {}),
    }


def task_cut_reason(result: Any) -> str | None:
    """How a failed task's turn was cut short (its round cap, deadline or
    budget, a stop), when a cut ended it. ``None`` for a task that completed,
    or that failed with an error, even one raised after its turn began and
    named on its ``loop_end_reason`` (RFC §23.3)."""
    reason = (getattr(result, "metadata", None) or {}).get("loop_end_reason")
    if reason is None or getattr(result, "error", None) != TaskCutShortError.message_for(reason):
        return None
    return reason


def finished_task_fields(
    response: str | None, failure: BaseException | None, context: dict[str, Any] | None
) -> dict[str, Any]:
    """The outcome of a delegated task that ran: completed with the worker's
    *response*, or failed with *failure*. A worker cut short keeps its last
    narration as the output and how its turn ended in the metadata (RFC
    §23.3); any other failure keeps nothing."""
    output, metadata = response, dict(context or {})
    if isinstance(failure, (TaskCutShortError, TaskTurnFailedError)):
        output = output or failure.narration
        metadata["loop_end_reason"] = failure.reason
    if failure is not None and was_reported(failure):
        # Its turn reported it to ON_ERROR in the child room: whoever reads
        # the task hands it on without a second report (RFC §23.3).
        metadata["error_reported"] = True
    return {
        "status": TaskStatus.COMPLETED if response else TaskStatus.FAILED,
        "output": output,
        "error": str(failure) if failure is not None else None,
        "metadata": metadata,
    }
