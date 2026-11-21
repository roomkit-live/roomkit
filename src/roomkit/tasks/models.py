"""Data models for background task delegation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from roomkit.core._failure_log import mark_reported, was_reported
from roomkit.core.exceptions import TaskCutShortError, TaskTurnFailedError
from roomkit.models.enums import TaskStatus
from roomkit.models.response_metadata import TURNS_KEY, TurnEntries
from roomkit.models.task import DelegatedTaskResult

logger = logging.getLogger("roomkit.tasks")


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
        brief: str | None = None,
        description: str | None = None,
        hand_back: str = "relay",
    ) -> None:
        self.id = id
        self.child_room_id = child_room_id
        self.parent_room_id = parent_room_id
        self.agent_id = agent_id
        self.task = task
        """What was asked: the label a hand-back, the StatusBus and a turn's
        notes name the task by (RFC §23.3 step 4)."""
        self.brief = brief
        """What the worker is told, when it says more than the task."""
        self.description = description or brief or task
        """What the worker reads: the brief, else the task, with the room's
        conversation when the delegation hands it over."""
        self.hand_back = hand_back
        """How the result reaches the room: ``relay`` (the notified agent says
        it in a turn of its own) or ``post`` (posted as it is, as its message)."""
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: DelegatedTaskResult | None = None
        self._done: asyncio.Event | None = None
        self._start_time = time.monotonic()
        # Whether the delegation posts the task on the StatusBus (RFC §23.3):
        # off for a caller that follows its tasks there itself.
        self._post_status = True
        # Whether ON_TASK_DELEGATED fired for it: only an announced task's
        # end is reported (RFC §23.3).
        self._announced = False

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


# The keys of a task's metadata only the framework writes: how its worker's
# turn ended, whether its failure was reported, and the turn's record (RFC
# §23.3 step 6). A context that carries one, which a model writes on a
# delegate_task call, never stands for the framework's.
_TASK_END_KEYS = frozenset({"loop_end_reason", "error_reported", TURNS_KEY})


def _context_metadata(context: dict[str, Any] | None) -> dict[str, Any]:
    """The task's metadata the *context* starts, without the keys only the
    framework writes."""
    return {key: value for key, value in (context or {}).items() if key not in _TASK_END_KEYS}


def cancelled_task_fields(
    context: dict[str, Any] | None,
    turns: TurnEntries | None = None,
    *,
    worker: str | None = None,
) -> dict[str, Any]:
    """The outcome of a delegated task cancelled from outside (its caller's
    timeout, the runner's ``cancel`` or ``close``): cancelled, with no output.
    A task whose worker's turn had begun (*worker*) carries it under
    ``turns``, ``cancelled``, over what its record held (*turns*), as a room
    turn's caller reads a cancelled read; one cut before carries none (RFC
    §23.3)."""
    metadata = _context_metadata(context)
    if worker is not None:
        entries = dict(turns or {})
        entries[worker] = {**entries.get(worker, {}), "loop_end_reason": "cancelled"}
        metadata[TURNS_KEY] = entries
    return {
        "status": TaskStatus.CANCELLED,
        "output": None,
        "error": "cancelled",
        "metadata": metadata,
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
    response: str | None,
    failure: BaseException | None,
    context: dict[str, Any] | None,
    turns: TurnEntries | None = None,
) -> dict[str, Any]:
    """The outcome of a delegated task that ran: completed with the worker's
    *response*, or failed with *failure*, kept as raised. A worker cut short
    keeps its last narration as the output and how its turn ended in the
    metadata (RFC §23.3); any other failure keeps nothing. Its turn's record
    (*turns*, by responder) goes under ``turns`` however it ended."""
    output, metadata = response, _context_metadata(context)
    if turns:
        metadata[TURNS_KEY] = dict(turns)
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
        "exception": _as_raised(failure),
        "metadata": metadata,
    }


def _as_raised(failure: BaseException | None) -> Exception | None:
    """What the worker's turn raised: a failed turn's own error, not the
    wrapper that carries its end and narration, marked reported when the
    wrapper was, so whoever hands it on reports it no second time."""
    if isinstance(failure, TaskTurnFailedError) and isinstance(failure.__cause__, Exception):
        cause = failure.__cause__
        return mark_reported(cause) if was_reported(failure) else cause
    return failure if isinstance(failure, Exception) else None
