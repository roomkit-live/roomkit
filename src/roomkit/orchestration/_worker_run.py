"""A strategy's worker delegation: one task handed to one agent, waited for
within the strategy's bound, and followed on the status bus (RFC §19.7.3,
§19.7.4, §23.3).

One sequence for the supervisor's workers (sequential, parallel, supervised,
per-worker) and the Loop's producer and reviewers: the worker is posted
pending, its task delegated and waited for, then one terminal entry is posted
however the delegation ends: completed, failed, timed out, raised or
cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from roomkit.core.exceptions import ToolTimeoutError
from roomkit.models.enums import TaskStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.orchestration.status_bus import StatusLevel, post_agent_lifecycle
from roomkit.tasks.models import failed_task_words, task_end_reason, task_work
from roomkit.tasks.status import RUN_ENTRY
from roomkit.tools.timeout import answer_within

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult


def task_input(event: RoomEvent) -> str:
    """The text a strategy's tasks carry for *event*: a plain text's body, else
    the empty string. A message it is empty for (a media, a composite, an empty
    text) is answered by the lead agent alone, whose own channel reads it
    (RFC §19.7.3, §19.7.4)."""
    content = event.content
    return content.body if isinstance(content, TextContent) else ""


def task_output(result: Any) -> str:
    """The text a strategy reads of a delegated task result: its work.

    A failed task reads as failed, with why from the closed list a model may
    read, never with its error, which is an exception's message for the logs
    and hooks, not for a model (RFC §9.3), nor with the narration a worker cut
    short keeps as its output (RFC §23.3). Accepts a real result, a duck-typed
    object, or ``None``.
    """
    if result is None:
        return ""
    if work := task_work(result):
        return work
    if getattr(result, "status", None) == TaskStatus.CANCELLED:
        return "The task was cancelled."
    failed = getattr(result, "error", None) or getattr(result, "status", None) == TaskStatus.FAILED
    return f"The task {failed_task_words(result)}." if failed else ""


def task_completed(result: Any) -> bool:
    """Whether a delegated task result reports ``completed`` status."""
    return getattr(result, "status", None) == TaskStatus.COMPLETED


@dataclass(frozen=True)
class WorkerOutcome:
    """How a worker's delegation ended.

    Attributes:
        output: What the strategy reads of it: the task's work, or that it
            failed, was cancelled or timed out (never an error's message).
        completed: Whether the task completed.
        task: The delegated task; ``None`` when it timed out.
        timed_out: Whether the strategy's bound cut the delegation.
    """

    output: str
    completed: bool
    task: DelegatedTask | None = None
    timed_out: bool = False

    @property
    def result(self) -> DelegatedTaskResult | None:
        """The task's result; ``None`` when it timed out."""
        return self.task.result if self.task is not None else None

    @property
    def end_reason(self) -> str | None:
        """Why the delegation did not complete, of the closed list a model
        may read (RFC §19.7.3): ``timeout`` when the strategy's bound cut it,
        whatever its task then ended with, else its task's end reason."""
        return "timeout" if self.timed_out else task_end_reason(self.result)


def _timed_out(timeout: float | None, task: DelegatedTask | None = None) -> WorkerOutcome:
    """The outcome of a delegation the strategy's *timeout* cut: read as
    timed out, by that bound."""
    return WorkerOutcome(f"The task timed out after {timeout:g}s.", False, task, timed_out=True)


class WorkerEnd(NamedTuple):
    """A worker delegation's terminal entry: its level, its detail, and what
    it adds to the status's metadata."""

    level: StatusLevel
    detail: str
    metadata: dict[str, Any] | None = None


def _completed_or_failed(outcome: WorkerOutcome) -> WorkerEnd:
    """A worker's terminal entry: completed or failed, with its output."""
    return WorkerEnd(
        StatusLevel.COMPLETED if outcome.completed else StatusLevel.FAILED, outcome.output
    )


@dataclass(frozen=True)
class WorkerStatus:
    """How a worker's delegation shows on the status bus.

    Attributes:
        metadata: Posted with every entry (the room, the strategy, the
            worker's role); a task that ran adds its ``task_id`` to its
            terminal entry.
        action: The entries' action.
        ended: The terminal entry of a delegation that returned, from its
            outcome; one that raised or was cancelled is posted failed.
    """

    metadata: dict[str, Any]
    action: str = "task"
    ended: Callable[[WorkerOutcome], WorkerEnd] = _completed_or_failed


async def run_worker(
    kit: RoomKit,
    room_id: str,
    worker_id: str,
    task: str,
    *,
    timeout: float | None,
    status: WorkerStatus | None = None,
    **delegation: Any,
) -> WorkerOutcome:
    """Delegate *task* to *worker_id* from *room_id* and wait for it.

    Bounded by *timeout* (``None``: unbounded): a task past it is cancelled
    and its delegation reads as failed. Posted on the status bus as *status*
    says, pending then one terminal entry whatever ends it; nothing is
    posted without one (a supervisor's own pass). *delegation* goes to
    :meth:`RoomKit.delegate` (``share_channels``,
    ``require_structured_result``, ``result_tool``).
    """
    post = _Posts(kit, worker_id, status)
    post(StatusLevel.PENDING, task)
    # The run learns its task once it ends, before the cut that timed it out
    # or cancelled it reaches the run: its terminal entry then names it.
    delegation = {**delegation, "on_complete": post.task_ended_then(delegation.get("on_complete"))}
    return await _posted(
        post, status, _delegate(kit, room_id, worker_id, task, timeout, delegation)
    )


async def follow_worker(
    kit: RoomKit, task: DelegatedTask, *, timeout: float | None, status: WorkerStatus
) -> WorkerOutcome:
    """Wait for *task*, a delegation the task runner runs in the background
    (``kit.delegate(wait=False)``), within *timeout*, past which the runner
    cancels it and it reads as failed; posted on the status bus as
    :func:`run_worker` posts, its pending entry once it was started."""
    post = _Posts(kit, task.agent_id, status, task_id=task.id)
    post(StatusLevel.PENDING, task.task)
    return await _posted(post, status, _await_started(kit, task, timeout))


async def _posted(
    post: _Posts, status: WorkerStatus | None, delegation: Awaitable[WorkerOutcome]
) -> WorkerOutcome:
    """*delegation*'s outcome, its terminal entry posted however it ends."""
    try:
        outcome = await delegation
    except asyncio.CancelledError:
        post(StatusLevel.FAILED, "cancelled", reason="cancelled")
        raise
    except Exception as exc:
        post(StatusLevel.FAILED, str(exc), reason="error")
        raise
    if status is not None:
        end = status.ended(outcome)
        reason = outcome.end_reason if end.level == StatusLevel.FAILED else None
        post(end.level, end.detail, task=outcome.task, metadata=end.metadata, reason=reason)
    return outcome


async def _await_started(
    kit: RoomKit, task: DelegatedTask, timeout: float | None
) -> WorkerOutcome:
    """A started delegation's outcome, the task runner cancelling it past
    *timeout* (it then ends cancelled, and reads as timed out)."""
    try:
        result = await answer_within(timeout, task.agent_id, task.wait())
    except ToolTimeoutError:
        await kit.task_runner.cancel(task.id)
        return _timed_out(timeout, task)
    return WorkerOutcome(task_output(result), task_completed(result), task)


async def _delegate(
    kit: RoomKit,
    room_id: str,
    worker_id: str,
    task: str,
    timeout: float | None,
    delegation: dict[str, Any],
) -> WorkerOutcome:
    """The delegation itself, inline, within *timeout*: a ``TimeoutError``
    the delegation raised itself is its own failure, not the bound's."""
    # The worker run posts the task's entries itself (RFC §23.3).
    answer = kit.delegate(room_id, worker_id, task, wait=True, post_status=False, **delegation)
    try:
        delegated = await answer_within(timeout, worker_id, answer)
    except ToolTimeoutError:
        return _timed_out(timeout)
    result = delegated.result
    return WorkerOutcome(task_output(result), task_completed(result), delegated)


class _Posts:
    """A worker delegation's entries on the status bus, as its status says,
    each naming the task once the run knows it (RFC §19.7.3)."""

    def __init__(
        self,
        kit: RoomKit,
        worker_id: str,
        status: WorkerStatus | None,
        *,
        task_id: str | None = None,
    ) -> None:
        self._kit = kit
        self._worker_id = worker_id
        self._status = status
        self._task_id = task_id

    def task_ended_then(
        self, on_complete: Callable[[DelegatedTaskResult], Awaitable[None]] | None
    ) -> Callable[[DelegatedTaskResult], Awaitable[None]]:
        """The completion callback that remembers the task the run's
        delegation ended, so the run's terminal entry names it however the
        run itself ends, then calls *on_complete*, the caller's, when given."""

        async def ended(result: DelegatedTaskResult) -> None:
            self._task_id = result.task_id
            if on_complete is not None:
                await on_complete(result)

        return ended

    def __call__(
        self,
        level: StatusLevel,
        detail: str,
        *,
        task: DelegatedTask | None = None,
        metadata: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        """Post an entry of the run; a failed terminal one names why under
        ``task_end_reason`` (*reason*), as a task's own entry does."""
        status = self._status
        if status is None:
            return
        metadata = {**status.metadata, **(metadata or {}), RUN_ENTRY: True}
        if reason is not None:
            metadata["task_end_reason"] = reason
        task_id = task.id if task is not None else self._task_id
        if task_id:
            metadata["task_id"] = task_id
        post_agent_lifecycle(
            self._kit,
            self._worker_id,
            level,
            action=status.action,
            detail=detail,
            metadata=metadata,
        )
