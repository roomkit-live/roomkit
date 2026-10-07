"""In-memory task runner using asyncio.create_task()."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.core._failure_log import log_failure
from roomkit.core.task_utils import (
    cancel_and_wait,
    held_runs,
    log_task_exception,
    shielded,
    start_named,
)
from roomkit.models.enums import TaskStatus
from roomkit.models.response_metadata import TurnEntries
from roomkit.tasks._child_status import record_task_end
from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.models import (
    DelegatedTask,
    DelegatedTaskResult,
    cancelled_task_fields,
    finished_task_fields,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.tasks")


class InMemoryTaskRunner(TaskRunner):
    """Default task runner — executes tasks as ``asyncio.Task`` instances."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._handles: dict[str, DelegatedTask] = {}
        # How each task ends cancelled. Whoever takes a task's entry ends the
        # task, so it ends once: the task itself once its work ran, or
        # ``cancel`` (RFC §23.3).
        self._cancelled_ends: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}

    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        fields = cancelled_task_fields(context)
        end = partial(self._finish, kit, task, fields, time.monotonic(), on_complete)
        bg = start_named(
            self._execute(kit, task, context=context, on_complete=on_complete),
            name=f"delegate:{task.id}",
        )
        bg.add_done_callback(log_task_exception)
        self._tasks[task.id] = bg
        self._handles[task.id] = task
        self._cancelled_ends[task.id] = end

    async def cancel(self, task_id: str) -> bool:
        bg = self._tasks.get(task_id)
        if task_id not in self._handles or bg is None:
            return False
        end = self._cancelled_ends.pop(task_id, None)
        if end is None:
            # Its work ran to its end, and it is ending as it stands.
            await asyncio.wait({bg})
            return False
        try:
            await cancel_and_wait(bg)
        finally:
            # It ends cancelled, to its end even if this call is cancelled.
            await shielded(end())
        return True

    async def close(self) -> None:
        # A task's end may delegate again: that task is cancelled in turn.
        # The ones this close runs under (a worker's tool closing the kit)
        # are spared: cancelling one would wait for itself.
        spared = set(held_runs())
        while running := [tid for tid, bg in self._tasks.items() if bg not in spared]:
            for task_id in running:
                await self.cancel(task_id)

    async def _execute(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        start = time.monotonic()
        task.status = TaskStatus.IN_PROGRESS
        fields = await self._run(kit, task, context)
        if self._cancelled_ends.pop(task.id, None) is None:
            # ``cancel`` took its end: it ends the task, cancelled.
            return
        # Its work ran: it ends as it stands, whatever cancels it now.
        await shielded(self._finish(kit, task, fields, start, on_complete))

    async def _run(
        self, kit: RoomKit, task: DelegatedTask, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Run the worker in the task's child room: the task's outcome."""
        agent_response: str | None = None
        failure: Exception | None = None
        turns: TurnEntries = {}
        try:
            # Update child room status
            room = await kit.get_room(task.child_room_id)
            if room is None:
                logger.warning(
                    "Task %s: child room %s not found",
                    task.id,
                    task.child_room_id,
                )
                # No worker ran.
                fields = finished_task_fields(None, None, context)
                return {**fields, "error": f"Child room {task.child_room_id} not found"}
            await kit.store.update_room(
                room.model_copy(
                    update={
                        "metadata": {
                            **room.metadata,
                            "task_status": TaskStatus.IN_PROGRESS,
                        },
                    }
                )
            )
            # Lazy import to avoid circular dependency
            from roomkit.core.mixins.delegation import run_agent_in_child_room

            agent_response = await run_agent_in_child_room(
                kit, task.child_room_id, task.task, turns=turns
            )
        except Exception as exc:
            log_failure(logger, exc, f"Task {task.id}")
            failure = exc
        return finished_task_fields(agent_response, failure, context, turns)

    async def _finish(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        fields: dict[str, Any],
        start: float,
        on_complete: OnCompleteCallback | None,
    ) -> None:
        """End *task* with its outcome *fields*: its child room's status, its
        completion callback, then its waiters."""
        result = DelegatedTaskResult(
            task_id=task.id,
            child_room_id=task.child_room_id,
            parent_room_id=task.parent_room_id,
            agent_id=task.agent_id,
            duration_ms=(time.monotonic() - start) * 1000,
            **fields,
        )
        await record_task_end(kit, result)

        # Run on_complete BEFORE setting result so hooks fire before waiters unblock
        if on_complete:
            try:
                await on_complete(result)
            except Exception:
                logger.exception("on_complete callback failed for task %s", task.id)

        # ALWAYS set result — callers of wait() depend on this
        task._set_result(result)

        self._tasks.pop(task.id, None)
        self._handles.pop(task.id, None)
