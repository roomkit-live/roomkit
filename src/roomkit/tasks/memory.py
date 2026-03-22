"""In-memory task runner using asyncio.create_task()."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import TaskStatus
from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.tasks")


class InMemoryTaskRunner(TaskRunner):
    """Default task runner — executes tasks as ``asyncio.Task`` instances."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._handles: dict[str, DelegatedTask] = {}

    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        bg = asyncio.create_task(
            self._execute(kit, task, context=context, on_complete=on_complete),
            name=f"delegate:{task.id}",
        )
        bg.add_done_callback(self._task_done)
        self._tasks[task.id] = bg
        self._handles[task.id] = task

    @staticmethod
    def _task_done(task: asyncio.Task[None]) -> None:
        """Log exceptions from background delegate tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Delegate task %s failed: %s", task.get_name(), exc)

    async def cancel(self, task_id: str) -> bool:
        handle = self._handles.get(task_id)
        bg = self._tasks.get(task_id)
        if handle is None or bg is None:
            return False
        handle.cancel()
        bg.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg
        self._tasks.pop(task_id, None)
        self._handles.pop(task_id, None)
        return True

    async def close(self) -> None:
        for task_id in list(self._tasks):
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
        agent_response: str | None = None
        error: str | None = None

        try:
            # Update child room status
            room = await kit.get_room(task.child_room_id)
            if room is None:
                logger.warning(
                    "Task %s: child room %s not found",
                    task.id,
                    task.child_room_id,
                )
                error = f"Child room {task.child_room_id} not found"
            else:
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

                agent_response = await run_agent_in_child_room(kit, task.child_room_id, task.task)
        except Exception as exc:
            logger.exception("Task %s failed: %s", task.id, exc)
            error = str(exc)

        elapsed = (time.monotonic() - start) * 1000
        status = TaskStatus.COMPLETED if agent_response else TaskStatus.FAILED

        result = DelegatedTaskResult(
            task_id=task.id,
            child_room_id=task.child_room_id,
            parent_room_id=task.parent_room_id,
            agent_id=task.agent_id,
            status=status,
            output=agent_response,
            error=error,
            duration_ms=elapsed,
            metadata=context or {},
        )

        # Update child room metadata
        try:
            room = await kit.get_room(task.child_room_id)
            if room is not None:
                await kit.store.update_room(
                    room.model_copy(
                        update={
                            "metadata": {
                                **room.metadata,
                                "task_status": status,
                                "task_result": agent_response,
                            },
                        }
                    )
                )
        except Exception:
            logger.exception("Task %s: failed to update child room metadata", task.id)

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
