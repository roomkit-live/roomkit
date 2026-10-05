"""A delegated task on the framework's StatusBus, and the tool that reads a room's
tasks (RFC §23.3, §23.4)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import TaskStatus
from roomkit.orchestration.status_bus import StatusEntry, StatusLevel, post_agent_lifecycle
from roomkit.tools.context import current_tool_room_id

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

TASK_ACTION = "task"
"""The action a task's entries carry, a delegation's as an orchestration worker's."""

TASK_STATUS_TOOL = "task_status"


def _task_metadata(room_id: str, task_id: str, child_room_id: str) -> dict[str, Any]:
    return {"room_id": room_id, "task_id": task_id, "child_room_id": child_room_id}


def post_task_pending(kit: RoomKit, task: DelegatedTask) -> None:
    """Post a delegated task as ``pending``, under its worker, once announced."""
    post_agent_lifecycle(
        kit,
        task.agent_id,
        StatusLevel.PENDING,
        action=TASK_ACTION,
        detail=task.task,
        metadata=_task_metadata(task.parent_room_id, task.id, task.child_room_id),
    )


def post_task_ended(kit: RoomKit, result: DelegatedTaskResult) -> None:
    """Post a delegated task's end: ``completed`` with its result, or ``failed``
    saying it failed or was cancelled, never the error's text (RFC §9.3)."""
    completed = result.status == TaskStatus.COMPLETED
    if completed:
        detail = result.output or ""
    else:
        detail = "cancelled" if result.status == TaskStatus.CANCELLED else "failed"
    post_agent_lifecycle(
        kit,
        result.agent_id,
        StatusLevel.COMPLETED if completed else StatusLevel.FAILED,
        action=TASK_ACTION,
        detail=detail,
        metadata={
            **_task_metadata(result.parent_room_id, result.task_id, result.child_room_id),
            "task_status": str(result.status),
            "duration_ms": round(result.duration_ms),
        },
    )


def room_tasks(entries: Iterable[StatusEntry]) -> list[dict[str, Any]]:
    """One line per task, from its entries in posting order: the task from its
    ``pending`` entry, its state and result from its latest.

    An orchestration worker run posts its ``pending`` entry before its task has
    an id: the run's terminal entry, which names the task, joins it.
    """
    tasks: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task_id = str(entry.metadata.get("task_id") or "")
        by_agent = f"agent:{entry.agent_id}"
        if task_id and task_id not in tasks and by_agent in tasks:
            tasks[task_id] = tasks.pop(by_agent)
        line = tasks.setdefault(
            task_id or by_agent, {"task_id": task_id, "agent": entry.agent_id, "task": ""}
        )
        if task_id:
            line["task_id"] = task_id
        if entry.status == StatusLevel.PENDING:
            line.update(task=entry.detail, status="running", since=entry.ts)
            line.pop("result", None)
            line.pop("ended", None)
            continue
        status = str(entry.metadata.get("task_status") or entry.status)
        line.update(status=status, ended=entry.ts)
        if status == str(TaskStatus.COMPLETED):
            line["result"] = entry.detail
    return list(tasks.values())


class TaskStatusTool:
    """``task_status``: the tasks of the room of the call, read from the
    framework's StatusBus (RFC §23.4).

    A tool an agent is given on its own — ``AIChannel(tools=[TaskStatusTool(kit)])``
    — independently of the delegation helpers. It answers with the tasks posted
    for the room of the call (delegations, orchestration workers): those running
    and those ended, the latest state of each, with its worker, the task, and its
    result when it completed; ``task_id`` narrows the answer to one task. It reads
    the bus for that room only: one room's tasks never show in another.

    Args:
        kit: The framework whose ``status_bus`` the tool reads.
        window: How many of the bus's latest entries are read.
        limit: How many tasks, the latest, an answer lists at most.
    """

    def __init__(self, kit: RoomKit, *, window: int = 500, limit: int = 10) -> None:
        self._kit = kit
        self._window = window
        self._limit = limit

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": TASK_STATUS_TOOL,
            "description": (
                "List the background tasks of this conversation: those still running and "
                "those that ended, with the agent working on each, what was asked, and the "
                "result once it completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Only this task (its id, from a delegation's answer).",
                    },
                },
                "required": [],
            },
        }

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        room_id = current_tool_room_id()
        if room_id is None:
            return json.dumps({"error": "task_status answers only within a conversation"})
        entries = await self._kit.status_bus.recent(self._window)
        tasks = room_tasks(
            e for e in entries if e.action == TASK_ACTION and e.metadata.get("room_id") == room_id
        )
        wanted = str(arguments.get("task_id") or "")
        if wanted:
            tasks = [t for t in tasks if t["task_id"] == wanted]
            if not tasks:
                return json.dumps({"tasks": [], "message": f"No task {wanted} here."})
        return json.dumps({"tasks": tasks[-self._limit :]}, ensure_ascii=False)
