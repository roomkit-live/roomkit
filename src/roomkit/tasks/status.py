"""A delegated task on the framework's StatusBus, and the tools that read a room's
tasks and cancel one of them (RFC §23.3, §23.4)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.enums import TaskStatus
from roomkit.orchestration.status_bus import StatusEntry, StatusLevel, post_agent_lifecycle
from roomkit.tools.context import current_tool_call, current_tool_room_id

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

TASK_ACTION = "task"
"""The action a task's entries carry, a delegation's as an orchestration worker's."""

TASK_STATUS_TOOL = "task_status"
CANCEL_TASK_TOOL = "cancel_task"


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
            # The whole outcome when the summary was cut (RFC §19.8).
            line["result"] = entry.metadata.get("result", entry.detail)
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
        # A channel chains its tools' handlers, the first to answer wins: a call
        # for another tool is declined, so the next handler serves it.
        if name != TASK_STATUS_TOOL:
            raise UnservedToolCallError(f"tool {name!r} is not served here")
        room_id = current_tool_room_id()
        if room_id is None:
            return json.dumps({"error": "task_status answers only within a conversation"})
        tasks = await _room_task_lines(self._kit, room_id, self._window)
        wanted = str(arguments.get("task_id") or "")
        if wanted:
            tasks = [t for t in tasks if t["task_id"] == wanted]
            if not tasks:
                return json.dumps({"tasks": [], "message": f"No task {wanted} here."})
        return json.dumps({"tasks": tasks[-self._limit :]}, ensure_ascii=False)


async def _room_task_lines(kit: RoomKit, room_id: str, window: int) -> list[dict[str, Any]]:
    """The tasks the bus lists for *room_id*, read from its latest *window* entries."""
    entries = await kit.status_bus.recent(window)
    return room_tasks(
        e for e in entries if e.action == TASK_ACTION and e.metadata.get("room_id") == room_id
    )


class CancelTaskTool:
    """``cancel_task``: cancel one task of the room of the call (RFC §23.4).

    A tool an agent is given on its own, as :class:`TaskStatusTool` —
    ``AIChannel(tools=[TaskStatusTool(kit), CancelTaskTool(kit)])`` — so that a
    request the person changed or dropped stops its work. It reaches only the
    tasks the StatusBus lists for the room of the call: never another room's,
    nor a strategy's worker run, which that strategy follows itself. A running
    task ends ``cancelled``; the agent that cancelled it is not handed the
    cancellation back, the tool's answer having told it.

    Args:
        kit: The framework whose tasks the tool cancels.
        window: How many of the bus's latest entries are read to find the task.
    """

    def __init__(self, kit: RoomKit, *, window: int = 500) -> None:
        self._kit = kit
        self._window = window

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": CANCEL_TASK_TOOL,
            "description": (
                "Cancel a background task of this conversation that is still running, "
                "when what it was asked to do is no longer wanted: the request changed "
                "or was dropped. Its result will not come back."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task to cancel (its id, from a delegation's answer).",
                    },
                },
                "required": ["task_id"],
            },
        }

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        # A channel chains its tools' handlers, the first to answer wins: a call
        # for another tool is declined, so the next handler serves it.
        if name != CANCEL_TASK_TOOL:
            raise UnservedToolCallError(f"tool {name!r} is not served here")
        room_id = current_tool_room_id()
        if room_id is None:
            return json.dumps({"error": "cancel_task answers only within a conversation"})
        task_id = str(arguments.get("task_id") or "")
        if not task_id:
            return json.dumps({"error": "task_id is required"})
        line = await self._line(room_id, task_id)
        if line is None:
            return _answer(task_id, "unknown", f"No task {task_id} here.")
        if line.get("status") == "running" and await self._cancel(task_id):
            return _answer(
                task_id, str(TaskStatus.CANCELLED), "Cancelled: its result will not come back."
            )
        # It ended meanwhile, or before the call: as it stands.
        ended = await self._line(room_id, task_id) or line
        return _answer(task_id, str(ended.get("status")), "The task had already ended.")

    async def _line(self, room_id: str, task_id: str) -> dict[str, Any] | None:
        lines = await _room_task_lines(self._kit, room_id, self._window)
        return next((t for t in lines if t["task_id"] == task_id), None)

    async def _cancel(self, task_id: str) -> bool:
        """Cancel for the channel that made the call, which is then not told twice."""
        call = current_tool_call()
        if call is not None and call.channel_id:
            return await self._kit._cancel_task_for(task_id, call.channel_id)
        return await self._kit.cancel_task(task_id)


def _answer(task_id: str, status: str, message: str) -> str:
    return json.dumps({"task_id": task_id, "status": status, "message": message})
