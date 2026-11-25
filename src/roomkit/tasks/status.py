"""A delegated task on the framework's StatusBus, its progress, and the tools that
read a room's tasks and cancel one of them (RFC §23.3, §23.4)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from roomkit.channels._discussion_turn import discussion_handles
from roomkit.core.exceptions import RoomNotFoundError, UnservedToolCallError
from roomkit.models.enums import TaskStatus
from roomkit.models.task import END_REASON_KEYS, end_reason_fields
from roomkit.orchestration.status_bus import (
    StatusBus,
    StatusEntry,
    StatusLevel,
    post_agent_lifecycle,
)
from roomkit.tasks._room_tasks import TaskReach, why_not_cancelled
from roomkit.tasks.models import end_reason_metadata
from roomkit.tools.context import current_tool_call, current_tool_room, current_tool_room_id

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.room import Room
    from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

TASK_ACTION = "task"
"""The action a task's entries carry, a delegation's as an orchestration worker's."""

RUN_ENTRY = "task_run"
"""The metadata key a strategy's worker run sets on its own entries, the
framework's alone (a turn's host refuses it): its terminal entry, whatever
action it posts under, ends the line of the task it names."""

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
    saying it failed or was cancelled, never the error's text (RFC §9.3), with
    why under ``task_end_reason``."""
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
            **end_reason_metadata(result),
            "duration_ms": round(result.duration_ms),
        },
    )


async def post_task_progress(kit: RoomKit, detail: str) -> bool:
    """From a tool a worker calls in its task's child room: say how far the task
    got ("12/30 s", "3 sources of 5"), for the room that delegated it (RFC §23.3).

    Posts an ``info`` entry under the worker, action ``task``, with the task's
    metadata; it does not end the task. Returns whether it posted: a call
    outside a task's child room posts nothing.
    """
    child_room_id = current_tool_room_id()
    if child_room_id is None:
        return False
    try:
        room = await kit.get_room(child_room_id)
    except RoomNotFoundError:
        return False
    metadata = room.metadata
    parent_room_id, task_id = metadata.get("parent_room_id"), metadata.get("task_id")
    if not parent_room_id or not task_id:
        return False
    post_agent_lifecycle(
        kit,
        str(metadata.get("task_agent_id") or "worker"),
        StatusLevel.INFO,
        action=TASK_ACTION,
        detail=detail,
        metadata=_task_metadata(str(parent_room_id), str(task_id), child_room_id),
    )
    return True


def room_tasks(entries: Iterable[StatusEntry]) -> list[dict[str, Any]]:
    """One line per task, from its entries in posting order: the task from its
    ``pending`` entry, its state from its latest terminal one, with its result
    when it completed and else why it did not (``end_reason``, a code of the
    closed list, and ``end_reason_text``, the runtime's words for it), its
    progress from its latest ``info`` one, which never ends it. Returned rows
    and their nested results can be edited independently of the source entries.

    An orchestration worker run posts its ``pending`` entry before its task has
    an id: the run's terminal entry, which names the task, joins it. A run
    posting under an action of its own (a Loop's ``iteration`` or ``review``)
    opens no line, but its terminal entry ends the line its worker's progress
    opened (RFC §19.7.3).
    """
    tasks: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task_id = str(entry.metadata.get("task_id") or "")
        if entry.action != TASK_ACTION:
            _end_run_line(tasks.get(task_id), entry)
            continue
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
            for key in ("result", "ended", *END_REASON_KEYS):
                line.pop(key, None)
            continue
        if entry.status == StatusLevel.INFO:
            # A task whose pending entry left the bus's window: running still.
            line.setdefault("status", "running")
            line.update(progress=entry.detail, progress_at=entry.ts)
            continue
        _end_line(line, str(entry.metadata.get("task_status") or entry.status), entry)
    return list(tasks.values())


def _end_line(line: dict[str, Any], status: str, entry: StatusEntry) -> None:
    """End *line* with *status* at *entry*, a terminal one: with its result
    when it completed, the whole outcome when the summary was cut (RFC §19.8),
    else with why it did not, as *entry* names it and never as an earlier end
    of the line did. A reason the closed list does not name gives none: an
    entry posted from elsewhere gives a line no words of its own (RFC §23.4)."""
    line.update(status=status, ended=entry.ts)
    for key in END_REASON_KEYS:
        line.pop(key, None)
    if status == str(TaskStatus.COMPLETED):
        line["result"] = deepcopy(entry.metadata.get("result", entry.detail))
    else:
        line.update(end_reason_fields(entry.metadata.get("task_end_reason")))


def _end_run_line(line: dict[str, Any] | None, entry: StatusEntry) -> None:
    """End *line*, a task's, with *entry*, the terminal entry its strategy's
    run posted under its own action: failed when the run says so, else
    completed (a review asking for a revision ended its task too)."""
    if line is None:
        return
    status = str(TaskStatus.FAILED if entry.status == StatusLevel.FAILED else TaskStatus.COMPLETED)
    _end_line(line, status, entry)


class TaskStatusTool:
    """``task_status``: the tasks of the room of the call, read from the
    framework's StatusBus (RFC §23.4).

    A tool an agent is given on its own — ``AIChannel(tools=[TaskStatusTool(kit)])``
    — independently of the delegation helpers. It answers with the tasks posted
    for the room of the call (delegations, orchestration workers): those running
    and those ended, the latest state of each, with its worker, the task, and its
    result when it completed or why it did not; ``task_id`` narrows the answer
    to one task. It reads the bus for that room only: one room's tasks never
    show in another.

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
        lines = await room_task_lines(self._kit.status_bus, room_id, self._window)
        tasks = _named_by_handle(lines, current_tool_room())
        wanted = str(arguments.get("task_id") or "")
        if wanted:
            tasks = [t for t in tasks if t["task_id"] == wanted]
            if not tasks:
                return json.dumps({"tasks": [], "message": f"No task {wanted} here."})
        return json.dumps({"tasks": tasks[-self._limit :]}, ensure_ascii=False)


def _named_by_handle(tasks: list[dict[str, Any]], room: Room | None) -> list[dict[str, Any]]:
    """*tasks* with each one's agent named as *room*, the room of the call,
    addresses it: a discussion's agent by its handle (RFC §19.7.5 rule 4),
    any other by its channel id."""
    handles = discussion_handles(room.metadata if room is not None else None)
    return [{**t, "agent": handles.get(t["agent"], t["agent"])} for t in tasks]


async def room_task_lines(bus: StatusBus, room_id: str, window: int) -> list[dict[str, Any]]:
    """The tasks *bus* lists for *room_id*, read from its latest *window*
    entries: a task's own, and a strategy run's terminal entry naming its
    task under the run's own action."""
    entries = await bus.recent(window)
    return room_tasks(e for e in entries if e.metadata.get("room_id") == room_id and _of_a_task(e))


def _of_a_task(entry: StatusEntry) -> bool:
    """Whether a task's line reads *entry*: one posted under the task action,
    or the end of a strategy's run that names its task (:data:`RUN_ENTRY`),
    never an entry a turn posted."""
    if entry.action == TASK_ACTION:
        return True
    metadata = entry.metadata
    return bool(metadata.get(RUN_ENTRY) and metadata.get("task_id")) and (
        entry.status != StatusLevel.PENDING
    )


class CancelTaskTool:
    """``cancel_task``: cancel one task of the room of the call (RFC §23.4).

    A tool an agent is given on its own, as :class:`TaskStatusTool` —
    ``AIChannel(tools=[TaskStatusTool(kit), CancelTaskTool(kit)])`` — so that a
    request the person changed or dropped stops its work. It reaches only the
    tasks the StatusBus lists for the room of the call: never another room's,
    nor a strategy's worker run, which that strategy follows itself. A running
    task ends ``cancelled``; the agent that cancelled it is not handed the
    cancellation back, the tool's answer having told it. A task of the room
    it may not cancel is answered still running, and why: whether a task the
    bus lists running is the room's, and whether it ended meanwhile, is what
    the framework knows of the tasks it runs for each room, never what the bus
    lists alone. A task the bus lists ended is answered as it lists it.

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
        if line.get("status") == "running":
            return await self._answer_running(room_id, task_id)
        # Ended: nothing to act on, so as the bus lists it, as task_status does.
        return _answer(task_id, str(line.get("status")), "The task had already ended.")

    async def _answer_running(self, room_id: str, task_id: str) -> str:
        """The answer for a task the bus lists running for *room_id*, from
        what the framework knows of it (RFC §23.4): cancelled when the room
        may cancel it, ended when it ended meanwhile or before, else still
        running and why, or no task of the room."""
        record = self._kit._room_tasks
        reach = record.reach(task_id, room_id)
        # The kit decides whether the call cancels, or joins a cancel whose
        # end is still being completed.
        if await self._cancel(task_id, room_id):
            return _answer(
                task_id, str(TaskStatus.CANCELLED), "Cancelled: its result will not come back."
            )
        if reach == "tool" or record.ended_with(task_id, room_id) is not None:
            return await self._answer_ended(room_id, task_id)
        return _out_of_reach(task_id, reach)

    async def _answer_ended(self, room_id: str, task_id: str) -> str:
        """The answer for a task of *room_id* that ended: with the status the
        framework recorded, else the bus's once it shows the end (a runner
        that let the task go before its end was recorded)."""
        status = self._kit._room_tasks.ended_with(task_id, room_id)
        if status is None:
            line = await self._line(room_id, task_id)
            ended = line is not None and line.get("status") != "running"
            status = str(line["status"]) if line is not None and ended else "ended"
        return _answer(task_id, status, "The task had already ended.")

    async def _line(self, room_id: str, task_id: str) -> dict[str, Any] | None:
        lines = await room_task_lines(self._kit.status_bus, room_id, self._window)
        return next((t for t in lines if t["task_id"] == task_id), None)

    async def _cancel(self, task_id: str, room_id: str) -> bool:
        """Cancel a task of *room_id* for the channel that made the call,
        which is then not told twice."""
        call = current_tool_call()
        channel_id = call.channel_id if call is not None else None
        return await self._kit._cancel_task_for(task_id, channel_id, room_id=room_id)


def _out_of_reach(task_id: str, reach: TaskReach | None) -> str:
    """The answer for a task the room lists running that its call may not
    cancel (RFC §23.4): a task of the room is still running, and why; any
    other (another room's, one never run by this framework), whoever posted
    its line, is no task here."""
    why = why_not_cancelled(reach)
    if why is None:
        return _answer(task_id, "unknown", f"No task {task_id} here.")
    return _answer(task_id, "running", f"Still running, and not cancellable here: {why}.")


def _answer(task_id: str, status: str, message: str) -> str:
    return json.dumps({"task_id": task_id, "status": status, "message": message})
