"""The kit's calls of task delegation, the status bus, the turn's host and a
room's observations, made on the Python core (roomkit-specs
``conformance/README.md``, "Calls").

A call ``status.<operation>`` is the operation of the kit's status bus
(RFC §19.8); ``status.post`` is the bus's ``post_async``, which returns once
the bus's subscribers ran. ``delegate`` returns the task, which ``task.wait`` names by its
label to wait for its result (RFC §23.3), and ``cancel_task`` to cancel it. A call
``host.<verb>`` is a verb of the turn's host (RFC §10.6.5), made by an action of a
scripted intelligence channel's turn: ``current_turn_host()`` there, which a call
made anywhere else lacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from roomkit import RoomKit, current_turn_host
from roomkit.models.event import EventContent
from roomkit.turns._error_codec import error_record
from tests.conformance._calls import Call, UnreadableError, read
from tests.conformance._script import complete_observation

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

_CONTENT = TypeAdapter(EventContent)

_STATUS = {"agent_id", "action", "status", "detail", "metadata"}

# Each call's arguments, beside `call` and `organization_id`.
TASK_ARGUMENTS: dict[str, set[str]] = {
    "status.post": _STATUS,
    "status.recent": {"n", "agent_id", "status"},
    "status.recent_text": {"n"},
    "update_room_metadata": {"room_id", "metadata"},
    "submit_feedback": {
        "room_id",
        "rating",
        "event_id",
        "channel_id",
        "comment",
        "dimension",
        "metadata",
    },
    "list_observations": {"room_id"},
    "list_tasks": {"room_id", "status"},
    "delegate": {
        "room_id",
        "agent_id",
        "task",
        "wait",
        "context",
        "share_channels",
        "notify",
        "post_status",
        "brief",
        "conversation",
        "hand_back",
    },
    "task.wait": {"task_id"},
    "cancel_task": {"task_id"},
    "start_task": {"room_id", "agent_id", "task", "hand_back"},
    "setup_background": {"agent_id", "worker", "hand_back"},
    "setup_task_tools": {"agent_id"},
    "host.send_event": {
        "room_id",
        "channel_id",
        "content",
        "event_type",
        "participant_id",
        "visibility",
        "response_visibility",
        "addressed_to",
        "parent_event_id",
        "idempotency_key",
        "metadata",
        "chain_depth",
        "standalone",
    },
    "host.delegate": {
        "room_id",
        "agent_id",
        "task",
        "context",
        "share_channels",
        "notify",
        "post_status",
        "brief",
        "conversation",
        "hand_back",
    },
    "host.wait_task": {"task_id"},
    "host.room_tasks": set(),
    "host.post_status": _STATUS,
    "host.update_room_metadata": {"room_id", "metadata"},
    "host.add_observation": {"observation"},
}


def _named(spec: Mapping[str, Any], names: set[str]) -> dict[str, Any]:
    return {name: spec[name] for name in names if name in spec}


async def _status_post(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    options = _named(spec, {"detail", "metadata"})
    return await kit.status_bus.post_async(
        spec["agent_id"], spec["action"], spec["status"], **options
    )


async def _status_recent(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.status_bus.recent(**_named(spec, {"n", "agent_id", "status"}))


async def _status_recent_text(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.status_bus.recent_text(**_named(spec, {"n"}))


async def _update_room_metadata(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.update_room_metadata(
        spec["room_id"], spec["metadata"], **_named(spec, {"organization_id"})
    )


async def _submit_feedback(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = TASK_ARGUMENTS["submit_feedback"] - {"room_id", "rating"} | {"organization_id"}
    return await kit.submit_feedback(spec["room_id"], spec["rating"], **_named(spec, names))


async def _list_observations(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.list_observations(spec["room_id"], **_named(spec, {"organization_id"}))


async def _list_tasks(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = {"status", "organization_id"}
    return await kit.list_tasks(spec["room_id"], **_named(spec, names))


async def _delegate(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = TASK_ARGUMENTS["delegate"] - {"room_id", "agent_id", "task"} | {"organization_id"}
    task = await kit.delegate(
        spec["room_id"], spec["agent_id"], spec["task"], **_named(spec, names)
    )
    harness.tasks[task.id] = task
    return task


async def _task_wait(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    task = harness.tasks.get(spec["task_id"])
    if task is None:
        raise UnreadableError(f"no task {spec['task_id']!r} was delegated")
    return await task.wait()


async def _cancel_task(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.cancel_task(spec["task_id"], **_named(spec, {"organization_id"}))


async def _start_task(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    options = _named(spec, {"hand_back", "organization_id"})
    task = await kit.start_task(spec["room_id"], spec["agent_id"], spec["task"], **options)
    harness.tasks[task.id] = task
    return task


async def _setup_background(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kit.setup_background(spec["agent_id"], **_named(spec, {"worker", "hand_back"}))


async def _setup_task_tools(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    kit.setup_task_tools(spec["agent_id"])


def _host() -> Any:
    host = current_turn_host()
    if host is None:
        raise UnreadableError("a host verb is made outside a turn: no turn's host")
    return host


async def _host_send_event(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    content = read(lambda: _CONTENT.validate_python(spec["content"]))
    names = TASK_ARGUMENTS["host.send_event"] - {"room_id", "channel_id", "content"}
    options = _named(spec, names | {"organization_id"})
    return await _host().send_event(spec["room_id"], spec["channel_id"], content, **options)


async def _host_delegate(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = TASK_ARGUMENTS["host.delegate"] - {"room_id", "agent_id", "task"}
    options = _named(spec, names)
    return await _host().delegate(spec["room_id"], spec["agent_id"], spec["task"], **options)


async def _host_wait_task(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    result = await _host().wait_task(spec["task_id"])
    if result is None:
        return None
    failure = result.exception
    return {
        "result": result,
        "exception": None if failure is None else error_record(failure),
    }


async def _host_room_tasks(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await _host().room_tasks()


async def _host_post_status(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    options = _named(spec, {"detail", "metadata"})
    return await _host().post_status(spec["agent_id"], spec["action"], spec["status"], **options)


async def _host_update_room_metadata(
    kit: RoomKit, harness: Harness, spec: Mapping[str, Any]
) -> Any:
    options = _named(spec, {"organization_id"})
    return await _host().update_room_metadata(spec["room_id"], spec["metadata"], **options)


async def _host_add_observation(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    template = spec["observation"]
    observation = read(
        lambda: complete_observation(template, template["room_id"], template["channel_id"])
    )
    return await _host().add_observation(observation)


TASK_CALLS: dict[str, Call] = {
    "status.post": _status_post,
    "status.recent": _status_recent,
    "status.recent_text": _status_recent_text,
    "update_room_metadata": _update_room_metadata,
    "submit_feedback": _submit_feedback,
    "list_observations": _list_observations,
    "list_tasks": _list_tasks,
    "delegate": _delegate,
    "task.wait": _task_wait,
    "cancel_task": _cancel_task,
    "start_task": _start_task,
    "setup_background": _setup_background,
    "setup_task_tools": _setup_task_tools,
    "host.send_event": _host_send_event,
    "host.delegate": _host_delegate,
    "host.wait_task": _host_wait_task,
    "host.room_tasks": _host_room_tasks,
    "host.post_status": _host_post_status,
    "host.update_room_metadata": _host_update_room_metadata,
    "host.add_observation": _host_add_observation,
}
