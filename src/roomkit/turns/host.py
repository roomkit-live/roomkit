"""How code on a turn's executor reaches a room: :class:`TurnHost` (RFC §10.6.5).

A host's tool, or a hook the executor fires, runs where the turn runs, which
may be another process than the room owner's. It reads the room from the
store, and changes it only through the turn's host, read with
:func:`current_turn_host`: inside a call of a host tool during a turn and
inside a hook the executor fires during it, and after the turn in work it
began, for the verbs :data:`BETWEEN_TURNS` names; ``None`` anywhere else, on
the owner included. Each verb is one ``host_call`` on the turn's link (a
``room_call`` after the turn, RFC §10.6.3), every argument is data, and the
owner runs it as its own code would, under what the turn gave the call (its
room, its chain depth, who asked) and with the reentrancy of its task reading the turn: a
``send_event`` that would wait on the room's own delivery from a task that
delivery needs returns with its completion detached.

A host acts for its turn's room: a verb naming another room, or another
organization than the room's, is refused. The verbs are enumerated here, each
with the record its arguments cross as; the owner validates what it receives
against it, and refuses an argument the record does not name.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from roomkit.core._reentrance import chain_floor
from roomkit.models._time import UtcDatetime
from roomkit.models.enums import EventType, Visibility
from roomkit.models.event import EventContent, RoomEvent
from roomkit.models.room import Room
from roomkit.models.task import DelegatedTaskResult, Observation
from roomkit.turns._error_codec import ROOMKIT_ERRORS, ErrorClasses
from roomkit.turns.wire import ErrorRecord

HostCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""Sends one verb and its arguments to the room owner; its result, as data."""


class _Verb(BaseModel):
    """A verb's arguments: an argument no record names is refused."""

    model_config = ConfigDict(extra="forbid")


class SendEvent(_Verb):
    """``send_event``: the arguments of ``RoomKit.send_event``. The event is
    at least as deep as the response of the turn that sends it (RFC §8.3)."""

    room_id: str
    channel_id: str
    content: EventContent
    event_type: EventType = EventType.MESSAGE
    chain_depth: int = 0
    participant_id: str | None = None
    metadata: dict[str, Any] | None = None
    visibility: str = Visibility.ALL
    provider: str | None = None
    response_visibility: str | None = None
    created_at: UtcDatetime | None = None
    parent_event_id: str | None = None
    idempotency_key: str | None = None
    addressed_to: list[str] | None = None
    organization_id: str | None = None
    standalone: bool = False


class Delegate(_Verb):
    """``delegate``: the arguments of ``RoomKit.delegate`` a background task
    reads, each data; the task's id answered."""

    room_id: str
    agent_id: str
    task: str
    context: dict[str, Any] | None = None
    share_channels: list[str] | None = None
    notify: str | None = None
    post_status: bool = True
    brief: str | None = None
    conversation: int = 0
    hand_back: str = "relay"


class WaitTask(_Verb):
    """``wait_task``: a task a host of the room delegated, and how long to wait."""

    task_id: str
    timeout: float | None = None


class PostStatus(_Verb):
    """``post_status``: an entry on the kit's status bus (RFC §19.8)."""

    agent_id: str
    action: str
    status: str
    detail: str = ""
    metadata: dict[str, Any] | None = None


class UpdateRoomMetadata(_Verb):
    """``update_room_metadata``: keys merged into a room's metadata, which
    the owner writes through ``kit.update_room_metadata``: a key of the
    framework's own (the ``roomkit:`` namespace, a key of the room's private
    state, orchestration's) is dropped there with a warning."""

    room_id: str
    metadata: dict[str, Any]
    organization_id: str | None = None


class AddObservation(_Verb):
    """``add_observation``: an observation stored for its room (RFC §5.11)."""

    observation: Observation


class RoomTasks(_Verb):
    """``room_tasks``: the turn's room's background tasks as its StatusBus
    lists them (RFC §23.4), a reader of the owner's process."""


class RoomVision(_Verb):
    """``room_vision``: what the turn's room's video last showed, as a turn's
    note (RFC §12.8.7), a reader of the owner's process."""


VERBS: dict[str, type[_Verb]] = {
    "send_event": SendEvent,
    "delegate": Delegate,
    "wait_task": WaitTask,
    "post_status": PostStatus,
    "update_room_metadata": UpdateRoomMetadata,
    "add_observation": AddObservation,
    "room_tasks": RoomTasks,
    "room_vision": RoomVision,
}
"""Every verb of a turn's host, by its name, with the record of its arguments."""

READERS = frozenset({"room_tasks", "room_vision"})
"""The verbs that read the owner's process for the turn's room and act for no
tool call: the owner serves them outside a call's context."""

BETWEEN_TURNS = frozenset({"send_event", "post_status", "update_room_metadata", "add_observation"})
"""The verbs a host still sends after its turn ended (RFC §10.6.5), for the
turn's room and at its chain depth; the others need the turn."""


class TurnHost:
    """What a host's tool or hook on a turn's executor may ask of the room
    owner, each verb one call across the turn backend (RFC §10.6.5)."""

    def __init__(
        self,
        call: HostCaller,
        *,
        between_turns: HostCaller | None = None,
        errors: ErrorClasses = ROOMKIT_ERRORS,
    ) -> None:
        self._call = call
        self._between_turns = between_turns
        self._errors = errors
        self._open = True

    @property
    def open(self) -> bool:
        """Whether the turn it serves still runs."""
        return self._open

    @property
    def reachable(self) -> bool:
        """Whether a verb can still reach the owner: during the turn, or
        after it for the verbs :data:`BETWEEN_TURNS` names."""
        return self._open or self._between_turns is not None

    def _close(self) -> None:
        """The turn ended: the host answers no more verbs. The executor's."""
        self._open = False

    async def send_event(
        self, room_id: str, channel_id: str, content: EventContent, **options: Any
    ) -> RoomEvent:
        """Send an event into a room, as ``RoomKit.send_event`` does: the
        same options, each a value the backend carries. The event is at
        least as deep as this context's floor, as the kit's own is: the
        body of an ASYNC hook on the executor sends at its trigger's depth
        after the turn too, which the owner no longer floors (RFC §8.3)."""
        options["chain_depth"] = max(options.get("chain_depth", 0), chain_floor())
        args = SendEvent(room_id=room_id, channel_id=channel_id, content=content, **options)
        return RoomEvent.model_validate(await self._verb("send_event", args))

    async def delegate(self, room_id: str, agent_id: str, task: str, **options: Any) -> str:
        """Delegate *task* to *agent_id* from *room_id*, in the background, as
        ``RoomKit.delegate`` does: the task's id, which :meth:`wait_task` waits."""
        args = Delegate(room_id=room_id, agent_id=agent_id, task=task, **options)
        return str(await self._verb("delegate", args))

    async def wait_task(
        self, task_id: str, timeout: float | None = None
    ) -> DelegatedTaskResult | None:
        """The result of a task a host of this room delegated, once it ends,
        its failure kept (``exception``, RFC §23.3); ``None`` when *timeout*
        passes first, or for a task the room's hosts did not delegate."""
        data = await self._verb("wait_task", WaitTask(task_id=task_id, timeout=timeout))
        if data is None:
            return None
        result = DelegatedTaskResult.model_validate(data["result"])
        if data.get("exception") is not None:
            failure = self._errors.rebuild(
                ErrorRecord.model_validate(data["exception"]), where="the room owner"
            )
            if isinstance(failure, Exception):
                result.exception = failure
        return result

    async def post_status(
        self,
        agent_id: str,
        action: str,
        status: str,
        *,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post an entry on the kit's status bus."""
        args = PostStatus(
            agent_id=agent_id, action=action, status=status, detail=detail, metadata=metadata
        )
        await self._verb("post_status", args)

    async def update_room_metadata(
        self, room_id: str, metadata: dict[str, Any], *, organization_id: str | None = None
    ) -> Room:
        """Merge *metadata* into the room's: the room as stored after."""
        args = UpdateRoomMetadata(
            room_id=room_id, metadata=metadata, organization_id=organization_id
        )
        return Room.model_validate(await self._verb("update_room_metadata", args))

    async def add_observation(self, observation: Observation) -> None:
        """Store *observation* for its room."""
        await self._verb("add_observation", AddObservation(observation=observation))

    async def room_tasks(self) -> list[dict[str, Any]]:
        """The turn's room's background tasks, as ``task_status`` reads them
        (RFC §23.4): state of the owner's process."""
        return list(await self._verb("room_tasks", RoomTasks()))

    async def room_vision(self) -> str | None:
        """What the turn's room's video last showed, as a turn's note (RFC
        §12.8.7); ``None`` when no camera of the room has a result."""
        note = await self._verb("room_vision", RoomVision())
        return None if note is None else str(note)

    async def _verb(self, verb: str, args: _Verb) -> Any:
        if self._open:
            return await self._call(verb, args.model_dump(mode="json"))
        if self._between_turns is None:
            raise RuntimeError(f"The turn this host served has ended: {verb} is not sent")
        if verb not in BETWEEN_TURNS:
            raise RuntimeError(
                f"{verb} needs a turn: after its turn ended, a host sends "
                f"{', '.join(sorted(BETWEEN_TURNS))} only (RFC §10.6.5)"
            )
        return await self._between_turns(verb, args.model_dump(mode="json"))


_TURN_HOST: contextvars.ContextVar[TurnHost | None] = contextvars.ContextVar(
    "roomkit_turn_host", default=None
)


def current_turn_host() -> TurnHost | None:
    """The host of the turn the current code runs in (RFC §10.6.5): inside a
    call of a host tool during a turn and inside a hook its executor fires
    during it; after the turn ended, in work it began (a background thinker),
    for the verbs :data:`BETWEEN_TURNS` names, where the backend reaches the
    owner outside a turn. ``None`` anywhere else, in a tool the room owner
    serves included."""
    host = _TURN_HOST.get()
    return host if host is not None and host.reachable else None


def set_turn_host(host: TurnHost | None) -> None:
    """Set the host of the turn the current context runs: an executor sets
    it for every turn it runs."""
    _TURN_HOST.set(host)
