"""The verbs of a turn's ``TurnHost``, run by the kit that owns the room
(RFC §10.6.5).

Each verb arrives as data, validated against the record of its arguments
(``roomkit.turns.host.VERBS``) before it reaches the kit, and answers with
data. It runs in the kit's own code, under what the turn gave the call (its
chain depth, who asked) and in the context the owner's task reading the turn
lent it, so its reentrancy is that task's.

A host acts for its turn's room, in the room's organization: a verb naming
another room, or another organization, is refused, and an event a turn sends
is at least as deep as the turn's response (RFC §8.3). After its turn ended,
work the turn began sends the verbs ``BETWEEN_TURNS`` names, for the same room
and at the depth its ``room_call`` carries (RFC §10.6.5).

The tasks the room's hosts delegate are kept for ``wait_task`` as long as
they run, and once ended until :data:`ENDED_TASKS_KEPT` more ended after them.

A verb the host declared (:meth:`KitHost.declare`) runs the host's code with
its arguments as data, under what the turn gave the call; its result goes
back as data. It is served during its turn only.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from roomkit.channels._video_hooks import room_vision_note
from roomkit.models.enums import TaskStatus
from roomkit.tasks.status import RUN_ENTRY, TASK_ACTION
from roomkit.turns._error_codec import error_record
from roomkit.turns.base import OwnerHost, response_depth
from roomkit.turns.host import (
    BETWEEN_TURNS,
    VERBS,
    AddObservation,
    DeclaredVerb,
    Delegate,
    PostStatus,
    RoomTasks,
    RoomVision,
    SendEvent,
    UpdateRoomMetadata,
    WaitTask,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.tasks.models import DelegatedTask
    from roomkit.turns.wire import RoomCall, TurnRequest

ENDED_TASKS_KEPT = 1024
"""How many ended tasks the hosts keep for a ``wait_task`` that has not come."""


@dataclass(frozen=True, slots=True)
class _Scope:
    """What a verb acts for: its turn's room and organization, the turn's
    context while it runs, and the depth an event it sends is at least while
    the turn runs (``None`` once it is over: what the verb names)."""

    room_id: str
    organization_id: str | None
    context: RoomContext | None
    chain_depth: int | None

    @classmethod
    def of(cls, request: TurnRequest) -> _Scope:
        room = request.context.room
        return cls(request.room_id, room.organization_id, request.context, response_depth(request))


Verb = Callable[[_Scope, Any], Awaitable[Any]]


class KitHost(OwnerHost):
    """The kit's side of its turns' hosts."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit
        self._tasks: OrderedDict[str, DelegatedTask] = OrderedDict()
        self._verbs: dict[str, Verb] = {
            "send_event": self._send_event,
            "delegate": self._delegate,
            "wait_task": self._wait_task,
            "post_status": self._post_status,
            "update_room_metadata": self._update_room_metadata,
            "add_observation": self._add_observation,
            "room_tasks": self._room_tasks,
            "room_vision": self._room_vision,
        }
        self._declared: dict[str, DeclaredVerb] = {}

    def declare(self, name: str, verb: DeclaredVerb) -> None:
        """Serve *verb* under *name* for the hosts of this kit's turns.

        Raises:
            ValueError: *name* is empty, one of RoomKit's verbs, or declared.
        """
        if not name:
            raise ValueError("A host verb needs a name")
        if name in VERBS or name in self._declared:
            raise ValueError(f"A turn's host already has a verb {name!r}")
        self._declared[name] = verb

    async def call(self, request: TurnRequest, verb: str, arguments: dict[str, Any]) -> Any:
        declared = self._declared.get(verb)
        if declared is not None:
            return to_jsonable_python(await declared(dict(arguments)), fallback=str)
        serve, args = self._parsed(verb, arguments)
        return to_jsonable_python(await serve(_Scope.of(request), args), fallback=str)

    async def call_between_turns(self, message: RoomCall) -> Any:
        if message.verb not in BETWEEN_TURNS:
            raise ValueError(f"{message.verb!r} needs a turn: it is not served between turns")
        room = await self._kit.store.get_room(message.room_id)
        if room is None:
            raise ValueError(f"No room {message.room_id!r}")
        serve, args = self._parsed(message.verb, message.arguments)
        # Work its turn began, which starts fresh once the turn is over
        # (RFC §8.3): the call's depth for its turn floors nothing.
        scope = _Scope(message.room_id, room.organization_id, None, None)
        return to_jsonable_python(await serve(scope, args), fallback=str)

    def _parsed(self, verb: str, arguments: dict[str, Any]) -> tuple[Verb, Any]:
        """The kit's code serving *verb*, and its arguments validated.

        Raises:
            ValueError: a verb or arguments the host does not know.
        """
        record = VERBS.get(verb)
        serve = self._verbs.get(verb)
        if record is None or serve is None:
            raise ValueError(f"No verb {verb!r} on a turn's host")
        try:
            args = record.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"The arguments of verb {verb!r} are not valid: {exc}") from exc
        return serve, args

    async def _send_event(self, scope: _Scope, args: SendEvent) -> Any:
        options = args.model_dump(exclude={"content", "chain_depth", "organization_id"})
        floor = scope.chain_depth if scope.chain_depth is not None else 0
        return await self._kit.send_event(
            content=args.content,
            chain_depth=max(args.chain_depth, floor),
            organization_id=_organization(scope, args.room_id, args.organization_id),
            **options,
        )

    async def _delegate(self, scope: _Scope, args: Delegate) -> str:
        _same_room(scope, args.room_id)
        task = await self._kit.delegate(**args.model_dump())
        self._keep(task)
        return task.id

    async def _wait_task(self, scope: _Scope, args: WaitTask) -> Any:
        task = self._tasks.get(args.task_id)
        if task is None or task.parent_room_id != scope.room_id:
            return None
        try:
            result = await task.wait(args.timeout)
        except TimeoutError:
            return None
        failure = result.exception
        return {
            "result": result,
            "exception": None if failure is None else error_record(failure),
        }

    async def _post_status(self, scope: _Scope, args: PostStatus) -> None:
        # A status names its room (RFC §19.8), whose hooks it fires: the
        # turn's, never another. A task's status is the framework's to post:
        # the room's task tools read it to know which tasks are the room's.
        if args.action == TASK_ACTION:
            raise ValueError(f"A turn's host posts no {TASK_ACTION!r} status: the framework does")
        metadata = dict(args.metadata or {})
        if RUN_ENTRY in metadata:
            # A worker run's entry ends its task's line: the framework's too.
            raise ValueError(f"A turn's host posts no {RUN_ENTRY!r} status: the framework does")
        _same_room(scope, metadata.setdefault("room_id", scope.room_id))
        self._kit.status_bus.post(
            args.agent_id, args.action, args.status, detail=args.detail, metadata=metadata
        )

    async def _update_room_metadata(self, scope: _Scope, args: UpdateRoomMetadata) -> Any:
        return await self._kit.update_room_metadata(
            args.room_id,
            args.metadata,
            organization_id=_organization(scope, args.room_id, args.organization_id),
        )

    async def _room_tasks(self, scope: _Scope, args: RoomTasks) -> list[dict[str, Any]]:
        return await self._kit._build_room_tasks_loader()(scope.room_id)

    async def _room_vision(self, scope: _Scope, args: RoomVision) -> str | None:
        # What the room's cameras last saw lives in this process (RFC §12.8.7).
        if scope.context is None:
            return None
        return room_vision_note(self._kit.channels, scope.context)

    async def _add_observation(self, scope: _Scope, args: AddObservation) -> None:
        _same_room(scope, args.observation.room_id)
        await self._kit.store.add_observation(args.observation)

    def _keep(self, task: DelegatedTask) -> None:
        """Keep *task* for ``wait_task``, letting go of the oldest ended ones
        past the bound."""
        self._tasks[task.id] = task
        ended = [tid for tid, kept in self._tasks.items() if _ended(kept)]
        for task_id in ended[: max(0, len(ended) - ENDED_TASKS_KEPT)]:
            del self._tasks[task_id]


def _same_room(scope: _Scope, room_id: str) -> None:
    """Refuse a verb for another room than its turn's."""
    if room_id != scope.room_id:
        raise ValueError(f"A turn's host acts for its room {scope.room_id!r}, not for {room_id!r}")


def _organization(scope: _Scope, room_id: str, named: str | None) -> str | None:
    """The organization a verb runs in: its turn's room's; one the verb names
    must be that one."""
    _same_room(scope, room_id)
    organization = scope.organization_id
    if named is not None and named != organization:
        raise ValueError(f"A turn's host acts in its room's organization, not in {named!r}")
    return organization


def _ended(task: DelegatedTask) -> bool:
    return task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
