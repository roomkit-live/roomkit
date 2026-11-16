"""A room's hooks end with the room (RFC §9.1, §17.2).

Room hooks, and the names a room enables on the turn executors, are kept in
this process by room id. A room the store deletes leaves them behind, and a
room created again under its id, by any organization, would run them. The
kit forgets them when it creates a room under the id, as it forgets the
id's strategy (RFC §19.7), and keeps those registered while no room held the
id: they wait for that room and become its own.

Another process may create the room again. The kit keeps the creation mark
(``created_at``) of the room an id's hooks and names belong to, and compares
it with the room in hand each time they are used (:meth:`RoomHookLifetime.settle`):
a later mark means the room they were registered for is gone, and they are
forgotten then; an earlier one is a room gone already, whose work runs none
of them. The hooks waiting for the room, and those registered below the kit,
become the first room's they run in.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.models.room import Room
    from roomkit.turns.owner import TurnClient


class Registration:
    """A kit's registration of a room hook or an enabled name, under way:
    the creation mark of the room its read found holding the id (``mark``,
    ``None`` for none), and whether the kit created a room under the id
    while it read."""

    __slots__ = ("_lifetime", "_room_id", "created", "mark")

    def __init__(self, lifetime: RoomHookLifetime, room_id: str) -> None:
        self._lifetime = lifetime
        self._room_id = room_id
        self.mark: datetime | None = None
        self.created = False

    @property
    def held(self) -> bool:
        """Whether a room held the id when it was read, or was created since."""
        return self.mark is not None or self.created

    def note(self, name: str) -> None:
        """The registration of *name* is done: it waits for the room created
        under the id unless a room held it when it was read, or was created
        since."""
        self._lifetime.registered(self._room_id, name, held=self.held, mark=self.mark)


class RoomHookLifetime:
    """Which of a room id's hooks wait for the room created under the id,
    and the forgetting of the others when the kit creates it."""

    def __init__(
        self,
        engine: HookEngine,
        executor_hooks: HookEngine,
        turns: TurnClient,
        names_forgotten: Callable[[str], None],
    ) -> None:
        self._engine = engine
        # The kit's own executor's engine: the room hooks given to it.
        self._executor_hooks = executor_hooks
        self._turns = turns
        # Told the room id whose enabled names a settle forgot.
        self._names_forgotten = names_forgotten
        # Room id → the names of its hooks registered while no room held it.
        self._waiting: dict[str, set[str]] = {}
        # Room id → the registrations reading whether a room holds it.
        self._registering: dict[str, list[Registration]] = {}
        # Room id → the creation mark of the room its hooks and enabled names
        # are registered for; only while it holds some.
        self._marks: dict[str, datetime] = {}

    @contextmanager
    def registering(self, room_id: str) -> Generator[Registration]:
        """A registration for *room_id*, from before its read of the room to
        its :meth:`Registration.note`: a room the kit creates meanwhile is
        one the registration counts as holding the id."""
        registration = Registration(self, room_id)
        self._registering.setdefault(room_id, []).append(registration)
        try:
            yield registration
        finally:
            under_way = self._registering[room_id]
            under_way.remove(registration)
            if not under_way:
                del self._registering[room_id]

    def registered(
        self, room_id: str, name: str, *, held: bool, mark: datetime | None = None
    ) -> None:
        """Note the hook or enabled name *name* of *room_id*: it waits for
        the room created under the id unless a room held it (*held*), the
        room of the creation mark *mark* when the registration read one."""
        if not held:
            self._waiting.setdefault(room_id, set()).add(name)
            return
        self._release_waiting(room_id, name)
        if mark is not None:
            self._marks[room_id] = mark

    def released(self, room_id: str, name: str) -> None:
        """*room_id*'s hook or enabled name *name* is removed or disabled."""
        self._release_waiting(room_id, name)
        if not self._holds(room_id):
            self._marks.pop(room_id, None)

    def _release_waiting(self, room_id: str, name: str) -> None:
        names = self._waiting.get(room_id)
        if names is None:
            return
        names.discard(name)
        if not names:
            del self._waiting[room_id]

    def _holds(self, room_id: str) -> bool:
        """Whether *room_id* holds a hook or an enabled name in this process."""
        return bool(
            room_id in self._waiting
            or self._engine._holds_room(room_id)
            or self._executor_hooks._holds_room(room_id)
            or self._turns.room_hook_names(room_id)
        )

    def settle(self, room: Room) -> bool:
        """*room* is in hand, read for its id's hooks or turns: whether the
        hooks and names of its id are *room*'s.

        Registered for an earlier room of the id, deleted since and created
        again, by this process or another, they are forgotten
        (:meth:`forget_earlier_room`), and those waiting for a room are
        *room*'s. With no mark yet, those waiting and those registered below
        the kit become *room*'s. *room* earlier than the room they are
        registered for is gone already: they are not its own (RFC §9.1,
        §17.2)."""
        mark = self._marks.get(room.id)
        if mark is None:
            if self._holds(room.id):
                self._waiting.pop(room.id, None)
                self._marks[room.id] = room.created_at
            return True
        if room.created_at == mark:
            return True
        if room.created_at < mark:
            return False
        if self.forget_earlier_room(room.id, room.created_at):
            self._names_forgotten(room.id)
        return True

    def forget_earlier_room(self, room_id: str, created_at: datetime) -> bool:
        """A room of creation mark *created_at* holds *room_id*, created by
        this process or found by it: forget the hooks and enabled names an
        earlier room of the id left, keep those waiting for this room, which
        are its own from now on. Whether an enabled name was forgotten."""
        for registration in self._registering.get(room_id, ()):
            registration.created = True
        keep = self._waiting.pop(room_id, set())
        # A hook registered below the kit, on an engine or given straight to
        # the executor, counts as one registered while a room held the id.
        self._engine._forget_room(room_id, keep)
        self._executor_hooks._forget_room(room_id, keep)
        disabled = False
        for name in self._turns.room_hook_names(room_id):
            if name not in keep:
                disabled = self._turns.disable_room_hook(room_id, name) or disabled
        if keep or room_id in self._registering:
            self._marks[room_id] = created_at
        else:
            self._marks.pop(room_id, None)
        return disabled
