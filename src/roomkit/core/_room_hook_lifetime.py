"""A room's hooks end with the room (RFC §9.1, §17.2).

Room hooks, and the names a room enables on the turn executors, are kept in
this process by room id. A room the store deletes leaves them behind, and a
room created again under its id, by any organization, would run them. The
kit forgets them when it creates a room under the id, as it forgets the
id's strategy (RFC §19.7), and keeps those registered while no room held the
id: they wait for that room and become its own.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.turns.owner import TurnClient


class Registration:
    """A kit's registration of a room hook or an enabled name, under way:
    whether its read found a room holding the id (``held``), and whether the
    kit created a room under the id while it read."""

    __slots__ = ("_lifetime", "_room_id", "created", "held")

    def __init__(self, lifetime: RoomHookLifetime, room_id: str) -> None:
        self._lifetime = lifetime
        self._room_id = room_id
        self.held = False
        self.created = False

    def note(self, name: str) -> None:
        """The registration of *name* is done: it waits for the room created
        under the id unless a room held it when it was read, or was created
        since."""
        self._lifetime.registered(self._room_id, name, held=self.held or self.created)


class RoomHookLifetime:
    """Which of a room id's hooks wait for the room created under the id,
    and the forgetting of the others when the kit creates it."""

    def __init__(self, engine: HookEngine, executor_hooks: HookEngine, turns: TurnClient) -> None:
        self._engine = engine
        # The kit's own executor's engine: the room hooks given to it.
        self._executor_hooks = executor_hooks
        self._turns = turns
        # Room id → the names of its hooks registered while no room held it.
        self._waiting: dict[str, set[str]] = {}
        # Room id → the registrations reading whether a room holds it.
        self._registering: dict[str, list[Registration]] = {}

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

    def registered(self, room_id: str, name: str, *, held: bool) -> None:
        """Note the hook or enabled name *name* of *room_id*: it waits for
        the room created under the id unless a room held it (*held*)."""
        if held:
            self.released(room_id, name)
        else:
            self._waiting.setdefault(room_id, set()).add(name)

    def released(self, room_id: str, name: str) -> None:
        """*room_id*'s hook or enabled name *name* is removed or disabled."""
        names = self._waiting.get(room_id)
        if names is None:
            return
        names.discard(name)
        if not names:
            del self._waiting[room_id]

    def forget_earlier_room(self, room_id: str) -> bool:
        """The store has just created a room under *room_id*: forget the
        hooks and enabled names an earlier room of the id left, keep those
        waiting for this room, which are its own from now on. Whether an
        enabled name was forgotten."""
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
        return disabled
