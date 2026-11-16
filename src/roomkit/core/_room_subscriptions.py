"""A subscription to a room's ephemerals ends with the room (RFC §8.4, §17.2).

The kit's subscriptions are kept by room id on the realtime backend, which
knows nothing of rooms. A room the store deletes leaves them behind, and a
room created again under its id, by any organization, would publish to them.
The kit ends them when it creates a room under the id, and keeps those made
while no room held the id: they wait for that room and become its own, as a
room's hooks do (``roomkit.core._room_hook_lifetime``).
"""

from __future__ import annotations


class RoomSubscriptions:
    """The kit's subscriptions to rooms' ephemerals, by room id."""

    def __init__(self) -> None:
        # Room id → the subscriptions made for the room that holds the id.
        self._held: dict[str, set[str]] = {}
        # Room id → the subscriptions made while no room held it.
        self._waiting: dict[str, set[str]] = {}
        # Subscription id → its room id.
        self._room_of: dict[str, str] = {}

    def made(self, room_id: str, subscription_id: str, *, held: bool) -> None:
        """Note the subscription *subscription_id* to *room_id*'s ephemerals:
        the room's when a room held the id (*held*), waiting for the room
        created under it otherwise."""
        rooms = self._held if held else self._waiting
        rooms.setdefault(room_id, set()).add(subscription_id)
        self._room_of[subscription_id] = room_id

    def ended(self, subscription_id: str) -> None:
        """*subscription_id* was unsubscribed."""
        room_id = self._room_of.pop(subscription_id, None)
        if room_id is None:
            return
        for rooms in (self._held, self._waiting):
            subscriptions = rooms.get(room_id)
            if subscriptions is not None and subscription_id in subscriptions:
                subscriptions.discard(subscription_id)
                if not subscriptions:
                    del rooms[room_id]

    def earlier_room(self, room_id: str) -> list[str]:
        """The store has just created a room under *room_id*: the
        subscriptions an earlier room of the id left, to end, forgotten
        here; those waiting for this room are its own from now on."""
        ending = self._held.pop(room_id, set())
        for subscription_id in ending:
            del self._room_of[subscription_id]
        waiting = self._waiting.pop(room_id, None)
        if waiting:
            self._held[room_id] = waiting
        return sorted(ending)
