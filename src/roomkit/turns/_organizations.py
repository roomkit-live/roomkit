"""The organizations an executor serves, and what a room asks of the executor
that takes its work (RFC §10.6.7, §17.2).

An executor declares the organizations whose rooms it serves: exact ids, the
host's own (UUIDs, say), and ``"*"`` for every one. One that declares none
serves the rooms without an organization alone, so an executor started
without a declaration never takes a tenant's turns. Like the channels it
declares, the declaration is data, told to the room owners of other processes
as it is.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.turns.wire import ChannelReleased

if TYPE_CHECKING:
    from roomkit.turns.wire import RoomAttached, RoomDetached, TurnRequest

EVERY_ORGANIZATION = "*"


def served(organizations: Collection[str] | None) -> frozenset[str] | None:
    """*organizations* as an executor declares them; ``None`` when it
    declares none.

    Raises:
        TypeError: One id given as a string, or an entry that is not a
            string (``None`` among them).
        ValueError: An empty collection, or an entry holding a ``*`` but
            ``"*"`` itself (ids are exact, never prefixes).
    """
    if organizations is None:
        return None
    if isinstance(organizations, str):
        raise TypeError("organizations is a collection of organization ids, not one id")
    declared = frozenset(organizations)
    if not all(isinstance(entry, str) for entry in declared):
        raise TypeError(
            "organizations entries are organization ids (str); omit organizations "
            "to serve the rooms without an organization, or declare '*'"
        )
    if not declared:
        raise ValueError(
            "organizations names at least one organization ('*' for every one); "
            "omit it to serve the rooms without an organization"
        )
    for entry in declared:
        if EVERY_ORGANIZATION in entry and entry != EVERY_ORGANIZATION:
            raise ValueError(
                f"organizations entry {entry!r}: an organization id is exact, and '*' "
                "stands alone for every organization"
            )
    return declared


def serves_organization(declared: frozenset[str] | None, organization_id: str | None) -> bool:
    """Whether an executor declaring *declared* serves a room of
    *organization_id* (``None``: a room without an organization)."""
    if declared is None:
        return organization_id is None
    return EVERY_ORGANIZATION in declared or organization_id in declared


def concerns(
    declared: frozenset[str] | None, message: RoomAttached | RoomDetached | ChannelReleased
) -> bool:
    """Whether *message* concerns an executor declaring *declared*, one that
    declares its channel: a release, the channel's in all its rooms, concerns
    each; an attach or a detach, one serving the room's organization (RFC
    §10.6.7)."""
    if isinstance(message, ChannelReleased):
        return True
    return serves_organization(declared, message.organization_id)


@dataclass(frozen=True, slots=True)
class RoomNeeds:
    """What a room asks of the executor that takes its turns, channel calls
    and committed events: the room hooks it defines (RFC §10.6.6), and the
    room's organization it serves (RFC §10.6.7)."""

    room_hooks: tuple[str, ...] = ()
    organization_id: str | None = None

    @classmethod
    def of_turn(cls, request: TurnRequest) -> RoomNeeds:
        """What *request*'s room asks: its hooks, and its organization as its
        context carries it."""
        return cls(tuple(request.room_hooks), request.context.room.organization_id)

    def describe(self) -> str:
        """What a refusal names beside the channel: `` with room hooks [...]``
        and `` in organization '...'``, each when it applies."""
        hooks = f" with room hooks {list(self.room_hooks)}" if self.room_hooks else ""
        organization = (
            f" in organization {self.organization_id!r}"
            if self.organization_id is not None
            else ""
        )
        return hooks + organization
