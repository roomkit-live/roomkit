"""What the processes of the Redis turn backend say of themselves (RFC
§10.6.7, §10.6.8), as the backend reads it: what an executor declares, and
the processes alive as one read of the registry saw them.

A turn counts as alive while its executor beats and its process still runs
it: a turn whose task ended without its ``end`` is soon lost, not waited for
forever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from roomkit.turns._hosting import Hosted
from roomkit.turns._organizations import RoomNeeds, serves_organization

if TYPE_CHECKING:
    from roomkit.turns.broker import RegistryView

logger = logging.getLogger("roomkit.turns")


@dataclass(frozen=True, slots=True)
class Declaration:
    """What an executor of a process declares to the owners (RFC §10.6.7):
    its channels, names and prefixes as it was given them, its room hooks,
    and the organizations it serves (``None``: the rooms without one; a
    beat that names none, an executor of a release before them, reads so)."""

    node: str
    executor_id: str
    channels: Hosted
    room_hooks: frozenset[str]
    organizations: frozenset[str] | None = None

    @property
    def member(self) -> str:
        """The executor, as one process runs it."""
        return member_of(self.node, self.executor_id)

    def takes(self, channel_id: str, needs: RoomNeeds) -> bool:
        return (
            self.channels.covers(channel_id)
            and serves_organization(self.organizations, needs.organization_id)
            and all(h in self.room_hooks for h in needs.room_hooks)
        )

    def dump(self) -> str:
        return json.dumps(
            {
                "node": self.node,
                "executor_id": self.executor_id,
                "channels": sorted(self.channels.entries),
                "room_hooks": sorted(self.room_hooks),
                "organizations": (
                    None if self.organizations is None else sorted(self.organizations)
                ),
            }
        )

    @classmethod
    def load(cls, raw: str) -> Declaration:
        data = json.loads(raw)
        return cls(
            data["node"],
            data["executor_id"],
            Hosted(frozenset(data["channels"])),
            frozenset(data["room_hooks"]),
            None if data.get("organizations") is None else frozenset(data["organizations"]),
        )


def member_of(node: str, executor_id: str) -> str:
    """An executor as one process runs it: ``<node>:<executor id>``."""
    return f"{node}:{executor_id}"


@dataclass(slots=True)
class View:
    """The processes alive as one read saw them, on the server's clock."""

    now: int = 0
    beats: dict[str, int] = field(default_factory=dict)
    declared: dict[str, Declaration] = field(default_factory=dict)
    owners: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, read: RegistryView) -> View:
        """What *read*, a read of the registry, saw."""
        view = cls(now=read.now)
        for member, (beat, declared) in read.executors.items():
            view.beats[member] = beat
            if declared:
                try:
                    view.declared[member] = Declaration.load(declared)
                except (ValueError, KeyError, TypeError):
                    logger.warning("The declaration of executor %s is malformed", member)
        view.owners = dict(read.owners)
        return view
