"""A discussion's configuration as a room stores it (RFC §19.7.5 rule 16), apart
from the strategy: the channels read a stored handle through it, and importing
:mod:`roomkit.orchestration` from there runs into the AI channel it loads. No
dependency inside RoomKit but its leaf modules.

Every process that serves the room routes and queues by it, whether or not its
host installed the discussion there; ``done`` is the host's and stays in the
processes that installed it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, Field, ValidationError

from roomkit._handles import is_handle
from roomkit.core._room_metadata import DISCUSSION_CONFIG

logger = logging.getLogger("roomkit.orchestration.discussion")


class StoredDiscussion(BaseModel):
    """What a process needs to follow a room's discussion."""

    agents: list[str]
    handles: dict[str, str] | None = None
    """The name each agent is addressed by, by channel id. None (a
    configuration stored without handles): each agent's channel id."""
    people: list[str] | None = None
    addressed_only: bool = False
    everyone: list[str] | None = None
    max_turns: int | None = None
    max_depth: int
    silent_token: str = "(silent)"
    rules: str | None = None
    """How the room works, as each turn's notes give it (rule 14). None: the
    default, as for a configuration stored before rules existed."""
    dispatch: bool = False
    """A dispatch policy decides who takes a person's unaddressed message: it
    waits for the process holding the lease (rule 18)."""
    attached: list[str] = Field(default_factory=list)
    """The agents installs of the discussion attached to the room, which any
    process uninstalling it detaches (RFC §19.7)."""

    def same_as(self, other: StoredDiscussion | None) -> bool:
        """Whether *other* configures the same discussion, whatever agents
        its installs attached."""
        if other is None:
            return False
        return self.model_dump(exclude={"attached"}) == other.model_dump(exclude={"attached"})

    @classmethod
    def stored(cls, metadata: Mapping[str, Any] | None) -> Self | None:
        """The configuration stored in a room's *metadata*, if one is. What
        does not read as what ``Discussion()`` writes, strictly, or what it
        would refuse, reads as none, with a warning, never an error: this runs
        for every context the kit builds, and an error would break every
        message of the room."""
        config, refusal = cls._read(metadata)
        if refusal is not None:
            logger.warning("A stored discussion reads as none: %s", refusal)
        return config

    @classmethod
    def valid(cls, metadata: Mapping[str, Any] | None) -> Self | None:
        """:meth:`stored` with no warning: what a reader of its handles asks."""
        return cls._read(metadata)[0]

    @classmethod
    def _read(cls, metadata: Mapping[str, Any] | None) -> tuple[Self | None, str | None]:
        """The configuration *metadata* stores, or what keeps it from being one."""
        raw = (metadata or {}).get(DISCUSSION_CONFIG)
        if not raw:
            return None, None
        try:
            config = cls.model_validate(raw, strict=True)
        except ValidationError as error:
            return None, f"{error.error_count()} fields do not read"
        refusal = config._refusal()
        return (None, refusal) if refusal is not None else (config, None)

    def _refusal(self) -> str | None:
        """What ``Discussion()`` would refuse in this configuration, if anything."""
        if not self.agents:
            return "it holds no agent"
        if len(set(self.agents)) != len(self.agents):
            return "its agents are not distinct"
        if self.max_turns is not None and self.max_turns < 1:
            return "max_turns is below 1"
        # 1 is not refused: :meth:`of` writes the kit's max_chain_depth when
        # the strategy gives none, and a kit may hold 1.
        if self.max_depth < 1:
            return "max_depth is below 1"
        if not self.silent_token.strip():
            return "silent_token is blank"
        if self.rules is not None and not self.rules.strip():
            return "rules is blank"
        if any(agent not in self.agents for agent in self.everyone or ()):
            return "everyone names a channel that is not an agent"
        return self._handles_refusal()

    def _handles_refusal(self) -> str | None:
        given = self.handles or {}
        if any(agent not in self.agents for agent in given):
            return "handles names a channel that is not an agent"
        if not all(is_handle(handle) for handle in given.values()):
            return "a handle cannot name an agent"
        # An agent with no handle stored is addressed by its channel id.
        addressed = [given.get(agent, agent).casefold() for agent in self.agents]
        if len(set(addressed)) != len(addressed):
            return "two agents are addressed alike"
        return None
