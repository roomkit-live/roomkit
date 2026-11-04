"""A discussion's configuration, stored with the room (RFC §19.7.5 rule 16).

Every process that serves the room routes and queues by it, whether or not
its host installed the discussion there; ``done`` is the host's and stays in
the processes that installed it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from roomkit._handles import is_handle
from roomkit.channels.declared import id_of
from roomkit.core._room_metadata import DISCUSSION_CONFIG

if TYPE_CHECKING:
    from .strategy import Discussion

logger = logging.getLogger("roomkit.orchestration.discussion")

CONFIG_KEY = DISCUSSION_CONFIG
"""The room metadata key the configuration is stored under."""


class DiscussionConfig(BaseModel):
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

    def same_as(self, other: DiscussionConfig | None) -> bool:
        """Whether *other* configures the same discussion, whatever agents
        its installs attached."""
        if other is None:
            return False
        return self.model_dump(exclude={"attached"}) == other.model_dump(exclude={"attached"})

    @classmethod
    def of(cls, strategy: Discussion, default_depth: int) -> DiscussionConfig:
        return cls(
            agents=[id_of(a) for a in strategy.agents()],
            handles=strategy.handles(),
            people=strategy.people,
            addressed_only=strategy.addressed_only,
            everyone=strategy.everyone,
            max_turns=strategy.max_turns,
            max_depth=strategy.max_depth or default_depth,
            silent_token=strategy.silent_token,
            rules=strategy.rules,
            dispatch=strategy.dispatch is not None,
        )

    @classmethod
    def stored(cls, metadata: Mapping[str, Any] | None) -> DiscussionConfig | None:
        """The configuration stored in a room's *metadata*, if one is. What
        the constructor would refuse reads as none, with a warning, never an
        error: this runs for every context the kit builds, and an error would
        break every message of the room."""
        raw = (metadata or {}).get(CONFIG_KEY)
        if not raw:
            return None
        try:
            config = cls.model_validate(raw)
        except ValidationError as error:
            logger.warning(
                "A stored discussion reads as none: %d fields do not read", error.error_count()
            )
            return None
        refusal = config._refusal()
        if refusal is not None:
            logger.warning("A stored discussion reads as none: %s", refusal)
            return None
        return config

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
