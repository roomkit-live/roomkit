"""A discussion's configuration, stored with the room (RFC §19.7.5 rule 16).

Every process that serves the room routes and queues by it, whether or not
its host installed the discussion there; ``done`` is the host's and stays in
the processes that installed it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from roomkit.channels.declared import id_of
from roomkit.core._room_metadata import DISCUSSION_CONFIG

if TYPE_CHECKING:
    from .strategy import Discussion

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
        """The configuration stored in a room's *metadata*, if one is."""
        raw = (metadata or {}).get(CONFIG_KEY)
        if not raw:
            return None
        try:
            return cls.model_validate(raw)
        except ValidationError:
            return None
