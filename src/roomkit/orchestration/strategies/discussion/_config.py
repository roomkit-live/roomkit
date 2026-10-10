"""A discussion's configuration, stored with the room (RFC §19.7.5 rule 16).

Every process that serves the room routes and queues by it, whether or not
its host installed the discussion there; ``done`` is the host's and stays in
the processes that installed it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .strategy import Discussion

CONFIG_KEY = "_discussion"
"""The room metadata key the configuration is stored under."""


class DiscussionConfig(BaseModel):
    """What a process needs to follow a room's discussion."""

    agents: list[str]
    people: list[str] | None = None
    addressed_only: bool = False
    everyone: list[str] | None = None
    max_turns: int | None = None
    max_depth: int
    silent_token: str = "(silent)"

    @classmethod
    def of(cls, strategy: Discussion, default_depth: int) -> DiscussionConfig:
        return cls(
            agents=[a.channel_id for a in strategy.agents()],
            people=strategy.people,
            addressed_only=strategy.addressed_only,
            everyone=strategy.everyone,
            max_turns=strategy.max_turns,
            max_depth=strategy.max_depth or default_depth,
            silent_token=strategy.silent_token,
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
