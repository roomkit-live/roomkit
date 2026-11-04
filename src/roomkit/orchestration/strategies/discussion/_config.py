"""A discussion's configuration, stored with the room (RFC §19.7.5 rule 16), as
the strategy writes it (:mod:`roomkit._discussion_config`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._discussion_config import StoredDiscussion
from roomkit.channels.declared import id_of
from roomkit.core._room_metadata import DISCUSSION_CONFIG

if TYPE_CHECKING:
    from .strategy import Discussion

CONFIG_KEY = DISCUSSION_CONFIG
"""The room metadata key the configuration is stored under."""


class DiscussionConfig(StoredDiscussion):
    """What a process needs to follow a room's discussion."""

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
