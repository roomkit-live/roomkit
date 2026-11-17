"""Extended-reasoning event payload (RFC §9.2 ``ON_AI_THINKING``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ThinkingEvent:
    """A block of an intelligence channel's extended reasoning, as it closed.

    Carried to ``ON_AI_THINKING`` hooks, once per block, when the block
    closes: the model starts its answer's text or composes a call, or its
    generation ends, fails or stops (RFC §8.4). The same reasoning is also
    published as ephemeral events for live UIs; the hook is for hosts that
    want to observe or record it without a realtime backend.
    """

    room_id: str
    """The room whose turn is reasoning."""

    channel_id: str
    """The intelligence channel doing the reasoning."""

    thinking: str
    """The block's whole reasoning text."""

    round_index: int = 0
    """Tool-loop round this reasoning belongs to."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the reasoning was observed."""
