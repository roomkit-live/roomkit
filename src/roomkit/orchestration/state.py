"""Conversation state models for multi-agent orchestration.

Tracks conversation progress within a room: which phase we're in,
which agent is active, and the full history of transitions.
Stored in Room.metadata["_conversation_state"].
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.room import Room


@unique
class ConversationPhase(StrEnum):
    """Built-in conversation phases.

    Users can use custom string values — routing and state do not
    restrict phases to this enum.
    """

    INTAKE = "intake"
    QUALIFICATION = "qualification"
    HANDLING = "handling"
    ESCALATION = "escalation"
    RESOLUTION = "resolution"
    FOLLOWUP = "followup"


_STATE_KEY = "_conversation_state"


class PhaseTransition(BaseModel):
    """Audit record for a phase change."""

    from_phase: str
    to_phase: str
    from_agent: str | None = None
    to_agent: str | None = None
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """Tracks conversation progress within a room.

    All fields are optional with sensible defaults so rooms
    without orchestration have zero overhead.
    """

    phase: str = ConversationPhase.INTAKE
    active_agent_id: str | None = None
    previous_agent_id: str | None = None
    handoff_count: int = 0
    phase_started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase_history: list[PhaseTransition] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    def transition(
        self,
        to_phase: str,
        to_agent: str | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ConversationState:
        """Create a new state with a phase transition recorded.

        Returns a new instance (immutable pattern) — the original
        is never modified.
        """
        record = PhaseTransition(
            from_phase=self.phase,
            to_phase=to_phase,
            from_agent=self.active_agent_id,
            to_agent=to_agent,
            reason=reason,
            metadata=metadata or {},
        )
        agent_changed = to_agent != self.active_agent_id
        return self.model_copy(
            update={
                "phase": to_phase,
                "previous_agent_id": self.active_agent_id,
                "active_agent_id": to_agent,
                "handoff_count": self.handoff_count + (1 if agent_changed else 0),
                "phase_started_at": datetime.now(UTC),
                "phase_history": [*self.phase_history, record],
            }
        )


def get_conversation_state(room: Room) -> ConversationState:
    """Extract typed ConversationState from room metadata.

    Returns a fresh default state if no orchestration state exists.
    """
    raw = room.metadata.get(_STATE_KEY)
    if raw is None:
        return ConversationState()
    return ConversationState.model_validate(raw)


def set_conversation_state(room: Room, state: ConversationState) -> Room:
    """Return a room copy with updated conversation state.

    Does NOT persist — the caller must save via ``store.update_room()``.
    """
    return room.model_copy(
        update={
            "metadata": {
                **room.metadata,
                _STATE_KEY: state.model_dump(mode="json"),
            }
        }
    )
