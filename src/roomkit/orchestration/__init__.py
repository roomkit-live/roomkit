"""Multi-agent conversation orchestration for RoomKit.

Phase 1 — ConversationState (foundation).
Router, Handoff, and Pipeline will be added in subsequent phases.
"""

from roomkit.orchestration.state import (
    ConversationPhase,
    ConversationState,
    PhaseTransition,
    get_conversation_state,
    set_conversation_state,
)

__all__ = [
    # State (Phase 1)
    "ConversationPhase",
    "ConversationState",
    "PhaseTransition",
    "get_conversation_state",
    "set_conversation_state",
]
