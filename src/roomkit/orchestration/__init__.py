"""Multi-agent conversation orchestration for RoomKit.

Provides ConversationState, ConversationRouter, HandoffHandler, and
supporting models for multi-agent routing, handoff, and pipeline workflows.
"""

from roomkit.orchestration.handoff import (
    HANDOFF_TOOL,
    HandoffHandler,
    HandoffMemoryProvider,
    HandoffRequest,
    HandoffResult,
    setup_handoff,
)
from roomkit.orchestration.router import (
    ConversationRouter,
    RoutingConditions,
    RoutingRule,
)
from roomkit.orchestration.state import (
    ConversationPhase,
    ConversationState,
    PhaseTransition,
    get_conversation_state,
    set_conversation_state,
)

__all__ = [
    # State
    "ConversationPhase",
    "ConversationState",
    "PhaseTransition",
    "get_conversation_state",
    "set_conversation_state",
    # Router
    "ConversationRouter",
    "RoutingConditions",
    "RoutingRule",
    # Handoff
    "HANDOFF_TOOL",
    "HandoffHandler",
    "HandoffMemoryProvider",
    "HandoffRequest",
    "HandoffResult",
    "setup_handoff",
]
