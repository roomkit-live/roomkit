"""Conversation router for multi-agent orchestration.

Routes events to the appropriate agent based on conversation state,
rules, and affinity. Implemented as a BEFORE_BROADCAST sync hook —
does NOT modify EventRouter internals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.models.hook import HookResult
from roomkit.orchestration.handoff import HandoffHandler, setup_handoff
from roomkit.orchestration.state import ConversationState, get_conversation_state

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.router")


class RoutingConditions(BaseModel):
    """Conditions for a routing rule to match.

    All specified conditions are ANDed — every non-None field must match.
    """

    phases: set[str] | None = None
    channel_types: set[ChannelType] | None = None
    intents: set[str] | None = None
    source_channel_ids: set[str] | None = None
    custom: Callable[[RoomEvent, RoomContext, ConversationState], bool] | None = Field(
        default=None, exclude=True
    )

    model_config = {"arbitrary_types_allowed": True}


class RoutingRule(BaseModel):
    """A routing rule mapping conditions to an agent."""

    agent_id: str
    conditions: RoutingConditions
    priority: int = 0

    model_config = {"arbitrary_types_allowed": True}


class ConversationRouter:
    """Routes events to the appropriate agent.

    Installed as a BEFORE_BROADCAST sync hook (priority -100) that stamps
    routing metadata on events. The EventRouter reads this metadata to
    skip non-targeted intelligence channels.

    Usage::

        router = ConversationRouter(
            rules=[...],
            default_agent_id="agent-triage",
            supervisor_id="agent-supervisor",
        )

        kit.hook(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=-100,
        )(router.as_hook())
    """

    def __init__(
        self,
        rules: list[RoutingRule] | None = None,
        default_agent_id: str | None = None,
        supervisor_id: str | None = None,
    ) -> None:
        self._rules = sorted(rules or [], key=lambda r: r.priority)
        self._default_agent_id = default_agent_id
        self._supervisor_id = supervisor_id

    def select_agent(
        self,
        event: RoomEvent,
        context: RoomContext,
        state: ConversationState,
    ) -> str | None:
        """Determine which agent should handle this event.

        Priority order:
        1. If active_agent_id is set and the agent is still in the room
           -> sticky affinity (agent keeps handling)
        2. Evaluate rules in priority order -> first match wins
        3. Fall back to default_agent_id
        """
        # Don't route events FROM intelligence channels (prevent loops)
        source_binding = context.get_binding(event.source.channel_id)
        if source_binding and source_binding.category == ChannelCategory.INTELLIGENCE:
            return None

        # Agent affinity: if an agent is actively handling, stick with it
        if state.active_agent_id is not None:
            binding = context.get_binding(state.active_agent_id)
            if binding is not None:
                return state.active_agent_id

        # Evaluate rules
        for rule in self._rules:
            if self._matches(rule.conditions, event, context, state):
                return rule.agent_id

        return self._default_agent_id

    def _matches(
        self,
        conditions: RoutingConditions,
        event: RoomEvent,
        context: RoomContext,
        state: ConversationState,
    ) -> bool:
        """Check if all conditions match (AND logic)."""
        if conditions.phases is not None and state.phase not in conditions.phases:
            return False

        if (
            conditions.channel_types is not None
            and event.source.channel_type not in conditions.channel_types
        ):
            return False

        if (
            conditions.source_channel_ids is not None
            and event.source.channel_id not in conditions.source_channel_ids
        ):
            return False

        if conditions.intents is not None:
            event_intent = (event.metadata or {}).get("intent")
            if event_intent not in conditions.intents:
                return False

        return conditions.custom is None or conditions.custom(event, context, state)

    def as_hook(self) -> Callable[..., Any]:
        """Return a BEFORE_BROADCAST sync hook function.

        The hook stamps ``event.metadata`` with routing information
        that EventRouter uses to filter intelligence channels.
        """

        async def conversation_router(event: RoomEvent, context: RoomContext) -> HookResult:
            # Set room_id for handoff tool handler (ContextVar inherited by
            # asyncio tasks spawned during broadcast).
            from roomkit.orchestration.handoff import _room_id_var

            _room_id_var.set(event.room_id)

            state = get_conversation_state(context.room)
            selected = self.select_agent(event, context, state)

            if selected is None:
                # No routing — all agents process (backward compatible)
                return HookResult.allow()

            # Build always-process list (supervisor, etc.)
            # Uses list (not set) — survives JSON serialization cleanly.
            always_process: list[str] = []
            if self._supervisor_id:
                always_process.append(self._supervisor_id)

            # Stamp routing metadata on the event
            updated_metadata = {
                **(event.metadata or {}),
                "_routed_to": selected,
                "_always_process": always_process,
            }
            modified_event = event.model_copy(update={"metadata": updated_metadata})

            logger.debug(
                "Routed event to %s (supervisor=%s)",
                selected,
                self._supervisor_id,
                extra={
                    "room_id": event.room_id,
                    "event_id": event.id,
                    "routed_to": selected,
                },
            )

            return HookResult.modify(modified_event)

        conversation_router.__name__ = "conversation_router"
        return conversation_router

    def install(
        self,
        kit: RoomKit,
        agents: list[AIChannel],
        *,
        agent_aliases: dict[str, str] | None = None,
        phase_map: dict[str, str] | None = None,
        hook_priority: int = -100,
    ) -> HandoffHandler:
        """Wire routing and handoff in one call.

        Registers this router as a ``BEFORE_BROADCAST`` sync hook,
        builds a ``HandoffHandler``, and calls ``setup_handoff``
        on every agent.

        Returns the ``HandoffHandler`` for further customisation.
        """
        kit.hook(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=hook_priority,
        )(self.as_hook())

        handler = HandoffHandler(
            kit=kit,
            router=self,
            agent_aliases=agent_aliases,
            phase_map=phase_map,
        )
        for agent in agents:
            setup_handoff(agent, handler)

        return handler
