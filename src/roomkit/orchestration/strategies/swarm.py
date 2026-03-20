"""Swarm orchestration strategy.

Every agent can hand off to every other agent — no linear constraints.
Routing relies on sticky agent affinity from ``ConversationState``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.handoff import (
    HandoffHandler,
    build_handoff_tool,
    setup_handoff,
)
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import (
    ConversationState,
    set_conversation_state,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.swarm")


class Swarm(Orchestration):
    """Swarm orchestration strategy.

    Every agent can hand off to every other agent with no phase
    constraints. Routing uses sticky agent affinity — once an agent
    is active, it keeps handling until it hands off.

    Example::

        kit = RoomKit(
            orchestration=Swarm(
                agents=[sales, support, billing],
                entry="sales",
            ),
        )
        room = await kit.create_room()
    """

    def __init__(
        self,
        agents: list[Agent],
        entry: str | None = None,
    ) -> None:
        """Initialise the swarm strategy.

        Args:
            agents: All participating agents.
            entry: Channel ID of the entry-point agent. Defaults to
                the first agent's channel ID.
        """
        self._agents = list(agents)
        self._entry = entry or (agents[0].channel_id if agents else None)

    def agents(self) -> list[Agent]:
        """Return all swarm agents."""
        return list(self._agents)

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire swarm routing and bidirectional handoff into the room."""
        router = ConversationRouter(
            default_agent_id=self._entry,
        )

        handler = HandoffHandler(
            kit=kit,
            router=router,
        )
        handler.greeting_map = {
            a.channel_id: g
            for a in self._agents
            if (g := getattr(a, "greeting", None)) is not None
        }
        handler.agents = {a.channel_id: a for a in self._agents}

        # Install router as room-scoped BEFORE_BROADCAST hook
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=router.as_hook(),
                priority=-100,
                name=f"swarm_router_{room_id}",
            ),
        )

        # Wire bidirectional handoff — each agent can reach all others
        for agent in self._agents:
            if any(t.name == "handoff_conversation" for t in agent._injected_tools):
                continue

            targets = [
                (a.channel_id, getattr(a, "description", None))
                for a in self._agents
                if a.channel_id != agent.channel_id
            ]
            tool = build_handoff_tool(targets)
            setup_handoff(agent, handler, tool=tool)

        # Set initial conversation state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase="swarm",
            active_agent_id=self._entry,
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)
