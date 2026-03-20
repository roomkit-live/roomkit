"""Loop orchestration strategy.

An agent produces output, a reviewer evaluates it, and the cycle
repeats until the reviewer approves or max iterations are reached.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.handoff import (
    HandoffHandler,
    build_handoff_tool,
    setup_handoff,
)
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

logger = logging.getLogger("roomkit.orchestration.strategies.loop")

_APPROVE_TOOL = AITool(
    name="approve_output",
    description=(
        "Approve the current output and end the review loop. "
        "Call this when the output meets quality standards."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the output is approved",
            },
        },
        "required": ["reason"],
    },
)


class Loop(Orchestration):
    """Loop orchestration strategy.

    The producing agent generates output, then the reviewer evaluates
    it. If the reviewer calls ``approve_output``, the loop ends.
    Otherwise, the reviewer's feedback is routed back to the producer
    for another iteration, up to ``max_iterations``.

    Example::

        kit = RoomKit(
            orchestration=Loop(
                agent=writer,
                reviewer=editor,
                max_iterations=3,
            ),
        )
        room = await kit.create_room()
    """

    def __init__(
        self,
        agent: Agent,
        reviewer: Agent,
        max_iterations: int = 3,
    ) -> None:
        """Initialise the loop strategy.

        Args:
            agent: The producing agent.
            reviewer: The reviewing agent.
            max_iterations: Maximum number of produce-review cycles.
        """
        self._agent = agent
        self._reviewer = reviewer
        self._max_iterations = max_iterations

    def agents(self) -> list[Agent]:
        """Return both the producer and reviewer agents."""
        return [self._agent, self._reviewer]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire loop routing, handoff, and approval tools."""
        producer_id = self._agent.channel_id
        reviewer_id = self._reviewer.channel_id

        stages = [
            PipelineStage(
                phase=producer_id,
                agent_id=producer_id,
                next=reviewer_id,
            ),
            PipelineStage(
                phase=reviewer_id,
                agent_id=reviewer_id,
                next=None,
                can_return_to={producer_id},
            ),
        ]
        cp = ConversationPipeline(stages=stages)
        router = cp.to_router()

        handler = HandoffHandler(
            kit=kit,
            router=router,
            phase_map=cp.get_phase_map(),
            allowed_transitions=cp.get_allowed_transitions(),
        )
        handler.agents = {
            producer_id: self._agent,
            reviewer_id: self._reviewer,
        }

        # Install router as room-scoped BEFORE_BROADCAST hook
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=router.as_hook(),
                priority=-100,
                name=f"loop_router_{room_id}",
            ),
        )

        # Wire handoff: producer → reviewer, reviewer → producer
        if not any(t.name == "handoff_conversation" for t in self._agent._injected_tools):
            reviewer_desc = getattr(self._reviewer, "description", None)
            tool = build_handoff_tool([(reviewer_id, reviewer_desc)])
            setup_handoff(self._agent, handler, tool=tool)

        if not any(t.name == "handoff_conversation" for t in self._reviewer._injected_tools):
            tool = build_handoff_tool([(producer_id, getattr(self._agent, "description", None))])
            setup_handoff(self._reviewer, handler, tool=tool)

        # Inject approve_output tool into reviewer
        self._inject_approve_tool(kit, room_id)

        # Install AFTER_BROADCAST hook to drive the loop cycle
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=self._make_loop_hook(kit, room_id, producer_id, reviewer_id),
                priority=0,
                name=f"loop_cycle_{room_id}",
            ),
        )

        # Set initial conversation state with loop metadata
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase=producer_id,
            active_agent_id=producer_id,
            context={
                "_loop_iteration": 0,
                "_loop_approved": False,
                "_loop_max_iterations": self._max_iterations,
            },
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    def _inject_approve_tool(self, kit: RoomKit, room_id: str) -> None:
        """Inject the approve_output tool into the reviewer."""
        from roomkit.orchestration.handoff import _room_id_var

        if any(t.name == "approve_output" for t in self._reviewer._injected_tools):
            return

        self._reviewer._injected_tools.append(_APPROVE_TOOL)

        original = self._reviewer.tool_handler

        async def approve_aware_handler(name: str, arguments: dict[str, Any]) -> str:
            if name == "approve_output":
                rid = _room_id_var.get() or room_id
                room = await kit.get_room(rid)
                state = get_conversation_state(room)
                state = state.transition(
                    to_phase=state.phase,
                    reason="Output approved by reviewer",
                    metadata={"approved_reason": arguments.get("reason", "")},
                )
                ctx = dict(state.context)
                ctx["_loop_approved"] = True
                state = state.model_copy(update={"context": ctx})
                room = set_conversation_state(room, state)
                await kit.store.update_room(room)
                return json.dumps({"status": "approved", "reason": arguments.get("reason", "")})
            if original:
                return await original(name, arguments)
            return json.dumps({"error": f"Unknown tool: {name}"})

        self._reviewer.tool_handler = approve_aware_handler

    def _make_loop_hook(
        self,
        kit: RoomKit,
        room_id: str,
        producer_id: str,
        reviewer_id: str,
    ) -> Any:
        """Create the AFTER_BROADCAST hook that drives the review cycle."""
        max_iter = self._max_iterations

        async def loop_cycle_hook(event: RoomEvent, context: RoomContext) -> None:
            # Only act on intelligence-sourced events in this room
            from roomkit.models.enums import ChannelCategory

            source_binding = context.get_binding(event.source.channel_id)
            if source_binding is None or source_binding.category != ChannelCategory.INTELLIGENCE:
                return

            room = await kit.get_room(event.room_id)
            state = get_conversation_state(room)

            if state.context.get("_loop_approved", False):
                return

            iteration = state.context.get("_loop_iteration", 0)
            if iteration >= max_iter:
                logger.info(
                    "Loop max iterations (%d) reached for room %s",
                    max_iter,
                    event.room_id,
                )
                return

            # Producer finished → route to reviewer (increment iteration)
            if event.source.channel_id == producer_id and state.active_agent_id == producer_id:
                new_iter = iteration + 1
                state = state.transition(
                    to_phase=reviewer_id,
                    to_agent=reviewer_id,
                    reason=f"Producer output ready for review (iteration {new_iter})",
                )
                ctx = dict(state.context)
                ctx["_loop_iteration"] = new_iter
                state = state.model_copy(update={"context": ctx})
                room = set_conversation_state(room, state)
                await kit.store.update_room(room)

            # Reviewer finished (without approving) → route back to producer
            elif event.source.channel_id == reviewer_id and state.active_agent_id == reviewer_id:
                if not state.context.get("_loop_approved", False):
                    state = state.transition(
                        to_phase=producer_id,
                        to_agent=producer_id,
                        reason="Reviewer feedback — another iteration needed",
                    )
                    room = set_conversation_state(room, state)
                    await kit.store.update_room(room)

        loop_cycle_hook.__name__ = f"loop_cycle_{room_id}"
        return loop_cycle_hook
