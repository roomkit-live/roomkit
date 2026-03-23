"""Loop orchestration strategy.

An agent produces output, a reviewer evaluates it, and the cycle
repeats until the reviewer approves or max iterations are reached.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
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
    it. If the reviewer approves, the loop ends. Otherwise, the
    reviewer's feedback is routed back to the producer for another
    iteration, up to ``max_iterations``.

    Examples::

        # Framework-driven: auto_cycle runs the loop automatically
        Loop(
            agent=writer,
            reviewer=editor,
            max_iterations=3,
            auto_cycle=True,
        )

        # Tool-based: agents hand off via handoff_conversation tools
        Loop(
            agent=writer,
            reviewer=editor,
            max_iterations=3,
        )
    """

    def __init__(
        self,
        agent: Agent,
        reviewer: Agent,
        max_iterations: int = 3,
        *,
        auto_cycle: bool = False,
    ) -> None:
        """Initialise the loop strategy.

        Args:
            agent: The producing agent.
            reviewer: The reviewing agent.
            max_iterations: Maximum number of produce-review cycles.
            auto_cycle: If ``True``, the framework drives the loop
                automatically using child rooms. Each agent runs in
                isolation, content passes between them, no handoff
                tools needed. If ``False`` (default), agents use
                ``handoff_conversation`` and ``approve_output`` tools.
        """
        self._agent = agent
        self._reviewer = reviewer
        self._max_iterations = max_iterations
        self._auto_cycle = auto_cycle

    def agents(self) -> list[Agent]:
        """Return agents to attach to the room."""
        if self._auto_cycle:
            # In auto_cycle mode, a supervisor-like agent presents results.
            # But we don't need agents in the room — the on_event wrapper
            # on the producer handles everything.
            return [self._agent]
        return [self._agent, self._reviewer]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire loop routing and tools."""
        if self._auto_cycle:
            await self._install_auto_cycle(kit, room_id)
        else:
            await self._install_handoff_loop(kit, room_id)

    # -- Auto-cycle (framework-driven) ----------------------------------------

    async def _install_auto_cycle(self, kit: RoomKit, room_id: str) -> None:
        """Wrap the producer's on_event to run the full loop inline."""
        producer = self._agent
        reviewer = self._reviewer
        max_iter = self._max_iterations
        original_on_event = producer.on_event

        # Register reviewer on the kit (not attached to room)
        if reviewer.channel_id not in kit.channels:
            kit.register_channel(reviewer)

        async def auto_cycle_on_event(
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            # Only intercept in the parent room (not in child rooms)
            current_room = context.room.id if context.room else event.room_id
            if current_room != room_id:
                return await original_on_event(event, binding, context)
            # Only intercept user messages (skip self-loop and AI sources)
            if event.source.channel_id == producer.channel_id:
                return ChannelOutput.empty()
            if event.source.channel_type == ChannelType.AI:
                return ChannelOutput.empty()
            if event.source.channel_type == ChannelType.SYSTEM:
                return await original_on_event(event, binding, context)

            rid = context.room.id if context.room else room_id
            return await _run_auto_cycle(
                kit=kit,
                room_id=rid,
                producer=producer,
                reviewer=reviewer,
                original_on_event=original_on_event,
                event=event,
                binding=binding,
                context=context,
                max_iterations=max_iter,
            )

        producer.on_event = auto_cycle_on_event  # type: ignore[assignment]

        # Set initial state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase=producer.channel_id,
            active_agent_id=producer.channel_id,
            context={
                "_loop_iteration": 0,
                "_loop_approved": False,
                "_loop_max_iterations": max_iter,
            },
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    # -- Handoff-based loop (original) ----------------------------------------

    async def _install_handoff_loop(self, kit: RoomKit, room_id: str) -> None:
        """Wire handoff-based loop with tools (original behavior)."""
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


# ---------------------------------------------------------------------------
# Auto-cycle execution
# ---------------------------------------------------------------------------


async def _run_auto_cycle(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewer: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    max_iterations: int,
) -> ChannelOutput:
    """Run the full produce/review loop inline using child rooms."""
    # Extract user's request
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body
    if not user_message:
        return await original_on_event(event, binding, context)

    current_input = user_message
    approved = False
    final_output = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        logger.info("[loop] Iteration %d/%d — producer", iteration, max_iterations)

        # Run producer in child room via kit.delegate()
        delegated = await kit.delegate(room_id, producer.channel_id, current_input, wait=True)
        producer_output = delegated.result.output if delegated.result else ""
        if not producer_output:
            logger.warning("[loop] Producer returned empty output")
            break

        logger.info("[loop] Iteration %d/%d — reviewer", iteration, max_iterations)

        # Run reviewer in child room with producer's output
        review_input = (
            f"Review the following content and decide if it meets quality standards.\n"
            f"If approved, your response MUST contain the word APPROVED.\n"
            f"If not approved, provide specific feedback for revision.\n\n"
            f"--- Content to review ---\n{producer_output}"
        )
        delegated = await kit.delegate(room_id, reviewer.channel_id, review_input, wait=True)
        reviewer_output = delegated.result.output if delegated.result else ""
        if not reviewer_output:
            logger.warning("[loop] Reviewer returned empty output")
            break

        # Check if reviewer approved
        if "APPROVED" in reviewer_output.upper():
            approved = True
            final_output = producer_output
            logger.info("[loop] Approved at iteration %d", iteration)
            break

        # Not approved — use reviewer's feedback as next input
        current_input = (
            f"Revise your previous work based on this feedback:\n\n"
            f"--- Your previous output ---\n{producer_output}\n\n"
            f"--- Reviewer feedback ---\n{reviewer_output}"
        )
        final_output = producer_output

    # Update loop state
    room = await kit.get_room(room_id)
    state = get_conversation_state(room)
    ctx = dict(state.context)
    ctx["_loop_approved"] = approved
    ctx["_loop_iteration"] = iteration
    state = state.model_copy(update={"context": ctx})
    room = set_conversation_state(room, state)
    await kit.store.update_room(room)

    # Return the final output directly — no pass 2 needed
    status = "approved" if approved else f"max iterations ({max_iterations}) reached"
    logger.info("[loop] Complete (%s) — returning result", status)

    result_event = RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(
            channel_id=producer.channel_id,
            channel_type=ChannelType.AI,
        ),
        content=TextContent(body=final_output),
    )
    return ChannelOutput(responded=True, response_events=[result_event])
