"""Loop orchestration strategy.

An agent produces output, a reviewer evaluates it, and the cycle
repeats until the reviewer approves or max iterations are reached.
The framework controls the flow — agents just produce content.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.state import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.loop")


class Loop(Orchestration):
    """Loop orchestration strategy.

    The producing agent generates output, then the reviewer evaluates
    it. If the reviewer approves, the loop ends. Otherwise, the
    reviewer's feedback is routed back to the producer for another
    iteration, up to ``max_iterations``.

    The framework drives the cycle using child rooms — agents never
    call handoff tools. Each agent runs in isolation and receives
    only the input it needs.

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
        """Return the producer — it presents results to the user."""
        return [self._agent]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire the framework-driven loop."""
        producer = self._agent
        reviewer = self._reviewer
        max_iter = self._max_iterations
        original_on_event = producer.on_event

        # Register reviewer on the kit (not attached to room)
        if reviewer.channel_id not in kit.channels:
            kit.register_channel(reviewer)

        async def loop_on_event(
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            # Only intercept in the parent room (not in child rooms)
            current_room = context.room.id if context.room else event.room_id
            if current_room != room_id:
                return await original_on_event(event, binding, context)
            # Skip self-loop, AI sources, and system events
            if event.source.channel_id == producer.channel_id:
                return ChannelOutput.empty()
            if event.source.channel_type in (ChannelType.AI, ChannelType.SYSTEM):
                return await original_on_event(event, binding, context)

            return await _run_loop(
                kit=kit,
                room_id=context.room.id if context.room else room_id,
                producer=producer,
                reviewer=reviewer,
                event=event,
                max_iterations=max_iter,
            )

        producer.on_event = loop_on_event  # type: ignore[assignment]

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


# ---------------------------------------------------------------------------
# Loop execution
# ---------------------------------------------------------------------------


async def _run_loop(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewer: Agent,
    event: RoomEvent,
    max_iterations: int,
) -> ChannelOutput:
    """Run the full produce/review loop using child rooms."""
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body
    if not user_message:
        return ChannelOutput.empty()

    current_input = user_message
    approved = False
    final_output = ""
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        logger.info("[loop] Iteration %d/%d — producer", iteration, max_iterations)

        delegated = await kit.delegate(room_id, producer.channel_id, current_input, wait=True)
        producer_output = delegated.result.output if delegated.result else ""
        if not producer_output:
            logger.warning("[loop] Producer returned empty output")
            break

        logger.info("[loop] Iteration %d/%d — reviewer", iteration, max_iterations)

        review_input = (
            "Review the following content and decide if it meets quality standards.\n"
            "If approved, your response MUST contain the word APPROVED.\n"
            "If not approved, provide specific feedback for revision.\n\n"
            f"--- Content to review ---\n{producer_output}"
        )
        delegated = await kit.delegate(room_id, reviewer.channel_id, review_input, wait=True)
        reviewer_output = delegated.result.output if delegated.result else ""
        if not reviewer_output:
            logger.warning("[loop] Reviewer returned empty output")
            break

        if "APPROVED" in reviewer_output.upper():
            approved = True
            final_output = producer_output
            logger.info("[loop] Approved at iteration %d", iteration)
            break

        current_input = (
            f"Revise your previous work based on this feedback:\n\n"
            f"--- Your previous output ---\n{producer_output}\n\n"
            f"--- Reviewer feedback ---\n{reviewer_output}"
        )
        final_output = producer_output

    # Update state
    room = await kit.get_room(room_id)
    state = get_conversation_state(room)
    ctx = dict(state.context)
    ctx["_loop_approved"] = approved
    ctx["_loop_iteration"] = iteration
    state = state.model_copy(update={"context": ctx})
    room = set_conversation_state(room, state)
    await kit.store.update_room(room)

    status = "approved" if approved else f"max iterations ({max_iterations}) reached"
    logger.info("[loop] Complete (%s) — returning result", status)

    result_event = RoomEvent(
        room_id=room_id,
        type=event.type,
        source=EventSource(
            channel_id=producer.channel_id,
            channel_type=ChannelType.AI,
        ),
        content=TextContent(body=final_output),
    )
    return ChannelOutput(responded=True, response_events=[result_event])
