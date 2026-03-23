"""Loop orchestration strategy.

An agent produces output, reviewers evaluate it, and the cycle
repeats until all reviewers approve or max iterations are reached.
The framework controls the flow — agents just produce content.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

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
from roomkit.orchestration.strategies.supervisor import WorkerStrategy

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.loop")


class Loop(Orchestration):
    """Loop orchestration strategy.

    The producing agent generates output, then reviewers evaluate it.
    If all reviewers approve, the loop ends. Otherwise, feedback is
    routed back to the producer for revision.

    Examples::

        # Single reviewer
        Loop(agent=writer, reviewers=[editor], max_iterations=3)

        # Multiple reviewers — sequential (chained)
        Loop(
            agent=coder,
            reviewers=[security, perf, style],
            strategy="sequential",
        )

        # Multiple reviewers — parallel (fan-out)
        Loop(
            agent=coder,
            reviewers=[security, perf, style],
            strategy="parallel",
        )

        # Voice — async delivery
        Loop(
            agent=writer,
            reviewers=[editor],
            async_delivery=True,
        )
    """

    def __init__(
        self,
        agent: Agent,
        reviewers: list[Agent] | None = None,
        reviewer: Agent | None = None,
        max_iterations: int = 3,
        *,
        strategy: WorkerStrategy | str | None = None,
        async_delivery: bool = False,
    ) -> None:
        """Initialise the loop strategy.

        Args:
            agent: The producing agent.
            reviewers: List of reviewing agents. For multiple reviewers,
                use *strategy* to control execution order.
            reviewer: Single reviewer (convenience, same as
                ``reviewers=[reviewer]``).
            max_iterations: Maximum number of produce-review cycles.
            strategy: How reviewers execute when there are multiple:

                - ``"sequential"``: reviewers chain — each sees the
                  previous reviewer's feedback.
                - ``"parallel"``: reviewers fan-out — all review
                  independently, feedback combined.
                - ``None`` (default): sequential for multiple reviewers,
                  single reviewer doesn't need a strategy.

            async_delivery: If ``True``, the loop runs in the background
                and results are delivered via ``kit.deliver()`` when
                ready. The conversation continues uninterrupted.
        """
        self._agent = agent

        # Accept either reviewers=[...] or reviewer=single
        if reviewers and reviewer:
            msg = "Provide either 'reviewers' or 'reviewer', not both"
            raise ValueError(msg)
        if reviewer:
            self._reviewers = [reviewer]
        elif reviewers:
            self._reviewers = list(reviewers)
        else:
            msg = "At least one reviewer is required"
            raise ValueError(msg)

        self._max_iterations = max_iterations
        self._strategy = WorkerStrategy(strategy) if strategy else None
        self._async_delivery = async_delivery

    def agents(self) -> list[Agent]:
        """Return the producer — it presents results to the user."""
        if self._async_delivery:
            return []
        return [self._agent]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire the framework-driven loop."""
        producer = self._agent
        reviewers = self._reviewers
        max_iter = self._max_iterations
        strategy = self._strategy
        async_delivery = self._async_delivery
        original_on_event = producer.on_event

        # Register all reviewers on the kit (not attached to room)
        for rev in reviewers:
            if rev.channel_id not in kit.channels:
                kit.register_channel(rev)

        # Also register producer if async (not attached to room)
        if async_delivery and producer.channel_id not in kit.channels:
            kit.register_channel(producer)

        if async_delivery:
            self._install_async_loop(kit, room_id)
        else:

            async def loop_on_event(
                event: RoomEvent,
                binding: ChannelBinding,
                context: RoomContext,
            ) -> ChannelOutput:
                current_room = context.room.id if context.room else event.room_id
                if current_room != room_id:
                    return await original_on_event(event, binding, context)
                if event.source.channel_id == producer.channel_id:
                    return ChannelOutput.empty()
                if event.source.channel_type in (ChannelType.AI, ChannelType.SYSTEM):
                    return await original_on_event(event, binding, context)

                return await _run_loop(
                    kit=kit,
                    room_id=current_room,
                    producer=producer,
                    reviewers=reviewers,
                    strategy=strategy,
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

    # -- Async delivery (voice) -----------------------------------------------

    def _install_async_loop(self, kit: RoomKit, room_id: str) -> None:
        """Inject delegate_workers tool into RealtimeVoiceChannel for async loop."""
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        producer = self._agent
        reviewers = self._reviewers
        max_iter = self._max_iterations
        strategy = self._strategy

        voice_channel: RealtimeVoiceChannel | None = None
        for ch in kit.channels.values():
            if isinstance(ch, RealtimeVoiceChannel):
                voice_channel = ch
                break

        if voice_channel is None:
            logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
            return

        reviewer_roles = ", ".join(getattr(r, "role", None) or r.channel_id for r in reviewers)
        tool_def = {
            "name": "delegate_loop",
            "description": (
                f"Submit work for review by specialists ({reviewer_roles}). "
                f"The producer will create content and reviewers will evaluate it. "
                f"Pass the topic or task description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The topic or task",
                    },
                },
                "required": ["task"],
            },
        }

        if voice_channel._tools is None:
            voice_channel._tools = []
        voice_channel._tools.append(tool_def)

        original_handler = voice_channel.tool_handler
        _running = False

        async def async_loop_handler(name: str, arguments: dict[str, Any]) -> str:
            nonlocal _running

            if name != "delegate_loop":
                if original_handler:
                    return await original_handler(name, arguments)
                return json.dumps({"error": f"Unknown tool: {name}"})

            if _running:
                return json.dumps(
                    {
                        "status": "already_running",
                        "message": "Loop is already running.",
                    }
                )

            task_desc = arguments.get("task", "")
            _running = True

            asyncio.create_task(
                _async_loop_and_deliver(
                    kit=kit,
                    room_id=room_id,
                    producer=producer,
                    reviewers=reviewers,
                    strategy=strategy,
                    task_desc=task_desc,
                    max_iterations=max_iter,
                    on_done=lambda: _clear(),
                )
            )

            return json.dumps(
                {
                    "status": "started",
                    "message": "Loop is running. Results will be delivered when ready.",
                }
            )

        def _clear() -> None:
            nonlocal _running
            _running = False

        voice_channel.tool_handler = async_loop_handler


# ---------------------------------------------------------------------------
# Loop execution
# ---------------------------------------------------------------------------


async def _run_loop(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    event: RoomEvent,
    max_iterations: int,
) -> ChannelOutput:
    """Run the full produce/review loop using child rooms."""
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body
    if not user_message:
        return ChannelOutput.empty()

    result = await _execute_loop(
        kit=kit,
        room_id=room_id,
        producer=producer,
        reviewers=reviewers,
        strategy=strategy,
        task_desc=user_message,
        max_iterations=max_iterations,
    )

    result_event = RoomEvent(
        room_id=room_id,
        type=event.type,
        source=EventSource(
            channel_id=producer.channel_id,
            channel_type=ChannelType.AI,
        ),
        content=TextContent(body=result["output"]),
        metadata={
            "_loop_approved": result["approved"],
            "_loop_iteration": result["iteration"],
        },
    )
    return ChannelOutput(responded=True, response_events=[result_event])


async def _async_loop_and_deliver(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    task_desc: str,
    max_iterations: int,
    on_done: Any,
) -> None:
    """Background: run loop → deliver results via kit.deliver()."""
    try:
        result = await _execute_loop(
            kit=kit,
            room_id=room_id,
            producer=producer,
            reviewers=reviewers,
            strategy=strategy,
            task_desc=task_desc,
            max_iterations=max_iterations,
        )

        status = "approved" if result["approved"] else "max iterations reached"
        logger.info("[loop] Complete (%s), delivering results", status)

        await kit.deliver(
            room_id,
            f"The review loop has completed ({status}).\n\n{result['output']}",
        )
    except Exception:
        logger.exception("[loop] Async loop failed")
    finally:
        on_done()


async def _execute_loop(
    *,
    kit: RoomKit,
    room_id: str,
    producer: Agent,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    task_desc: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Core loop logic shared by sync and async modes."""
    current_input = task_desc
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

        # Run reviewers
        logger.info("[loop] Iteration %d/%d — reviewers", iteration, max_iterations)
        review_results = await _run_reviewers(kit, room_id, reviewers, strategy, producer_output)

        # Check if ALL reviewers approved
        all_approved = all(r["approved"] for r in review_results)
        if all_approved:
            approved = True
            final_output = producer_output
            logger.info("[loop] All reviewers approved at iteration %d", iteration)
            break

        # Combine feedback from reviewers who didn't approve
        feedback_parts = []
        for r in review_results:
            if not r["approved"]:
                reviewer_name = r["reviewer"]
                feedback_parts.append(f"[{reviewer_name}]: {r['feedback']}")

        combined_feedback = "\n\n".join(feedback_parts)
        current_input = (
            f"Revise your previous work based on this feedback:\n\n"
            f"--- Your previous output ---\n{producer_output}\n\n"
            f"--- Reviewer feedback ---\n{combined_feedback}"
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

    return {"approved": approved, "iteration": iteration, "output": final_output}


async def _run_reviewers(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    strategy: WorkerStrategy | None,
    producer_output: str,
) -> list[dict[str, Any]]:
    """Run reviewers according to strategy."""
    review_prompt = (
        "Review the following content and decide if it meets quality standards.\n"
        "If approved, your response MUST contain the word APPROVED.\n"
        "If not approved, provide specific feedback for revision.\n\n"
        f"--- Content to review ---\n{producer_output}"
    )

    if len(reviewers) == 1 or strategy != WorkerStrategy.PARALLEL:
        # Sequential: each reviewer sees the content (+ previous feedback)
        return await _review_sequential(kit, room_id, reviewers, review_prompt)

    # Parallel: all reviewers see the same content
    return await _review_parallel(kit, room_id, reviewers, review_prompt)


async def _review_sequential(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    review_input: str,
) -> list[dict[str, Any]]:
    """Run reviewers sequentially — each sees previous feedback."""
    results: list[dict[str, Any]] = []
    current_input = review_input

    for reviewer in reviewers:
        delegated = await kit.delegate(room_id, reviewer.channel_id, current_input, wait=True)
        output = delegated.result.output if delegated.result else ""
        is_approved = "APPROVED" in output.upper() if output else False

        name = getattr(reviewer, "role", None) or reviewer.channel_id
        results.append(
            {
                "reviewer": name,
                "approved": is_approved,
                "feedback": output,
            }
        )

        # Next reviewer sees previous feedback appended
        if not is_approved and output:
            current_input = f"{current_input}\n\n--- {name} feedback ---\n{output}"

    return results


async def _review_parallel(
    kit: RoomKit,
    room_id: str,
    reviewers: list[Agent],
    review_input: str,
) -> list[dict[str, Any]]:
    """Run all reviewers in parallel on the same content."""

    async def _review_one(reviewer: Agent) -> dict[str, Any]:
        delegated = await kit.delegate(room_id, reviewer.channel_id, review_input, wait=True)
        output = delegated.result.output if delegated.result else ""
        is_approved = "APPROVED" in output.upper() if output else False
        name = getattr(reviewer, "role", None) or reviewer.channel_id
        return {"reviewer": name, "approved": is_approved, "feedback": output}

    results = await asyncio.gather(*[_review_one(r) for r in reviewers])
    return list(results)
