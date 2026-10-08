"""AIChannel mixin for mid-run steering (cancel, inject, update prompt)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._realtime_context import ending_cause, held_by
from roomkit.channels._turn_notes import without_header_copies
from roomkit.core.task_utils import CLOSE_WAIT_S
from roomkit.models.steering import Cancel, InjectMessage, SteeringDirective, UpdateSystemPrompt
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


class AISteeringMixin(_AIChannelContract):
    """Handles steering directives that modify a running tool loop."""

    channel_id: str
    _active_loops: dict[str, _ToolLoopContext]

    def _get_loop_ctx(self) -> _ToolLoopContext:
        """Get the current tool loop context (from contextvar or create default)."""
        ctx = _current_loop_ctx.get()
        if ctx is None:
            # Fallback for code paths outside a tool loop
            ctx = _ToolLoopContext()
        return ctx

    def steer(
        self,
        directive: SteeringDirective,
        *,
        loop_id: str | None = None,
        room_id: str | None = None,
    ) -> int:
        """Enqueue a steering directive for the tool loops it addresses (RFC §21.3).

        Safe to call from any coroutine. Cancel directives also set the
        fast-path cancel event so the loop can exit without waiting for
        the next drain point. One channel object serves every room it is
        bound to, so a host acting for one room addresses that room.

        Args:
            directive: The steering directive to enqueue.
            loop_id: The loop to target.
            room_id: The room whose loops to target: a ``Cancel`` reaches
                every loop of the room, any other directive the room's most
                recent one, and never a loop of another room.

        With neither, the directive reaches the most recently started loop,
        whatever its room.

        Returns:
            How many loops the directive reached. A loop is reachable once
            its turn has started (its response stream is read), so a
            directive that comes before reaches none.

        Raises:
            ValueError: *loop_id* and *room_id* both given.
        """
        if loop_id is not None and room_id is not None:
            raise ValueError("steer() addresses a loop_id or a room_id, not both")
        targets = self._steering_targets(directive, loop_id, room_id)
        if not targets:
            if room_id is None:
                logger.warning("steer() called with no active tool loop")
            else:
                logger.info("steer(): no active tool loop in room %s", room_id)
            return 0
        for ctx in targets:
            ctx.steering_queue.put_nowait(directive)
            if isinstance(directive, Cancel):
                ctx.cancel_event.set()
        return len(targets)

    async def _end_running_turns(self) -> None:
        """Cut every turn still running, as the channel closes (RFC §9.3):
        the calls it runs are cancelled and reported cancelled, no further
        round is asked, and the close waits for the turns to end, at most
        :data:`CLOSE_WAIT_S`. A close from inside a turn (one of its calls,
        one of its hooks, a task they started) spares the current task and the
        realtime call whose handler is closing (a reasoning backend's agent,
        RFC §12.4), and does not wait for that turn; a turn an earlier close
        cut is not cut or waited for again."""
        spared = {asyncio.current_task(), *held_by(ending_cause())}
        ending = [
            ctx.ended
            for ctx in list(self._active_loops.values())
            if not ctx.closing and self._cut_turn(ctx, spared)
        ]
        if not ending:
            return
        waits = asyncio.gather(*(ended.wait() for ended in ending))
        try:
            await asyncio.wait_for(waits, CLOSE_WAIT_S)
        except TimeoutError:
            logger.warning(
                "Channel %s closed with %d turn(s) still running",
                self.channel_id,
                sum(not ended.is_set() for ended in ending),
            )

    @staticmethod
    def _cut_turn(ctx: _ToolLoopContext, spared: set[asyncio.Task[Any] | None]) -> bool:
        """Cut one turn: no further round, its calls cancelled but the *spared*
        tasks. Whether the close waits for it: not when it holds one of them,
        or runs the close."""
        ctx.closing = True
        ctx.cancel_event.set()
        holds_spared = False
        for task in list(ctx.cancellable):
            if task in spared:
                holds_spared = True
            else:
                task.cancel()
        return not holds_spared and _current_loop_ctx.get() is not ctx

    def _steering_targets(
        self, directive: SteeringDirective, loop_id: str | None, room_id: str | None
    ) -> list[_ToolLoopContext]:
        """The running loops *directive* reaches, as :meth:`steer` addresses them."""
        if loop_id is not None:
            ctx = self._active_loops.get(loop_id)
            return [ctx] if ctx is not None else []
        loops = list(self._active_loops.values())
        if room_id is None:
            return loops[-1:]
        in_room = [ctx for ctx in loops if ctx.room_id == room_id]
        return in_room if isinstance(directive, Cancel) else in_room[-1:]

    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]:
        """Drain all pending steering directives, applying them to *context*.

        An injected message is the conversation's text: a copy of the turn's
        notes' header it holds is replaced (RFC §6.4).

        Returns:
            (updated_context, should_cancel)
        """
        should_cancel = False
        while not loop_ctx.steering_queue.empty():
            try:
                directive = loop_ctx.steering_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if isinstance(directive, Cancel):
                logger.info("Steering: cancel received — %s", directive.reason)
                should_cancel = True
            elif isinstance(directive, InjectMessage):
                logger.info("Steering: injecting %s message", directive.role)
                content = without_header_copies(directive.content)
                context.messages.append(AIMessage(role=directive.role, content=content))
            elif isinstance(directive, UpdateSystemPrompt):
                logger.info("Steering: appending to system prompt")
                context = context.model_copy(
                    update={"system_prompt": (context.system_prompt or "") + directive.append}
                )

        return context, should_cancel
