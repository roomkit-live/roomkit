"""AIChannel mixin for mid-run steering (cancel, inject, update prompt)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.models.steering import Cancel, InjectMessage, SteeringDirective, UpdateSystemPrompt
from roomkit.providers.ai.base import AIContext, AIMessage

if TYPE_CHECKING:
    from roomkit.channels.ai import _ToolLoopContext

logger = logging.getLogger("roomkit.channels.ai")


class AISteeringMixin:
    """Handles steering directives that modify a running tool loop."""

    _active_loops: dict[str, _ToolLoopContext]

    def _get_loop_ctx(self) -> _ToolLoopContext:
        """Get the current tool loop context (from contextvar or create default)."""
        from roomkit.channels.ai import _current_loop_ctx, _ToolLoopContext

        ctx = _current_loop_ctx.get()
        if ctx is None:
            # Fallback for code paths outside a tool loop
            ctx = _ToolLoopContext()
        return ctx

    def steer(self, directive: SteeringDirective, *, loop_id: str | None = None) -> None:
        """Enqueue a steering directive for the active tool loop.

        Safe to call from any coroutine. Cancel directives also set the
        fast-path cancel event so the loop can exit without waiting for
        the next drain point.

        Args:
            directive: The steering directive to enqueue.
            loop_id: Optional loop ID to target. If ``None``, targets the
                most recently started active loop.
        """
        if loop_id is not None:
            ctx = self._active_loops.get(loop_id)
        elif self._active_loops:
            ctx = next(reversed(self._active_loops.values()))
        else:
            ctx = None
        if ctx is None:
            logger.warning("steer() called with no active tool loop")
            return
        ctx.steering_queue.put_nowait(directive)
        if isinstance(directive, Cancel):
            ctx.cancel_event.set()

    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]:
        """Drain all pending steering directives, applying them to *context*.

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
                context.messages.append(AIMessage(role=directive.role, content=directive.content))
            elif isinstance(directive, UpdateSystemPrompt):
                logger.info("Steering: appending to system prompt")
                context = context.model_copy(
                    update={"system_prompt": (context.system_prompt or "") + directive.append}
                )

        return context, should_cancel
