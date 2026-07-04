"""Shared per-round decision rules for the AI tool loops (single definition).

The non-streaming loop (``AIGenerationMixin._run_tool_loop``) and the
streaming loop (``AIStreamingMixin._run_streaming_tool_loop``) share the
same business rules per round — force-stop ripcord, bounded empty-retry,
deadline/warn budget, assistant-message assembly, tool execution — but
differ in how a round is *generated* (blocking response vs streamed
deltas). Keeping the rules here, as the single base of both loop mixins,
guarantees a loop rule cannot exist in one path and be missing from the
other — a rule enforced by only one loop would silently cover only the
providers that use that generation mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.providers.ai.base import (
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
)
from roomkit.realtime.base import EphemeralEventType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.providers.ai.base import (
        AIContext,
        AIToolCall,
        AIToolResultPart,
        StreamToolCall,
    )

logger = logging.getLogger("roomkit.channels.ai")


# Corrective nudge re-injected when a generation round ends after tool calls
# without any final text (common with small local models): the tool results
# are in context, the model just failed to verbalize the answer. Re-prompting
# for the final answer recovers it. Bounded by ``max_empty_retries``.
_EMPTY_RETRY_NUDGE = (
    "You called tools and already have their results above. Now write your "
    "final answer to the user in plain text. Do not call any more tools."
)

# Injected when the anti-loop guard force-stops a stuck model. Tools are
# stripped from the next (final) generation so it cannot keep looping.
_FORCE_STOP_NUDGE = (
    "You have repeated the same tool call with identical arguments several "
    "times; it cannot produce anything new and further tool calls are "
    "disabled. Stop now and reply to the user in plain text with a summary of "
    "what you found and what remains, using the results already above."
)


@dataclass
class _ToolLoopState:
    """Per-invocation mutable state for one tool-loop run (either mode)."""

    deadline: float | None
    warn_after: int
    log_label: str
    empty_retries: int = 0
    force_stop_nudged: bool = False

    def deadline_exceeded(self) -> bool:
        """Whether the loop's wall-clock deadline has passed."""
        return self.deadline is not None and asyncio.get_running_loop().time() >= self.deadline

    def warn_if_needed(self, round_idx: int) -> None:
        """Log the soft budget warning when the loop hits ``warn_after`` rounds."""
        if round_idx == self.warn_after:
            logger.warning("%s reached %d rounds, still running", self.log_label, round_idx)


@runtime_checkable
class AIToolLoopRulesHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolLoopRulesMixin.

    Attributes provided by the host's ``__init__``:
        _tool_loop_timeout_seconds: Optional wall-clock timeout for the loop.
        _tool_loop_warn_after: Log a warning after this many rounds.
        _max_empty_retries: Bound for the empty-response re-prompt.

    Methods provided by other mixins:
        _apply_tool_filters: ``AIToolPolicyMixin`` — apply policy + gating filters.
        _publish_tool_event: ``AIEventsMixin`` — publish tool call events.
        _execute_tools_parallel: ``AIToolsMixin`` — execute tool calls concurrently.
    """

    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int

    def _apply_tool_filters(self, tools: list[Any]) -> list[Any]: ...
    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = ...,
    ) -> None: ...
    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        parent_span_id: str | None = ...,
    ) -> list[_ContentPart]: ...


class AIToolLoopRulesMixin:
    """Single-definition loop rules shared by both tool loops.

    Host contract: :class:`AIToolLoopRulesHost`.
    """

    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    _apply_tool_filters: Any  # see AIToolLoopRulesHost
    _publish_tool_event: Any  # see AIToolLoopRulesHost
    _execute_tools_parallel: Any  # see AIToolLoopRulesHost

    def _new_loop_state(self, log_label: str) -> _ToolLoopState:
        """Create the per-run loop state, computing the wall-clock deadline."""
        deadline = (
            asyncio.get_running_loop().time() + self._tool_loop_timeout_seconds
            if self._tool_loop_timeout_seconds
            else None
        )
        return _ToolLoopState(
            deadline=deadline,
            warn_after=self._tool_loop_warn_after,
            log_label=log_label,
        )

    def _prepare_round_context(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        state: _ToolLoopState,
        round_idx: int,
    ) -> AIContext:
        """Force-stop ripcord (nudge once + strip tools) or per-round tool re-filter.

        When the anti-loop guard set ``force_stop`` (the model keeps re-issuing
        a blocked identical call), inject the corrective nudge once and strip
        tools so the next generation must produce a plain-text answer.
        Otherwise re-apply the tool policy filters for the next round.
        """
        if loop_ctx.force_stop:
            if not state.force_stop_nudged:
                logger.warning("%s anti-loop force-stop at round %d", state.log_label, round_idx)
                context.messages.append(AIMessage(role="user", content=_FORCE_STOP_NUDGE))
                state.force_stop_nudged = True
            return context.model_copy(update={"tools": []})
        if loop_ctx.all_context_tools:
            return context.model_copy(
                update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
            )
        return context

    def _try_empty_retry(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        state: _ToolLoopState,
        *,
        had_tool_round: bool,
        final_text: str,
    ) -> bool:
        """Bounded re-prompt when the final answer is empty after tool rounds.

        Returns ``True`` when the caller should re-generate: the nudge has
        been appended and the retry counted. The deadline term is evaluated
        last so no clock read happens when an earlier term already fails.
        """
        if not (
            had_tool_round
            and not final_text.strip()
            and state.empty_retries < self._max_empty_retries
            and not loop_ctx.cancel_event.is_set()
            and not state.deadline_exceeded()
        ):
            return False
        state.empty_retries += 1
        logger.warning(
            "%s: empty response after tool round(s); re-prompting for final answer (retry %d/%d)",
            state.log_label,
            state.empty_retries,
            self._max_empty_retries,
        )
        context.messages.append(AIMessage(role="user", content=_EMPTY_RETRY_NUDGE))
        return True

    @staticmethod
    def _build_assistant_parts(
        thinking: str,
        signature: str | None,
        text: str,
        tool_calls: Sequence[AIToolCall | StreamToolCall],
    ) -> list[_ContentPart]:
        """Assemble the assistant message parts for a tool round."""
        parts: list[_ContentPart] = []
        if thinking or signature:
            parts.append(AIThinkingPart(thinking=thinking, signature=signature))
        if text:
            parts.append(AITextPart(text=text))
        for tc in tool_calls:
            parts.append(
                AIToolCallPart(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    metadata=tc.metadata,
                )
            )
        return parts

    async def _execute_round_tools(
        self,
        context: AIContext,
        tool_calls: list[Any],
        telemetry: Any,
        room_id: str | None,
        round_idx: int,
        *,
        parent_span_id: str | None = None,
    ) -> tuple[list[AIToolResultPart], int]:
        """Publish TOOL_CALL_START, execute the calls, append the tool message.

        The TOOL_CALL_END publish (and, in streaming, the persistence
        markers) stays at the call sites — their relative order around this
        helper differs legitimately between the two loops.
        """
        if room_id:
            await self._publish_tool_event(
                EphemeralEventType.TOOL_CALL_START,
                room_id,
                tool_calls,
                round_idx,
            )
        t0 = time.monotonic()
        result_parts = await self._execute_tools_parallel(
            tool_calls, telemetry, parent_span_id=parent_span_id
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        context.messages.append(AIMessage(role="tool", content=result_parts))
        return result_parts, duration_ms
