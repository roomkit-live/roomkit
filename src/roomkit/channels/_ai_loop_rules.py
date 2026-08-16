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

from roomkit.channels._tool_eviction import ToolEviction
from roomkit.models.streaming import LoopEndReason
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

# Every provider reports "I hit the output cap" in its own vocabulary, and
# RoomKit forwards the raw value rather than inventing a normalized one:
# OpenAI-compatible servers and Ollama's ``done_reason`` say ``length``,
# Anthropic's ``stop_reason`` says ``max_tokens``, Gemini's candidate says
# ``MAX_TOKENS``. A rule that knew only one spelling would cover only the
# providers using it, which is the failure mode this module exists to prevent.
_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


def _is_truncation(finish_reason: str | None) -> bool:
    """Whether a round ended by exhausting its output budget.

    Compared case-insensitively so Gemini's ``MAX_TOKENS`` and Anthropic's
    ``max_tokens`` are one entry rather than two.
    """
    return finish_reason is not None and finish_reason.lower() in _TRUNCATION_FINISH_REASONS


def final_round_reason(
    *,
    had_tool_round: bool,
    final_text: str,
    finish_reason: str | None,
    deadline_exceeded: bool,
) -> LoopEndReason:
    """Why a loop that reached its final-answer round is stopping there.

    The round produced no tool calls, so the loop ends here either way; what
    differs is whether the model answered. Ordered by the fix the reader
    would apply: raise the cap, raise the budget, change model. Text — or a
    turn that ran no tool at all — is a plain completion.

    The other exits (round cap, deadline at a round boundary, cancellation)
    are named at their own ``return``: they know their reason without asking.
    """
    if final_text.strip() or not had_tool_round:
        return "completed"
    if _is_truncation(finish_reason):
        return "truncated"
    if deadline_exceeded:
        return "timeout"
    return "empty_response"


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
    _eviction: ToolEviction

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
        declared_tools: list[Any] | None = ...,
        parent_span_id: str | None = ...,
        executed_arguments: dict[str, dict[str, Any]] | None = ...,
    ) -> list[_ContentPart]: ...


class AIToolLoopRulesMixin:
    """Single-definition loop rules shared by both tool loops.

    Host contract: :class:`AIToolLoopRulesHost`.
    """

    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _eviction: ToolEviction

    # Cross-mixin methods — Any annotations avoid MRO shadowing.
    _apply_tool_filters: Any  # see AIToolLoopRulesHost
    _publish_tool_event: Any  # see AIToolLoopRulesHost
    _execute_tools_parallel: Any  # see AIToolLoopRulesHost

    # Ceiling on the tool calls honoured from ONE generation. The loop already
    # bounds rounds, wall clock, identical repeats and result size; a single
    # round was the one unbounded axis, and a model can saturate its whole
    # output budget with tool calls. Observed: a 27B local model emitted 164
    # calls in one completion (154 of them byte-identical) until it hit
    # max_tokens, which cost 164 executions and 328 room events for one turn.
    # 32 is far above legitimate parallel fan-out (a strong model issues a
    # handful) and far below a degenerate run.
    _MAX_TOOL_CALLS_PER_ROUND = 32

    def _cap_round_tool_calls(self, tool_calls: list[Any], log_label: str) -> list[Any]:
        """Truncate a round's tool calls to ``_MAX_TOOL_CALLS_PER_ROUND``.

        Applied BEFORE the assistant message is assembled, so the transcript
        stays internally consistent: a dropped call is absent from the
        assistant message as well as from the results, and no provider sees a
        tool call with no matching result. The drop is deliberately invisible
        to the model — the calls it keeps are its own, in its own order — and
        loud in the log, which is where an operator diagnoses a looping model.
        """
        if len(tool_calls) <= self._MAX_TOOL_CALLS_PER_ROUND:
            return tool_calls
        kept = tool_calls[: self._MAX_TOOL_CALLS_PER_ROUND]
        dropped = tool_calls[self._MAX_TOOL_CALLS_PER_ROUND :]
        logger.warning(
            "%s: round requested %d tool calls, capped at %d — dropped %d (%s). "
            "A model emitting this many calls in one generation is looping.",
            log_label,
            len(tool_calls),
            self._MAX_TOOL_CALLS_PER_ROUND,
            len(dropped),
            ", ".join(sorted({tc.name for tc in dropped})),
        )
        return kept

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

        tools: list[Any] | None = None
        if loop_ctx.all_context_tools:
            tools = self._apply_tool_filters(loop_ctx.all_context_tools)

        # A result evicted mid-loop replaces itself with a preview that tells
        # the model to page the full output back with ``read_stored_result`` —
        # but that tool was only ever injected at ``_build_context`` time, i.e.
        # on the *next inbound event*, and every round here re-filters from the
        # frozen ``all_context_tools`` snapshot. So the tool was unreachable in
        # the very turn whose preview recommends it, and a one-shot automation
        # run (webhook, schedule) has no next event at all: its evicted content
        # was simply lost. Inject the definition per round instead — the
        # dispatch table already accepts the call unconditionally.
        if self._eviction.has_evicted:
            current = tools if tools is not None else list(context.tools or [])
            if all(t.name != "read_stored_result" for t in current):
                tools = [*current, ToolEviction.tool_definition()]

        if tools is not None:
            return context.model_copy(update={"tools": tools})
        return context

    def _try_empty_retry(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        state: _ToolLoopState,
        *,
        had_tool_round: bool,
        final_text: str,
        finish_reason: str | None = None,
    ) -> bool:
        """Bounded re-prompt when the final answer is empty after tool rounds.

        Returns ``True`` when the caller should re-generate: the nudge has
        been appended and the retry counted. The deadline term is evaluated
        last so no clock read happens when an earlier term already fails.

        A truncated round (see ``_is_truncation``) is a different failure and
        is not retried: the round did not fall silent, it ran out of output
        budget — typically a reasoning model that spent the whole cap inside
        its thinking block, so ``content`` arrives empty. Re-prompting under
        the same cap truncates again, so the nudge is skipped in favour of a
        log line naming the actual cause.
        """
        if had_tool_round and not final_text.strip() and _is_truncation(finish_reason):
            logger.warning(
                "%s: response truncated at the output cap before any final text "
                "(finish_reason=%s). Raise max_tokens, or disable the model's "
                "reasoning block if it is consuming the budget.",
                state.log_label,
                finish_reason,
            )
            return False
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
    ) -> tuple[list[AIToolResultPart], int, dict[str, dict[str, Any]]]:
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
        executed_arguments: dict[str, dict[str, Any]] = {}
        result_parts = await self._execute_tools_parallel(
            tool_calls,
            telemetry,
            declared_tools=context.tools,
            parent_span_id=parent_span_id,
            executed_arguments=executed_arguments,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        context.messages.append(AIMessage(role="tool", content=result_parts))
        return result_parts, duration_ms, executed_arguments
