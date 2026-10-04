"""Per-round decision rules for the AI tool loop.

The rules the loop (``AIStreamingMixin._run_streaming_tool_loop``) applies
to every round — force-stop ripcord, bounded empty-retry, deadline/warn
budget, assistant-message assembly, tool execution — apart from how a round
is generated and streamed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from roomkit.channels._tool_eviction import REREAD_TOOL, ToolEviction
from roomkit.channels._turn_budget import TurnBudget
from roomkit.models.streaming import LoopEndReason
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.ai.tool_calls import (
    is_malformed_call,
    is_natural_stop,
    is_truncation,
    malformed_call_nudge,
)
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools._outcome import OutcomeKind, ToolOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.models.tool_call import ContinuationPolicy
    from roomkit.providers.ai.base import (
        AIContext,
    )
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.tools.context import _ToolLoopContext

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


# Corrective nudge re-injected when a generation round ends after tool calls
# without any final text (common with small local models): the tool results
# are in context, the model just failed to verbalize the answer. Re-prompting
# for the final answer recovers it. Bounded by ``max_empty_retries``.
_EMPTY_RETRY_NUDGE = (
    "You called tools and already have their results above. Now write your "
    "final answer to the user in plain text. Do not call any more tools."
)

# Injected when the anti-loop guard force-stops a stuck model. The next
# generation is the last: none of its calls runs.
_FORCE_STOP_NUDGE = (
    "You have repeated the same tool call with identical arguments several "
    "times; it cannot produce anything new and further tool calls are "
    "disabled. Stop now and reply to the user in plain text with a summary of "
    "what you found and what remains, using the results already above. "
    "A rejected tool call did not execute. Report its failure and the correction "
    "needed; never claim an action succeeded without a successful tool result."
)


def _accumulate_usage(total: dict[str, int], round_usage: dict[str, Any]) -> None:
    """Add one round's token counters into a turn's running totals.

    Every integer counter is carried, not just input and output: cache reads
    and writes are what tell a re-read prefix apart from fresh input, and the
    two are billed an order of magnitude apart. A turn that only reports its
    final round's usage under-counts a multi-round loop by every round but
    the last.
    """
    for counter, value in round_usage.items():
        if isinstance(value, int):
            total[counter] = total.get(counter, 0) + value


# How a tool loop can stop short of its final answer: every end but
# ``completed``, and ``cancelled``, which someone chose. A turn constrained to
# a response schema that ends this way has no checked document to deliver;
# ``unfinished`` included, since the host's own policy judged its last text no
# answer.
_CUT_SHORT: frozenset[str] = frozenset(
    {
        "max_rounds",
        "timeout",
        "budget_exceeded",
        "force_stopped",
        "empty_response",
        "truncated",
        "unfinished",
        "error",
    }
)


def require_schema_answer(context: AIContext, reason: LoopEndReason) -> None:
    """Fail a turn constrained to a response schema that ended without its answer.

    Its last text is the model's narration of a tool round, not the document
    (RFC §6.7), so the turn raises rather than delivering it. A cancelled turn
    is left alone: someone stopped it on purpose.

    Raises:
        ResponseSchemaError: ``truncated``, naming why the loop stopped.
    """
    if context.response_schema is not None and reason in _CUT_SHORT:
        raise ResponseSchemaError(
            f"the tool loop stopped ({reason}) before a final answer in the response schema",
            reason="truncated",
        )


def interrupts_turn(exc: ProviderError, *, after_round: bool) -> bool:
    """Whether a provider error interrupts the turn rather than failing it.

    Once a round ran, its calls and text already reached the room: the turn
    is kept, ends ``error`` and is reported (RFC §6.4). A schema check that
    refuses the final answer is the answer failing, not an interruption
    (RFC A.9).
    """
    return after_round and not isinstance(exc, ResponseSchemaError)


def turn_span_status(reason: LoopEndReason) -> str:
    """The status of the ``llm.generate`` span of a turn that reached its end.

    A turn cancelled between rounds ends ``cancelled``, one the provider
    interrupted after a round ends ``error``: neither is ``ok`` (RFC §6.4).
    """
    if reason in ("cancelled", "error"):
        return reason
    return "ok"


def final_round_reason(
    *,
    had_tool_round: bool,
    final_text: str,
    finish_reason: str | None,
    limit: LoopEndReason | None,
    force_stopped: bool = False,
    unfinished: bool = False,
) -> LoopEndReason:
    """Why a loop that reached its final-answer round is stopping there.

    The round produced no tool calls, so the loop ends here either way; what
    differs is whether the model answered. Ordered by the fix the reader
    would apply: raise the cap, raise the budget, change model. Text — or a
    turn that ran no tool at all — is a plain completion.

    ``force_stopped`` outranks the text, and is the reason this parameter
    exists: the anti-loop ripcord reaches this round precisely by demanding
    prose (and running none of the round's calls), so its exit produced text
    and read as ``completed`` — the one loop-cut this function could not tell
    from an answer. A caller then delivered a cut turn's summary as the result.

    *limit* is the loop's own limit the turn has passed (its deadline, its
    budget): a round that failed to answer past one ends on that limit.

    *unfinished* is the channel's continuation policy still asking once no
    continuation may run: an answer that did not act ends ``unfinished`` (or
    on the limit that stopped it), never ``completed`` (RFC §6.4).

    The other exits (round cap, a limit at a round boundary, cancellation)
    are named at their own ``return``: they know their reason without asking.
    """
    if force_stopped:
        return "force_stopped"
    if is_malformed_call(finish_reason):
        # Its call never ran: no answer, whatever the round said (RFC §6.4).
        return limit or "empty_response"
    if unfinished:
        return limit or "unfinished"
    if final_text.strip() or not had_tool_round:
        return "completed"
    if is_truncation(finish_reason):
        return "truncated"
    if limit is not None:
        return limit
    return "empty_response"


def _empty_round_nudge(
    *, had_tool_round: bool, final_text: str, finish_reason: str | None, log_label: str
) -> str | None:
    """What to tell a model whose round ended with no call to run, or ``None``
    to end the turn there.

    A call the provider could not parse never reached the loop: the model is
    told it did not run, on any round and whatever the round said, so it can
    issue it again (RFC §6.4). An empty answer after tool rounds gets the
    plain nudge. A truncated round is a different failure and is not retried:
    it ran out of room, its output cap (typically a reasoning model that spent
    the whole cap thinking) or its context window, and the same room runs out
    again.
    """
    if is_malformed_call(finish_reason):
        return malformed_call_nudge(finish_reason)
    if final_text.strip() or not had_tool_round:
        return None
    if is_truncation(finish_reason):
        logger.warning(
            "%s: response truncated before any final text (finish_reason=%s). "
            "Raise max_tokens or shorten the context, or disable the model's "
            "reasoning block if it is consuming the budget.",
            log_label,
            finish_reason,
        )
        return None
    return _EMPTY_RETRY_NUDGE


@dataclass
class _ToolLoopState:
    """Per-invocation mutable state for one tool-loop run (either mode)."""

    deadline: float | None
    warn_after: int
    log_label: str
    timeout_seconds: float | None = None
    budget: TurnBudget | None = None
    billed_tokens: int = 0
    spent: float = 0.0
    empty_retries: int = 0
    force_stop_nudged: bool = False

    def count(self, total: dict[str, int], usage: dict[str, Any]) -> None:
        """Count one generation's usage: into the turn's *total*, and against
        its budget when it has one."""
        _accumulate_usage(total, usage)
        if self.budget is not None:
            self.billed_tokens += self.budget.tokens_of(usage)
            self.spent += self.budget.cost_of(usage)

    def deadline_exceeded(self) -> bool:
        """Whether the loop's wall-clock deadline has passed."""
        return self.deadline is not None and asyncio.get_running_loop().time() >= self.deadline

    def limit_passed(self) -> LoopEndReason | None:
        """The limit of the loop's own the turn has reached, or ``None``: its
        wall-clock deadline, then its budget (RFC §6.4). Past either, the loop
        asks for no further generation."""
        if self.deadline_exceeded():
            return "timeout"
        if self.budget is not None and self.budget.reached(self.billed_tokens, self.spent):
            return "budget_exceeded"
        return None

    def limit_reached(self, rounds: int) -> LoopEndReason | None:
        """The limit passed at a round boundary after *rounds* tool rounds ran,
        logged. The loop asks it there, before running the round's calls."""
        limit = self.limit_passed()
        if limit == "timeout":
            logger.warning(
                "%s timeout after %d tool rounds (%.0fs)",
                self.log_label,
                rounds,
                self.timeout_seconds,
            )
        elif limit == "budget_exceeded":
            logger.warning(
                "%s reached the turn's budget after %d tool rounds (%d tokens, %.4f)",
                self.log_label,
                rounds,
                self.billed_tokens,
                self.spent,
            )
        return limit

    def warn_if_needed(self, rounds: int) -> None:
        """Log the soft warning once *rounds* tool rounds have run, at ``warn_after``."""
        if rounds == self.warn_after:
            logger.warning("%s reached %d tool rounds, still running", self.log_label, rounds)


def _aborted_results(tool_calls: list[Any]) -> list[AIToolResultPart]:
    """A failed result for each call of a round that was aborted mid-run."""
    body = json.dumps({"error": "Tool call aborted"})
    return [ToolOutcome(OutcomeKind.CANCELLED, body).as_part(tc.id, tc.name) for tc in tool_calls]


class AIToolLoopRulesMixin(_AIChannelContract):
    """The tool loop's rules, each defined once.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _tool_loop_timeout_seconds: float | None
    _tool_loop_warn_after: int
    _max_empty_retries: int
    _continuation: ContinuationPolicy | None
    _eviction: ToolEviction

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

    def _new_loop_state(
        self, log_label: str, timeout_seconds: float | None, budget: TurnBudget | None = None
    ) -> _ToolLoopState:
        """Create the per-run loop state, its wall-clock deadline
        *timeout_seconds* from now (``None``: no deadline)."""
        deadline = (
            asyncio.get_running_loop().time() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        return _ToolLoopState(
            deadline=deadline,
            warn_after=self._tool_loop_warn_after,
            log_label=log_label,
            timeout_seconds=timeout_seconds,
            budget=budget,
        )

    def _prepare_round_context(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        state: _ToolLoopState,
        round_idx: int,
    ) -> AIContext:
        """The round's context: its tools re-filtered, and the force-stop nudge
        once the anti-loop guard has pulled the ripcord.

        The declaration holds from round to round (RFC §6.4): a provider
        caches a request as a prefix, tools first, and one that gains, loses
        or reorders a tool is billed as if nothing were cached. So the forced
        final round keeps its tools, told that no further call will run (the
        loop ends there, running none of its calls), and a tool shows up only
        where one must: a reveal, a skill activation. On a provider that holds
        tools unseen, round 0 also fixes the turn's declaration from the
        room's, and may place before the turn's input the exchange that
        reopens what it holds (``_open_turn_declaration``).
        """
        if loop_ctx.force_stop and not state.force_stop_nudged:
            logger.warning("%s anti-loop force-stop at round %d", state.log_label, round_idx)
            context.messages.append(AIMessage(role="user", content=_FORCE_STOP_NUDGE))
            state.force_stop_nudged = True

        # An empty resolved toolset is a real one (``None`` means the loop was
        # built without context): its re-filter declares nothing.
        shown = (
            self._apply_tool_filters(loop_ctx.all_context_tools)
            if loop_ctx.all_context_tools is not None
            else list(context.tools or [])
        )
        if loop_ctx.first_shown is None:
            self._open_turn_declaration(context, loop_ctx, shown)
        tools = self._held_declaration(loop_ctx, shown)
        # ``_build_context`` declares the large-result re-read from the first
        # round; a loop built without it gets it here, as the preview of a
        # result evicted mid-loop tells the model to page it back with it.
        if REREAD_TOOL not in loop_ctx.withdrawn_tools:
            tools = self._eviction.with_reread_tool(tools)
        return context.model_copy(update={"tools": tools})

    def _try_round_again(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        state: _ToolLoopState,
        *,
        had_tool_round: bool,
        final_text: str,
        finish_reason: str | None = None,
    ) -> Literal["retry", "unfinished"] | None:
        """Another try at a round that ended without a call to run (RFC §6.4).

        An empty answer after tool rounds, or a call that could not be parsed,
        is tried again with the empty round's nudge; an answer the channel's
        continuation policy finds unfinished, with the policy's instruction.
        Both share one bound. Returns ``"retry"`` when the caller should
        re-generate (the instruction appended, the try counted),
        ``"unfinished"`` when the policy still asks and no try may run, and
        ``None`` when the round stands. The deadline term is evaluated last so
        no clock read happens when an earlier term already fails.
        """
        nudge = _empty_round_nudge(
            had_tool_round=had_tool_round,
            final_text=final_text,
            finish_reason=finish_reason,
            log_label=state.log_label,
        )
        continuation = nudge is None
        if continuation:
            nudge = self._continuation_nudge(context, final_text, finish_reason)
        if nudge is None:
            return None
        if not (
            state.empty_retries < self._max_empty_retries
            and not loop_ctx.cancel_event.is_set()
            and state.limit_passed() is None
        ):
            return "unfinished" if continuation else None
        state.empty_retries += 1
        logger.warning(
            "%s: %s (finish_reason=%s); re-prompting (retry %d/%d)",
            state.log_label,
            "the continuation policy asked to go on" if continuation else "no answer",
            finish_reason,
            state.empty_retries,
            self._max_empty_retries,
        )
        if final_text.strip():
            # What the model said before its call failed to parse, or the
            # answer the policy continues: it goes on from there.
            context.messages.append(AIMessage(role="assistant", content=final_text))
        context.messages.append(AIMessage(role="user", content=nudge))
        return "retry"

    def _continuation_nudge(
        self, context: AIContext, final_text: str, finish_reason: str | None
    ) -> str | None:
        """What the channel's continuation policy says of a round the model
        ended itself on text, without a call, a tool declared: the instruction
        to go on, or ``None`` when the answer stands (RFC §6.4).

        A cancelled or force-stopped round never reaches it: the loop ends on
        those before it asks for another try.
        """
        policy = self._continuation
        if policy is None or not context.tools:
            return None
        if not final_text.strip() or not is_natural_stop(finish_reason):
            return None
        return policy(final_text)

    async def _execute_round_tools(
        self,
        context: AIContext,
        tool_calls: list[Any],
        telemetry: TelemetryProvider,
        room_id: str | None,
        round_idx: int,
        *,
        parent_span_id: str | None = None,
        answered: Sequence[AIToolResultPart] = (),
    ) -> tuple[list[AIToolResultPart], int]:
        """Publish TOOL_CALL_START, execute the calls, append the tool message,
        the results of the round's calls the provider served (*answered*) first.

        The TOOL_CALL_END publish and the persistence markers stay with the
        caller, which orders them around this helper.
        """
        if room_id:
            await self._publish_tool_event(
                EphemeralEventType.TOOL_CALL_START,
                room_id,
                tool_calls,
                round_idx,
            )
        t0 = time.monotonic()
        try:
            result_parts = await self._execute_tools_parallel(
                tool_calls,
                telemetry,
                declared_tools=context.tools,
                parent_span_id=parent_span_id,
            )
        except BaseException:
            # Aborted mid-round (a turn cancelled while a tool ran): the START
            # published above gets its END, or a live surface spins forever.
            if room_id:
                await self._publish_tool_event(
                    EphemeralEventType.TOOL_CALL_END,
                    room_id,
                    _aborted_results(tool_calls),
                    round_idx,
                )
            raise
        duration_ms = int((time.monotonic() - t0) * 1000)
        context.messages.append(AIMessage(role="tool", content=[*answered, *result_parts]))
        return result_parts, duration_ms
