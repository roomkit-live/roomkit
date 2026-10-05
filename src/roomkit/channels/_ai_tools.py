"""AIChannel mixin for tool execution, dispatch, and skill tool handlers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_stream_external_tools import report_cut
from roomkit.channels._sandbox_handlers import handle_sandbox_command
from roomkit.channels._served_tools import CollisionLog, declared_once
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    ALREADY_ACTIVE_NOTE,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
)
from roomkit.channels._skill_handlers import (
    activation_ack,
    handle_activate_skill,
    handle_read_reference,
    handle_run_script,
    tools_hint,
)
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tool_eviction import ToolEviction, kept_whole
from roomkit.channels._tool_registry import (
    ChannelRegistry,
    ToolSource,
    channel_tool,
    schema_tool,
)
from roomkit.channels._tool_search import (
    normalize_max_results,
    related_family_tools,
    render_find_payload,
    render_list_payload,
    search_catalogue,
    search_tool_defs,
)
from roomkit.channels._tool_search_constants import (
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)
from roomkit.core.exceptions import (
    ChannelRefusalError,
    ToolFailedError,
    ToolRefusedError,
    UnservedToolCallError,
)
from roomkit.models.enums import ChannelType
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.models.tool_call import ToolCallEvent, ToolCallVerdict
from roomkit.providers.ai.base import (
    AIImagePart,
    AIProvider,
    AITextPart,
    AITool,
    AIToolResultPart,
)
from roomkit.providers.ai.tool_calls import partial_call_error
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX, TOOL_SANDBOX_BASH
from roomkit.skills.models import missing_required_tools, missing_tools_error
from roomkit.telemetry.base import SpanKind, TelemetryProvider
from roomkit.telemetry.redaction import redact
from roomkit.tools._outcome import OutcomeKind, ToolOutcome, kept_in_tool_memory, read_outcome
from roomkit.tools._turn_calls import AnnouncedCall, reporting
from roomkit.tools.context import ToolCallContext, _current_tool_call
from roomkit.tools.result import (
    GateRefusal,
    as_tool_result,
    call_id_in_flight_error,
    cancelled_tool_error,
    declined_answer,
    failure_detail,
    pre_execution_denial,
    read_tool_call_verdict,
    result_text,
    tool_failure,
    unknown_tool_error,
    unserved_tool_error,
)
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.tools.validation import (
    fold_hoisted_arguments,
    rewritten_arguments_error,
    validate_tool_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels._task_planner import PlanUpdatedCallback
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.models.tool_call import ToolCallCallback, ToolCallObserver
    from roomkit.providers.ai.base import AIImagePart, AITextPart
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools._human_input_channel import ChannelHumanInput
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.tools.external import BeforeToolCallback, ExternalToolHandler

    ToolResult = str | list[AITextPart | AIImagePart]
    ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


@dataclass(frozen=True)
class _CallRound:
    """What every call of one round shares: its handler, telemetry, room and declarations."""

    handler: Any
    telemetry: TelemetryProvider
    room_id: str | None
    declared_tools: list[AITool] | None
    parent_span_id: str | None


def _refused_with(
    error: dict[str, Any], detail: str | None = None, *, arguments: dict[str, Any] | None = None
) -> GateRefusal:
    """A gate's refusal of a call: *error* as the model reads it, *detail* for
    the observers, *arguments* as the gate had them."""
    return GateRefusal(json.dumps(error), detail, arguments)


def _log_answer(name: str, result: Any, started: float) -> None:
    elapsed = (time.monotonic() - started) * 1000
    if result is None:
        logger.info("Tool %s: nothing served it (%.0f ms)", name, elapsed)
        return
    size = len(result) if isinstance(result, str) else -1
    logger.info("Tool %s returned %d chars in %.0f ms", name, size, elapsed)
    logger.debug("Tool %s result: %s", name, redact(_preview(result)))


def _tool_name(tool: AITool) -> str:
    return tool.name


def _partial_call_error(tc: Any) -> dict[str, Any]:
    """What the model reads for a call whose arguments do not read."""
    garbled = getattr(tc, "garbled", False)
    logger.warning(
        "Provider %s tool call %s (%s): it does not run",
        "sent unreadable arguments for" if garbled else "cut",
        tc.name,
        tc.id,
    )
    return partial_call_error(tc.name, garbled=garbled)


class AIToolsMixin(_AIChannelContract):
    """Parallel tool execution, skill tool definitions, and dispatch routing.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _provider: AIProvider
    _user_tool_handler: ToolHandler | None
    _user_tools: list[AITool]
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _human_input: ChannelHumanInput
    _collisions: CollisionLog
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _realtime: RealtimeBackend | None
    _plan_updated_hook: PlanUpdatedCallback | None
    _tool_call_hook: ToolCallCallback | None
    _tool_observer_hook: ToolCallObserver | None
    _before_tool_call_hook: BeforeToolCallback | None
    _external_tool_handler: ExternalToolHandler | None
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_miss_hint: str | None
    channel_id: str

    def _tool_parameters(
        self, name: str, declared_tools: list[AITool] | None = None
    ) -> dict[str, Any] | None:
        """Return the declared JSON-Schema ``parameters`` for tool *name*.

        ``None`` when the tool's schema is not known to this channel (infra,
        skill, or sandbox tools) — those skip argument validation.
        """
        if declared_tools is None:
            room_id = self._get_loop_ctx().room_id
            declared_tools = [*self._user_tools, *self._orchestration_tools(room_id)]
        for tool in declared_tools:
            if tool.name == name:
                return tool.parameters
        return None

    def _recover_deferred_tool(self, name: str, call_id: str) -> AITool | None:
        """A find_tools reveal applied at call time, for an exact-name call.

        Small models routinely skip the two-step discovery protocol and call a
        catalogue tool they saw (via list_tools, or a prior turn) without
        revealing it first. The name being exact, the call is trivially
        recoverable: let the call proceed, and reveal the tool as find_tools
        would have once the tool answered the call *call_id*, as the room's
        tool memory keeps any tool used (``_settle_recovery``) — provided it
        passes what a reveal is subject to (tool policy, glob-aware skill
        gating). The execution guard applies that rule again to the call, but
        a name it refuses must not even be recovered. A call refused before
        it ran reveals nothing, and touches no other call's reveal.

        Returns the catalogue tool (its schema keeps argument validation
        fail-closed) or ``None`` when the name is not recoverable.
        """
        loop_ctx = self._get_loop_ctx()
        if not loop_ctx.tool_search_active:
            # Inactive search declares the whole filtered catalogue — an
            # undeclared name is either filtered out or unknown, never deferred.
            return None
        tool = next((t for t in loop_ctx.all_context_tools or () if t.name == name), None)
        if tool is None:
            return None
        # What a reveal is subject to, read without revealing: the window is
        # left as it is until the call settles.
        if not self._reachable_tools([tool]):
            return None
        loop_ctx.pending_recoveries[call_id] = name
        return tool

    def _declared_schema(
        self, name: str, call_id: str, declared_tools: list[AITool] | None
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        """The schema a call to *name* is validated against, or why it is undeclared.

        Once the turn's toolset is resolved, a call must name a tool the round
        declared, an empty declaration included, or one Tool Search recovers
        from the turn's catalogue (RFC §6.4); the channel's own tools that
        Tool Search never hides answer for themselves, the person's tools
        included. Its sandbox commands it can hide, so they are recovered and
        validated as a host tool is; one the policy or a skill keeps from the turn is
        left to the gate, which refuses it in its own words. A loop built
        without context (``all_context_tools`` is ``None``) has no
        declaration to hold the call to.
        """
        params = self._tool_parameters(name, declared_tools)
        loop_ctx = self._get_loop_ctx()
        # A tool held unseen that nothing referenced is not callable yet: it
        # goes through recovery like a tool Tool Search hides (RFC §6.4).
        referenced = loop_ctx.referenced
        declared_names = {
            tool.name
            for tool in declared_tools or []
            if not tool.defer_loading or tool.name in referenced
        }
        channel_managed = name in self._channel_tool_names()
        # The channel's own tools answer for themselves, when the turn offers
        # them: Tool Search's only while it hides the catalogue (RFC §6.4).
        offered = {tool.name for tool in loop_ctx.all_context_tools or ()}
        always_shown = (
            channel_managed and name in self._never_hidden(loop_ctx.room_id) and name in offered
        )
        resolved = bool(declared_names) or loop_ctx.all_context_tools is not None
        if not resolved or name in declared_names or always_shown:
            return params, None
        recovered = self._recover_deferred_tool(name, call_id)
        if recovered is None and channel_managed and name in offered:
            # A sandbox command or human-input tool the policy or a skill keeps
            # from the turn: the gate below refuses it, in its own words.
            return params, None
        if recovered is None:
            return None, self._undeclared_tool_error(name)
        # The model skipped find_tools but named a real catalogue tool: the
        # reveal happened at call time instead of ahead of it, and every
        # guard after this one still applies.
        logger.info("Recovered deferred catalogue tool %s at call time", name)
        return recovered.parameters, None

    def _undeclared_tool_error(self, name: str) -> dict[str, str]:
        """Actionable payload for an undeclared call that could not be recovered."""
        loop_ctx = self._get_loop_ctx()
        refusal = self._unavailable_refusal(name)
        if refusal is not None:
            # In the catalogue but kept from the round (tool policy or skill
            # gating): its cause, as every gate words it (RFC §21.1). A
            # find_tools reveal would be dropped by the same filter, so no
            # retry hint: the refusal is the answer. Its gate logs the cause.
            return refusal
        logger.warning("Provider requested undeclared tool %s", name)
        return unknown_tool_error(name, searching=loop_ctx.tool_search_active)

    def _unavailable_refusal(self, name: str) -> dict[str, str] | None:
        """Why a tool of the turn's catalogue the round did not declare is
        refused: as the gate driving the turn words it (a reasoning backend's
        voice session), else by this channel's policy or skill gating; ``None``
        when the catalogue does not hold *name* or nothing refuses it."""
        loop_ctx = self._get_loop_ctx()
        if (cause := loop_ctx.unavailable_tools.get(name)) is not None:
            return {"error": cause}
        if any(t.name == name for t in loop_ctx.all_context_tools or ()):
            return self._gate_refusal(name)
        return None

    async def _fire_tool_refusal(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: str,
        room_id: str | None,
        *,
        detail: str | None = None,
        cancelled: bool = False,
        refused: bool = False,
    ) -> None:
        """Fire ON_TOOL_CALL for a call that failed, was refused or was cancelled.

        *refused* marks a call a gate or its handler refused before it ran
        (``ToolCallEvent.refused``), apart from one that failed.

        *detail* is a raised call's full failure, or the error of a
        BEFORE_TOOL_USE hook that failed closed, for the observers only
        (``ToolCallEvent.error_detail``). *cancelled* marks a call a stop or
        the turn's cancellation interrupted, as every channel marks it.

        The refusal paths below return before the handler runs, and a handler
        that raises jumps past the firing that follows it — so without this,
        ON_TOOL_CALL only ever reported the calls that worked. A host auditing
        tool use saw a denied tool as a tool that was never called, which reads
        the same as an agent that never tried.

        Observational by construction: it reaches the ASYNC observers of
        ON_TOOL_CALL and no further. A SYNC hook is the one that can *serve* a
        call, so a refused call must not reach one — otherwise the refusal
        would hide the side effect instead of preventing it. A hook that raises
        must not turn a refusal into a crash either: the refusal is already on
        its way to the model.
        """
        loop_ctx = self._get_loop_ctx()
        if self._tool_observer_hook is None or loop_ctx.was_reported(tc.id):
            return
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tc.id,
            name=tc.name,
            arguments=arguments,
            result=result,
            room_id=room_id,
            is_error=True,
            cancelled=cancelled,
            refused=refused,
            error_detail=detail,
        )
        try:
            await self._tool_observer_hook(event)
        except Exception:
            logger.debug(
                "ON_TOOL_CALL observation failed for refused tool %s", tc.name, exc_info=True
            )
        # Claimed where the observers heard it; an observer hook that reports
        # nothing has made the call's report all the same.
        loop_ctx.claim_report(tc.id)

    async def _failed_call(
        self,
        tc: Any,
        arguments: dict[str, Any],
        room_id: str | None,
        body: str,
        *,
        detail: str | None = None,
        refused: bool = False,
    ) -> ToolResult:
        """The model's copy of a call that was refused or failed, its observers told.

        Fired on the body before eviction and the repeated-result note shape
        the model's copy of it.
        """
        await self._fire_tool_refusal(tc, arguments, body, room_id, detail=detail, refused=refused)
        return self._bound_tool_result(tc.name, body, tc.id)

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: TelemetryProvider,
        *,
        declared_tools: list[AITool] | None = None,
        parent_span_id: str | None = None,
    ) -> list[AIToolResultPart]:
        """Execute tool calls concurrently and return result parts.

        A channel without a handler still serves its calls through its own
        dispatcher: past the gate, a call nothing serves is unserved, which
        ON_TOOL_CALL's hooks may serve (RFC §9.3).
        """
        # Capture the invocation-scoped room once. The channel object is shared
        # across rooms, while the loop context is copied into every task spawned
        # by gather below.
        scope = _CallRound(
            handler=self._channel_tool_handler,
            telemetry=telemetry,
            room_id=self._get_loop_ctx().room_id,
            declared_tools=declared_tools,
            parent_span_id=parent_span_id,
        )
        calls = self._get_loop_ctx().calls
        tasks = [
            asyncio.create_task(
                self._serve_announced(tc, calls.entry_of(tc) or calls.announce(tc), scope)
            )
            for tc in tool_calls
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # gather propagates a failed gate or a cancelled child immediately;
            # its siblings otherwise keep running after the loop has ended.
            # Own them through cleanup, without cancelling a finalizer twice.
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return list(results)

    async def _serve_announced(
        self, tc: Any, entry: AnnouncedCall, scope: _CallRound
    ) -> AIToolResultPart:
        """One call of a round as a call of its own: its reports are its, a
        duplicate of its id is refused, and its end rides its start marker
        as soon as it is known, so a turn cut later closes it as it ended."""
        started = time.monotonic()
        with reporting(entry):
            if entry.duplicate:
                part = await self._refuse_id_in_flight(tc, scope)
            else:
                part = await self._run_call(tc, scope)
        if entry.marker is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
            entry.marker.ended = call_end_marker(tc, entry.marker, part, duration_ms)
        return part

    async def _run_call(self, tc: Any, scope: _CallRound) -> AIToolResultPart:
        """One call of a round, through its gates to the part the model reads."""
        # INFO names the call and its argument keys; the values can carry
        # personal data: DEBUG shows them only with content logging on.
        logger.info(
            "Executing tool %s (call %s) with %s",
            tc.name,
            tc.id,
            ", ".join(sorted(tc.arguments)) or "no arguments",
        )
        logger.debug("Tool %s arguments: %s", tc.name, redact(_preview(tc.arguments)))
        gated = await self._gate_call(tc, scope)
        if isinstance(gated, AIToolResultPart):
            return gated
        call_arguments, arguments = gated
        return await self._serve_gated_call(tc, call_arguments, arguments, scope)

    async def _refuse_id_in_flight(self, tc: Any, scope: _CallRound) -> AIToolResultPart:
        """The part of a call made under an id an earlier call of its round
        holds: refused before any gate, as a realtime session refuses it,
        and reported as a call of its own (RFC §9.3, §12.4)."""
        body = call_id_in_flight_error(tc.id)
        await self._fire_tool_refusal(tc, tc.arguments, body, scope.room_id, refused=True)
        return ToolOutcome(OutcomeKind.REFUSED, body).as_part(tc.id, tc.name)

    async def _reject_call(
        self, tc: Any, scope: _CallRound, stopped: GateRefusal
    ) -> AIToolResultPart:
        """The part of a call a gate stopped, its observers told."""
        self._settle_recovery(tc.id, kept=False)
        # Refusals never reach the handler's guard. Count their raw
        # attempts here; successful calls are counted only by the
        # handler, using the effective payload after folds and hooks.
        guard = self._repeated_call_guard(tc.name, tc.arguments)
        body = guard or stopped.body
        # Reported with the arguments the gate had, as a call that ran is
        # with those it ran with; the model's when it stopped before them.
        arguments = stopped.arguments if stopped.arguments is not None else tc.arguments
        self._record_call_arguments(tc, arguments, scope)
        await self._fire_tool_refusal(
            tc, arguments, body, scope.room_id, detail=stopped.detail, refused=True
        )
        # Bounded as any outcome the model reads (RFC §21.5): a hook's reason
        # can be as large as a result.
        bounded = self._bound_tool_result(tc.name, body, tc.id)
        return ToolOutcome(OutcomeKind.REFUSED, bounded).as_part(tc.id, tc.name)

    def _record_call_arguments(
        self, tc: Any, arguments: dict[str, Any], scope: _CallRound
    ) -> None:
        """Keep the arguments *tc* runs with, or that its gate had when it
        stopped it, for its END row and for its report if the turn cuts it:
        every report of a call carries those (RFC §9.3). Snapshot before user
        code sees them, so persistence tells the model's request from what
        executed."""
        entry = self._get_loop_ctx().calls.entry_for(tc.id)
        if entry is None:
            return
        entry.arguments = dict(arguments)
        if entry.marker is not None:
            entry.marker.ran_with = dict(arguments)

    async def _gate_call(
        self, tc: Any, scope: _CallRound
    ) -> tuple[dict[str, Any], dict[str, Any]] | AIToolResultPart:
        """The call's arguments as the model sent them (repaired) and as they
        run, or the part of the gate that stopped it."""
        params, stopped = self._declared_schema_gate(tc, scope.declared_tools)
        call_arguments = tc.arguments
        if stopped is None:
            # Execution guard: policy and skill gating, the listing filter's
            # rule (RFC §21.1), re-checked on the call itself before its
            # arguments are read, so a refused tool never names its schema.
            refusal = self._gate_refusal(tc.name)
            stopped = _refused_with(refusal) if refusal is not None else None
        if stopped is None:
            call_arguments, stopped = self._model_arguments(tc, params)
        if stopped is None:
            ran = await self._before_tool_use(tc, call_arguments, params, scope.room_id)
            if not isinstance(ran, GateRefusal):
                return call_arguments, ran
            stopped = ran
        return await self._reject_call(tc, scope, stopped)

    def _declared_schema_gate(
        self, tc: Any, declared_tools: list[AITool] | None
    ) -> tuple[dict[str, Any] | None, GateRefusal | None]:
        """The schema the call is checked against, or why the call cannot run
        at all: its arguments unreadable, withdrawn for the turn, or not declared."""
        if getattr(tc, "partial", False):
            return None, _refused_with(_partial_call_error(tc))
        # A tool withdrawn for the turn (by BEFORE_AI_GENERATION or
        # AFTER_TOOL_ROUND) is gone, the channel's own included: no exemption
        # below may bring it back.
        if tc.name in self._get_loop_ctx().withdrawn_tools:
            logger.warning("Provider called %s, withdrawn for this turn", tc.name)
            return None, _refused_with(
                {"error": f"Tool '{tc.name}' is not available in this turn."}
            )
        # The declared check (fail-closed): the schema the call's arguments
        # are validated against, once the policy and skill gating admit it.
        params, undeclared = self._declared_schema(tc.name, tc.id, declared_tools)
        return params, (_refused_with(undeclared) if undeclared is not None else None)

    def _model_arguments(
        self, tc: Any, params: dict[str, Any] | None
    ) -> tuple[dict[str, Any], GateRefusal | None]:
        """The model's arguments, repaired into the schema's shape, or why
        they do not fit it."""
        call_arguments = tc.arguments
        if params is None:
            return call_arguments, None
        # Repair before validating: a model that flattened a hub tool's
        # ``params`` gets its call folded back into shape instead of
        # spending a round on an error it can only fix by re-issuing.
        folded, fold_error = fold_hoisted_arguments(params, call_arguments)
        if fold_error is not None:
            logger.warning("Tool %s arguments ambiguous: %s", tc.name, fold_error)
            return call_arguments, _refused_with(
                {"error": f"Invalid arguments for '{tc.name}': {fold_error}"}
            )
        if folded is not None:
            logger.info(
                "Tool %s: folded hoisted arguments %s into its container (model=%s)",
                tc.name,
                sorted(set(call_arguments) - set(folded)),
                self._provider.model_name,
            )
            call_arguments = folded
        arg_error = validate_tool_arguments(params, call_arguments)
        if arg_error is not None:
            logger.warning("Tool %s arguments rejected: %s", tc.name, arg_error)
            return call_arguments, _refused_with(
                {"error": f"Invalid arguments for '{tc.name}': {arg_error}"},
                arguments=call_arguments,
            )
        return call_arguments, None

    async def _before_tool_use(
        self,
        tc: Any,
        call_arguments: dict[str, Any],
        params: dict[str, Any] | None,
        room_id: str | None,
    ) -> dict[str, Any] | GateRefusal:
        """The arguments the call runs with once BEFORE_TOOL_USE ran, or its denial."""
        # Pre-execution gate: BEFORE_TOOL_USE hook can deny the tool call,
        # or hand back rewritten arguments (a redaction hook putting real
        # values back before the tool acts on the model's tokenised text).
        # The handler and ON_TOOL_CALL read ``arguments``, so they report
        # what actually ran; the usage record keeps the model's own.
        arguments = call_arguments
        if self._before_tool_call_hook is not None:
            pre_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=tc.id,
                name=tc.name,
                arguments=arguments,
                result=None,
                room_id=room_id,
            )
            decision = await self._before_tool_call_hook(pre_event)
            if not decision:
                logger.info("Tool %s denied by BEFORE_TOOL_USE hook", tc.name)
                denial = pre_execution_denial(tc.name, decision.reason)
                return _refused_with({"error": denial}, decision.detail, arguments=arguments)
            if decision.arguments is not None:
                arguments = decision.arguments

        # Validated again after the hooks, an edit in place included: the
        # event is frozen but its arguments dict is not.
        invalid = rewritten_arguments_error(tc.name, params, arguments)
        if invalid is not None:
            logger.warning("Tool %s: %s", tc.name, invalid)
            return _refused_with({"error": invalid}, arguments=arguments)
        return arguments

    async def _serve_gated_call(
        self,
        tc: Any,
        call_arguments: dict[str, Any],
        arguments: dict[str, Any],
        scope: _CallRound,
    ) -> AIToolResultPart:
        """A call past its gates, served and judged, as the model reads it."""
        self._record_call_arguments(tc, arguments, scope)
        outcome = await self._judged_call(tc, arguments, scope)
        self._settle_served_call(tc.id, served=not outcome.failed)
        kept = kept_in_tool_memory(outcome.kind)
        self._settle_recovery(tc.id, kept=kept)
        references = [] if outcome.failed else self._reference_shown(self._get_loop_ctx())
        if kept:
            self._remember_call(scope.room_id, tc.name, call_arguments, outcome.answer)
        return self._model_part(tc, outcome, references=references)

    async def _judged_call(
        self, tc: Any, arguments: dict[str, Any], scope: _CallRound
    ) -> ToolOutcome:
        """The handler's answer to the call once ON_TOOL_CALL judged it, or
        the call's failure; its result bounded for the model."""
        telemetry = scope.telemetry
        tool_span_id = telemetry.start_span(
            SpanKind.LLM_TOOL_CALL,
            f"tool.{tc.name}",
            parent_id=scope.parent_span_id,
            attributes={"tool.name": tc.name, "tool.id": tc.id},
        )
        try:
            # Set contextvar so HumanInputToolHandler can read
            # room_id / tool_call_id / channel_id without protocol changes.
            _tc_ctx = ToolCallContext(
                room_id=scope.room_id or "",
                tool_call_id=tc.id,
                channel_id=self.channel_id,
            )
            started = time.monotonic()
            result = await self._serve_call(scope.handler, tc.name, arguments, _tc_ctx)
            _log_answer(tc.name, result, started)
            hook = await self._apply_tool_call_hook(tc, arguments, result, _tc_ctx, scope.room_id)
            # The call's own answer, neither replaced nor withheld by a hook.
            served = hook.kind is OutcomeKind.SERVED and hook.recorded is result
            bounded = self._bound_tool_result(tc.name, hook.result, tc.id, served=served)
            judged = replace(hook, result=bounded)
            telemetry.end_span(tool_span_id)
            return judged
        except asyncio.CancelledError:
            # Reported, if no report was made, when the turn ends.
            telemetry.end_span(tool_span_id, status="cancelled")
            raise
        except Exception as exc:
            telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
            return await self._raised_outcome(tc, arguments, scope.room_id, exc)

    async def _report_unreported_calls(self, loop_ctx: _ToolLoopContext) -> None:
        """Report, cancelled, each call the turn announced and no report
        claimed: a stop, a cancellation or a transport that stopped reading cut
        it before its result. Every channel reports such a call once (RFC §9.3):
        one whose outcome the model already read with that outcome, one its
        external handler was deciding through that handler."""
        for entry in loop_ctx.calls.unreported():
            with reporting(entry):
                await self._report_cut_call(loop_ctx, entry)

    async def _report_cut_call(self, loop_ctx: _ToolLoopContext, entry: AnnouncedCall) -> None:
        """Report one call the turn cut before its report: with the outcome
        the model read, through the handler that was deciding it, or
        cancelled, with the arguments it ran with."""
        tc = entry.as_ran()
        if entry.known is not None:
            await self._report_known_outcome(loop_ctx, entry.known)
            return
        handler = self._external_tool_handler
        if handler is not None and entry.external:
            # The report is claimed where the observers hear it; a handler
            # that reports nothing has still made it.
            await report_cut(handler, tc, loop_ctx.room_id)
            loop_ctx.claim_report(tc.id)
            return
        body = cancelled_tool_error(tc.name, "The turn ended before its result.")
        await self._fire_tool_refusal(tc, tc.arguments, body, loop_ctx.room_id, cancelled=True)

    async def _report_known_outcome(
        self, loop_ctx: _ToolLoopContext, event: ToolCallEvent
    ) -> None:
        """Report a call whose outcome the model already read, its own report
        cut before the observers heard it: to them alone, with that outcome
        (RFC §9.3)."""
        if self._tool_observer_hook is None:
            return
        try:
            await self._tool_observer_hook(event)
        except Exception:
            logger.warning(
                "ON_TOOL_CALL observation failed for tool %s", event.name, exc_info=True
            )
        loop_ctx.claim_report(event.tool_call_id)

    async def _raised_outcome(
        self, tc: Any, arguments: dict[str, Any], room_id: str | None, exc: Exception
    ) -> ToolOutcome:
        """The outcome of a call whose serving or judging raised *exc*, reported."""
        if isinstance(exc, ToolRefusedError):
            # The failure below with the message kept. A handler that
            # refuses a call has words for the model — a host tunes them
            # for a small one — and the generic wrapper would replace them
            # with its own sentence, which is how the reason gets lost.
            logger.info("Tool %s refused: %s", tc.name, exc.message)
            body = await self._failed_call(tc, arguments, room_id, exc.message, refused=True)
            return ToolOutcome(
                OutcomeKind.REFUSED,
                body,
                recorded=exc.message,
            )
        if isinstance(exc, ToolFailedError):
            # A failure in the handler's words: the tool ran, and this is
            # what the model reads of it; the observers read it as the detail.
            logger.warning("Tool %s failed: %s", tc.name, exc.message)
            body = await self._failed_call(tc, arguments, room_id, exc.message, detail=exc.message)
            return ToolOutcome(OutcomeKind.FAILED, body, recorded=exc.message)
        logger.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
        # The class, never the message (RFC §9.3): it goes to the log above
        # and to the observers, not to the model. The memory records what the
        # model saw, not a success the handler returned before a hook raised.
        recorded = tool_failure(tc.name, exc)
        body = await self._failed_call(
            tc, arguments, room_id, recorded, detail=failure_detail(exc)
        )
        return ToolOutcome(OutcomeKind.FAILED, body, recorded=recorded)

    def _model_part(
        self, tc: Any, outcome: ToolOutcome, *, references: list[str]
    ) -> AIToolResultPart:
        """What the model reads of a call: its result, noted when this tool
        already gave that answer this turn, and the held tools it makes
        callable."""
        # The hash is taken on the recorded outcome, so the memory keeps the
        # tool's own output and only the model's copy carries the note, and
        # the hash stays stable: annotating before hashing would make every
        # repeat look new, and so would an evicted copy, whose placeholder id
        # is unique per call.
        if isinstance(outcome.result, str):
            hashed = outcome.answer if isinstance(outcome.answer, str) else outcome.result
            noted = self._repeated_result_note(tc.name, outcome.result, outcome=hashed)
            outcome = replace(outcome, result=noted)
        return outcome.as_part(tc.id, tc.name, references=references)

    def _skill_tools(self) -> list[AITool]:
        """Build the list of AITool definitions for skill operations."""
        tools = [schema_tool(ACTIVATE_SKILL_SCHEMA), schema_tool(READ_REFERENCE_SCHEMA)]
        if self._script_executor:
            tools.append(schema_tool(RUN_SCRIPT_SCHEMA))
        return tools

    def _register_channel_tools(self) -> None:
        """Register the tools the channel serves itself, each with its traits.

        A handler may be sync or async: the dispatcher awaits what needs it.
        """
        served: list[tuple[AITool, Any]] = [
            (ToolEviction.tool_definition(), self._handle_read_tool_result)
        ]
        if self._planner is not None:
            served.append((TaskPlanner.tool_definition(), self._handle_plan_tasks))
        if self._skills:
            served += [
                (schema_tool(ACTIVATE_SKILL_SCHEMA), self._handle_activate_skill),
                (schema_tool(READ_REFERENCE_SCHEMA), self._handle_read_reference),
                (schema_tool(RUN_SCRIPT_SCHEMA), self._handle_run_script),
            ]
        # Tool Search discovery tools are channel-managed (they reshape the
        # visible tool surface, not the world). Registered unless explicitly
        # disabled; they are only ever injected into context when active.
        if self._tool_search is not False:
            find, inventory = search_tool_defs()
            served += [(find, self._handle_find_tools), (inventory, self._handle_list_tools)]
        for definition, serve in served:
            self._registry.register(channel_tool(definition, serve), owner=self)

    # Identical-call ceiling for regular tools: the 3rd repeat short-circuits.
    # Two identical executions can be legitimate (retry after a transient
    # failure); a model issuing the same call a third time is looping — the
    # observed failure mode is a small model re-running one find_tools query
    # for an entire turn and never answering.
    _REPEAT_CALL_LIMIT = 3
    # After the guard has BLOCKED the same call this many extra times and the
    # model still re-issues it, the advisory clearly isn't landing — force-stop
    # the loop. Small models otherwise ignore the error and hammer the same
    # call to the round limit (observed: sandbox_bash({}) called 37×).
    _REPEAT_FORCE_STOP_AT = 3
    # The same ceiling on the OTHER axis: how many identical RESULTS from one
    # tool before the model is told. Matched to ``_REPEAT_CALL_LIMIT`` for the
    # same reason — a second identical answer is ordinary (a retry, a poll, two
    # rows deleted), a third is a pattern.
    _REPEAT_RESULT_LIMIT = 3
    # Marker on this module's own advisory results, so a repeated advisory does
    # not get annotated as a repeated result. It already says what is wrong.
    _ADVISORY_MARKER = "these EXACT arguments"

    def _repeated_result_note(self, name: str, result: str, *, outcome: str | None = None) -> str:
        """Append a note when a tool returns an answer it already gave this turn.

        ``outcome`` is what the tool gave, when ``result`` (the model's copy)
        differs from it: an evicted copy carries a per-call id, so identical
        answers are recognised on the outcome. It defaults to ``result``.

        The blind spot in ``_repeated_call_guard``: it keys on the arguments, so
        a model that permutes them is never told anything. Measured on a stuck
        turn — 54 calls, 44 distinct argument sets, **25 distinct results**, one
        of them (`{"cards":[],"total":0}`) returned 23 times. The model narrated
        "let me confirm" at every round because nothing in what it read said the
        confirmation had already arrived, twenty-two times.

        **Annotates, never blocks**, and that asymmetry is deliberate. Identical
        results are not by themselves a fault: six deletions each answering
        ``{"success": true}`` are six correct operations with one result, and
        short-circuiting the sixth would destroy real work to save latency.
        Blocking stays with the argument guard, which cannot mistake legitimate
        work for a loop. This one only supplies the missing fact and lets the
        model act on it.
        """
        if self._ADVISORY_MARKER in result:
            return result
        hashed = result if outcome is None else outcome
        digest = hashlib.sha256(hashed.encode("utf-8", "replace")).hexdigest()
        counts = self._get_loop_ctx().repeated_results
        key = (name, digest)
        counts[key] = count = counts.get(key, 0) + 1
        if count < self._REPEAT_RESULT_LIMIT:
            return result
        # The only witness. The note rides on the tool result handed to the
        # model, which is downstream of the ON_TOOL_CALL hook the audit trail
        # listens on and absent from the turn-start context snapshot — so
        # neither of the two places an operator would look can show that this
        # fired. Logging it is what makes the guard observable at all.
        logger.warning(
            "Anti-loop: '%s' returned an identical result %d times this turn", name, count
        )
        return (
            f"{result}\n\n[identical result: '{name}' has now returned exactly this "
            f"{count} times this turn, for different arguments. Varying the arguments "
            f"is not finding anything new — this answer is settled. Use it and move "
            f"on, or answer with what you have.]"
        )

    def _repeated_call_guard(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Short-circuit a tool call repeated with identical arguments this turn."""
        try:
            key = (name, json.dumps(arguments or {}, sort_keys=True, default=str))
        except (TypeError, ValueError):
            return None
        loop_ctx = self._get_loop_ctx()
        counts = loop_ctx.repeated_calls
        counts[key] = count = counts.get(key, 0) + 1
        # A pure tool reads what cannot change within the turn (Tool Search's
        # fixed catalogue): an identical repeat never says anything new, so it
        # short-circuits at 2.
        traits = self._registry.traits(name, loop_ctx.room_id)
        limit = 2 if traits is not None and traits.pure else self._REPEAT_CALL_LIMIT
        if count < limit:
            return None
        # The model is ignoring the advisory and re-issuing anyway — pull the
        # ripcord so the loop force-ends with a plain-text answer.
        if count >= limit + self._REPEAT_FORCE_STOP_AT:
            loop_ctx.force_stop = True
        return json.dumps(
            {
                "error": (
                    f"You already called '{name}' with these EXACT arguments "
                    f"{count - 1} time(s) this turn — repeating it cannot yield "
                    "anything new."
                ),
                "hint": (
                    "STOP repeating this call. Use the results you already "
                    "have, try genuinely different arguments, or answer the "
                    "user now with what you know."
                ),
            }
        )

    def _sandbox_tool_names(self) -> frozenset[str]:
        """The names the attached sandbox declares, the ones the channel serves.

        By exact name, never by prefix: a host tool that merely starts with
        ``sandbox_`` is the host's (RFC §21.1).
        """
        if self._sandbox is None:
            return frozenset()
        return frozenset(
            tdef["name"]
            for tdef in self._sandbox.tool_definitions()
            if tdef["name"].startswith(SANDBOX_TOOL_PREFIX)
        )

    def _channel_tool_names(self) -> set[str]:
        """The tools this channel serves itself, before any host handler.

        Its own dispatch, its sandbox's commands and its human-input tools:
        a host tool under one of these names would be declared with the
        host's schema and served by the channel (RFC §21.1).
        """
        return self._channel_own_names() | self._human_input.declared_names

    def _channel_own_names(self) -> set[str]:
        """The tools the channel's own features serve: its dispatch's and its
        sandbox's commands."""
        names = {e.name for e in self._registry.entries(None, source=ToolSource.CHANNEL)}
        return names | self._sandbox_tool_names()

    def _served_tool_names(self, room_id: str | None) -> set[str]:
        """The tools the channel and orchestration serve in *room_id*, before
        any host handler: a host tool under one of these is not declared."""
        return self._channel_tool_names() | self._registry.names(room_id)

    def _declared_once(self, tools: list[AITool], room_id: str | None) -> list[AITool]:
        """The host's part of a turn's toolset in *room_id*: no tool under a name
        the channel or orchestration serves there, and each name once (RFC
        §21.1, :func:`declared_once`)."""
        served = self._served_tool_names(room_id)
        return declared_once(tools, _tool_name, served, self._collisions)

    async def _channel_tool_handler(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Unified tool dispatcher: channel-managed -> sandbox -> person -> user tools.

        The outcomes the channel decides itself (a repeat the guard stops, a
        tool outside the turn's toolset) are refusals, raised as
        :class:`ToolRefusedError` so they carry the failure marker (RFC §9.3).
        :class:`UnservedToolCallError` when nothing serves the call:
        ON_TOOL_CALL's hooks may then serve it.
        """
        guard = self._repeated_call_guard(name, arguments)
        if guard is not None:
            raise ChannelRefusalError(guard)
        entry = self._registry.lookup(name, self._get_loop_ctx().room_id)
        if entry is not None and entry.serve is not None:
            result = entry.serve(arguments)
            # Support both sync and async handlers
            if asyncio.iscoroutine(result):
                result = await result
            return as_tool_result(result)
        # The sandbox's own tools, by exact name, before user/MCP tools
        if self._sandbox is not None and name in self._sandbox_tool_names():
            return await handle_sandbox_command(name, arguments or {}, self._sandbox)
        # Provider responses are untrusted and may name a tool outside the
        # turn's resolved toolset. Once context construction has resolved
        # that invocation-scoped set, fail closed, before anything may serve
        # it, a hook included. ``None`` preserves direct internal loops built
        # without context; [] is a real deny-all set.
        context_tools = self._get_loop_ctx().all_context_tools
        if context_tools is not None and name not in {t.name for t in context_tools}:
            raise ChannelRefusalError(
                json.dumps({"error": f"Tool '{name}' is not available in the current turn."})
            )
        # The person's tools, before the host's handler (RFC §9.3).
        if self._human_input.serves(name):
            return as_tool_result(await self._human_input.serve(name, arguments))
        if self._user_tool_handler is None:
            raise UnservedToolCallError(name)
        # The host's answer alone may be the "not mine" envelope (RFC §21.4):
        # the channel's own tools answer what they ran, a failing command's
        # error included.
        answer = await self._user_tool_handler(name, arguments)
        return as_tool_result(declined_answer(answer, name))

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        loop_ctx = self._get_loop_ctx()
        skill = self._skills.get_skill(skill_name) if skill_name else None
        if skill_name and skill is None:
            own = self._channel_tool_names()
            reachable = self._reachable_tools(loop_ctx.all_context_tools or ())
            result_str, matching = tools_hint(
                result_str,
                skill_name,
                self._skills,
                (t.name for t in reachable if t.name not in own),
            )
            if not matching:
                # Nothing to reveal: the call named no skill, refused (RFC §9.3).
                raise ToolRefusedError(result_str)
            self._defer_reveal(loop_ctx, matching)
            return result_str
        if skill is not None and (missing := self._missing_required_tools(skill)):
            # Refused as every door refuses it (RFC §24.3): nothing opens.
            raise ToolRefusedError(missing_tools_error(missing))
        # Recorded once the call's outcome is known: an ON_TOOL_CALL hook that
        # blocks the call, or a failure, must open no gate (_settle_served_call).
        already_active = self._skill_activation.is_active(loop_ctx.room_id, skill_name)
        self._defer_activation(loop_ctx, skill_name)
        if skill is None or not already_active:
            return result_str
        # Already active: _build_context put these very instructions in front of
        # the model before the turn started, so the body just built above would
        # be a second copy of rules it already holds. Ack instead.
        return activation_ack(skill, ALREADY_ACTIVE_NOTE, already_active=True)

    def _missing_required_tools(self, skill: Skill) -> list[str]:
        """The tools *skill* requires that this turn does not offer once its
        tool policy is applied, skill gating aside (RFC §24.3); none to check
        outside a turn, which resolved no toolset."""
        base = self._get_loop_ctx().all_context_tools
        if base is None or self._skills is None:
            return []
        admitted = (tool.name for tool in base if self._policy_allows(tool.name))
        # Its own gates are the skill's to open; a closed one nothing opens.
        closed = self._skills.closed_tool_names(self._activated_skill_names() | {skill.name})
        match = self._skills.requires_match
        return missing_required_tools(skill.metadata, admitted, closed, match=match)

    def _defer_activation(self, loop_ctx: _ToolLoopContext, skill_name: str) -> None:
        """Hold an activation until its call is served, or record it now.

        Inside the tool loop the call's outcome is not known yet: an
        ON_TOOL_CALL hook may still block it. Outside one (a direct call)
        nothing can, and the activation is recorded at once.
        """
        call = _current_tool_call.get()
        if call is not None and call.tool_call_id:
            loop_ctx.pending_activations[call.tool_call_id] = skill_name
        else:
            self._record_activation(loop_ctx, skill_name)

    def _defer_reveal(self, loop_ctx: _ToolLoopContext, names: list[str]) -> None:
        """Hold the reveal a ``find_tools`` call or an activation's hint asked
        for until its call is served, as an activation is held, or reveal now
        outside a loop."""
        call = _current_tool_call.get()
        if call is not None and call.tool_call_id:
            if names:
                loop_ctx.pending_reveals[call.tool_call_id] = names
        else:
            self._reveal(loop_ctx, names)

    def _reveal(self, loop_ctx: _ToolLoopContext, names: list[str]) -> None:
        """Reveal *names* as ``find_tools`` reveals its matches: the reveal
        window swapped, what Tool Search never defers left out, and kept for
        the rest of the session (RFC §24.4)."""
        never = self._never_deferred(loop_ctx)
        revealed = {name for name in names if name not in never}
        if not revealed:
            return
        # A tool a served recovery revealed was used: the swap keeps it.
        loop_ctx.revealed_tools = revealed | loop_ctx.recovered_tools
        self._tool_usage.record_revealed(loop_ctx.room_id, revealed)

    def _settle_served_call(self, tool_call_id: str, *, served: bool) -> None:
        """Record the activation, or the reveal (a ``find_tools`` call's, an
        activation hint's), a served call asked for; drop a refused one."""
        loop_ctx = self._get_loop_ctx()
        skill_name = loop_ctx.pending_activations.pop(tool_call_id, None)
        if skill_name is not None and served:
            self._record_activation(loop_ctx, skill_name)
        names = loop_ctx.pending_reveals.pop(tool_call_id, None)
        if names is not None and served:
            self._reveal(loop_ctx, names)

    def _settle_recovery(self, tool_call_id: str, *, kept: bool) -> None:
        """Reveal the tool a call recovered for the turn's next rounds when the
        room's tool memory keeps the call, which re-reveals it on later turns
        as any tool used; a call it does not keep reveals nothing (RFC §6.4).
        The reveal outlasts a ``find_tools`` of the round (``_reveal``)."""
        loop_ctx = self._get_loop_ctx()
        name = loop_ctx.pending_recoveries.pop(tool_call_id, None)
        if name is not None and kept:
            loop_ctx.recovered_tools.add(name)
            loop_ctx.revealed_tools.add(name)

    def _record_activation(self, loop_ctx: _ToolLoopContext, skill_name: str) -> None:
        # For this turn, so gated tools become visible on the next round...
        loop_ctx.activated_skills.add(skill_name)
        # ... and for the rest of the conversation, so the body can ride the
        # system prompt instead of being re-fetched every turn.
        self._skill_activation.activate(loop_ctx.room_id, skill_name)

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        return await handle_run_script(arguments, self._skills, self._script_executor)

    def _tool_search_catalogue(self, loop_ctx: _ToolLoopContext) -> list[dict[str, Any]]:
        """The turn's reachable tools as score-able dicts (name + description + tags).

        Only what the policy allows and no skill gates (RFC §21.1): a match
        the model can never call is a false promise, and listing it discloses
        what the policy hides.
        """
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "tags": getattr(t, "tags", []) or [],
            }
            for t in self._reachable_tools(loop_ctx.offered_tools())
        ]

    async def _handle_find_tools(self, arguments: dict[str, Any]) -> str:
        """Reveal catalogue tools matching a query for the rest of the loop,
        once the call is served (RFC §6.4).

        The matches swap the reveal window when the call's outcome is known
        (``_settle_served_call``): a call an ON_TOOL_CALL hook blocks, or one
        that fails, reveals nothing, and a search that finds nothing keeps the
        window as it was. The next round's tool re-filter exposes the matches.
        No ``provider.reconfigure``: the text loop re-sends its tool list every
        round.
        """
        loop_ctx = self._get_loop_ctx()
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps(
                {
                    "error": "query is required",
                    "hint": "Pass a short natural-language description.",
                }
            )
        catalogue = self._tool_search_catalogue(loop_ctx)
        max_results = normalize_max_results(
            arguments.get("max_results"), self._tool_search_threshold
        )
        # Declared already, never named: every tool Tool Search never defers
        # (RFC §6.4, §21.1).
        exclude = self._never_deferred(loop_ctx)
        matches = search_catalogue(catalogue, query, max_results, exclude_names=exclude)
        # Revealed once the call is served, then for the rest of the session
        # (ToolUsageMemory, through _reveal): a tool found in turn N is often
        # only called in turn N+1, after the user confirms.
        self._defer_reveal(loop_ctx, [m["name"] for m in matches if m.get("name")])
        # Compact result (name + short description). The matched tools' full
        # schemas reach the model via the next round's re-filtered tool list
        # (loop_ctx.revealed_tools), so inlining them here would only risk
        # overflowing the tool-result size limit on verbose tools.
        return render_find_payload(
            matches,
            miss_hint=self._tool_search_miss_hint,
            related=related_family_tools(catalogue, matches, exclude_names=exclude),
        )

    async def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        """List the turn's catalogue (name + short description). Reveals nothing."""
        loop_ctx = self._get_loop_ctx()
        category = str(arguments.get("category", "")).strip()
        catalogue = self._tool_search_catalogue(loop_ctx)
        return render_list_payload(catalogue, category, exclude_names=TOOL_SEARCH_INFRA_TOOL_NAMES)

    def _remember_call(
        self, room_id: str | None, name: str, call_arguments: dict[str, Any], result: Any
    ) -> None:
        """Remember a call (the tool's own answer, success or error, never the
        eviction placeholder an oversized one became for the model) so later
        turns show "tools you've already used" and re-reveal it under Tool
        Search. Which calls are kept is :func:`kept_in_tool_memory`'s.

        With the model's own arguments, never a BEFORE_TOOL_USE rewrite: the
        digest goes back into the next turn's prompt, and a hook that
        de-tokenises (``<EMAIL_1>`` to the real address) would put there the
        very value it kept from the model. Infra/discovery tools are filtered
        inside ``record()``.
        """
        self._tool_usage.record(room_id, name, call_arguments, result)

    async def _apply_tool_call_hook(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: ToolResult | None,
        call_ctx: ToolCallContext,
        room_id: str | None,
    ) -> ToolOutcome:
        """Run ON_TOOL_CALL on the outcome the model will read, whole.

        After a text-only model's flattening, so the hook sees the shape the
        model reads; before eviction, so everything read_stored_result can
        page back has passed through the hook and a redacting hook covers the
        full text, not a preview of it.

        The call's structured copy (MCP structuredContent, which the handler
        left on *call_ctx*) is read here, once the handler returned, and only
        the outcome carries it on: a call that fails before or during this
        step keeps none. The hook sees the copy and may replace it; a BLOCK
        withholds the result and drops the copy. Eviction never touches it:
        UI surfaces need the payload whole.

        A call nothing served (*result* ``None``) reaches the hooks with no
        result: one may serve it with the result it supplies, and if none does
        the call failed, reported once to the observers (RFC §9.3).
        """
        shaped = None if result is None else self._shape_for_model(tc.name, result, tc.id)
        structured = None if result is None else call_ctx.structured_content
        verdict = await self._tool_call_verdict(tc, arguments, shaped, structured, room_id)
        reading = read_tool_call_verdict(tc.name, verdict, shaped)
        kind = read_outcome(reading)
        if kind is not OutcomeKind.UNSERVED:
            # Judged, so reported by the chain's observers: claimed, if the
            # chain did not claim it itself, so the turn's end reports it no
            # second time.
            self._get_loop_ctx().claim_report(tc.id)
        if kind is OutcomeKind.BLOCKED:
            # The memory keeps the hook's reason, before eviction swaps a placeholder in.
            return ToolOutcome(kind, reading.result, recorded=reading.result)
        if verdict is not None and verdict.replaces_structured:
            structured = verdict.structured_content
        if kind is OutcomeKind.UNSERVED:
            body = unserved_tool_error(tc.name)
            detail = verdict.error_detail if verdict is not None else None
            await self._fire_tool_refusal(tc, arguments, body, room_id, detail=detail)
            return ToolOutcome(kind, body)
        # The memory keeps the answer itself, before eviction swaps a placeholder in.
        recorded = reading.result if reading.replaced else result
        return ToolOutcome(kind, reading.result, recorded=recorded, structured=structured)

    async def _tool_call_verdict(
        self,
        tc: Any,
        arguments: dict[str, Any],
        result: ToolResult | None,
        structured: dict[str, Any] | None,
        room_id: str | None,
    ) -> ToolCallVerdict | None:
        """ON_TOOL_CALL's SYNC chain on one call's outcome, as a verdict."""
        if self._tool_call_hook is None:
            return None
        # The call's one report, claimed between the chain and its observers.
        claim = partial(self._get_loop_ctx().claim_report, tc.id)
        verdict = await self._tool_call_hook(
            ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                tool_call_id=tc.id,
                name=tc.name,
                arguments=arguments,
                result=result,
                room_id=room_id,
                structured_content=structured,
            ),
            claim=claim,
        )
        if verdict is None or isinstance(verdict, ToolCallVerdict):
            return verdict
        return ToolCallVerdict(result=verdict)  # a bare override, read as a verdict

    async def _serve_call(
        self,
        handler: Any,
        name: str,
        arguments: dict[str, Any],
        call_ctx: ToolCallContext,
    ) -> ToolResult | None:
        """The handler's answer to one call, run in its tool call context.

        ``None`` when nothing served it: the dispatcher found no handler, or
        the handlers all answered that the tool is not theirs (RFC §21.4), the
        answer a composition passes a call on for.
        """
        token = _current_tool_call.set(call_ctx)
        timeout = self._call_timeout(name, call_ctx.room_id or None)
        try:
            answer = await answer_within(timeout, name, handler(name, arguments))
        except UnservedToolCallError:
            return None
        finally:
            _current_tool_call.reset(token)
        return as_tool_result(answer)

    def _call_timeout(self, name: str, room_id: str | None) -> float | None:
        """The bound of one call to *name* (RFC §21.6): the channel's, unless
        the tool keeps a bound of its own."""
        return self._registry.bound(
            name, room_id, self._tool_timeouts, own=name in self._own_bound_tools()
        )

    def _own_bound_tools(self) -> set[str]:
        """The tools outside the registry that carry a bound of their own: a
        person's answer under its handler's timeout, a sandbox command under its
        ``timeout`` argument."""
        names = set(self._human_input.names)
        if self._sandbox is not None:
            names.add(TOOL_SANDBOX_BASH)
        return names

    def _shape_for_model(self, name: str, result: ToolResult, tool_call_id: str) -> ToolResult:
        """A text-only model gets the text of a content-part result, the way it
        gets a message's (``_extract_content``): an image it cannot take would
        fail the request."""
        if isinstance(result, list) and not self._provider.supports_vision:
            return AIToolResultPart(tool_call_id=tool_call_id, name=name, result=result).as_text()
        return result

    def _bound_tool_result(
        self, name: str, result: ToolResult, tool_call_id: str, *, served: bool = False
    ) -> ToolResult:
        """The copy of a tool's outcome the model reads, evicted when oversized.

        Every outcome goes through it (a result, a hook's override, a refusal,
        an error): whichever path a 500 KB body takes, it must not reach the
        provider whole, save a result the model reads whole (``kept_whole``:
        a skill's instructions, which a 20 KB skill would otherwise see
        evicted), and only as *served* by the call itself: a refusal, a block
        or a hook's replacement is bounded (RFC §21.5). References are data
        and still evict.
        """
        result = self._shape_for_model(name, result, tool_call_id)
        if served and kept_whole(name):
            return result
        return self._maybe_truncate_result(result, tool_call_id)

    def _bound_provider_result(self, name: str, result: str, tool_call_id: str) -> str:
        """The copy of a provider-served call's outcome the model reads and
        its END row keeps, bounded as every outcome is (RFC §21.5)."""
        bounded = self._bound_tool_result(name, result, tool_call_id)
        return bounded if isinstance(bounded, str) else result_text(bounded)

    # -- Extracted tool handlers (delegate to focused modules) -----------------

    def _handle_read_tool_result(self, arguments: dict[str, Any]) -> str:
        """Delegate to ToolEviction."""
        return self._eviction.handle_read(arguments)

    async def _handle_plan_tasks(self, arguments: dict[str, Any]) -> str:
        """Delegate to TaskPlanner."""
        if self._planner is None:
            return json.dumps({"error": "Planning is not enabled"})
        room_id = self._get_loop_ctx().room_id
        return await self._planner.handle_plan_tasks(
            arguments,
            realtime=self._realtime,
            room_id=room_id,
            channel_id=self.channel_id,
            # RFC §9.2 ON_PLAN_UPDATED — the hook surface for the plan the
            # ephemeral event carries to live UIs.
            on_plan_updated=self._plan_updated_hook,
        )


_PREVIEW_CHARS = 500


def _preview(value: Any) -> str:
    """A bounded one-line rendering of a tool payload for DEBUG logs."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    if len(text) <= _PREVIEW_CHARS:
        return text
    return f"{text[:_PREVIEW_CHARS]}… ({len(text)} chars)"


def call_end_marker(
    tc: Any, marker: ToolCallStartMarker, part: AIToolResultPart, duration_ms: int
) -> ToolCallEndMarker:
    """The end of a call that ran or was refused, as its row stores it: the
    arguments it ran with (its start marker's), its outcome as the model
    reads it."""
    is_error = part.is_error
    return ToolCallEndMarker(
        tool_name=tc.name,
        tool_id=tc.id,
        arguments=marker.ran_with if marker.ran_with is not None else tc.arguments,
        result=part.result,
        status="failed" if is_error else "completed",
        duration_ms=duration_ms,
        # ``error`` is text; a failure that answered with content parts
        # flattens the way any text consumer of that result would.
        error=part.as_text() if is_error else None,
        structured_content=part.structured_content,
        outcome=part.outcome,
    )
