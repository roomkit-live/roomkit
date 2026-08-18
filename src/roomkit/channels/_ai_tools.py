"""AIChannel mixin for tool execution, dispatch, and skill tool handlers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._sandbox_handlers import handle_sandbox_command
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    ALREADY_ACTIVE_NOTE,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._skill_handlers import (
    activation_ack,
    handle_activate_skill,
    handle_read_reference,
    handle_run_script,
)
from roomkit.channels._tool_search import (
    normalize_max_results,
    related_family_tools,
    render_find_payload,
    render_list_payload,
    search_catalogue,
)
from roomkit.channels._tool_search_constants import (
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
)
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import (
    AIImagePart,
    AIProvider,
    AITextPart,
    AITool,
    AIToolResultPart,
)
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX
from roomkit.telemetry.base import SpanKind
from roomkit.tools.context import ToolCallContext, _current_tool_call
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels._task_planner import TaskPlanner
    from roomkit.channels._tool_eviction import ToolEviction
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.channels.ai import _ContentPart, _ToolLoopContext
    from roomkit.models.tool_call import ToolCallCallback
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.policy import ToolPolicy

    ToolResult = str | list[AITextPart | AIImagePart]
    ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class AIToolsHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolsMixin.

    Attributes provided by the host's ``__init__``:
        _provider: AI provider — read for the model id in fold diagnostics.
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _user_tool_handler: User-provided tool handler for fallback dispatch.
        _skills: Skill registry for gated tool resolution.
        _script_executor: Script executor for skill scripts.
        _sandbox: Sandbox executor for ad-hoc command execution.
        _eviction: Tool result eviction / truncation strategy.
        _skill_activation: Per-room record of the skills active in a conversation.
        _planner: Optional task planner.
        _realtime: Realtime backend for ephemeral events.
        _tool_call_hook: Optional unified ON_TOOL_CALL hook callback.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        _effective_tool_policy: ``AIToolPolicyMixin`` property — resolved policy.
        _SKILL_INFRA_TOOLS: ``AIToolPolicyMixin`` class var — infra tool names.
        _gated_tool_names: ``AIToolPolicyMixin`` property — gated tool names.
        _maybe_truncate_result: ``AIResilienceMixin`` — truncate large results.
        _get_loop_ctx: ``AISteeringMixin`` — returns current tool-loop context.
        _apply_tool_filters: ``AIToolPolicyMixin`` — policy / skill-gating /
            Tool Search visibility filter.
    """

    _provider: AIProvider
    _tool_handler: Any
    _user_tool_handler: Any
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _realtime: RealtimeBackend | None
    _plan_updated_hook: Any  # ON_PLAN_UPDATED callback — injected by register_channel
    _tool_call_hook: ToolCallCallback | None
    _before_tool_call_hook: Any
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_miss_hint: str | None
    channel_id: str

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None: ...
    @property
    def _gated_tool_names(self) -> set[str]: ...

    _SKILL_INFRA_TOOLS: frozenset[str]

    def _maybe_truncate_result(
        self,
        result: str | list[AITextPart | AIImagePart],
        tool_call_id: str = ...,
    ) -> str | list[AITextPart | AIImagePart]: ...
    def _get_loop_ctx(self) -> _ToolLoopContext: ...
    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]: ...


class AIToolsMixin:
    """Parallel tool execution, skill tool definitions, and dispatch routing.

    Host contract: :class:`AIToolsHost`.
    """

    _provider: AIProvider
    _tool_handler: Any
    _user_tool_handler: Any
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _realtime: RealtimeBackend | None
    _plan_updated_hook: Any  # ON_PLAN_UPDATED callback — injected by register_channel
    _tool_call_hook: ToolCallCallback | None
    _before_tool_call_hook: Any
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_miss_hint: str | None
    channel_id: str

    # Cross-mixin methods — Any annotations avoid MRO shadowing
    _effective_tool_policy: Any  # see AIToolsHost
    _SKILL_INFRA_TOOLS: Any  # see AIToolsHost
    _gated_tool_names: Any  # see AIToolsHost
    _maybe_truncate_result: Any  # see AIToolsHost
    _get_loop_ctx: Any  # see AIToolsHost
    _apply_tool_filters: Any  # see AIToolsHost
    extra_tools: Any  # AIChannel property: user + orchestration-injected tools

    def _tool_parameters(
        self, name: str, declared_tools: list[AITool] | None = None
    ) -> dict[str, Any] | None:
        """Return the declared JSON-Schema ``parameters`` for tool *name*.

        ``None`` when the tool's schema is not known to this channel (infra,
        skill, or sandbox tools) — those skip argument validation.
        """
        for tool in declared_tools if declared_tools is not None else self.extra_tools:
            if tool.name == name:
                return tool.parameters
        return None

    def _recover_deferred_tool(self, name: str) -> AITool | None:
        """A find_tools reveal applied at call time, for an exact-name call.

        Small models routinely skip the two-step discovery protocol and call a
        catalogue tool they saw (via list_tools, or a prior turn) without
        revealing it first. The name being exact, the call is trivially
        recoverable: reveal the tool as find_tools would have and let the call
        proceed — provided it survives the same visibility filter a reveal is
        subject to. That filter is the authority on eligibility (tool policy,
        glob-aware skill gating); the execution guards re-check policy and
        exact-name gating, but glob gating is enforced only by the filter, so
        recovery must not bypass it.

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
        # Reveal first: while Tool Search is active the filter keeps only
        # pinned/revealed/sticky names, so the eligibility probe needs the name
        # in the reveal set. Rolled back when the probe fails.
        loop_ctx.revealed_tools.add(name)
        if not self._apply_tool_filters([tool]):
            loop_ctx.revealed_tools.discard(name)
            return None
        # Parity with _handle_find_tools: the reveal persists across turns.
        self._tool_usage.record_revealed(loop_ctx.room_id, {name})
        return tool

    def _undeclared_tool_error(self, name: str) -> dict[str, str]:
        """Actionable payload for an undeclared call that could not be recovered."""
        loop_ctx = self._get_loop_ctx()
        if any(t.name == name for t in loop_ctx.all_context_tools or ()):
            # In the catalogue but filtered out (tool policy or skill gating):
            # a find_tools reveal would be dropped by the same filter, so no
            # retry hint — the refusal is the answer.
            return {
                "error": (
                    f"Tool '{name}' exists but is not available to this agent "
                    "(blocked by the tool policy or gated behind a skill)."
                )
            }
        if loop_ctx.tool_search_active:
            return {
                "error": f"Unknown tool '{name}': no tool by that name exists.",
                "hint": (
                    "Check the spelling, or call find_tools(query=<the task>) "
                    "to discover the right tool."
                ),
            }
        return {"error": f"Unknown tool '{name}': it is not declared"}

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        declared_tools: list[AITool] | None = None,
        parent_span_id: str | None = None,
        executed_arguments: dict[str, dict[str, Any]] | None = None,
    ) -> list[_ContentPart]:
        """Execute tool calls concurrently and return result parts."""
        if self._tool_handler is None:
            raise RuntimeError("_execute_tools_parallel called without a tool handler")
        handler = self._tool_handler
        # Capture the invocation-scoped room once. The channel object is shared
        # across rooms, while the loop context is copied into every task spawned
        # by gather below.
        room_id = self._get_loop_ctx().room_id

        async def _run_one(tc: Any) -> AIToolResultPart:
            logger.info("Executing tool: %s(%s)", tc.name, tc.id)

            # Execution guard: argument validation against the declared schema
            # (fail-closed) — reject malformed calls before any other gate.
            params = self._tool_parameters(tc.name, declared_tools)
            declared_names = {tool.name for tool in declared_tools or []}
            channel_managed = (
                tc.name in self._SKILL_INFRA_TOOLS
                or tc.name in TOOL_SEARCH_INFRA_TOOL_NAMES
                or tc.name.startswith(SANDBOX_TOOL_PREFIX)
            )
            if declared_names and tc.name not in declared_names and not channel_managed:
                recovered = self._recover_deferred_tool(tc.name)
                if recovered is None:
                    logger.warning("Provider requested undeclared tool %s", tc.name)
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(self._undeclared_tool_error(tc.name)),
                    )
                # The model skipped find_tools but named a real catalogue tool:
                # the reveal happened at call time instead of ahead of it, and
                # every guard below still applies.
                logger.info("Recovered deferred catalogue tool %s at call time", tc.name)
                params = recovered.parameters
            call_arguments = tc.arguments
            if params is not None:
                # Repair before validating: a model that flattened a hub tool's
                # ``params`` gets its call folded back into shape instead of
                # spending a round on an error it can only fix by re-issuing.
                folded, fold_error = fold_hoisted_arguments(params, call_arguments)
                if fold_error is not None:
                    logger.warning("Tool %s arguments ambiguous: %s", tc.name, fold_error)
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(
                            {"error": f"Invalid arguments for '{tc.name}': {fold_error}"}
                        ),
                    )
                if folded is not None:
                    logger.info(
                        "Tool %s: folded hoisted arguments %s into 'params' (model=%s)",
                        tc.name,
                        sorted(folded["params"]),
                        self._provider.model_name,
                    )
                    call_arguments = folded
                arg_error = validate_tool_arguments(params, call_arguments)
                if arg_error is not None:
                    logger.warning("Tool %s arguments rejected: %s", tc.name, arg_error)
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(
                            {"error": f"Invalid arguments for '{tc.name}': {arg_error}"}
                        ),
                    )

            # Execution guard: policy deny (defense-in-depth, role-aware)
            # Sandbox tools are exempt — they are channel-managed, not user-managed.
            effective_policy = self._effective_tool_policy
            if (
                tc.name not in self._SKILL_INFRA_TOOLS
                and tc.name not in TOOL_SEARCH_INFRA_TOOL_NAMES
                and not tc.name.startswith(SANDBOX_TOOL_PREFIX)
                and effective_policy
                and not effective_policy.is_allowed(tc.name)
            ):
                logger.warning("Tool %s blocked by policy", tc.name)
                return AIToolResultPart(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=json.dumps(
                        {"error": f"Tool '{tc.name}' is not permitted by the agent's tool policy."}
                    ),
                )

            # Execution guard: skill gating
            if tc.name not in self._SKILL_INFRA_TOOLS and tc.name in self._gated_tool_names:
                logger.warning("Tool %s blocked by skill gating", tc.name)
                return AIToolResultPart(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=json.dumps(
                        {
                            "error": (
                                f"Tool '{tc.name}' is gated by a skill. "
                                "Activate the skill first using activate_skill."
                            ),
                        }
                    ),
                )

            # Pre-execution gate: BEFORE_TOOL_USE hook can deny the tool call,
            # or hand back rewritten arguments (a redaction hook putting real
            # values back before the tool acts on the model's tokenised text).
            # Everything downstream — the handler, ON_TOOL_CALL, the usage
            # record — reads ``arguments``, so it reports what actually ran.
            arguments = call_arguments
            arguments_rewritten = False
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
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(
                            {"error": f"Tool '{tc.name}' denied by pre-execution hook."}
                        ),
                    )
                if decision.arguments is not None:
                    arguments = decision.arguments
                    arguments_rewritten = True

            # Validate the payload after every hook, even when it did not
            # explicitly return a replacement. ToolCallEvent is frozen but its
            # nested dict is mutable, so an in-place edit must not bypass this
            # fail-closed boundary either. No fold here, deliberately: these
            # arguments come from user code, and repairing a hook's output
            # would hide the hook's bug instead of naming it. The model's own
            # call was already folded above, so a hook that rewrites nothing
            # arrives here in the repaired shape.
            if params is not None:
                arg_error = validate_tool_arguments(params, arguments)
                if arg_error is not None:
                    qualifier = "rewritten " if arguments_rewritten else ""
                    logger.warning(
                        "Tool %s %sarguments rejected: %s", tc.name, qualifier, arg_error
                    )
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(
                            {
                                "error": (
                                    f"Invalid {qualifier}arguments for '{tc.name}': {arg_error}"
                                )
                            }
                        ),
                    )

            tool_span_id = telemetry.start_span(
                SpanKind.LLM_TOOL_CALL,
                f"tool.{tc.name}",
                parent_id=parent_span_id,
                attributes={"tool.name": tc.name, "tool.id": tc.id},
            )
            structured_content: dict[str, Any] | None = None
            if executed_arguments is not None:
                # Snapshot the post-hook payload before handing it to user
                # code. Streaming persistence can then distinguish what the
                # model requested from what actually executed.
                executed_arguments[tc.id] = dict(arguments)
            try:
                # Set contextvar so HumanInputToolHandler can read
                # room_id / tool_call_id / channel_id without protocol changes.
                _tc_ctx = ToolCallContext(
                    room_id=room_id or "",
                    tool_call_id=tc.id,
                    channel_id=self.channel_id,
                )
                _tc_tok = _current_tool_call.set(_tc_ctx)
                try:
                    result = await handler(tc.name, arguments)
                finally:
                    _current_tool_call.reset(_tc_tok)
                # Capture the handler's structured result (MCP structuredContent)
                # BEFORE eviction — the string below may become a placeholder,
                # but UI surfaces need the structured payload verbatim.
                structured_content = _tc_ctx.structured_content
                # A skill's instructions are binding rules the model must hold
                # whole — never a head/tail preview behind a read_stored_result
                # pointer, which is what eviction would make of a body over the
                # threshold (a 20 KB skill crosses it). Every other tool still
                # evicts, references included: those are data, and paginating
                # data is exactly what eviction is for.
                if tc.name != TOOL_ACTIVATE_SKILL:
                    result = self._maybe_truncate_result(result, tc.id)

                # Fire unified ON_TOOL_CALL hook (if framework injected callback)
                if self._tool_call_hook is not None:
                    event = ToolCallEvent(
                        channel_id=self.channel_id,
                        channel_type=ChannelType.AI,
                        tool_call_id=tc.id,
                        name=tc.name,
                        arguments=arguments,
                        result=result,
                        room_id=room_id,
                    )
                    override = await self._tool_call_hook(event)
                    if override is not None:
                        result = override

                telemetry.end_span(tool_span_id)
            except Exception as exc:
                telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
                logger.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
                result = f"Error executing tool '{tc.name}': {exc}"
            # Remember this call (final result, success or error) so later turns
            # can show "tools you've already used" and re-reveal it under Tool
            # Search. Infra/discovery tools are filtered inside record().
            self._tool_usage.record(room_id, tc.name, arguments, result)
            # Annotate an answer this tool already gave this turn. Runs on the
            # recorded result, so the memory above keeps the tool's own output
            # and only the model's copy carries the note — and the hash stays
            # stable, since annotating before hashing would make every repeat
            # look new.
            if isinstance(result, str):
                result = self._repeated_result_note(tc.name, result)
            return AIToolResultPart(
                tool_call_id=tc.id,
                name=tc.name,
                result=result,
                structured_content=structured_content,
            )

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
        return list(results)

    def _skill_tools(self) -> list[AITool]:
        """Build the list of AITool definitions for skill operations."""

        def _to_ai_tool(schema: dict[str, Any]) -> AITool:
            return AITool(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )

        tools = [_to_ai_tool(ACTIVATE_SKILL_SCHEMA), _to_ai_tool(READ_REFERENCE_SCHEMA)]
        if self._script_executor:
            tools.append(_to_ai_tool(RUN_SCRIPT_SCHEMA))
        return tools

    # Dispatch table for channel-managed and skill tools.
    # Sync handlers are wrapped to match the async signature.
    @property
    def _channel_tool_dispatch(self) -> dict[str, Any]:
        dispatch: dict[str, Any] = {
            "read_stored_result": self._handle_read_tool_result,
            "plan_tasks": self._handle_plan_tasks,
        }
        if self._skills:
            dispatch[TOOL_ACTIVATE_SKILL] = self._handle_activate_skill
            dispatch[TOOL_READ_REFERENCE] = self._handle_read_reference
            dispatch[TOOL_RUN_SCRIPT] = self._handle_run_script
        # Tool Search discovery tools are channel-managed (they reshape the
        # visible tool surface, not the world). Registered unless explicitly
        # disabled; they are only ever injected into context when active.
        if self._tool_search is not False:
            dispatch[TOOL_FIND_TOOLS] = self._handle_find_tools
            dispatch[TOOL_LIST_TOOLS] = self._handle_list_tools
        return dispatch

    # Identical-call ceiling for regular tools: the 3rd repeat short-circuits.
    # Two identical executions can be legitimate (retry after a transient
    # failure); a model issuing the same call a third time is looping — the
    # observed failure mode is a small model re-running one find_tools query
    # for an entire turn and never answering.
    _REPEAT_CALL_LIMIT = 3
    # Pure within a turn (they read the fixed catalogue, mutate nothing): an
    # identical repeat can never say anything new, so it short-circuits at 2.
    _REPEAT_PURE_TOOLS = frozenset({TOOL_FIND_TOOLS, TOOL_LIST_TOOLS})
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

    def _repeated_result_note(self, name: str, result: str) -> str:
        """Append a note when a tool returns an answer it already gave this turn.

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
        digest = hashlib.sha256(result.encode("utf-8", "replace")).hexdigest()
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
        limit = 2 if name in self._REPEAT_PURE_TOOLS else self._REPEAT_CALL_LIMIT
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

    async def _channel_tool_handler(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Unified tool dispatcher: channel-managed -> sandbox -> skill -> user tools."""
        guard = self._repeated_call_guard(name, arguments)
        if guard is not None:
            return guard
        handler = self._channel_tool_dispatch.get(name)
        if handler is not None:
            result = handler(arguments)
            # Support both sync and async handlers
            if asyncio.iscoroutine(result):
                return str(await result)
            return str(result)
        # Sandbox tools — dispatched by prefix before user/MCP tools
        if self._sandbox is not None and name.startswith(SANDBOX_TOOL_PREFIX):
            return await handle_sandbox_command(name, arguments or {}, self._sandbox)
        if self._user_tool_handler:
            # Provider responses are untrusted and may name a tool outside the
            # turn's resolved toolset. Once context construction has resolved
            # that invocation-scoped set, fail closed instead of forwarding a
            # guessed name to a shared host handler. ``None`` preserves direct
            # internal loops built without context; [] is a real deny-all set.
            context_tools = self._get_loop_ctx().all_context_tools
            if context_tools is not None and name not in {t.name for t in context_tools}:
                return json.dumps(
                    {"error": f"Tool '{name}' is not available in the current turn."}
                )
            result = await self._user_tool_handler(name, arguments)
            # A multimodal result (content-part list, e.g. a screenshot) must
            # reach the provider intact — str() would flatten it to its repr.
            if isinstance(result, list):
                return result
            return str(result)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        loop_ctx = self._get_loop_ctx()
        skill = self._skills.get_skill(skill_name) if skill_name else None
        if skill_name and skill is None:
            # A known-but-unavailable skill already carries its reason in the
            # error — a "this is not a skill" tools hint would contradict it.
            if self._skills.get_unavailable_reason(skill_name) is None:
                # Small models routinely confuse skills with TOOLS ("activate the
                # Spotify skill" when SpotifySearch/... are tools). Turn the dead
                # end into the right outcome: reveal the matching tools and say so.
                wanted = skill_name.lower()
                matching = sorted(
                    t.name for t in loop_ctx.all_context_tools or () if wanted in t.name.lower()
                )
                if matching:
                    loop_ctx.revealed_tools.update(matching)
                    data = json.loads(result_str)
                    data["tools_hint"] = (
                        f"{skill_name!r} is not a skill, but these TOOLS match and are "
                        f"now in your tool list — call one directly instead: "
                        f"{', '.join(matching[:8])}."
                    )
                    result_str = json.dumps(data)
            return result_str
        # Track activation so gated tools become visible on next round
        loop_ctx.activated_skills.add(skill_name)
        # ... and for the rest of the conversation, so the body can ride the
        # system prompt instead of being re-fetched every turn.
        if skill is None or self._skill_activation.activate(loop_ctx.room_id, skill_name):
            return result_str
        # Already active: _build_context put these very instructions in front of
        # the model before the turn started, so the body just built above would
        # be a second copy of rules it already holds. Ack instead.
        return activation_ack(skill, ALREADY_ACTIVE_NOTE, already_active=True)

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

    @staticmethod
    def _tool_search_catalogue(loop_ctx: _ToolLoopContext) -> list[dict[str, Any]]:
        """The turn's full tool list as score-able dicts (name + description + tags)."""
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "tags": getattr(t, "tags", []) or [],
            }
            for t in loop_ctx.all_context_tools or ()
        ]

    async def _handle_find_tools(self, arguments: dict[str, Any]) -> str:
        """Reveal catalogue tools matching a query for the rest of the loop.

        Mutates ``loop_ctx.revealed_tools`` (swap window); the next round's
        tool re-filter exposes the matches. No ``provider.reconfigure`` — the
        text loop re-sends its tool list every round.
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
        exclude = self._tool_search_pinned | TOOL_SEARCH_INFRA_TOOL_NAMES
        matches = search_catalogue(catalogue, query, max_results, exclude_names=exclude)
        loop_ctx.revealed_tools = {m["name"] for m in matches if m.get("name")}
        # Reveals persist across turns via ToolUsageMemory (the tool's own
        # description promises "the rest of the session") — a tool found in
        # turn N is often only called in turn N+1, after the user confirms.
        self._tool_usage.record_revealed(loop_ctx.room_id, loop_ctx.revealed_tools)
        # Compact result (name + short description). The matched tools' full
        # schemas reach the model via the next round's re-filtered tool list
        # (loop_ctx.revealed_tools), so inlining them here would only risk
        # overflowing the tool-result size limit on verbose tools.
        return render_find_payload(
            matches,
            miss_hint=self._tool_search_miss_hint,
            related=related_family_tools(catalogue, matches),
        )

    async def _handle_list_tools(self, arguments: dict[str, Any]) -> str:
        """List the turn's catalogue (name + short description). Reveals nothing."""
        loop_ctx = self._get_loop_ctx()
        category = str(arguments.get("category", "")).strip()
        catalogue = self._tool_search_catalogue(loop_ctx)
        return render_list_payload(catalogue, category, exclude_names=TOOL_SEARCH_INFRA_TOOL_NAMES)

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
