"""AIChannel mixin for tool execution, dispatch, and skill tool handlers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._sandbox_handlers import handle_sandbox_command
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._skill_handlers import (
    handle_activate_skill,
    handle_read_reference,
    handle_run_script,
)
from roomkit.channels._tool_search import (
    normalize_max_results,
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
from roomkit.providers.ai.base import AITool, AIToolResultPart
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX
from roomkit.telemetry.base import SpanKind
from roomkit.tools.human_input import ToolCallContext, _current_tool_call

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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

    ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

logger = logging.getLogger("roomkit.channels.ai")


@runtime_checkable
class AIToolsHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolsMixin.

    Attributes provided by the host's ``__init__``:
        _tool_handler: Tool call handler (or ``None`` if tools disabled).
        _user_tool_handler: User-provided tool handler for fallback dispatch.
        _skills: Skill registry for gated tool resolution.
        _script_executor: Script executor for skill scripts.
        _sandbox: Sandbox executor for ad-hoc command execution.
        _eviction: Tool result eviction / truncation strategy.
        _planner: Optional task planner.
        _realtime: Realtime backend for ephemeral events.
        _current_room_id: Current room ID for tool-call hook events.
        _tool_call_hook: Optional unified ON_TOOL_CALL hook callback.
        channel_id: Unique identifier for this channel.

    Properties / methods provided by other mixins:
        _effective_tool_policy: ``AIToolPolicyMixin`` property — resolved policy.
        _SKILL_INFRA_TOOLS: ``AIToolPolicyMixin`` class var — infra tool names.
        _gated_tool_names: ``AIToolPolicyMixin`` property — gated tool names.
        _maybe_truncate_result: ``AIResilienceMixin`` — truncate large results.
        _get_loop_ctx: ``AISteeringMixin`` — returns current tool-loop context.
    """

    _tool_handler: Any
    _user_tool_handler: Any
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _planner: TaskPlanner | None
    _realtime: RealtimeBackend | None
    _current_room_id: str | None
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

    def _maybe_truncate_result(self, result: str, tool_call_id: str = ...) -> str: ...
    def _get_loop_ctx(self) -> _ToolLoopContext: ...


class AIToolsMixin:
    """Parallel tool execution, skill tool definitions, and dispatch routing.

    Host contract: :class:`AIToolsHost`.
    """

    _tool_handler: Any
    _user_tool_handler: Any
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _planner: TaskPlanner | None
    _realtime: RealtimeBackend | None
    _current_room_id: str | None
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

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        parent_span_id: str | None = None,
    ) -> list[_ContentPart]:
        """Execute tool calls concurrently and return result parts."""
        if self._tool_handler is None:
            raise RuntimeError("_execute_tools_parallel called without a tool handler")
        handler = self._tool_handler

        async def _run_one(tc: Any) -> AIToolResultPart:
            logger.info("Executing tool: %s(%s)", tc.name, tc.id)

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

            # Pre-execution gate: BEFORE_TOOL_USE hook can deny the tool call
            if self._before_tool_call_hook is not None:
                from roomkit.models.tool_call import ToolCallEvent

                pre_event = ToolCallEvent(
                    channel_id=self.channel_id,
                    channel_type=ChannelType.AI,
                    tool_call_id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    result=None,
                    room_id=self._current_room_id,
                )
                allowed = await self._before_tool_call_hook(pre_event)
                if not allowed:
                    logger.info("Tool %s denied by BEFORE_TOOL_USE hook", tc.name)
                    return AIToolResultPart(
                        tool_call_id=tc.id,
                        name=tc.name,
                        result=json.dumps(
                            {"error": f"Tool '{tc.name}' denied by pre-execution hook."}
                        ),
                    )

            tool_span_id = telemetry.start_span(
                SpanKind.LLM_TOOL_CALL,
                f"tool.{tc.name}",
                parent_id=parent_span_id,
                attributes={"tool.name": tc.name, "tool.id": tc.id},
            )
            try:
                # Set contextvar so HumanInputToolHandler can read
                # room_id / tool_call_id / channel_id without protocol changes.
                _tc_ctx = ToolCallContext(
                    room_id=self._current_room_id or "",
                    tool_call_id=tc.id,
                    channel_id=self.channel_id,
                )
                _tc_tok = _current_tool_call.set(_tc_ctx)
                try:
                    result = await handler(tc.name, tc.arguments)
                finally:
                    _current_tool_call.reset(_tc_tok)
                result = self._maybe_truncate_result(result, tc.id)

                # Fire unified ON_TOOL_CALL hook (if framework injected callback)
                if self._tool_call_hook is not None:
                    from roomkit.models.tool_call import ToolCallEvent

                    event = ToolCallEvent(
                        channel_id=self.channel_id,
                        channel_type=ChannelType.AI,
                        tool_call_id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        result=result,
                        room_id=self._current_room_id,
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
            self._tool_usage.record(self._current_room_id, tc.name, tc.arguments, result)
            return AIToolResultPart(
                tool_call_id=tc.id,
                name=tc.name,
                result=result,
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

    async def _channel_tool_handler(self, name: str, arguments: dict[str, Any]) -> str:
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
            return str(await self._user_tool_handler(name, arguments))
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        loop_ctx = self._get_loop_ctx()
        if skill_name and self._skills.get_skill(skill_name) is None:
            # Small models routinely confuse skills with TOOLS ("activate the
            # Spotify skill" when SpotifySearch/... are tools). Turn the dead
            # end into the right outcome: reveal the matching tools and say so.
            wanted = skill_name.lower()
            matching = sorted(
                t.name for t in loop_ctx.all_context_tools if wanted in t.name.lower()
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
        return result_str

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
            for t in loop_ctx.all_context_tools
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
        # Compact result (name + short description). The matched tools' full
        # schemas reach the model via the next round's re-filtered tool list
        # (loop_ctx.revealed_tools), so inlining them here would only risk
        # overflowing the tool-result size limit on verbose tools.
        return render_find_payload(matches, miss_hint=self._tool_search_miss_hint)

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
        return await self._planner.handle_plan_tasks(
            arguments,
            realtime=self._realtime,
            room_id=self._current_room_id,
            channel_id=self.channel_id,
        )
