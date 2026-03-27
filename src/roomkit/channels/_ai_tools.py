"""AIChannel mixin for tool execution, dispatch, and skill tool handlers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

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
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AITool, AIToolResultPart
from roomkit.telemetry.base import SpanKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.channels._task_planner import TaskPlanner
    from roomkit.channels._tool_eviction import ToolEviction
    from roomkit.channels.ai import _ContentPart
    from roomkit.models.tool_call import ToolCallCallback
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry

    ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

logger = logging.getLogger("roomkit.channels.ai")


class AIToolsMixin:
    """Parallel tool execution, skill tool definitions, and dispatch routing."""

    _tool_handler: Any
    _user_tool_handler: Any
    _skills: SkillRegistry | None
    _script_executor: ScriptExecutor | None
    _eviction: ToolEviction
    _planner: TaskPlanner | None
    _realtime: RealtimeBackend | None
    _current_room_id: str | None
    _tool_call_hook: ToolCallCallback | None
    channel_id: str

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
            effective_policy = self._effective_tool_policy  # type: ignore[attr-defined]
            if (
                tc.name not in self._SKILL_INFRA_TOOLS  # type: ignore[attr-defined]
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
            if tc.name not in self._SKILL_INFRA_TOOLS and tc.name in self._gated_tool_names:  # type: ignore[attr-defined]
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

            tool_span_id = telemetry.start_span(
                SpanKind.LLM_TOOL_CALL,
                f"tool.{tc.name}",
                parent_id=parent_span_id,
                attributes={"tool.name": tc.name, "tool.id": tc.id},
            )
            try:
                result = await handler(tc.name, tc.arguments)
                result = self._maybe_truncate_result(result, tc.id)  # type: ignore[attr-defined]

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
        return dispatch

    async def _channel_tool_handler(self, name: str, arguments: dict[str, Any]) -> str:
        """Unified tool dispatcher: channel-managed -> skill -> user tools."""
        handler = self._channel_tool_dispatch.get(name)
        if handler is not None:
            result = handler(arguments)
            # Support both sync and async handlers
            if asyncio.iscoroutine(result):
                return str(await result)
            return str(result)
        if self._user_tool_handler:
            return str(await self._user_tool_handler(name, arguments))
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        # Track activation so gated tools become visible on next round
        self._get_loop_ctx().activated_skills.add(skill_name)  # type: ignore[attr-defined]
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
