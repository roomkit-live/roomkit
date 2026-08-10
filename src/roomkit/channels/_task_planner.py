"""Structured task planning for AI agents."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.providers.ai.base import AITool
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType, RealtimeBackend

logger = logging.getLogger("roomkit.channels.ai")

PlanUpdatedCallback = Callable[[str, list[dict[str, Any]]], Awaitable[None]]
"""Framework-injected ON_PLAN_UPDATED callback: ``(room_id, tasks)``."""

_STATUS_ICONS = {
    "completed": "[x]",
    "in_progress": "[-]",
    "blocked": "[!]",
    "pending": "[ ]",
}
_MAX_ROOMS = 100
_MAX_TASKS = 100
_MAX_TITLE_LENGTH = 500


class TaskPlanner:
    """Manages room-scoped structured task plans for an AI agent.

    Provides the ``plan_tasks`` tool for creating/updating plans,
    formats the plan into system prompt context, and publishes
    ephemeral events for real-time UI rendering.
    """

    def __init__(self) -> None:
        self._plans: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    @property
    def current_plan(self) -> list[dict[str, Any]] | None:
        """Plan stored outside a room context, retained for compatibility."""
        return self.plan_for(None)

    def plan_for(self, room_id: str | None) -> list[dict[str, Any]] | None:
        """Return the plan for one room without exposing another room's state."""
        key = room_id or ""
        plan = self._plans.get(key)
        if plan is not None:
            self._plans.move_to_end(key)
        return plan

    async def handle_plan_tasks(
        self,
        arguments: dict[str, Any],
        *,
        realtime: RealtimeBackend | None = None,
        room_id: str | None = None,
        channel_id: str | None = None,
        on_plan_updated: PlanUpdatedCallback | None = None,
    ) -> str:
        """Store a task plan, publish an ephemeral update and fire the hook.

        ``on_plan_updated`` is the framework-injected ON_PLAN_UPDATED callback
        (RFC §9.2); it runs whether or not a realtime backend is configured.
        """
        tasks, error = self._validated_tasks(arguments.get("tasks", []))
        if error is not None:
            return json.dumps({"error": error})

        key = room_id or ""
        self._plans[key] = tasks
        self._plans.move_to_end(key)
        while len(self._plans) > _MAX_ROOMS:
            self._plans.popitem(last=False)

        if realtime and room_id:
            try:
                await realtime.publish_to_room(
                    room_id,
                    EphemeralEvent(
                        room_id=room_id,
                        type=EphemeralEventType.CUSTOM,
                        user_id=channel_id or "",
                        channel_id=channel_id,
                        data={"type": "plan_updated", "tasks": tasks},
                    ),
                )
            except Exception:
                logger.debug("Failed to publish plan ephemeral event", exc_info=True)

        if on_plan_updated is not None and room_id:
            try:
                await on_plan_updated(room_id, tasks)
            except Exception:
                logger.debug("ON_PLAN_UPDATED hook failed", exc_info=True)

        counts = {
            s: sum(1 for t in tasks if t.get("status") == s)
            for s in ("pending", "in_progress", "completed", "blocked")
        }
        return json.dumps({"status": "ok", "task_count": len(tasks), **counts})

    @staticmethod
    def _validated_tasks(value: Any) -> tuple[list[dict[str, Any]], str | None]:
        """Validate model-authored plan data before it can alter room state.

        ``plan_tasks`` is channel-managed, so the generic tool guard does not
        validate its nested JSON schema. Treat the model's arguments as
        untrusted here: a malformed item must become a tool error, not a value
        stored in the room cache that breaks every later context rebuild.
        """
        if not isinstance(value, list):
            return [], "Invalid plan: 'tasks' must be an array"
        if len(value) > _MAX_TASKS:
            return [], f"Invalid plan: at most {_MAX_TASKS} tasks are allowed"

        tasks: list[dict[str, Any]] = []
        for index, task in enumerate(value):
            if not isinstance(task, dict):
                return [], f"Invalid plan: task {index} must be an object"
            title = task.get("title")
            if not isinstance(title, str):
                return [], f"Invalid plan: task {index} title must be a string"
            if len(title) > _MAX_TITLE_LENGTH:
                return [], (
                    f"Invalid plan: task {index} title must be at most "
                    f"{_MAX_TITLE_LENGTH} characters"
                )
            status = task.get("status")
            if not isinstance(status, str) or status not in _STATUS_ICONS:
                return [], (
                    f"Invalid plan: task {index} status must be one of {', '.join(_STATUS_ICONS)}"
                )
            # Retain only the declared contract. Model-authored extra fields
            # may contain arbitrarily large nested data and must not reach the
            # room cache, realtime payload, or next-turn prompt.
            tasks.append({"title": title, "status": status})
        return tasks, None

    @staticmethod
    def format_plan_prompt(tasks: list[dict[str, Any]]) -> str:
        """Format the current plan as a system prompt block."""
        lines = ["\n\n## Current Task Plan"]
        for t in tasks:
            icon = _STATUS_ICONS.get(t.get("status", "pending"), "[ ]")
            title = t.get("title", "Untitled")
            status = t.get("status", "pending")
            lines.append(f"- {icon} {title} ({status})")
        return "\n".join(lines)

    @staticmethod
    def tool_definition() -> AITool:
        """Return the AITool definition for plan_tasks."""
        return AITool(
            name="plan_tasks",
            description=(
                "Create or update a structured task plan. "
                "Use this to break down complex work into steps and track progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "maxItems": _MAX_TASKS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "maxLength": _MAX_TITLE_LENGTH,
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "blocked"],
                                },
                            },
                            "required": ["title", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
        )
