"""Structured task planning for AI agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from roomkit.providers.ai.base import AITool
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType, RealtimeBackend

logger = logging.getLogger("roomkit.channels.ai")

_STATUS_ICONS = {
    "completed": "[x]",
    "in_progress": "[-]",
    "blocked": "[!]",
    "pending": "[ ]",
}


class TaskPlanner:
    """Manages a structured task plan for an AI agent.

    Provides the ``_plan_tasks`` tool for creating/updating plans,
    formats the plan into system prompt context, and publishes
    ephemeral events for real-time UI rendering.
    """

    def __init__(self) -> None:
        self.current_plan: list[dict[str, Any]] | None = None

    async def handle_plan_tasks(
        self,
        arguments: dict[str, Any],
        *,
        realtime: RealtimeBackend | None = None,
        room_id: str | None = None,
        channel_id: str | None = None,
    ) -> str:
        """Store a task plan and publish an ephemeral update event."""
        tasks = arguments.get("tasks", [])
        self.current_plan = tasks

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

        counts = {
            s: sum(1 for t in tasks if t.get("status") == s)
            for s in ("pending", "in_progress", "completed", "blocked")
        }
        return json.dumps({"status": "ok", "task_count": len(tasks), **counts})

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
        """Return the AITool definition for _plan_tasks."""
        return AITool(
            name="_plan_tasks",
            description=(
                "Create or update a structured task plan. "
                "Use this to break down complex work into steps and track progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "blocked"],
                                },
                            },
                            "required": ["title", "status"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        )
