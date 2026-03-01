"""Tool integration for agent delegation.

Mirrors the handoff pattern in ``orchestration/handoff.py``:
tool definition, handler class, and ``setup_delegation()`` wiring.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.orchestration.handoff import _room_id_var
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.delivery import BackgroundTaskDeliveryStrategy

logger = logging.getLogger("roomkit.tasks")


# -- Tool definition ----------------------------------------------------------

DELEGATE_TOOL = AITool(
    name="delegate_task",
    description=(
        "Delegate a task to a background agent. The agent works in its own "
        "room and you can continue the current conversation. Use when: "
        "a task needs a specialist, the user wants something done in the "
        "background, or you need to run work in parallel."
    ),
    parameters={
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Target agent ID for the delegated task",
            },
            "task": {
                "type": "string",
                "description": "Clear description of what the agent should do",
            },
            "context": {
                "type": "object",
                "description": ("Optional context to pass to the agent (e.g. email, repo URL)"),
            },
            "share_channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Channel IDs from the current room to share with the background agent"
                ),
            },
        },
        "required": ["agent", "task"],
    },
)


def build_delegate_tool(targets: list[tuple[str, str | None]]) -> AITool:
    """Build a delegate tool with constrained agent enum.

    Args:
        targets: List of ``(agent_id, description_or_none)`` pairs.
            When empty, returns the generic :data:`DELEGATE_TOOL`.

    Returns:
        An :class:`AITool` whose ``agent`` parameter has an ``enum``
        restricting the AI to only the listed agent IDs.
    """
    if not targets:
        return DELEGATE_TOOL

    target_ids = [t[0] for t in targets]
    target_lines = []
    for agent_id, desc in targets:
        if desc:
            target_lines.append(f"  - {agent_id}: {desc}")
        else:
            target_lines.append(f"  - {agent_id}")

    description = "Delegate a task to a background agent.\nAvailable agents:\n" + "\n".join(
        target_lines
    )

    params = {
        "type": "object",
        "properties": {
            **DELEGATE_TOOL.parameters["properties"],
            "agent": {
                "type": "string",
                "enum": target_ids,
                "description": "Target agent ID for the delegated task",
            },
        },
        "required": DELEGATE_TOOL.parameters["required"],
    }

    return AITool(
        name="delegate_task",
        description=description,
        parameters=params,
    )


# -- DelegateHandler ----------------------------------------------------------


class DelegateHandler:
    """Processes ``delegate_task`` tool calls by calling ``kit.delegate()``."""

    def __init__(
        self,
        kit: RoomKit,
        *,
        notify: str | None = None,
        default_share_channels: list[str] | None = None,
        delivery_strategy: BackgroundTaskDeliveryStrategy | None = None,
    ) -> None:
        self._kit = kit
        self._notify = notify
        self._default_share_channels = default_share_channels or []
        self._delivery_strategy = delivery_strategy

    async def handle(
        self,
        room_id: str,
        calling_agent_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a delegate_task tool call."""
        agent_id = arguments.get("agent", "")
        task_text = arguments.get("task", "")
        context = arguments.get("context") or {}
        share_channels = arguments.get("share_channels") or self._default_share_channels

        task = await self._kit.delegate(
            room_id=room_id,
            agent_id=agent_id,
            task=task_text,
            context=context,
            share_channels=share_channels,
            notify=self._notify or calling_agent_id,
            delivery_strategy=self._delivery_strategy,
        )

        return {
            "status": "delegated",
            "task_id": task.id,
            "child_room_id": task.child_room_id,
            "agent_id": agent_id,
        }


# -- Wiring -------------------------------------------------------------------


def setup_delegation(
    channel: AIChannel,
    handler: DelegateHandler,
    *,
    tool: AITool | None = None,
) -> None:
    """Wire delegation into an AIChannel's tool chain.

    Injects the delegate tool and wraps the tool handler to intercept
    ``delegate_task`` calls. Same pattern as ``setup_handoff()``.

    Args:
        channel: The AI channel to wire delegation into.
        handler: The delegate handler that processes tool calls.
        tool: Optional custom delegate tool (e.g. from :func:`build_delegate_tool`).
    """
    if any(t.name == "delegate_task" for t in channel._extra_tools):
        msg = f"setup_delegation() already called for channel '{channel.channel_id}'"
        raise RuntimeError(msg)

    channel._extra_tools.append(tool or DELEGATE_TOOL)

    original = channel._tool_handler

    async def delegate_aware_handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "delegate_task":
            room_id = _room_id_var.get()
            if room_id is None:
                return json.dumps({"error": "No orchestration context (room_id unavailable)"})
            result = await handler.handle(
                room_id=room_id,
                calling_agent_id=channel.channel_id,
                arguments=arguments,
            )
            return json.dumps(result)
        if original:
            return await original(name, arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    channel._tool_handler = delegate_aware_handler
