"""Tool integration for agent delegation.

Mirrors the handoff pattern in ``orchestration/handoff.py``:
tool definition, handler class, and ``setup_delegation()`` wiring.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.orchestration.handoff import _room_id_var
from roomkit.providers.ai.base import AITool
from roomkit.tasks.cache import CompletedTaskCache

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
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
    """Processes ``delegate_task`` tool calls by calling ``kit.delegate()``.

    Supports optional dedup via :class:`CompletedTaskCache` and
    per-room concurrency control via asyncio locks.

    Args:
        kit: The RoomKit instance.
        notify: Channel ID to notify on completion (defaults to calling agent).
        default_share_channels: Channels shared with background agents by default.
        delivery_strategy: How to deliver results back proactively.
        cache: Optional completed-task cache for dedup.
        serialize_per_room: If True, only one delegation per room at a time.
    """

    def __init__(
        self,
        kit: RoomKit,
        *,
        notify: str | None = None,
        default_share_channels: list[str] | None = None,
        delivery_strategy: BackgroundTaskDeliveryStrategy | None = None,
        cache: CompletedTaskCache | None = None,
        serialize_per_room: bool = False,
    ) -> None:
        self._kit = kit
        self._notify = notify
        self._default_share_channels = default_share_channels or []
        self._delivery_strategy = delivery_strategy
        self._cache = cache
        self._serialize_per_room = serialize_per_room
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _get_room_lock(self, room_id: str) -> asyncio.Lock:
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        return self._room_locks[room_id]

    async def handle(
        self,
        room_id: str,
        calling_agent_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a delegate_task tool call."""
        agent_id = arguments.get("agent", "")
        task_text = arguments.get("task", "")

        # Gap 13: Check cache for recently completed matching task
        if self._cache is not None:
            cached = self._cache.get(room_id, agent_id, task_text)
            if cached is not None:
                logger.info(
                    "Returning cached result for task '%s' -> agent '%s'",
                    task_text[:60],
                    agent_id,
                )
                return {**cached, "from_cache": True}

        context = arguments.get("context") or {}
        share_channels = arguments.get("share_channels") or self._default_share_channels

        # Gap 15: Inject previous task context from cache
        if self._cache is not None:
            recent = self._cache.recent_context(room_id, limit=3)
            if recent:
                context = {
                    **context,
                    "previous_tasks": recent,
                }

        # Gap 14: Serialize delegations per room
        if self._serialize_per_room:
            lock = self._get_room_lock(room_id)
            async with lock:
                return await self._do_delegate(
                    room_id, calling_agent_id, agent_id, task_text, context, share_channels
                )

        return await self._do_delegate(
            room_id, calling_agent_id, agent_id, task_text, context, share_channels
        )

    async def _do_delegate(
        self,
        room_id: str,
        calling_agent_id: str,
        agent_id: str,
        task_text: str,
        context: dict[str, Any],
        share_channels: list[str],
    ) -> dict[str, Any]:
        task = await self._kit.delegate(
            room_id=room_id,
            agent_id=agent_id,
            task=task_text,
            context=context,
            share_channels=share_channels,
            notify=self._notify or calling_agent_id,
            delivery_strategy=self._delivery_strategy,
        )

        result = {
            "status": "delegated",
            "task_id": task.id,
            "child_room_id": task.child_room_id,
            "agent_id": agent_id,
        }

        # Store in cache when task completes (fire-and-forget)
        if self._cache is not None:
            cache = self._cache
            bg_tasks = self._background_tasks

            async def _cache_on_complete() -> None:
                try:
                    await task.wait(timeout=600.0)
                    cache.put(room_id, agent_id, task_text, result)
                except Exception:
                    pass  # timeout or error — don't cache

            t = asyncio.create_task(_cache_on_complete())
            bg_tasks.add(t)
            t.add_done_callback(bg_tasks.discard)

        return result


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
    if any(t.name == "delegate_task" for t in channel._injected_tools):
        msg = f"setup_delegation() already called for channel '{channel.channel_id}'"
        raise RuntimeError(msg)

    channel._injected_tools.append(tool or DELEGATE_TOOL)

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


def _aitool_to_dict(tool: AITool) -> dict[str, Any]:
    """Convert an AITool to the dict format used by RealtimeVoiceChannel."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def setup_realtime_delegation(
    channel: RealtimeVoiceChannel,
    handler: DelegateHandler,
    *,
    tool: AITool | None = None,
) -> None:
    """Wire delegation into a RealtimeVoiceChannel's tool chain.

    Injects the delegate tool dict into ``channel._tools`` and wraps
    ``channel._tool_handler`` to intercept ``delegate_task`` calls.
    Room ID is resolved from the current voice session via
    ``get_current_voice_session()`` + ``channel.session_rooms``.

    Args:
        channel: The realtime voice channel to wire delegation into.
        handler: The delegate handler that processes tool calls.
        tool: Optional custom delegate tool (e.g. from :func:`build_delegate_tool`).
    """
    from roomkit.channels.realtime_voice import get_current_voice_session

    tool_def = _aitool_to_dict(tool or DELEGATE_TOOL)

    # Guard against double setup
    if channel._tools and any(t.get("name") == "delegate_task" for t in channel._tools):
        msg = f"setup_realtime_delegation() already called for channel '{channel.channel_id}'"
        raise RuntimeError(msg)

    if channel._tools is None:
        channel._tools = [tool_def]
    else:
        channel._tools.append(tool_def)

    original = channel._tool_handler

    async def delegate_aware_handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "delegate_task":
            session = get_current_voice_session()
            room_id: str | None = None
            if session is not None:
                room_id = channel.session_rooms.get(session.id)
            if room_id is None:
                return json.dumps({"error": "No voice session context (room_id unavailable)"})
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
