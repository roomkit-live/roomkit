"""Supervisor orchestration strategy.

A supervisor agent delegates tasks to worker agents via ``kit.delegate()``.
Workers are registered on the kit but NOT attached to the parent room.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import (
    ConversationState,
    set_conversation_state,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.supervisor")


class Supervisor(Orchestration):
    """Supervisor orchestration strategy.

    The supervisor handles all user interaction. Workers are registered
    on the kit (so ``delegate()`` can find them) but are NOT attached
    to the parent room — they run in child rooms.

    The supervisor receives ``delegate_to_<worker>`` tools that wrap
    ``kit.delegate()``.

    Example::

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=manager,
                workers=[researcher, coder],
            ),
        )
        room = await kit.create_room()
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: list[Agent],
        *,
        wait_for_result: bool = False,
        result_timeout: float = 120.0,
    ) -> None:
        """Initialise the supervisor strategy.

        Args:
            supervisor: The agent that handles user interaction and
                delegates tasks.
            workers: Agents that run delegated tasks in child rooms.
            wait_for_result: If ``True``, delegation tool calls block
                until the worker completes and return the worker's
                output directly.  If ``False`` (default), the tool
                returns immediately and the result is injected into
                the supervisor's system prompt later.
            result_timeout: Maximum seconds to wait for a worker result
                when *wait_for_result* is ``True`` (default 120s).
        """
        self._supervisor = supervisor
        self._workers = list(workers)
        self._wait_for_result = wait_for_result
        self._result_timeout = result_timeout

    def agents(self) -> list[Agent]:
        """Return only the supervisor — workers are not attached to the room."""
        return [self._supervisor]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire supervisor routing and delegation tools."""
        router = ConversationRouter(
            default_agent_id=self._supervisor.channel_id,
        )

        # Install router as room-scoped BEFORE_BROADCAST hook
        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=router.as_hook(),
                priority=-100,
                name=f"supervisor_router_{room_id}",
            ),
        )

        # Register workers on the kit (not attached to room)
        for worker in self._workers:
            if worker.channel_id not in kit.channels:
                kit.register_channel(worker)

        # Inject delegation tools into supervisor
        self._inject_delegation_tools(kit, room_id)

        # Set initial conversation state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase="supervisor",
            active_agent_id=self._supervisor.channel_id,
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    def _inject_delegation_tools(self, kit: RoomKit, room_id: str) -> None:
        """Create and inject per-worker delegation tools."""
        from roomkit.orchestration.handoff import _room_id_var

        any_new = False
        for worker in self._workers:
            tool_name = f"delegate_to_{worker.channel_id}"

            # Skip if already injected
            if any(t.name == tool_name for t in self._supervisor._injected_tools):
                continue

            any_new = True
            desc = getattr(worker, "description", None) or f"Worker agent {worker.channel_id}"
            tool = AITool(
                name=tool_name,
                description=f"Delegate a task to {worker.channel_id}. {desc}",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of the task to delegate",
                        },
                    },
                    "required": ["task"],
                },
            )
            self._supervisor._injected_tools.append(tool)

        # Only wrap the tool handler once — guard against double install
        if not any_new:
            return

        # Wrap the tool handler to intercept delegation calls
        original = self._supervisor.tool_handler
        tool_to_worker = {f"delegate_to_{w.channel_id}": w.channel_id for w in self._workers}

        wait = self._wait_for_result

        async def delegation_handler(name: str, arguments: dict[str, Any]) -> str:
            worker_id = tool_to_worker.get(name)
            if worker_id is not None:
                rid = _room_id_var.get() or room_id
                task_desc = arguments.get("task", "")
                try:
                    if wait:
                        # Run worker inline to avoid async scheduling issues
                        # when called from inside a streaming tool loop.
                        output = await _run_worker_inline(kit, rid, worker_id, task_desc)
                        return json.dumps(
                            {
                                "status": "completed",
                                "worker": worker_id,
                                "result": output or "",
                            }
                        )
                    delegated = await kit.delegate(
                        rid,
                        worker_id,
                        task_desc,
                        notify=self._supervisor.channel_id,
                    )
                    return json.dumps(
                        {
                            "status": "delegated",
                            "task_id": delegated.id,
                            "worker": worker_id,
                        }
                    )
                except Exception as exc:
                    logger.exception("Delegation to %s failed", worker_id)
                    return json.dumps({"error": str(exc)})
            if original:
                return await original(name, arguments)
            return json.dumps({"error": f"Unknown tool: {name}"})

        self._supervisor.tool_handler = delegation_handler


async def _run_worker_inline(
    kit: RoomKit,
    parent_room_id: str,
    worker_id: str,
    task_desc: str,
) -> str | None:
    """Run a worker agent inline and return its text output.

    Creates a temporary child room, sends the task as a message,
    and collects the worker's response — all synchronously within
    the caller's coroutine (no background task).
    """
    import time
    from uuid import uuid4

    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import (
        ChannelCategory,
        ChannelType,
        EventStatus,
        EventType,
        TaskStatus,
    )
    from roomkit.models.event import EventSource, RoomEvent, TextContent

    task_id = f"task-{uuid4().hex[:12]}"
    child_room_id = f"{parent_room_id}::inline-{uuid4().hex[:12]}"
    start = time.monotonic()

    # Fire ON_TASK_DELEGATED hook
    delegated_event = RoomEvent(
        room_id=parent_room_id,
        source=EventSource(channel_id=worker_id, channel_type=ChannelType.AI),
        content=TextContent(body=f"[Task delegated to {worker_id}] {task_desc}"),
        type=EventType.TASK_DELEGATED,
        status=EventStatus.DELIVERED,
        visibility="internal",
        metadata={"task_id": task_id, "child_room_id": child_room_id, "agent_id": worker_id},
    )
    parent_context = await kit._build_context(parent_room_id)
    await kit._hook_engine.run_async_hooks(
        parent_room_id, HookTrigger.ON_TASK_DELEGATED, delegated_event, parent_context
    )

    await kit.create_room(
        room_id=child_room_id,
        metadata={"parent_room_id": parent_room_id, "task_agent_id": worker_id},
        orchestration=None,  # no orchestration — worker runs standalone
    )
    await kit.attach_channel(child_room_id, worker_id, category=ChannelCategory.INTELLIGENCE)

    # Store the task as a message
    task_event = RoomEvent(
        room_id=child_room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body=task_desc),
    )
    task_event = await kit.store.add_event_auto_index(child_room_id, task_event)

    # Build context with the task event visible
    room = await kit.get_room(child_room_id)
    bindings = await kit.store.list_bindings(child_room_id)
    recent = await kit.store.list_events(child_room_id, offset=0, limit=50)
    context = RoomContext(room=room, bindings=bindings, recent_events=recent)

    # Broadcast — the worker picks up the task
    router = kit._get_router()
    source_binding = ChannelBinding(
        channel_id="system",
        room_id=child_room_id,
        channel_type=ChannelType.SYSTEM,
    )
    result = await router.broadcast(task_event, source_binding, context)

    # Collect response: check synchronous output first, then streaming
    agent_output: str | None = None
    for output in result.outputs.values():
        if output.responded and output.response_events:
            for resp in output.response_events:
                if isinstance(resp.content, TextContent) and resp.content.body:
                    agent_output = resp.content.body
                    break

    if agent_output is None:
        for sr in result.streaming_responses:
            parts: list[str] = []
            async for delta in sr.stream:
                parts.append(delta)
            text = "".join(parts)
            if text:
                agent_output = text
                break

    # Fire ON_TASK_COMPLETED hook
    elapsed = (time.monotonic() - start) * 1000
    status = TaskStatus.COMPLETED if agent_output else TaskStatus.FAILED
    completed_event = RoomEvent(
        room_id=parent_room_id,
        source=EventSource(channel_id=worker_id, channel_type=ChannelType.AI),
        content=TextContent(body=agent_output or ""),
        type=EventType.TASK_COMPLETED,
        status=EventStatus.DELIVERED,
        visibility="internal",
        metadata={
            "task_id": task_id,
            "child_room_id": child_room_id,
            "agent_id": worker_id,
            "task_status": status,
            "duration_ms": elapsed,
        },
    )
    try:
        parent_context = await kit._build_context(parent_room_id)
        await kit._hook_engine.run_async_hooks(
            parent_room_id, HookTrigger.ON_TASK_COMPLETED, completed_event, parent_context
        )
    except Exception:
        logger.exception("Failed to fire ON_TASK_COMPLETED for %s", task_id)

    return agent_output
