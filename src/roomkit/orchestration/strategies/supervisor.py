"""Supervisor orchestration strategy.

A supervisor agent delegates tasks to worker agents via ``kit.delegate()``.
Workers are registered on the kit but NOT attached to the parent room.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import StrEnum
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


class WorkerStrategy(StrEnum):
    """How workers are executed when delegated to."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class Supervisor(Orchestration):
    """Supervisor orchestration strategy.

    The supervisor handles all user interaction. Workers are registered
    on the kit (so ``delegate()`` can find them) but are NOT attached
    to the parent room — they run in child rooms.

    Examples::

        # Sequential: researcher → writer (chained)
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
        )

        # Parallel: technical + business (fan-out)
        Supervisor(
            supervisor=coordinator,
            workers=[technical, business],
            strategy="parallel",
        )

        # Manual: AI decides via per-worker tools (default)
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
        )
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: list[Agent],
        *,
        strategy: WorkerStrategy | str | None = None,
        wait_for_result: bool = True,
    ) -> None:
        """Initialise the supervisor strategy.

        Args:
            supervisor: The agent that handles user interaction and
                delegates tasks.
            workers: Agents that run delegated tasks in child rooms.
            strategy: Deterministic execution pattern for workers.

                - ``"sequential"``: workers run in order, each receiving
                  the previous worker's output. One ``delegate_workers``
                  tool is injected.
                - ``"parallel"``: all workers run concurrently on the
                  same task. One ``delegate_workers`` tool is injected.
                - ``None`` (default): per-worker ``delegate_to_<id>``
                  tools are injected and the AI decides when to call
                  them.

            wait_for_result: When *strategy* is ``None``, controls
                whether delegation runs inline (``True``, default) or
                in the background (``False``).  Ignored when *strategy*
                is set (always inline).
        """
        self._supervisor = supervisor
        self._workers = list(workers)
        self._strategy = WorkerStrategy(strategy) if strategy else None
        self._wait_for_result = wait_for_result

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

        # Inject tools based on strategy
        if self._strategy is not None:
            self._inject_strategy_tool(kit, room_id)
        else:
            self._inject_per_worker_tools(kit, room_id)

        # Set initial conversation state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase="supervisor",
            active_agent_id=self._supervisor.channel_id,
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)

    # -- Strategy-based tool (sequential / parallel) --------------------------

    def _inject_strategy_tool(self, kit: RoomKit, room_id: str) -> None:
        """Inject a single ``delegate_workers`` tool for deterministic execution."""
        from roomkit.orchestration.handoff import _room_id_var

        tool_name = "delegate_workers"

        if any(t.name == tool_name for t in self._supervisor._injected_tools):
            return

        worker_names = ", ".join(w.channel_id for w in self._workers)
        mode = self._strategy
        tool = AITool(
            name=tool_name,
            description=(
                f"Delegate a task to workers ({worker_names}). "
                f"Workers run {mode}. Call this tool with the task description."
            ),
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

        original = self._supervisor.tool_handler
        strategy = self._strategy
        workers = self._workers
        # Lock prevents concurrent duplicate calls when asyncio.gather
        # runs multiple tool calls from the same AI response in parallel.
        # Cache is keyed by task description so different tasks run fresh
        # but the same task (retried by the AI in the same turn) is deduped.
        _lock = asyncio.Lock()
        _cache: dict[str, str] = {}

        async def strategy_handler(name: str, arguments: dict[str, Any]) -> str:
            if name != tool_name:
                if original:
                    return await original(name, arguments)
                return json.dumps({"error": f"Unknown tool: {name}"})

            rid = _room_id_var.get() or room_id
            task_desc = arguments.get("task", "")

            async with _lock:
                # Return cached result if same task already ran in this turn
                if task_desc in _cache:
                    return _cache[task_desc]

                try:
                    if strategy == WorkerStrategy.SEQUENTIAL:
                        result = await _run_sequential(kit, rid, workers, task_desc)
                    else:
                        result = await _run_parallel(kit, rid, workers, task_desc)
                    _cache[task_desc] = result
                    return result
                except Exception as exc:
                    logger.exception("Strategy delegation failed")
                    return json.dumps({"error": str(exc)})

        self._supervisor.tool_handler = strategy_handler

    # -- Per-worker tools (manual mode) ---------------------------------------

    def _inject_per_worker_tools(self, kit: RoomKit, room_id: str) -> None:
        """Inject per-worker ``delegate_to_<id>`` tools (AI decides)."""
        from roomkit.orchestration.handoff import _room_id_var

        any_new = False
        for worker in self._workers:
            tool_name = f"delegate_to_{worker.channel_id}"

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

        if not any_new:
            return

        original = self._supervisor.tool_handler
        tool_to_worker = {f"delegate_to_{w.channel_id}": w.channel_id for w in self._workers}
        wait = self._wait_for_result
        pending: set[str] = set()

        async def delegation_handler(name: str, arguments: dict[str, Any]) -> str:
            worker_id = tool_to_worker.get(name)
            if worker_id is not None:
                rid = _room_id_var.get() or room_id
                task_desc = arguments.get("task", "")
                try:
                    if wait:
                        delegated = await kit.delegate(
                            rid,
                            worker_id,
                            task_desc,
                            wait=True,
                            notify=self._supervisor.channel_id,
                        )
                        result = delegated.result
                        return json.dumps(
                            {
                                "status": result.status if result else "failed",
                                "worker": worker_id,
                                "result": (result.output or result.error or "") if result else "",
                            }
                        )

                    if worker_id in pending:
                        return json.dumps(
                            {
                                "status": "already_running",
                                "worker": worker_id,
                                "message": (
                                    f"{worker_id} is already working on this. "
                                    "Do NOT call this tool again. "
                                    "Tell the user to ask again shortly."
                                ),
                            }
                        )

                    delegated = await kit.delegate(
                        rid,
                        worker_id,
                        task_desc,
                        notify=self._supervisor.channel_id,
                    )
                    pending.add(worker_id)

                    original_set = delegated._set_result

                    def _patched_set(r: Any, *, _wid: str = worker_id) -> None:
                        pending.discard(_wid)
                        original_set(r)

                    delegated._set_result = _patched_set  # type: ignore[assignment]

                    return json.dumps(
                        {
                            "status": "delegated",
                            "task_id": delegated.id,
                            "worker": worker_id,
                            "message": (
                                f"Task dispatched to {worker_id}. "
                                "It is running in the background. "
                                "Do NOT call this tool again. "
                                "Tell the user to ask again shortly "
                                "for results."
                            ),
                        }
                    )
                except Exception as exc:
                    logger.exception("Delegation to %s failed", worker_id)
                    return json.dumps({"error": str(exc)})
            if original:
                return await original(name, arguments)
            return json.dumps({"error": f"Unknown tool: {name}"})

        self._supervisor.tool_handler = delegation_handler


# ---------------------------------------------------------------------------
# Strategy execution helpers
# ---------------------------------------------------------------------------


async def _run_sequential(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
) -> str:
    """Run workers in order, each receiving the previous output."""
    current_input = task_desc
    results: list[dict[str, str]] = []

    for worker in workers:
        delegated = await kit.delegate(
            room_id,
            worker.channel_id,
            current_input,
            wait=True,
        )
        output = ""
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
        results.append({"worker": worker.channel_id, "output": output})
        # Next worker receives this worker's output
        current_input = output

    return json.dumps({"status": "completed", "results": results})


async def _run_parallel(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
) -> str:
    """Run all workers concurrently on the same task."""

    async def _delegate_one(worker: Agent) -> dict[str, str]:
        delegated = await kit.delegate(
            room_id,
            worker.channel_id,
            task_desc,
            wait=True,
        )
        output = ""
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
        return {"worker": worker.channel_id, "output": output}

    results = await asyncio.gather(*[_delegate_one(w) for w in workers])
    return json.dumps({"status": "completed", "results": list(results)})
