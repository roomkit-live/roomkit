"""Tool-based delegation wiring for the supervisor.

Mixin for :class:`Supervisor`: injects either a single ``delegate_workers``
tool (deterministic sequential/parallel execution) or per-worker
``delegate_to_<id>`` tools (the AI decides). Host attributes are declared as
annotations; they are set in ``Supervisor.__init__``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _STRATEGY_TOOL_NAME,
    WorkerStrategy,
    _post_worker_status,
    logger,
)
from roomkit.orchestration.strategies.supervisor.delegate import _async_run_and_deliver
from roomkit.orchestration.strategies.supervisor.execution import _run_parallel
from roomkit.orchestration.strategies.supervisor.prompts import _format_supervised_digest
from roomkit.orchestration.strategies.supervisor.results import _format_supervisor_review
from roomkit.orchestration.strategies.supervisor.supervised import (
    _run_supervised_sequential,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class _ToolInjectionMixin:
    """Inject delegation tools onto the supervisor agent."""

    _supervisor: Agent
    _workers: list[Agent]
    _strategy: WorkerStrategy | None
    _share_channels: list[str]
    _async_delivery: bool
    _task_timeout: float
    _max_revisions: int
    _wait_for_result: bool

    def _inject_strategy_tool(self, kit: RoomKit, room_id: str) -> None:
        """Inject a single ``delegate_workers`` tool for deterministic execution."""
        from roomkit.orchestration.handoff import _room_id_var

        tool_name = _STRATEGY_TOOL_NAME

        if any(t.name == tool_name for t in self._supervisor._injected_tools):
            return

        worker_roles = ", ".join(getattr(w, "role", None) or w.channel_id for w in self._workers)
        tool = AITool(
            name=tool_name,
            description=(
                f"Delegate a task to ALL workers ({worker_roles}) at once. "
                f"Call this tool exactly ONCE with the topic. "
                f"All workers run automatically in {self._strategy} mode. "
                f"Do NOT split into separate calls per worker."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The topic or task — sent to all workers as-is",
                    },
                },
                "required": ["task"],
            },
        )
        self._supervisor._injected_tools.append(tool)

        original = self._supervisor.tool_handler
        strategy = self._strategy
        supervisor = self._supervisor
        workers = self._workers
        share_channels = self._share_channels
        async_delivery = self._async_delivery
        task_timeout = self._task_timeout
        max_revisions = self._max_revisions
        _lock = asyncio.Lock()
        # Per-room dedup: prevents duplicate calls within the same turn
        _dedup_cache: dict[str, tuple[str, float]] = {}  # room_id → (result, timestamp)
        # Per-room running flag for async_delivery mode — prevents re-dispatch
        # while the background pipeline is still in flight.
        _running: set[str] = set()
        dedup_window = 30.0

        async def strategy_handler(name: str, arguments: dict[str, Any]) -> str:
            if name != tool_name:
                if original:
                    return await original(name, arguments)
                return json.dumps({"error": f"Unknown tool: {name}"})

            rid = _room_id_var.get() or room_id
            # The supervisor owns this tool, but the supervised flow re-invokes the
            # SAME supervisor for dispatch/review inside its own ``::task-`` child
            # rooms. There it must answer the dispatch/review prompt directly —
            # calling delegate_workers again recurses the whole pipeline
            # (delegate_workers within delegate_workers). Refuse from a sub-task room.
            if "::task-" in rid:
                return json.dumps(
                    {
                        "error": (
                            "delegate_workers is unavailable here: you are already running "
                            "inside a delegated step. Respond directly to the instruction "
                            "with the requested text — do not call delegate_workers."
                        )
                    }
                )
            task_desc = arguments.get("task", "")

            async with _lock:
                cached = _dedup_cache.get(rid)
                if cached is not None and (time.monotonic() - cached[1]) < dedup_window:
                    return cached[0]

                # Async dispatch: fire-and-return so the supervisor's
                # tool loop doesn't block on worker execution. Workers
                # post lifecycle events to the status bus and their
                # combined output is delivered back to the room via
                # kit.deliver(), re-triggering the supervisor.
                if async_delivery:
                    if rid in _running:
                        return json.dumps(
                            {
                                "status": "already_running",
                                "message": (
                                    "Workers are already running for this room. "
                                    "Do NOT call this tool again. "
                                    "Use check_status_bus to see progress."
                                ),
                            }
                        )
                    _running.add(rid)

                    def _clear(
                        *,
                        success: bool = True,
                        _rid: str = rid,
                    ) -> None:
                        """Release the room and evict the dedup entry on failure.

                        A failed pipeline must drop its cached
                        ``"dispatched"`` response — otherwise callers get
                        that stale string for the full dedup window and
                        never learn the run failed.
                        """
                        _running.discard(_rid)
                        if not success:
                            _dedup_cache.pop(_rid, None)

                    # Create the task + populate dedup atomically with the
                    # _running flag. If create_task raises (shutdown race)
                    # we must release _running so the room isn't
                    # permanently marked busy.
                    try:
                        task = asyncio.create_task(
                            _async_run_and_deliver(
                                kit=kit,
                                room_id=rid,
                                strategy=strategy,
                                workers=workers,
                                task_desc=task_desc,
                                share_channels=share_channels,
                                on_done=_clear,
                            )
                        )
                        task.add_done_callback(log_task_exception)
                    except BaseException:
                        _running.discard(rid)
                        raise

                    dispatched_response = json.dumps(
                        {
                            "status": "dispatched",
                            "workers": [w.channel_id for w in workers],
                            "message": (
                                "Workers are running in the background. "
                                "Use check_status_bus to follow progress. "
                                "Their combined results will arrive as a new "
                                "message in this room when they are done."
                            ),
                        }
                    )
                    now = time.monotonic()
                    _dedup_cache[rid] = (dispatched_response, now)
                    stale = [k for k, (_, ts) in _dedup_cache.items() if now - ts >= dedup_window]
                    for k in stale:
                        del _dedup_cache[k]
                    return dispatched_response

                try:
                    if strategy == WorkerStrategy.SEQUENTIAL:
                        # Hub & spoke: every worker output returns to the
                        # supervisor, which validates it (rework up to
                        # max_revisions) and frames the next worker's task.
                        # The digest goes back so the supervisor summarizes.
                        steps = await _run_supervised_sequential(
                            kit,
                            rid,
                            supervisor,
                            workers,
                            task_desc,
                            max_revisions=max_revisions,
                            share_channels=share_channels,
                            task_timeout=task_timeout,
                        )
                        review = _format_supervised_digest(task_desc, steps, max_revisions)
                    else:
                        # Parallel: all workers run on the same task; the
                        # supervisor reviews the combined result and delivers.
                        raw = await _run_parallel(
                            kit,
                            rid,
                            workers,
                            task_desc,
                            share_channels=share_channels,
                            task_timeout=task_timeout,
                        )
                        review = _format_supervisor_review(task_desc, raw, workers)
                    now = time.monotonic()
                    _dedup_cache[rid] = (review, now)
                    # Evict expired entries to prevent unbounded growth
                    stale = [k for k, (_, ts) in _dedup_cache.items() if now - ts >= dedup_window]
                    for k in stale:
                        del _dedup_cache[k]
                    return review
                except Exception as exc:
                    logger.exception("Strategy delegation failed")
                    return json.dumps({"error": str(exc)})

        self._supervisor.tool_handler = strategy_handler

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
        share_channels = self._share_channels
        pending: set[str] = set()

        async def delegation_handler(name: str, arguments: dict[str, Any]) -> str:
            worker_id = tool_to_worker.get(name)
            if worker_id is not None:
                rid = _room_id_var.get() or room_id
                task_desc = arguments.get("task", "")
                try:
                    if wait:
                        _post_worker_status(
                            kit,
                            worker_id,
                            StatusLevel.PENDING,
                            detail=task_desc,
                            metadata={"room_id": rid, "mode": "per_worker_wait"},
                        )
                        try:
                            delegated = await kit.delegate(
                                rid,
                                worker_id,
                                task_desc,
                                wait=True,
                                notify=self._supervisor.channel_id,
                                share_channels=share_channels,
                            )
                        except Exception as exc:
                            _post_worker_status(
                                kit,
                                worker_id,
                                StatusLevel.FAILED,
                                detail=str(exc),
                                metadata={"room_id": rid, "mode": "per_worker_wait"},
                            )
                            raise
                        result = delegated.result
                        result_status = result.status if result else "failed"
                        result_output = (result.output or result.error or "") if result else ""
                        _post_worker_status(
                            kit,
                            worker_id,
                            StatusLevel.COMPLETED
                            if result_status == "completed"
                            else StatusLevel.FAILED,
                            detail=result_output,
                            metadata={
                                "room_id": rid,
                                "mode": "per_worker_wait",
                                "task_id": delegated.id,
                            },
                        )
                        return json.dumps(
                            {
                                "status": result_status,
                                "worker": worker_id,
                                "result": result_output,
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
                        share_channels=share_channels,
                    )
                    pending.add(worker_id)
                    _post_worker_status(
                        kit,
                        worker_id,
                        StatusLevel.PENDING,
                        detail=task_desc,
                        metadata={
                            "room_id": rid,
                            "mode": "per_worker_async",
                            "task_id": delegated.id,
                        },
                    )

                    original_set = delegated._set_result
                    _bus_kit = kit
                    _bus_room = rid
                    _bus_task_id = delegated.id

                    def _patched_set(r: Any, *, _wid: str = worker_id or "") -> None:
                        pending.discard(_wid)
                        output = ""
                        ok = False
                        if r is not None:
                            output = getattr(r, "output", None) or getattr(r, "error", None) or ""
                            ok = getattr(r, "status", None) == "completed"
                        _post_worker_status(
                            _bus_kit,
                            _wid,
                            StatusLevel.COMPLETED if ok else StatusLevel.FAILED,
                            detail=output,
                            metadata={
                                "room_id": _bus_room,
                                "mode": "per_worker_async",
                                "task_id": _bus_task_id,
                            },
                        )
                        original_set(r)

                    delegated._set_result = _patched_set  # ty: ignore[invalid-assignment]

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
