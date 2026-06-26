"""Strategy-tool delegation wiring for the supervisor.

Mixin for :class:`Supervisor`: injects a single ``delegate_workers`` tool that
runs the whole team in the configured strategy (the supervised sequential flow,
or parallel). Host attributes are declared as annotations; they are set in
``Supervisor.__init__``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.orchestration.strategies.supervisor._common import (
    _STRATEGY_TOOL_NAME,
    WorkerStrategy,
    _fallthrough,
    _is_subtask_room,
    logger,
)
from roomkit.orchestration.strategies.supervisor.delegate import _async_run_and_deliver
from roomkit.orchestration.strategies.supervisor.execution import _run_parallel
from roomkit.orchestration.strategies.supervisor.prompts import _format_supervised_digest
from roomkit.orchestration.strategies.supervisor.results import (
    _format_supervisor_review,
    _worker_roles_csv,
)
from roomkit.orchestration.strategies.supervisor.supervised import (
    _run_supervised_sequential,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class _StrategyToolMixin:
    """Inject the single ``delegate_workers`` tool (deterministic execution)."""

    _supervisor: Agent
    _workers: list[Agent]
    _strategy: WorkerStrategy | None
    _share_channels: list[str]
    _async_delivery: bool
    _task_timeout: float
    _max_revisions: int

    def _inject_strategy_tool(self, kit: RoomKit, room_id: str) -> None:
        """Inject a single ``delegate_workers`` tool for deterministic execution."""
        from roomkit.orchestration.handoff import _room_id_var

        tool_name = _STRATEGY_TOOL_NAME

        if any(t.name == tool_name for t in self._supervisor._injected_tools):
            return

        worker_roles = _worker_roles_csv(self._workers)
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
                return await _fallthrough(original, name, arguments)

            rid = _room_id_var.get() or room_id
            # The supervisor owns this tool, but the supervised flow re-invokes the
            # SAME supervisor for dispatch/review inside its own ``::task-`` child
            # rooms. There it must answer the dispatch/review prompt directly —
            # calling delegate_workers again recurses the whole pipeline
            # (delegate_workers within delegate_workers). Refuse from a sub-task room.
            if _is_subtask_room(rid):
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
