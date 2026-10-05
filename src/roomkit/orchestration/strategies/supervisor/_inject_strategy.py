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
import weakref
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.orchestration._background import start_background_run
from roomkit.orchestration._call_room import in_call_room
from roomkit.orchestration.strategies.supervisor._common import (
    _STRATEGY_TOOL_NAME,
    WorkerStrategy,
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
from roomkit.tasks.status import TASK_STATUS_TOOL
from roomkit.tools.context import current_tool_allowed_names

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


def _follow_hint(what: str) -> str:
    """Tell the supervisor to use ``task_status`` to *what*, only when its turn
    has it (RFC §23.4): never point an agent at a tool it was not given."""
    if TASK_STATUS_TOOL in (current_tool_allowed_names() or set()):
        return f"Use {TASK_STATUS_TOOL} to {what}. "
    return ""


class _StrategyToolMixin:
    """Inject the single ``delegate_workers`` tool (deterministic execution)."""

    _supervisor: Agent
    _workers: list[Agent]
    _strategy: WorkerStrategy | None
    _share_channels: list[str]
    _async_delivery: bool
    _task_timeout: float
    _max_revisions: int

    def _strategy_tool(self) -> AITool:
        """The ``delegate_workers`` declaration: the whole team, in one call."""
        worker_roles = _worker_roles_csv(self._workers)
        return AITool(
            name=_STRATEGY_TOOL_NAME,
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

    def _inject_strategy_tool(self, kit: RoomKit, room_id: str) -> None:
        """Declare ``delegate_workers`` in *room_id*'s turns, and serve it there.

        The tool is set up for the installed room (RFC §19.7): not in the
        supervisor's other rooms, nor in the ``::task-`` rooms where the
        supervised flow runs the supervisor to frame and judge, where it must
        answer instead of delegating again. It runs this install's team: a
        second room's install, with its own workers, serves its own room.
        """
        server = _StrategyToolServer(
            kit,
            self._supervisor,
            self._workers,
            self._strategy,
            share_channels=self._share_channels,
            async_delivery=self._async_delivery,
            task_timeout=self._task_timeout,
            max_revisions=self._max_revisions,
        )
        serve = in_call_room(_STRATEGY_TOOL_NAME, server.serve)
        entry = orchestration_tool(self._strategy_tool(), serve, waits=True)
        self._supervisor._registry.register(entry, room_id=room_id, owner=self)


_SUBTASK_REFUSAL = json.dumps(
    {
        "error": (
            "delegate_workers is unavailable here: you are already running "
            "inside a delegated step. Respond directly to the instruction "
            "with the requested text — do not call delegate_workers."
        )
    }
)
"""What ``delegate_workers`` answers inside a ``::task-`` room."""


class _StrategyToolServer:
    """Serves ``delegate_workers``: runs the team for the room of the call."""

    # How long a room's answer is served again to a repeated call.
    _DEDUP_WINDOW = 30.0

    def __init__(
        self,
        kit: RoomKit,
        supervisor: Agent,
        workers: list[Agent],
        strategy: WorkerStrategy | None,
        *,
        share_channels: list[str],
        async_delivery: bool,
        task_timeout: float,
        max_revisions: int,
    ) -> None:
        self._kit = kit
        self._supervisor = supervisor
        self._workers = workers
        self._strategy = strategy
        self._share_channels = share_channels
        self._async_delivery = async_delivery
        self._task_timeout = task_timeout
        self._max_revisions = max_revisions
        # One lock per room, held while its pipeline runs: another room's call
        # does not wait on it. A lock lives while a call holds it.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        # Per-room dedup: prevents duplicate calls within the same turn
        self._dedup_cache: dict[str, tuple[str, float]] = {}  # room_id → (result, timestamp)
        # Per-room running flag for async_delivery mode — prevents re-dispatch
        # while the background pipeline is still in flight.
        self._running: set[str] = set()

    async def serve(self, rid: str, name: str, arguments: dict[str, Any]) -> str:
        """Answer one ``delegate_workers`` call made in room *rid*."""
        # The supervisor owns this tool, but the supervised flow re-invokes the
        # SAME supervisor for dispatch/review inside its own ``::task-`` child
        # rooms. There it must answer the dispatch/review prompt directly —
        # calling delegate_workers again recurses the whole pipeline
        # (delegate_workers within delegate_workers). Refuse from a sub-task room.
        if _is_subtask_room(rid):
            return _SUBTASK_REFUSAL
        task_desc = arguments.get("task", "")
        lock = self._locks.get(rid)
        if lock is None:
            lock = self._locks[rid] = asyncio.Lock()
        async with lock:
            cached = self._dedup_cache.get(rid)
            if cached is not None and (time.monotonic() - cached[1]) < self._DEDUP_WINDOW:
                return cached[0]
            if self._async_delivery:
                return self._dispatch(rid, task_desc)
            return await self._run(rid, task_desc)

    def _dispatch(self, rid: str, task_desc: str) -> str:
        """Start the team for *rid* in the background, and answer at once.

        Fire-and-return so the supervisor's tool loop doesn't block on worker
        execution. Workers post lifecycle events to the status bus and their
        combined output is handed back to the supervisor as an instruction
        (tasks.handback), which it answers.
        """
        if rid in self._running:
            return json.dumps(
                {
                    "status": "already_running",
                    "message": (
                        "Workers are already running for this room. "
                        "Do NOT call this tool again. " + _follow_hint("see progress")
                    ).strip(),
                }
            )
        self._start_team(rid, task_desc)
        dispatched_response = json.dumps(
            {
                "status": "dispatched",
                "workers": [w.channel_id for w in self._workers],
                "message": (
                    "Workers are running in the background. "
                    + _follow_hint("follow progress")
                    + "Their combined results will be handed to you when they are done."
                ),
            }
        )
        self._remember(rid, dispatched_response)
        return dispatched_response

    def _start_team(self, rid: str, task_desc: str) -> None:
        """Run the team for *rid* in the background, the room marked busy
        until it ends."""
        self._running.add(rid)

        def _clear(*, success: bool = True, _rid: str = rid) -> None:
            """Release the room and evict its dedup entry, whatever the outcome.

            The cached ``"dispatched"`` answer belongs to the run that ended: a
            dispatch after it (the supervisor's follow-up to the results it is
            handed, or to a failure) starts a new run, rather than reading
            that stale answer for the rest of the dedup window and promising
            results that never come.
            """
            self._running.discard(_rid)
            self._dedup_cache.pop(_rid, None)

        # Start the run + populate dedup atomically with the _running flag.
        # If the start raises (shutdown race) we must release _running so
        # the room isn't permanently marked busy.
        try:
            start_background_run(
                self._kit,
                _async_run_and_deliver(
                    kit=self._kit,
                    room_id=rid,
                    supervisor_id=self._supervisor.channel_id,
                    supervisor=self._supervisor,
                    strategy=self._strategy,
                    workers=self._workers,
                    task_desc=task_desc,
                    share_channels=self._share_channels,
                    max_revisions=self._max_revisions,
                    task_timeout=self._task_timeout,
                    on_done=_clear,
                ),
            )
        except BaseException:
            self._running.discard(rid)
            raise

    async def _run(self, rid: str, task_desc: str) -> str:
        """Run the team for *rid*, and answer with what the supervisor reviews."""
        try:
            if self._strategy == WorkerStrategy.SEQUENTIAL:
                # Hub & spoke: every worker output returns to the supervisor,
                # which validates it (rework up to max_revisions) and frames
                # the next worker's task. The digest goes back so the
                # supervisor summarizes.
                steps = await _run_supervised_sequential(
                    self._kit,
                    rid,
                    self._supervisor,
                    self._workers,
                    task_desc,
                    max_revisions=self._max_revisions,
                    share_channels=self._share_channels,
                    task_timeout=self._task_timeout,
                )
                review = _format_supervised_digest(task_desc, steps, self._max_revisions)
            else:
                # Parallel: all workers run on the same task; the supervisor
                # reviews the combined result and delivers.
                raw = await _run_parallel(
                    self._kit,
                    rid,
                    self._workers,
                    task_desc,
                    share_channels=self._share_channels,
                    task_timeout=self._task_timeout,
                )
                review = _format_supervisor_review(task_desc, raw, self._workers)
        except Exception:
            # Raised on: the channel reads it as any failed call, the class for
            # the model and the message for the observers (RFC §9.3).
            logger.exception("Strategy delegation failed")
            raise
        self._remember(rid, review)
        return review

    def _remember(self, rid: str, response: str) -> None:
        """Serve *response* again to a repeat in *rid* within the dedup window,
        and evict expired entries to prevent unbounded growth."""
        now = time.monotonic()
        self._dedup_cache[rid] = (response, now)
        stale = [k for k, (_, ts) in self._dedup_cache.items() if now - ts >= self._DEDUP_WINDOW]
        for k in stale:
            del self._dedup_cache[k]
