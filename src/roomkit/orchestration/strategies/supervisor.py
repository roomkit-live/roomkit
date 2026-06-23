"""Supervisor orchestration strategy.

A supervisor agent delegates tasks to worker agents via ``kit.delegate()``.
Workers are registered on the kit but NOT attached to the parent room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import (
    ConversationState,
    set_conversation_state,
)
from roomkit.orchestration.status_bus import StatusLevel, post_agent_lifecycle
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.supervisor")

# Local alias — the Supervisor strategy specifically tracks worker
# lifecycle, but the underlying helper is shared with Pipeline / Swarm /
# Loop so all strategies emit events under the same conventions.
_post_worker_status = post_agent_lifecycle


class WorkerStrategy(StrEnum):
    """How workers are executed when delegated to."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# Per-worker delegation budget. Each delegated task is bounded individually,
# so the supervisor's chain is governed by a per-task limit rather than one
# global timeout (a single global can never be coherent: it must cover the
# *sum* of every worker, which scales with team size). A worker that exceeds
# this fails on its own budget and the chain keeps going.
_DEFAULT_TASK_TIMEOUT_SECONDS = 120.0

# Max supervisor→worker round-trips per step in supervised mode: the supervisor
# validates each worker's output and, if it's not acceptable, sends it back with
# feedback for a rework. Bounded so a worker that can't satisfy the supervisor
# doesn't loop forever — on exhaustion the step is delivered flagged unvalidated.
_DEFAULT_MAX_REVISIONS = 3


class Supervisor(Orchestration):
    """Supervisor orchestration strategy.

    The supervisor handles all user interaction. Workers are registered
    on the kit (so ``delegate()`` can find them) but are NOT attached
    to the parent room — they run in child rooms.

    Examples::

        # Framework-driven: auto-delegate with task refinement
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
            auto_delegate=True,
        )

        # Framework-driven: workers get raw user message
        Supervisor(
            supervisor=coordinator,
            workers=[technical, business],
            strategy="parallel",
            auto_delegate=True,
            refine_task=False,
        )

        # Tool-based: AI decides when to delegate
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
        )

        # Manual: per-worker tools, AI decides everything
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
        auto_delegate: bool = False,
        async_delivery: bool = False,
        refine_task: bool = True,
        refine_instruction: str | None = None,
        delegation_message: str | None = "I'm dispatching my team to work on this.",
        wait_for_result: bool = True,
        share_channels: list[str] | None = None,
        task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
    ) -> None:
        """Initialise the supervisor strategy.

        Args:
            supervisor: The agent that handles user interaction and
                delegates tasks.
            workers: Agents that run delegated tasks in child rooms.
            strategy: Deterministic execution pattern for workers.

                - ``"sequential"``: workers run in order, each receiving
                  the previous worker's output.
                - ``"parallel"``: all workers run concurrently on the
                  same task.
                - ``None`` (default): per-worker ``delegate_to_<id>``
                  tools are injected and the AI decides when to call
                  them.

            auto_delegate: If ``True``, the framework triggers workers
                automatically — no tool needed. For sync channels (CLI),
                blocks until results are ready. For async channels
                (voice), runs in background. Requires *strategy*.

            async_delivery: If ``True``, worker delegation returns
                immediately and results are delivered back to the
                room via ``kit.deliver()`` when they complete. This
                keeps the supervisor's tool-loop clock bounded by
                its own reasoning time rather than aggregated worker
                wall-clock time. Applies to:

                - strategy-tool mode (``strategy`` set): the
                  ``delegate_workers`` tool fires background workers
                  and returns ``{"status": "dispatched", ...}``
                  immediately. Use ``check_status_bus`` to follow
                  progress.
                - voice ``auto_delegate`` mode: injects a
                  ``delegate_workers`` tool on the voice channel;
                  supervisor is not attached (voice handles UI).

                Lifecycle events (``pending`` / ``completed`` /
                ``failed``) are posted to ``kit.status_bus`` for
                every worker regardless of mode.
                If ``False`` (default), delegation blocks until
                results are ready (sync mode).

            refine_task: Controls whether the framework extracts a clean
                topic from the user's message before sending to workers.

            refine_instruction: Custom instruction for topic extraction.
                Overrides the default.

            delegation_message: Message injected into the conversation
                when workers are dispatched (async mode only). Set to
                ``None`` to disable. Default: "I'm dispatching my team
                to work on this."

            wait_for_result: When *strategy* is ``None``, controls
                whether delegation runs inline (``True``, default) or
                in the background (``False``).  Ignored when *strategy*
                is set or *auto_delegate* is ``True``.

            share_channels: Channel IDs from the parent room to share
                with every child room created during delegation.  For
                example, passing ``["system", "ws-status"]`` attaches
                those channels to each worker's child room so the
                worker can emit events visible on those channels.

            task_timeout: Per-worker delegation budget in seconds
                (default 120). Each delegated task is bounded
                individually so the chain is governed per-task rather
                than by a single global timeout that can never be
                coherent across team sizes. A worker that exceeds it is
                recorded as failed and the run continues.

            max_revisions: In supervised sequential mode, the maximum
                number of supervisor→worker rework round-trips per step
                (default 3). The supervisor validates each worker's
                output; if unacceptable it sends feedback for a rework,
                up to this many times. On exhaustion the step is
                delivered flagged as unvalidated rather than looping.
        """
        self._supervisor = supervisor
        self._workers = list(workers)
        self._strategy = WorkerStrategy(strategy) if strategy else None
        self._auto_delegate = auto_delegate
        self._async_delivery = async_delivery
        self._refine_task = refine_task
        self._refine_instruction = refine_instruction
        self._delegation_message = delegation_message
        self._wait_for_result = wait_for_result
        self._share_channels = list(share_channels) if share_channels else []
        self._task_timeout = task_timeout
        self._max_revisions = max_revisions

        if auto_delegate and self._strategy is None:
            msg = "auto_delegate=True requires strategy to be set"
            raise ValueError(msg)

    def agents(self) -> list[Agent]:
        """Return agents to attach to the room.

        ``async_delivery`` removes the supervisor from the room ONLY
        in voice ``auto_delegate`` mode, where a
        ``RealtimeVoiceChannel`` handles user interaction directly.
        In strategy-tool or per-worker-tool modes the supervisor
        still drives the conversation via its tool loop; background
        dispatch only affects worker delegation, not user routing.
        """
        if self._async_delivery and self._auto_delegate:
            return []
        return [self._supervisor]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire supervisor routing and delegation tools."""
        # Router only needed when the supervisor agent is in the room.
        # In voice async_delivery mode (auto_delegate=True), the voice
        # channel owns routing and the supervisor is not attached.
        supervisor_in_room = not (self._async_delivery and self._auto_delegate)
        if supervisor_in_room:
            router = ConversationRouter(
                default_agent_id=self._supervisor.channel_id,
            )
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

        # Wire delegation based on mode
        if self._auto_delegate:
            self._install_auto_delegate(kit, room_id)
        elif self._strategy is not None:
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

    # -- Auto-delegate (framework-driven, no tools) ---------------------------

    def _install_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Install framework-driven delegation (sync or async)."""
        if self._async_delivery:
            self._install_async_auto_delegate(kit, room_id)
        else:
            self._install_sync_auto_delegate(kit, room_id)

    def _install_sync_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Wrap supervisor's on_event — blocks until workers complete."""
        supervisor = self._supervisor
        strategy = self._strategy
        workers = self._workers
        refine = self._refine_task
        refine_instruction = self._refine_instruction
        share_channels = self._share_channels
        max_revisions = self._max_revisions
        task_timeout = self._task_timeout
        original_on_event = supervisor.on_event

        async def auto_delegate_on_event(
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            if event.source.channel_id == supervisor.channel_id:
                return ChannelOutput.empty()
            if event.source.channel_type == _ChannelType.AI:
                return ChannelOutput.empty()

            rid = context.room.id if context.room else room_id
            # Only the parent room drives delegation. Inside a child task room
            # (e.g. a supervisor review room created by the supervised loop, or
            # any delegated worker room), the supervisor must run NORMALLY —
            # otherwise the review prompt would be treated as a fresh user task
            # and re-trigger delegation, recursing without bound.
            if "::task-" in rid:
                return await original_on_event(event, binding, context)

            if refine:
                return await _two_pass_delegate(
                    kit,
                    rid,
                    supervisor,
                    original_on_event,
                    event,
                    binding,
                    context,
                    strategy,
                    workers,
                    instruction=refine_instruction,
                    share_channels=share_channels,
                    max_revisions=max_revisions,
                    task_timeout=task_timeout,
                )
            return await _one_pass_delegate(
                kit,
                rid,
                supervisor,
                original_on_event,
                event,
                binding,
                context,
                strategy,
                workers,
                share_channels=share_channels,
                max_revisions=max_revisions,
                task_timeout=task_timeout,
            )

        supervisor.on_event = auto_delegate_on_event  # ty: ignore[invalid-assignment]

    def _install_async_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Inject delegate_workers tool into RealtimeVoiceChannel.

        The tool handler runs workers in the background and returns
        immediately. Results are delivered via kit.deliver().
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        strategy = self._strategy
        workers = self._workers
        share_channels = self._share_channels

        # Find the RealtimeVoiceChannel in registered channels
        voice_channel: RealtimeVoiceChannel | None = None
        for ch in kit.channels.values():
            if isinstance(ch, RealtimeVoiceChannel):
                voice_channel = ch
                break

        if voice_channel is None:
            logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
            return

        # Build tool definition
        worker_roles = ", ".join(getattr(w, "role", None) or w.channel_id for w in workers)
        tool_def = {
            "name": "delegate_workers",
            "description": (
                f"Delegate analysis to specialist workers ({worker_roles}). "
                f"Call when the user requests analysis, research, or investigation. "
                f"Workers run in {strategy} mode. Pass the topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The topic to analyze",
                    },
                },
                "required": ["task"],
            },
        }

        # Add tool to voice channel
        if voice_channel._tools is None:
            voice_channel._tools = []
        voice_channel._tools.append(tool_def)

        # Wrap tool handler for async delegation
        original_handler = voice_channel.tool_handler
        _running = False
        _lock = asyncio.Lock()

        async def async_tool_handler(name: str, arguments: dict[str, Any]) -> str:
            nonlocal _running

            if name != "delegate_workers":
                if original_handler:
                    return await original_handler(name, arguments)
                return json.dumps({"error": f"Unknown tool: {name}"})

            async with _lock:
                if _running:
                    return json.dumps(
                        {
                            "status": "already_running",
                            "message": "Workers are already running.",
                        }
                    )

                task_desc = arguments.get("task", "")
                _running = True

                # Launch workers inside the lock so _running and the
                # task creation are atomic. If create_task raises
                # (shutdown race), release _running so the voice
                # channel isn't permanently stuck in already_running.
                try:
                    task = asyncio.create_task(
                        _async_run_and_deliver(
                            kit=kit,
                            room_id=room_id,
                            strategy=strategy,
                            workers=workers,
                            task_desc=task_desc,
                            share_channels=share_channels,
                            on_done=_clear,
                        )
                    )
                    task.add_done_callback(log_task_exception)
                except BaseException:
                    _running = False
                    raise

            return json.dumps(
                {
                    "status": "dispatched",
                    "workers": worker_roles,
                    "message": "Workers are running. Results will be delivered when ready.",
                }
            )

        def _clear(**_: Any) -> None:
            """Accept but ignore ``success`` from _async_run_and_deliver.

            The voice path has no dedup cache to evict on failure, so
            success/failure doesn't change behaviour here.
            """
            nonlocal _running
            _running = False

        voice_channel.tool_handler = async_tool_handler

    # -- Strategy-based tool (sequential / parallel) --------------------------

    def _inject_strategy_tool(self, kit: RoomKit, room_id: str) -> None:
        """Inject a single ``delegate_workers`` tool for deterministic execution."""
        from roomkit.orchestration.handoff import _room_id_var

        tool_name = "delegate_workers"

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


# ---------------------------------------------------------------------------
# Auto-delegate execution helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Async tool handler helper
# ---------------------------------------------------------------------------


async def _async_run_and_deliver(
    *,
    kit: RoomKit,
    room_id: str,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    task_desc: str,
    share_channels: list[str] | None = None,
    on_done: Callable[..., None],
) -> None:
    """Background: run workers → deliver results via kit.deliver().

    Individual worker lifecycle events are posted to ``kit.status_bus``
    inside ``_run_sequential`` / ``_run_parallel``. This helper emits
    one additional terminal entry under ``agent_id="orchestration"``
    so subscribers can observe the pipeline as a whole.

    ``on_done`` is called in ``finally`` regardless of outcome. Callers
    that need to distinguish success from failure can accept the
    ``success`` keyword argument — e.g. to evict cached dispatch
    responses that should not be re-served after a failed pipeline.
    Callers that don't care simply accept no arguments (legacy shape).
    """
    pipeline_meta = {
        "room_id": room_id,
        "strategy": str(strategy) if strategy else None,
        "workers": [w.channel_id for w in workers],
    }
    pipeline_success = False
    try:
        worker_results = await _run_workers(
            kit,
            room_id,
            strategy,
            workers,
            task_desc,
            share_channels=share_channels,
        )
        results_text = _format_worker_results(worker_results)
        logger.info("[async_delegate] Workers completed, delivering results")

        await kit.deliver(
            room_id,
            f"Analysis results are ready. Here's what the analysts found:\n\n{results_text}",
        )
        _post_worker_status(
            kit,
            "orchestration",
            StatusLevel.COMPLETED,
            action="pipeline",
            detail=f"{len(workers)} worker(s) completed",
            metadata=pipeline_meta,
        )
        pipeline_success = True
    except Exception as exc:
        logger.exception("[async_delegate] Pipeline failed")
        _post_worker_status(
            kit,
            "orchestration",
            StatusLevel.FAILED,
            action="pipeline",
            detail=str(exc),
            metadata=pipeline_meta,
        )
    finally:
        # Pass success through when the callback accepts it; older
        # voice-path callbacks take no args and are called plain.
        try:
            on_done(success=pipeline_success)
        except TypeError:
            on_done()


# ---------------------------------------------------------------------------
# Sync auto-delegate helpers
# ---------------------------------------------------------------------------


def _build_pass1_instruction(workers: list[Agent]) -> str:
    """Build the default pass-1 instruction."""
    return (
        "Extract the core topic or subject from the user's request. "
        "Output only the topic, nothing else. No questions, no instructions, "
        "no formatting. Example: user says 'analyse anthropic' → 'Anthropic'"
    )


async def _two_pass_delegate(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    *,
    instruction: str | None = None,
    share_channels: list[str] | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> ChannelOutput:
    """Two-pass: supervisor formulates task → workers run (validated between
    steps by the supervisor in sequential mode) → supervisor presents."""
    # Pass 1: temporarily inject a task-formulation instruction
    pass1_instruction = instruction or _build_pass1_instruction(workers)
    original_prompt = supervisor._system_prompt
    supervisor._system_prompt = (
        f"{original_prompt}\n\n{pass1_instruction}" if original_prompt else pass1_instruction
    )
    try:
        pass1_output = await original_on_event(event, binding, context)
        refined_task = await _extract_output_text(pass1_output)
    finally:
        # Always restore the original prompt
        supervisor._system_prompt = original_prompt

    logger.debug("Pass 1 refined task: %s", refined_task[:200] if refined_task else "(empty)")

    if not refined_task:
        return pass1_output

    # Run workers with the refined task — supervised between steps in sequential.
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        refined_task,
        supervisor=supervisor,
        max_revisions=max_revisions,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )

    # Pass 2: inject worker results and generate final response
    results_text = _format_worker_results(worker_results)
    results_event = RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(body=f"Here are the results from your workers:\n\n{results_text}"),
    )

    # Ingest the results so the supervisor sees them in context
    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)


async def _one_pass_delegate(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    *,
    share_channels: list[str] | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> ChannelOutput:
    """One-pass: workers run on raw message (validated between steps by the
    supervisor in sequential mode) → supervisor presents."""
    # Extract user's raw message
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body

    if not user_message:
        return await original_on_event(event, binding, context)

    # Run workers with the raw user message — supervised between steps in sequential.
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        user_message,
        supervisor=supervisor,
        max_revisions=max_revisions,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )

    # Inject results into context and let supervisor present
    results_text = _format_worker_results(worker_results)
    results_event = RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(
            body=(
                f"The user asked: {user_message}\n\n"
                f"Here are the results from your workers:\n\n{results_text}"
            )
        ),
    )

    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)


# ---------------------------------------------------------------------------
# Strategy execution helpers
# ---------------------------------------------------------------------------


async def _run_workers(
    kit: RoomKit,
    room_id: str,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    task_desc: str,
    *,
    supervisor: Agent | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run workers according to strategy and return their reviewed results.

    Sequential goes through the supervised hub-&-spoke loop when a *supervisor*
    is given (every output returns to the supervisor, which validates it and
    frames the next worker's task); parallel runs all workers on the same task.
    """
    if strategy == WorkerStrategy.SEQUENTIAL and supervisor is not None:
        return await _run_supervised_sequential(
            kit,
            room_id,
            supervisor,
            workers,
            task_desc,
            max_revisions=max_revisions,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    if strategy == WorkerStrategy.SEQUENTIAL:
        result_json = await _run_sequential(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    else:
        result_json = await _run_parallel(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    parsed = json.loads(result_json)
    return parsed.get("results", [])


def _worker_label(worker: Agent) -> str:
    """Human-readable label for a worker, used to attribute its output when
    composing the next worker's input and the supervisor's review brief."""
    return (
        getattr(worker, "role", None) or getattr(worker, "description", None) or worker.channel_id
    )


_PROFILE_CAP = 600


def _worker_profile(worker: Agent) -> str:
    """Role + the agent's own configured instructions, so the supervisor frames
    tasks knowing each worker's real capabilities (e.g. that the report agent
    publishes HTML to the website) rather than just a label. Prefers ``purpose``
    (the agent's concise own instructions, set by the host) and falls back to
    ``description``; the full runtime system prompt is far too large to inject."""
    role = getattr(worker, "role", None) or worker.channel_id
    detail = getattr(worker, "purpose", None) or getattr(worker, "description", None)
    if detail and detail.strip() and detail.strip() != role.strip():
        text = detail.strip()
        if len(text) > _PROFILE_CAP:
            text = text[:_PROFILE_CAP] + "…"
        return f"{role} — {text}"
    return role


# --- Supervised sequential (hub & spoke) --------------------------------------
#
# Every worker's output returns to the supervisor, which judges it and frames the
# next worker's task — workers never hand off to each other directly. Order is
# fixed; the supervisor owns each transition (validate + frame + rework). The
# supervisor runs in its own child rooms (visible in the timeline) for each
# review; the final user-facing summary is left to the supervisor's own turn
# (the digest below is returned as the ``delegate_workers`` tool result).

_VERDICT_INSTRUCTIONS = (
    "Respond with ONLY a JSON object, no other text, of exactly this shape:\n"
    '{"approved": true_or_false, "feedback": "what to fix if not approved, else empty", '
    '"next_task": "task for the next worker if approved and one exists, else empty"}'
)


def _parse_verdict(raw: str) -> dict[str, Any]:
    """Parse the supervisor's review JSON.

    Fails CLOSED (``approved=False``) on a parse miss: an unreadable verdict must
    not pass a step through unjudged. The miss is logged and fed back as feedback
    so the next attempt can correct itself; the surrounding rework loop is bounded
    by ``max_revisions``, so a persistently malformed verdict ends in an honest
    failure rather than a silent approval.
    """
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        obj = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        logger.warning("Supervisor verdict unparseable; rejecting by default: %r", raw[:200])
        return {
            "approved": False,
            "feedback": "Your verdict was unreadable. Respond with ONLY the JSON verdict.",
            "next_task": None,
        }
    next_task = obj.get("next_task")
    return {
        "approved": bool(obj.get("approved", False)),
        "feedback": str(obj.get("feedback") or ""),
        "next_task": (str(next_task).strip() or None) if next_task else None,
    }


async def _delegate_and_wait(
    kit: RoomKit,
    room_id: str,
    channel_id: str,
    task: str,
    *,
    share_channels: list[str] | None,
    task_timeout: float,
    require_structured_result: bool = False,
) -> tuple[str, bool]:
    """Delegate *task* to an agent and wait for its result, bounded by
    *task_timeout*. Returns ``(output, completed_ok)``. When
    *require_structured_result* is set, ``output`` is the JSON-encoded
    ``submit_result`` payload (the worker is forced to hand its work back via
    the tool); otherwise it is the agent's free text."""
    try:
        delegated = await asyncio.wait_for(
            kit.delegate(
                room_id,
                channel_id,
                task,
                wait=True,
                share_channels=share_channels,
                require_structured_result=require_structured_result,
            ),
            timeout=task_timeout,
        )
    except TimeoutError:
        return (f"(timed out after {task_timeout:.0f}s)", False)
    result = delegated.result
    if not result:
        return ("", False)
    return (result.output or result.error or "", result.status == "completed")


def _render_result(output: str) -> str:
    """Render a worker's structured ``submit_result`` payload (JSON) as readable
    text for the supervisor's review and the final digest. Falls back to the raw
    string when it isn't a structured payload."""
    try:
        payload = json.loads(output)
    except (ValueError, TypeError):
        return output
    if not isinstance(payload, dict) or "status" not in payload:
        return output
    if payload.get("by") == "orchestration":
        last = str(payload.get("last_output") or "")[:500]
        return (
            f"status: failed (orchestration: {payload.get('reason')})\n"
            f"the worker never returned a structured result; its last raw output:\n{last}"
        )
    parts = [f"status: {payload.get('status')}"]
    if payload.get("summary"):
        parts.append(f"summary: {payload['summary']}")
    if payload.get("data"):
        parts.append(f"data: {json.dumps(payload['data'], ensure_ascii=False)}")
    if payload.get("deliverables"):
        parts.append(f"deliverables: {json.dumps(payload['deliverables'], ensure_ascii=False)}")
    if payload.get("reason"):
        parts.append(f"reason: {payload['reason']}")
    return "\n".join(parts)


async def _supervisor_dispatch(
    kit: RoomKit,
    supervisor: Agent,
    room_id: str,
    *,
    goal: str,
    workers: list[Agent],
    share_channels: list[str] | None,
    task_timeout: float,
) -> str:
    """The supervisor takes the lead: knowing its full team (each worker's role +
    description), it reads the user goal and frames the FIRST worker's task, so the
    chain starts from a supervisor-authored brief rather than the raw user message.
    Falls back to the raw goal if the supervisor returns nothing."""
    roster = "\n".join(f"- {_worker_profile(w)}" for w in workers)
    prompt = (
        "You are the supervisor of a worker team that runs in this FIXED order. Know "
        "your team and what each member does:\n"
        f"{roster}\n\n"
        "A user gave you the goal below. Frame a clear, specific, self-contained task "
        "for your FIRST worker so it can start — addressed to its role, stating exactly "
        "what to produce. Respect each worker's own job: do not strip a standing "
        "responsibility such as publishing a report or sending a message.\n\n"
        f"User goal:\n{goal}\n\n"
        f"First worker — {_worker_profile(workers[0])}.\n\n"
        "Respond with ONLY the task text for that worker — no preamble, no JSON."
    )
    framed, _ok = await _delegate_and_wait(
        kit,
        room_id,
        supervisor.channel_id,
        prompt,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )
    return (framed or "").strip() or goal


async def _supervisor_review(
    kit: RoomKit,
    supervisor: Agent,
    room_id: str,
    *,
    goal: str,
    worker: Agent,
    output: str,
    next_worker: Agent | None,
    share_channels: list[str] | None,
    task_timeout: float,
) -> dict[str, Any]:
    """Run the supervisor (in its own child room) to judge a worker's output and,
    if approved, frame the next worker's task — in one call. Returns the parsed
    verdict ``{approved, feedback, next_task}``."""
    if next_worker is not None:
        next_clause = (
            "If you APPROVE, write 'next_task' as a clear, self-contained task for the "
            f"next worker — {_worker_label(next_worker)} — framed for ITS role, carrying "
            "whatever of this output it needs as input."
        )
    else:
        next_clause = "This was the LAST worker; leave 'next_task' empty."
    prompt = (
        "You are the supervisor reviewing ONE step of your team's work. Judge it "
        "STRICTLY — the team's final answer is only as good as what you let through.\n\n"
        f"User goal:\n{goal}\n\n"
        f"Worker that just finished — {_worker_label(worker)}:\n{output}\n\n"
        "APPROVE only if the output genuinely fulfills the worker's part of the goal: "
        "correct, complete, and directly usable. REJECT (approved=false) if the worker "
        "gave up or claimed it couldn't find anything / that the subject doesn't exist / "
        "that data is missing, asked a question back instead of delivering, returned "
        "status=failed, or produced something vague, off-topic, or not actually answering "
        "the user's intent. Well-formatted text that does not do the job is still a reject. "
        "When you reject, give precise, actionable feedback on exactly what to fix. "
        f"{next_clause}\n\n"
        f"{_VERDICT_INSTRUCTIONS}"
    )
    raw, _ok = await _delegate_and_wait(
        kit,
        room_id,
        supervisor.channel_id,
        prompt,
        share_channels=share_channels,
        task_timeout=task_timeout,
    )
    return _parse_verdict(raw)


def _compose_rework(task: str, output: str, feedback: str) -> str:
    """Re-frame a worker's task after the supervisor rejected its output."""
    return (
        f"{task}\n\n"
        "--- Revision requested by the supervisor ---\n"
        f"Your previous attempt was NOT accepted. Feedback:\n{feedback}\n\n"
        f"Your previous output (for reference):\n{output}\n\n"
        "Produce a corrected, complete result that addresses the feedback."
    )


def _format_supervised_digest(goal: str, steps: list[dict[str, Any]], max_revisions: int) -> str:
    """Brief handed back to the supervisor's own turn so it writes the final
    user-facing summary — each step's validated output + validation status."""
    aborted = bool(steps) and not steps[-1]["approved"]
    if aborted:
        intro = (
            "Your team could NOT complete the task. The final step below FAILED after "
            f"{max_revisions} attempts, so the chain was STOPPED — later workers did not "
            "run. Tell the user HONESTLY that the task could not be completed: name the "
            "step that failed and why, and summarize what was accomplished before it. Do "
            "NOT fabricate a finished result or a deliverable that does not exist."
        )
    else:
        intro = (
            "Your team has finished and you have reviewed each step. Deliver ONE final "
            "summary to the user: what each step accomplished and the outcome. Reference "
            "any deliverables (published reports/artifacts) by their link."
        )
    lines = [intro, "", f"User request:\n{goal}", "", "Reviewed work:"]
    for step in steps:
        status = "validated" if step["approved"] else f"FAILED after {max_revisions} attempts"
        lines.append(f"\n--- {step['role']} ({status}) ---\n{step['output'] or '(no output)'}")
    return "\n".join(lines)


def _compose_supervised_handoff(framing: str, prior_steps: list[dict[str, Any]]) -> str:
    """Build the next worker's task: the supervisor's framing PLUS the team's
    validated work embedded verbatim.

    The supervisor's ``next_task`` says WHAT the next worker must do, but it
    references prior results in prose ("build the report from the analyst's
    data") — an LLM won't reliably paste the content. So the supervisor curates
    the instruction and the code carries the data: each prior worker's rendered
    result is attached. Without this the next worker gets a task pointing at data
    it never sees and reports it as missing."""
    if not prior_steps:
        return framing
    blocks = [framing, "", "--- Work already completed by the team (build on this) ---"]
    for step in prior_steps:
        blocks.append(f"\n[{step['role']}]:\n{step['output'] or '(no output)'}")
    return "\n".join(blocks)


async def _run_supervised_sequential(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    workers: list[Agent],
    task_desc: str,
    *,
    max_revisions: int,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Hub & spoke sequential: fixed worker order, but every output returns to the
    supervisor, which validates it (rework up to *max_revisions*) and frames the
    next worker's task. Returns the reviewed steps
    (``{worker, role, output, approved}``) for the caller to present/summarize."""
    steps: list[dict[str, Any]] = []
    # The supervisor leads: it reads the user goal + its own instructions and
    # frames the FIRST worker's task, rather than forwarding the raw user message.
    task = (
        await _supervisor_dispatch(
            kit,
            supervisor,
            room_id,
            goal=task_desc,
            workers=workers,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
        if workers
        else task_desc
    )
    for i, worker in enumerate(workers):
        next_worker = workers[i + 1] if i + 1 < len(workers) else None
        revisions = 0
        approved = False
        rendered = ""
        verdict: dict[str, Any] = {}
        while True:
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.PENDING,
                detail=task,
                metadata={"room_id": room_id, "strategy": "supervised"},
            )
            # The worker MUST hand its work back via submit_result (forced
            # structure + a guaranteed result); ``output`` is the JSON payload.
            output, ok = await _delegate_and_wait(
                kit,
                room_id,
                worker.channel_id,
                task,
                share_channels=share_channels,
                task_timeout=task_timeout,
                require_structured_result=True,
            )
            rendered = _render_result(output)
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.COMPLETED if ok else StatusLevel.FAILED,
                detail=rendered,
                metadata={"room_id": room_id, "strategy": "supervised"},
            )
            if not ok:
                # The delegation itself FAILED — a timeout, or a provider error
                # like an exhausted credit balance. This is infrastructure, not
                # content quality: re-running the same task won't fix it, and it
                # must NEVER be waved through the supervisor's review (which can't
                # repair infra and might approve the error blob, making the run
                # look successful). Leave the step unapproved and stop the chain;
                # the supervisor reports the failure in its final summary.
                break
            verdict = await _supervisor_review(
                kit,
                supervisor,
                room_id,
                goal=task_desc,
                worker=worker,
                output=rendered,
                next_worker=next_worker,
                share_channels=share_channels,
                task_timeout=task_timeout,
            )
            if verdict["approved"]:
                approved = True
                break
            revisions += 1
            if revisions >= max_revisions:
                break
            task = _compose_rework(task, rendered, verdict["feedback"])
        steps.append(
            {
                "worker": worker.channel_id,
                "role": _worker_label(worker),
                "output": rendered,
                "approved": approved,
            }
        )
        if not approved:
            # Exhausted max_revisions without an accepted result. Continuing on
            # unvalidated work would only propagate the failure, so the chain
            # STOPS here — the supervisor reports the failure to the user in its
            # final summary (see _format_supervised_digest) rather than handing
            # broken input to the next worker.
            break
        # Approved: the supervisor frames the next worker's task, with the team's
        # validated work embedded so the next worker receives the actual data, not
        # just a prose reference to it ("use the analyst's data").
        framing = verdict.get("next_task") or task_desc
        task = _compose_supervised_handoff(framing, steps)
    return steps


def _compose_sequential_input(task_desc: str, prior_steps: list[tuple[str, str]]) -> str:
    """Build a worker's input for sequential delegation.

    The first worker (no prior steps) gets the task unchanged. Every later
    worker gets the original task plus each prior worker's labeled output, so
    the goal and accumulated work survive the chain — a worker handed only its
    predecessor's raw output has no task to act on and just converses with it.
    """
    if not prior_steps:
        return task_desc
    blocks = [f"Original task:\n{task_desc}", "", "Work already completed by the team:"]
    for label, output in prior_steps:
        blocks.append(f"\n--- {label} ---\n{output or '(no output)'}")
    blocks.append("\nBuild on the work above to complete your part of the original task.")
    return "\n".join(blocks)


def _format_supervisor_review(task_desc: str, result_json: str, workers: list[Agent]) -> str:
    """Re-present the workers' combined results to the supervisor as a review
    brief: the original request, then each worker's labeled output, with an
    instruction to verify the work and deliver one final answer (or flag what's
    missing). Returns the raw payload unchanged if it can't be parsed."""
    try:
        parsed = json.loads(result_json)
    except (ValueError, TypeError):
        return result_json
    labels = {w.channel_id: _worker_label(w) for w in workers}
    lines = [
        "Your team has finished. Review their work against the user's request "
        "below, then deliver ONE final answer to the user. If the work is "
        "incomplete or wrong, say what's missing — do not invent content.",
        "",
        f"User request:\n{task_desc}",
        "",
        "Team output:",
    ]
    for item in parsed.get("results", []):
        cid = item.get("worker", "")
        label = labels.get(cid) or cid or "worker"
        lines.append(f"\n--- {label} ---\n{item.get('output') or '(no output)'}")
    return "\n".join(lines)


async def _run_sequential(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> str:
    """Run workers in order, each receiving the original task plus every prior
    worker's output (see :func:`_compose_sequential_input`).

    Each delegation is bounded by *task_timeout*; a worker that exceeds it is
    recorded as failed and the chain continues so the rest of the team — and
    the supervisor's review — still runs on partial results."""
    results: list[dict[str, str]] = []
    prior_steps: list[tuple[str, str]] = []

    for worker in workers:
        worker_input = _compose_sequential_input(task_desc, prior_steps)
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=worker_input,
            metadata={"room_id": room_id, "strategy": "sequential"},
        )
        try:
            delegated = await asyncio.wait_for(
                kit.delegate(
                    room_id,
                    worker.channel_id,
                    worker_input,
                    wait=True,
                    share_channels=share_channels,
                ),
                timeout=task_timeout,
            )
        except TimeoutError:
            timeout_msg = f"Worker timed out after {task_timeout:.0f}s"
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=timeout_msg,
                metadata={"room_id": room_id, "strategy": "sequential"},
            )
            results.append({"worker": worker.channel_id, "output": timeout_msg})
            prior_steps.append((_worker_label(worker), timeout_msg))
            continue
        except Exception as exc:
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=str(exc),
                metadata={"room_id": room_id, "strategy": "sequential"},
            )
            raise
        output = ""
        status_ok = False
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
            status_ok = delegated.result.status == "completed"
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.COMPLETED if status_ok else StatusLevel.FAILED,
            detail=output,
            metadata={
                "room_id": room_id,
                "strategy": "sequential",
                "task_id": delegated.id,
            },
        )
        results.append({"worker": worker.channel_id, "output": output})
        prior_steps.append((_worker_label(worker), output))

    return json.dumps({"status": "completed", "results": results})


async def _run_parallel(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> str:
    """Run all workers concurrently on the same task. Each is bounded by
    *task_timeout*; one that exceeds it is recorded as failed without aborting
    its siblings."""

    async def _delegate_one(worker: Agent) -> dict[str, str]:
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=task_desc,
            metadata={"room_id": room_id, "strategy": "parallel"},
        )
        try:
            delegated = await asyncio.wait_for(
                kit.delegate(
                    room_id,
                    worker.channel_id,
                    task_desc,
                    wait=True,
                    share_channels=share_channels,
                ),
                timeout=task_timeout,
            )
        except TimeoutError:
            timeout_msg = f"Worker timed out after {task_timeout:.0f}s"
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=timeout_msg,
                metadata={"room_id": room_id, "strategy": "parallel"},
            )
            return {"worker": worker.channel_id, "output": timeout_msg}
        except Exception as exc:
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=str(exc),
                metadata={"room_id": room_id, "strategy": "parallel"},
            )
            raise
        output = ""
        status_ok = False
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
            status_ok = delegated.result.status == "completed"
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.COMPLETED if status_ok else StatusLevel.FAILED,
            detail=output,
            metadata={
                "room_id": room_id,
                "strategy": "parallel",
                "task_id": delegated.id,
            },
        )
        return {"worker": worker.channel_id, "output": output}

    results = await asyncio.gather(*[_delegate_one(w) for w in workers])
    return json.dumps({"status": "completed", "results": list(results)})


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


async def _extract_output_text(output: ChannelOutput) -> str:
    """Extract plain text from a ChannelOutput.

    Handles both synchronous response_events and streaming responses.
    For streaming, drains the stream to collect the full text.
    """
    # Check synchronous response first
    if output.response_events:
        for resp in output.response_events:
            if isinstance(resp.content, TextContent) and resp.content.body:
                return resp.content.body

    # Drain streaming response if present
    if output.response_stream is not None:
        parts: list[str] = []
        async for chunk in output.response_stream:
            if isinstance(chunk, str):
                parts.append(chunk)
        return "".join(parts)

    return ""


def _format_worker_results(results: list[dict[str, Any]]) -> str:
    """Format worker results as readable text for the supervisor's presentation.

    Prefers the worker's role as the label and, when the step carries a
    supervised verdict, surfaces its validation status so the supervisor's final
    summary can flag anything it could not validate.
    """
    parts: list[str] = []
    for r in results:
        label = r.get("role") or r.get("worker", "unknown")
        output = r.get("output", "")
        suffix = ""
        if "approved" in r:
            suffix = " (validated)" if r["approved"] else " (UNVALIDATED)"
        parts.append(f"--- {label}{suffix} ---\n{output}")
    return "\n\n".join(parts)
