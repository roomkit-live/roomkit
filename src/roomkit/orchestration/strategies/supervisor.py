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
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.strategies.supervisor")


def _post_worker_status(
    kit: RoomKit,
    worker_id: str,
    level: StatusLevel,
    *,
    action: str = "task",
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Post a worker lifecycle event to ``kit.status_bus``.

    Observability only — swallows exceptions so that a broken bus
    never blocks a delegation.
    """
    try:
        kit.status_bus.post(
            worker_id,
            action,
            level,
            detail=detail[:200],
            metadata=metadata or {},
        )
    except Exception:
        logger.debug("status_bus.post failed for %s", worker_id, exc_info=True)


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

            # Launch workers in background — tool returns immediately
            task = asyncio.create_task(
                _async_run_and_deliver(
                    kit=kit,
                    room_id=room_id,
                    strategy=strategy,
                    workers=workers,
                    task_desc=task_desc,
                    share_channels=share_channels,
                    on_done=lambda: _clear(),
                )
            )
            task.add_done_callback(log_task_exception)

            return json.dumps(
                {
                    "status": "dispatched",
                    "workers": worker_roles,
                    "message": "Workers are running. Results will be delivered when ready.",
                }
            )

        def _clear() -> None:
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
        workers = self._workers
        share_channels = self._share_channels
        async_delivery = self._async_delivery
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

                    def _clear(_rid: str = rid) -> None:
                        _running.discard(_rid)

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
                        result = await _run_sequential(
                            kit,
                            rid,
                            workers,
                            task_desc,
                            share_channels=share_channels,
                        )
                    else:
                        result = await _run_parallel(
                            kit,
                            rid,
                            workers,
                            task_desc,
                            share_channels=share_channels,
                        )
                    now = time.monotonic()
                    _dedup_cache[rid] = (result, now)
                    # Evict expired entries to prevent unbounded growth
                    stale = [k for k, (_, ts) in _dedup_cache.items() if now - ts >= dedup_window]
                    for k in stale:
                        del _dedup_cache[k]
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
    on_done: Callable[[], None],
) -> None:
    """Background: run workers → deliver results via kit.deliver().

    Individual worker lifecycle events are posted to ``kit.status_bus``
    inside ``_run_sequential`` / ``_run_parallel``. This helper emits
    one additional terminal entry under ``agent_id="orchestration"``
    so subscribers can observe the pipeline as a whole.
    """
    pipeline_meta = {
        "room_id": room_id,
        "strategy": str(strategy) if strategy else None,
        "workers": [w.channel_id for w in workers],
    }
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
) -> ChannelOutput:
    """Two-pass: supervisor formulates task → workers run → supervisor presents."""
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

    # Run workers with the refined task
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        refined_task,
        share_channels=share_channels,
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
) -> ChannelOutput:
    """One-pass: workers run on raw message → supervisor presents."""
    # Extract user's raw message
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body

    if not user_message:
        return await original_on_event(event, binding, context)

    # Run workers with the raw user message
    worker_results = await _run_workers(
        kit,
        room_id,
        strategy,
        workers,
        user_message,
        share_channels=share_channels,
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
    share_channels: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run workers according to strategy and return raw results."""
    if strategy == WorkerStrategy.SEQUENTIAL:
        result_json = await _run_sequential(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
        )
    else:
        result_json = await _run_parallel(
            kit,
            room_id,
            workers,
            task_desc,
            share_channels=share_channels,
        )
    parsed = json.loads(result_json)
    return parsed.get("results", [])


async def _run_sequential(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
) -> str:
    """Run workers in order, each receiving the previous output."""
    current_input = task_desc
    results: list[dict[str, str]] = []

    for worker in workers:
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=current_input,
            metadata={"room_id": room_id, "strategy": "sequential"},
        )
        try:
            delegated = await kit.delegate(
                room_id,
                worker.channel_id,
                current_input,
                wait=True,
                share_channels=share_channels,
            )
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
        current_input = output

    return json.dumps({"status": "completed", "results": results})


async def _run_parallel(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
) -> str:
    """Run all workers concurrently on the same task."""

    async def _delegate_one(worker: Agent) -> dict[str, str]:
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=task_desc,
            metadata={"room_id": room_id, "strategy": "parallel"},
        )
        try:
            delegated = await kit.delegate(
                room_id,
                worker.channel_id,
                task_desc,
                wait=True,
                share_channels=share_channels,
            )
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


def _format_worker_results(results: list[dict[str, str]]) -> str:
    """Format worker results as readable text for the supervisor."""
    parts: list[str] = []
    for r in results:
        worker = r.get("worker", "unknown")
        output = r.get("output", "")
        parts.append(f"--- {worker} ---\n{output}")
    return "\n\n".join(parts)
