"""The ``Supervisor`` orchestration strategy class.

Wires supervisor routing and one of three delegation modes (framework
auto-delegate, a single strategy tool, or per-worker tools). The execution
helpers live in the sibling modules of this package.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_MAX_REVISIONS,
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    WorkerStrategy,
    _post_worker_status,
    logger,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _async_run_and_deliver,
    _one_pass_delegate,
    _two_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.execution import _run_parallel
from roomkit.orchestration.strategies.supervisor.prompts import _format_supervised_digest
from roomkit.orchestration.strategies.supervisor.results import _format_supervisor_review
from roomkit.orchestration.strategies.supervisor.supervised import _run_supervised_sequential
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


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
