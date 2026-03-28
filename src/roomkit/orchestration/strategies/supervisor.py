"""Supervisor orchestration strategy.

A supervisor agent delegates tasks to worker agents via ``kit.delegate()``.
Workers are registered on the kit but NOT attached to the parent room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import HookRegistration
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

            async_delivery: If ``True``, workers run in the background
                and results are delivered via ``kit.deliver()`` when
                ready. The conversation continues uninterrupted. For
                voice channels, injects a ``delegate_workers`` tool
                that the AI calls when delegation is needed.
                If ``False`` (default), blocks until results are
                ready (sync mode).

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

        if auto_delegate and self._strategy is None:
            msg = "auto_delegate=True requires strategy to be set"
            raise ValueError(msg)

    def agents(self) -> list[Agent]:
        """Return agents to attach to the room.

        In async_delivery mode, the supervisor is NOT attached — the
        conversation channel (e.g. RealtimeVoiceChannel) handles user
        interaction independently.
        """
        if self._async_delivery:
            return []
        return [self._supervisor]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire supervisor routing and delegation tools."""
        # Router only needed when supervisor agent is in the room
        if not self._async_delivery:
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

        async def async_tool_handler(name: str, arguments: dict[str, Any]) -> str:
            nonlocal _running

            if name != "delegate_workers":
                if original_handler:
                    return await original_handler(name, arguments)
                return json.dumps({"error": f"Unknown tool: {name}"})

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
            asyncio.create_task(
                _async_run_and_deliver(
                    kit=kit,
                    room_id=room_id,
                    strategy=strategy,
                    workers=workers,
                    task_desc=task_desc,
                    on_done=lambda: _clear(),
                )
            )

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
        _lock = asyncio.Lock()
        # Per-room dedup: prevents duplicate calls within the same turn
        _dedup_cache: dict[str, tuple[str, float]] = {}  # room_id → (result, timestamp)
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

                try:
                    if strategy == WorkerStrategy.SEQUENTIAL:
                        result = await _run_sequential(kit, rid, workers, task_desc)
                    else:
                        result = await _run_parallel(kit, rid, workers, task_desc)
                    _dedup_cache[rid] = (result, time.monotonic())
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

                    def _patched_set(r: Any, *, _wid: str = worker_id or "") -> None:
                        pending.discard(_wid)
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
    on_done: Any,
) -> None:
    """Background: run workers → deliver results via kit.deliver()."""
    try:
        worker_results = await _run_workers(kit, room_id, strategy, workers, task_desc)
        results_text = _format_worker_results(worker_results)
        logger.info("[async_delegate] Workers completed, delivering results")

        await kit.deliver(
            room_id,
            f"Analysis results are ready. Here's what the analysts found:\n\n{results_text}",
        )
    except Exception:
        logger.exception("[async_delegate] Pipeline failed")
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
    worker_results = await _run_workers(kit, room_id, strategy, workers, refined_task)

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
) -> ChannelOutput:
    """One-pass: workers run on raw message → supervisor presents."""
    # Extract user's raw message
    user_message = ""
    if isinstance(event.content, TextContent):
        user_message = event.content.body

    if not user_message:
        return await original_on_event(event, binding, context)

    # Run workers with the raw user message
    worker_results = await _run_workers(kit, room_id, strategy, workers, user_message)

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
) -> list[dict[str, str]]:
    """Run workers according to strategy and return raw results."""
    if strategy == WorkerStrategy.SEQUENTIAL:
        result_json = await _run_sequential(kit, room_id, workers, task_desc)
    else:
        result_json = await _run_parallel(kit, room_id, workers, task_desc)
    parsed = json.loads(result_json)
    return parsed.get("results", [])


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
