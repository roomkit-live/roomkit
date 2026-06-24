"""Delegation entry points that wire worker execution to the supervisor.

The framework-driven auto-delegate helpers (one-pass / two-pass), the
background runner that delivers results via ``kit.deliver``, and the strategy
dispatcher that routes to the supervised, sequential, or parallel runner.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_MAX_REVISIONS,
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    WorkerStrategy,
    _post_worker_status,
    logger,
)
from roomkit.orchestration.strategies.supervisor.execution import (
    _run_parallel,
    _run_sequential,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _extract_output_text,
    _format_worker_results,
    _present_worker_results,
)
from roomkit.orchestration.strategies.supervisor.supervised import (
    _run_supervised_sequential,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


def _build_pass1_instruction(workers: list[Agent]) -> str:
    """Build the default pass-1 instruction."""
    return (
        "Extract the core topic or subject from the user's request. "
        "Output only the topic, nothing else. No questions, no instructions, "
        "no formatting. Example: user says 'analyse anthropic' → 'Anthropic'"
    )


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
    Callers that don't care simply accept no arguments.
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
    results_event = RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(body=_present_worker_results(worker_results)),
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
    results_event = RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(
            body=(f"The user asked: {user_message}\n\n{_present_worker_results(worker_results)}")
        ),
    )

    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)
