"""Delegation entry points that wire worker execution to the supervisor.

The framework-driven auto-delegate helpers (one-pass / two-pass), the
background runner that hands results back to the supervisor, and the strategy
dispatcher that routes to the supervised, sequential, or parallel runner.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard
from uuid import uuid4

from roomkit.core._failure_log import mark_reported
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.core._requester import asked_by, asking_label, task_heading
from roomkit.core.event_router import StreamingResponse, stream_record
from roomkit.core.mixins._child_execution import persist_tool_calls
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.enums import EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import ResponseMetadata, recorded_turn_end
from roomkit.orchestration._background import (
    BackgroundRun,
    background_failure_text,
    run_in_background,
)
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
from roomkit.tasks.handback import bounded, workers_text
from roomkit.tools.fence import fence

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


def _build_pass1_instruction() -> str:
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
    supervisor_id: str,
    supervisor: Agent,
    strategy: WorkerStrategy | None,
    workers: list[Agent],
    task_desc: str,
    share_channels: list[str] | None = None,
    max_revisions: int = _DEFAULT_MAX_REVISIONS,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
    on_done: Callable[..., None],
) -> None:
    """Background: run workers, then hand their results back to *supervisor_id*.

    The workers run as the install's synchronous delegation runs them, within
    its bounds (*max_revisions*, *task_timeout*): a sequential team through
    the supervised hub-&-spoke flow, *supervisor* validating each step.

    A pipeline that fails hands its failure back the same way, so the
    supervisor, which told the user results would follow, can say the work
    could not be completed (RFC §19.7.3). The steps are every strategy's
    background run (:func:`run_in_background`).

    Each worker's lifecycle is posted to ``kit.status_bus`` by its
    delegation (:func:`~roomkit.orchestration._worker_run.run_worker`). This
    run posts one additional terminal entry under ``agent_id="orchestration"``
    so subscribers can observe the pipeline as a whole: ``COMPLETED`` once its
    results are handed back, ``FAILED`` when it raised, was cancelled, its
    work did not complete, or the hand-back reached nobody.

    ``on_done`` is called once with ``success=<bool>``, whether the run
    returned its results, so callers can distinguish success from failure — e.g. to
    evict cached dispatch responses that should not be re-served after a
    failed pipeline. It is called before the outcome is handed back: the
    supervisor's turn on it may dispatch again, and must find the room free
    rather than the stale ``dispatched`` answer.
    """
    pipeline_meta = {
        "room_id": room_id,
        "strategy": str(strategy) if strategy else None,
        "workers": [w.channel_id for w in workers],
    }

    def post(status: StatusLevel, detail: str) -> None:
        _post_worker_status(
            kit, "orchestration", status, action="pipeline", detail=detail, metadata=pipeline_meta
        )

    async def work() -> list[dict[str, Any]]:
        return await _run_workers(
            kit,
            room_id,
            strategy,
            workers,
            task_desc,
            supervisor=supervisor,
            max_revisions=max_revisions,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )

    run = BackgroundRun(
        room_id=room_id,
        notify=supervisor_id,
        work=work,
        told=_outcome_text,
        ended=_pipeline_ended,
        post=post,
        release=lambda succeeded: on_done(success=succeeded),
    )
    await run_in_background(kit, run)


def _why_failed(worker_results: list[dict[str, Any]]) -> str | None:
    """Why a run's work did not complete, or ``None`` when it did: a
    supervised step the supervisor left unvalidated (the chain stopped there,
    as the supervisor reads it within the turn), or no worker's task completed
    (RFC §19.7.3)."""
    if any("approved" in r and not r["approved"] for r in worker_results):
        return "a step was not validated"
    if not any(r.get("completed") for r in worker_results):
        return "no worker completed"
    return None


def _pipeline_ended(worker_results: list[dict[str, Any]]) -> tuple[StatusLevel, str]:
    """A background pipeline's terminal entry, once handed back: failed when
    its work did not complete, as a Loop whose producer failed (RFC §19.7.3)."""
    if (why := _why_failed(worker_results)) is not None:
        return StatusLevel.FAILED, why
    done = sum(1 for r in worker_results if r.get("completed"))
    return StatusLevel.COMPLETED, f"{done} worker(s) completed"


def _outcome_text(worker_results: list[dict[str, Any]] | None) -> str:
    """What the supervisor reads of its background workers (RFC §19.7.3,
    §23.3): each worker's output bounded, under a header that says whether
    the work completed; for a pipeline that failed (``None``), that the work
    could not be completed, without the failure's message, so the supervisor
    can tell the user."""
    if worker_results is None:
        return background_failure_text("workers")
    each_bounded = [{**r, "output": bounded(str(r.get("output") or ""))} for r in worker_results]
    header = (
        "[Your background workers completed. Share their results with the user.]"
        if _why_failed(worker_results) is None
        else "[Your background workers could not complete the work. Tell the user what failed.]"
    )
    return workers_text(header, _format_worker_results(each_bounded))


def _one_pass_results(user_message: str, worker_results: list[dict[str, Any]]) -> str:
    """What the supervisor presents from in one pass: the user's message as a
    ``<task>`` block, then each worker's output as a block of its own."""
    return (
        f"{task_heading('The user asked:')}\n{fence('task', user_message)}\n\n"
        f"{_present_worker_results(worker_results)}"
    )


def _results_event(event: RoomEvent, body: str) -> RoomEvent:
    """The workers' results, standing in for the event the supervisor answers.

    As deep as the event and in its thread, so the supervisor's answer stays
    one deeper than the event it answers (RFC §8.3, §19.7.3); naming that
    event in ``responds_to``, so the pass that is handed the stand-in can tell
    what it answers (RFC §8.5).
    """
    return RoomEvent(
        room_id=event.room_id,
        type=event.type,
        source=EventSource(channel_id="system", channel_type=_ChannelType.SYSTEM),
        content=TextContent(body=body),
        chain_depth=event.chain_depth,
        parent_event_id=event.parent_event_id,
        responds_to=event.id,
    )


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
    that can answer is given (every output returns to the supervisor, which
    validates it and frames the next worker's task); a supervisor without a
    model (a configuration-only agent, as a voice supervisor often is) cannot
    frame nor judge, and its chain runs unsupervised, each worker given the
    task and the work done before it (RFC §19.7.3). Parallel runs all workers
    on the same task.
    """
    if strategy == WorkerStrategy.SEQUENTIAL and _supervises(supervisor):
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


def _supervises(supervisor: Agent | None) -> TypeGuard[Agent]:
    """Whether *supervisor* can frame and judge a sequential team's steps."""
    return supervisor is not None and not supervisor.is_config_only


async def _formulate_task(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    original_on_event: Any,
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    instruction: str | None,
) -> _Pass1:
    """Pass 1: the supervisor turns the request into a task for its workers.

    The task-formulation instruction rides this call only, on a copy of the
    binding whose prompt is the one the turn would have had followed by the
    instruction (RFC §19.7.3). The supervisor serves every room it is attached
    to: its own prompt is never the carrier, so a room running meanwhile never
    reads another's instruction.
    """
    pass1_instruction = instruction or _build_pass1_instruction()
    _, settings = await supervisor._resolve_turn(binding, context)
    base = settings.get("system_prompt")
    prompt = f"{base}\n\n{pass1_instruction}" if base else pass1_instruction
    pass1_binding = binding.model_copy(
        update={"metadata": {**binding.metadata, "system_prompt": prompt}}
    )
    pass1_output = await original_on_event(event, pass1_binding, context)
    return await _pass1_task(kit, room_id, supervisor, event, pass1_output, context)


@dataclass
class _Pass1:
    """What the task-formulation pass gave: its output, the task it hands
    the workers, and how its turn ended."""

    output: ChannelOutput
    task: str = ""
    end: str | None = None
    record: dict[str, Any] = field(default_factory=dict)


async def _pass1_task(
    kit: RoomKit,
    room_id: str,
    supervisor: Agent,
    event: RoomEvent,
    output: ChannelOutput,
    context: RoomContext,
) -> _Pass1:
    """The task pass 1 hands on: its final answer, as every streamed turn is
    read, its tool calls stored in the room as any turn's (RFC §19.7.3). A
    pass that failed is reported as a streamed turn's failure is, then hands
    its error to the turn's caller, its stream read."""
    if output.error is not None or output.response_stream is None:
        return _Pass1(output, await _extract_output_text(output))
    stream = StreamingResponse(
        stream=output.response_stream,
        source_channel_id=supervisor.channel_id,
        source_channel_type=supervisor.channel_type,
        trigger_event=event,
        response_metadata=output.response_metadata,
    )
    correlation_id = uuid4().hex
    try:
        turn = await persist_tool_calls(kit, room_id, stream, context, correlation_id)
    except Exception as exc:
        await _report_pass1_failure(kit, exc, supervisor, stream, context, correlation_id)
        record = dict(stream_record(stream))
        return _Pass1(_read(output, error=exc), end=recorded_turn_end(record), record=record)
    return _Pass1(_read(output), turn.answer, turn.end, turn.record)


async def _report_pass1_failure(
    kit: RoomKit,
    exc: Exception,
    supervisor: Agent,
    stream: StreamingResponse,
    context: RoomContext,
    correlation_id: str,
) -> None:
    """Report pass 1's failure as every streamed turn's: one log line at its
    own level (whoever opened the turn may not receive it: ``send_event``, a
    delivery), ON_ERROR once, ``streaming``, with its tool rows' correlation;
    then marked, so the room turn it fails hands it on unreported (RFC
    §19.7.3, §15.2)."""
    what = f"Pass 1 of {supervisor.channel_id} in room {context.room.id}"
    await kit._report_stream_failure(exc, what, stream, context, correlation_id=correlation_id)
    mark_reported(exc)


def _with_record(pass1: _Pass1) -> ChannelOutput:
    """The pass's output carrying its turn's record (its end, its usage)."""
    if not pass1.record:
        return pass1.output
    # ``model_copy`` does not validate: the record is wrapped as the field's
    # own type, never left a plain dict.
    metadata = ResponseMetadata({**pass1.output.response_metadata, **pass1.record})
    return pass1.output.model_copy(update={"response_metadata": metadata})


def _read(output: ChannelOutput, *, error: Exception | None = None) -> ChannelOutput:
    """*output* once its stream was read here: nothing of it is left to
    deliver, never an empty stream handed on to the room's transports."""
    update: dict[str, Any] = {"response_stream": None}
    if error is not None:
        update["error"] = error
    return output.model_copy(update=update)


def _pass1_answer(supervisor: Agent, event: RoomEvent, pass1: _Pass1) -> ChannelOutput:
    """What the room reads of a pass that handed on no task: the supervisor's
    fallback when the pass stopped short of its answer, so the message it
    answered gets one, the turn's record on it (RFC §19.7.3); else the pass's
    own output, its error included, carrying the pass's record so the caller
    reads how it ended as of any turn (RFC §6.4). A pass stopped on purpose
    (``cancelled``), or that failed with an error, has no fallback."""
    if pass1.output.error is not None or pass1.end in (None, "completed", "cancelled"):
        return _with_record(pass1)
    fallback = RoomEvent(
        room_id=event.room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id=supervisor.channel_id, channel_type=_ChannelType.AI),
        content=TextContent(body=FALLBACK_FAILED),
        chain_depth=event.chain_depth + 1,
        parent_event_id=event.parent_event_id,
        responds_to=event.id,
        # The turn's whole record, as its last message would carry it (§6.4).
        metadata={**pass1.record, "loop_end_reason": pass1.end},
    )
    return ChannelOutput(
        responded=True,
        response_events=[fallback],
        response_metadata=pass1.output.response_metadata,
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
    pass1 = await _formulate_task(
        kit, room_id, supervisor, original_on_event, event, binding, context, instruction
    )
    refined_task = pass1.task

    logger.debug("Pass 1 refined task: %s", refined_task[:200] if refined_task else "(empty)")

    if not refined_task:
        return _pass1_answer(supervisor, event, pass1)

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
    results_event = _results_event(event, _present_worker_results(worker_results))

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

    # Run workers with the raw user message — supervised between steps in
    # sequential — naming its author when several people speak (RFC §19.7).
    with asked_by(asking_label(event, context, supervisor.channel_id)):
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
        results = _one_pass_results(user_message, worker_results)

    # Inject results into context and let supervisor present
    results_event = _results_event(event, results)

    try:
        await supervisor._memory.ingest(
            event.room_id, results_event, channel_id=supervisor.channel_id
        )
    except Exception:
        logger.warning("Failed to ingest worker results", exc_info=True)

    return await original_on_event(results_event, binding, context)
