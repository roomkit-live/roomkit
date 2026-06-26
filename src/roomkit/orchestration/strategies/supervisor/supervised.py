"""Supervised hub-&-spoke sequential flow.

Every worker output returns to the supervisor, which judges it and frames the
next worker's task — workers never hand off to each other directly. Order is
fixed; the supervisor owns each transition (validate + frame + rework) in its
own child rooms. The final user-facing summary is left to the supervisor's own
turn (see ``prompts._format_supervised_digest``).

``_delegate_and_wait`` and its three callers (``_supervisor_dispatch``,
``_supervisor_review``, ``_run_supervised_sequential``) are co-located so a
single test seam over ``_delegate_and_wait`` covers the whole flow.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    _STRATEGY_TOOL_NAME,
    _post_worker_status,
)
from roomkit.orchestration.strategies.supervisor.prompts import (
    _VERDICT_INSTRUCTIONS,
    _compose_rework,
    _compose_supervised_handoff,
    _parse_verdict,
)
from roomkit.orchestration.strategies.supervisor.results import (
    _render_result,
    _result_completed,
    _result_output,
    _worker_label,
    _worker_profile,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


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
    return (_result_output(result), _result_completed(result))


@contextlib.contextmanager
def _supervisor_without_strategy_tool(supervisor: Agent) -> Iterator[None]:
    """Run the supervisor for dispatch/review WITHOUT its ``delegate_workers`` tool.

    In strategy-tool mode the supervisor owns ``delegate_workers``. If it carried
    that tool into its own dispatch/review sub-runs it would try to delegate again
    instead of framing the task or judging the output — re-entering the pipeline
    (delegate_workers within delegate_workers) and stalling. The orchestrator has
    already decided to delegate; here the supervisor only frames and judges. A
    no-op in auto-delegate mode, where the supervisor has no such tool.
    """
    saved = supervisor._injected_tools
    supervisor._injected_tools = [t for t in saved if t.name != _STRATEGY_TOOL_NAME]
    try:
        yield
    finally:
        supervisor._injected_tools = saved


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
    with _supervisor_without_strategy_tool(supervisor):
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
    with _supervisor_without_strategy_tool(supervisor):
        raw, _ok = await _delegate_and_wait(
            kit,
            room_id,
            supervisor.channel_id,
            prompt,
            share_channels=share_channels,
            task_timeout=task_timeout,
        )
    return _parse_verdict(raw)


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
