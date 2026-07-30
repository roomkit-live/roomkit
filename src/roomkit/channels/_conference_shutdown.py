"""The one shutdown a conference channel has, and the report it produces.

``close()`` used to be a method like any other: every caller ran its own copy,
and two callers — a framework shutdown overlapping an integrator's close, the
ordinary way it happens — ran the teardown steps twice against books that
each pass was emptying under the other. The coordinator makes the shutdown a
*thing* rather than a call: the first ``close()`` creates it, every later or
concurrent ``close()`` joins it, a cancelled caller abandons only its own
wait, and once it ends its result is the channel's answer for good — replayed,
never re-earned. Retrying what a shutdown could not do (removing a session an
SFU refuses to release) is the operator's task its failure names.

The report is the other half. A close step that fails, times out, is abandoned
mid-cancellation or has to leave a resource open used to end up in whichever
of a log line, a flag or an unstructured string its author reached for — and
what only a log records, a caller reads as success. Every such outcome is a
:class:`CloseIssue` here, and the final :class:`ConferenceCloseError` is built
from nothing else.

See RFC section 12.10.4 — "Closing and shared state".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any

from roomkit.channels import _conference_activity
from roomkit.channels._conference_operations import ConferenceOperations, ConferenceResource
from roomkit.core.exceptions import ConferenceCloseError
from roomkit.core.task_utils import log_task_exception

logger = logging.getLogger("roomkit.channels.conference")


class CloseStatus(Enum):
    """What became of one closing step."""

    FAILED = "failed"
    """The step raised."""

    TIMED_OUT = "timed out"
    """The step outlived its budget and honoured the cancellation."""

    ABANDONED = "abandoned"
    """The step outlived its budget *and* its cancellation grace. It is still
    running somewhere, and nothing it was using has been freed on its account."""

    RETAINED = "retained"
    """A resource was left open because operations that outlived their budgets
    are still using it. It closes in the background once they truly end; this
    close does not wait for that, and does not claim it happened."""


@dataclass(frozen=True)
class CloseIssue:
    """One thing the close could not do, said once and structurally.

    ``step`` is the human name the report prints ("closing the conference
    backend"); ``component`` and ``operation`` are the stable identifiers an
    operator's tooling can match on without parsing prose.
    """

    component: str
    operation: str
    status: CloseStatus
    step: str
    detail: str = ""

    def render(self) -> str:
        text = f"{self.step} [{self.status.value}]"
        return f"{text}: {self.detail}" if self.detail else text


class ConferenceShutdownCoordinator:
    """Owns the close's task, its budget-keeping, and its report.

    The channel still decides the *order* of its shutdown — that knowledge is
    the channel's own. What lives here is everything the steps share: the one
    task concurrent closes join, the budget-and-grace discipline each bounded
    step runs under, the background tasks that outlive the close, and the
    issues the final raise is built from.
    """

    def __init__(self, channel_id: str, operations: ConferenceOperations) -> None:
        self._channel_id = channel_id
        self._operations = operations
        self._close_task: asyncio.Task[None] | None = None
        self._issues: list[CloseIssue] = []
        # Tasks that outlive the close: steps that ignored their cancellation,
        # and the deferred closes waiting for them. Referenced so they are
        # neither garbage-collected mid-flight nor left to dump their parting
        # exception into the loop's handler.
        self._background: set[asyncio.Task[Any]] = set()

    # -------------------------------------------------------------------------
    # The one shutdown
    # -------------------------------------------------------------------------

    async def close(self, run: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Run the channel's one shutdown, or join the one already running.

        Shielded, so a caller cancelled mid-wait abandons its wait and nothing
        else: the other callers, and the invariants the steps maintain, still
        depend on the shutdown finishing. Once the task has ended, awaiting it
        again replays its terminal result — an immediate return after a
        success, the same exception after a failure. ``close()`` reports what
        the one shutdown achieved; it never re-runs it.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                run(), name=f"roomkit-conference-close-{self._channel_id}"
            )
        await asyncio.shield(self._close_task)

    @property
    def started(self) -> bool:
        """Whether the channel's shutdown exists (running or finished)."""
        return self._close_task is not None

    # -------------------------------------------------------------------------
    # The report
    # -------------------------------------------------------------------------

    def record(
        self,
        *,
        component: str,
        operation: str,
        status: CloseStatus,
        step: str,
        detail: str = "",
    ) -> None:
        """Keep one stable issue for a step that did not complete."""
        issue = CloseIssue(
            component=component, operation=operation, status=status, step=step, detail=detail
        )
        if issue not in self._issues:
            self._issues.append(issue)

    @property
    def issues(self) -> list[CloseIssue]:
        return list(self._issues)

    def raise_for_failures(self, stuck: dict[str, list[str]]) -> None:
        """Fail the close for every retained session and recorded issue.

        The last thing the shutdown does, so nothing is skipped on a failure's
        account. Sessions, abandoned operations, retained resources and failed
        steps are reported in their own sections rather than one undifferenti-
        ated string: what an operator must go and remove (a bot still in a
        meeting) is a different task from what will resolve itself (a deferred
        close waiting on a wedged call).
        """
        if not stuck and not self._issues:
            return
        reports: list[str] = []
        if stuck:
            details = "; ".join(
                f"room {room_id}: session(s) {', '.join(sessions)}"
                for room_id, sessions in sorted(stuck.items())
            )
            total = sum(len(sessions) for sessions in stuck.values())
            reports.append(
                f"{total} bot session(s) could not be taken out — {details}. They remain "
                "on the channel's books and info() reports them"
            )
        for status, label in (
            (CloseStatus.RETAINED, "resource(s) retained under operations still running"),
            (CloseStatus.ABANDONED, "step(s) abandoned past their budget"),
            (CloseStatus.TIMED_OUT, "step(s) that outlived their budget"),
            (CloseStatus.FAILED, "close step failure(s)"),
        ):
            section = [issue.render() for issue in self._issues if issue.status is status]
            if section:
                reports.append(f"{label}: " + "; ".join(section))
        raise ConferenceCloseError(
            f"Conference channel {self._channel_id!r} did not close cleanly — "
            + " — ".join(reports),
            issues=tuple(self._issues),
        )

    # -------------------------------------------------------------------------
    # Bounded steps
    # -------------------------------------------------------------------------

    async def spend(
        self,
        step: Coroutine[Any, Any, Any],
        step_name: str,
        *,
        component: str,
        operation: str = "close",
    ) -> asyncio.Task[Any] | None:
        """Run one closing step on a budget it may not outlive.

        The steps this covers end in code the channel does not own — a
        backend's network call, a provider's shutdown — and the framework
        closes channels in sequence, so time spent here is spent holding
        every channel behind this one in its conference (RFC 12.10.4).

        Past the budget the step is cancelled, given a short grace to unwind,
        and then abandoned rather than waited for again: a provider that
        swallows the cancellation does not get to hold the shutdown. Nothing
        the survivor was using is freed on its account *by this method* — its
        leases hold the resources it touches, and the task is returned so a
        caller can treat it as a barrier of its own.

        A step that raises is recorded rather than propagated, so one failure
        never costs the steps after it their run.
        """
        budget = _conference_activity.DRAIN_TIMEOUT_S
        task = asyncio.ensure_future(step)
        _, pending = await asyncio.wait({task}, timeout=budget)
        if pending:
            task.cancel()
            _, pending = await asyncio.wait({task}, timeout=_conference_activity.CANCEL_GRACE_S)
        if pending:
            self.retain(task)
            self.record(
                component=component,
                operation=operation,
                status=CloseStatus.ABANDONED,
                step=step_name,
                detail="did not return within the budget or honour cancellation",
            )
            logger.error(
                "Conference channel %r abandoned a closing step after %.1fs: %s did not "
                "return and did not honour its cancellation. Nothing it was using has "
                "been freed on its account, and any session it failed to remove is "
                "still on the books",
                self._channel_id,
                budget,
                step_name,
            )
            return task
        if task.cancelled():
            self.record(
                component=component,
                operation=operation,
                status=CloseStatus.TIMED_OUT,
                step=step_name,
                detail="outlived the budget and was cancelled",
            )
            logger.error(
                "Conference channel %r cancelled a closing step after %.1fs: %s outlived "
                "the budget. Any session it failed to remove is still on the books",
                self._channel_id,
                budget,
                step_name,
            )
            return None
        if (failure := task.exception()) is not None:
            self.record(
                component=component,
                operation=operation,
                status=CloseStatus.FAILED,
                step=step_name,
                detail=f"{type(failure).__name__}: {failure}",
            )
            logger.error(
                "Conference channel %r: %s failed: %s. The close carries on; what the "
                "step failed to do is reported from the books, not from this log alone",
                self._channel_id,
                step_name,
                failure,
                exc_info=failure,
            )
            return None
        return None

    def retain(self, task: asyncio.Task[Any]) -> None:
        """Keep a task that outlives the close referenced until it truly ends."""
        self._background.add(task)
        task.add_done_callback(self._forget)

    def _forget(self, task: asyncio.Task[Any]) -> None:
        self._background.discard(task)
        if not task.cancelled():
            task.exception()

    # -------------------------------------------------------------------------
    # Closing resources by their leases
    # -------------------------------------------------------------------------

    async def close_resource(
        self,
        resources: ConferenceResource | tuple[ConferenceResource, ...],
        closer: Callable[[], Coroutine[Any, Any, None]],
        *,
        step: str,
        blockers: set[asyncio.Task[Any]] | None = None,
    ) -> None:
        """Close resources once nothing the channel admitted is using them.

        One closer for the resources it closes together, because their closing
        order is the closer's own knowledge (the pipeline before the STT).

        The registry's leases are the authority; ``blockers`` adds tasks whose
        work spans several backend calls (a teardown between its ``leave()``
        and its ``close_room()``), which a per-call lease cannot see across.

        A resource still in use past a short settling wait is *retained*: the
        close is deferred to a background task that waits however long the
        surviving operations take, the retention is recorded, and this close
        fails rather than claiming a resource it left open is closed
        (RFC 12.10.4).
        """
        if isinstance(resources, ConferenceResource):
            resources = (resources,)
        resources = tuple(r for r in resources if not self._operations.is_closed(r))
        if not resources:
            return
        blocked = {task for task in (blockers or set()) if not task.done()}
        idle = await self._operations.wait_idle(
            *resources, timeout=_conference_activity.CANCEL_GRACE_S
        )
        blocked = {task for task in blocked if not task.done()}
        component = "+".join(resource.value for resource in resources)
        if idle and not blocked:
            for resource in resources:
                self._operations.mark_closed(resource)
            await self.spend(closer(), step, component=component)
            return
        using: set[str] = set()
        for resource in resources:
            using.update(self._operations.holders(resource))
        using.update(task.get_name() for task in blocked)
        self.record(
            component=component,
            operation="close",
            status=CloseStatus.RETAINED,
            step=step,
            detail="operation(s) that outlived the budget are still using it: "
            + ", ".join(sorted(using)),
        )
        logger.error(
            "Conference channel %r is retaining its %s: %d operation(s) that outlived the "
            "closing budget are still using it (%s). It closes in the background once they "
            "end; this close() reports it as not closed",
            self._channel_id,
            component,
            len(using),
            ", ".join(sorted(using)),
        )
        deferred = asyncio.create_task(
            self._close_when_free(resources, closer, blocked),
            name=f"roomkit-deferred-close-{component}",
        )
        self.retain(deferred)
        deferred.add_done_callback(log_task_exception)

    async def _close_when_free(
        self,
        resources: tuple[ConferenceResource, ...],
        closer: Callable[[], Coroutine[Any, Any, None]],
        blockers: set[asyncio.Task[Any]],
    ) -> None:
        """The deferred half of :meth:`close_resource`, off the close's clock."""
        if blockers:
            await asyncio.gather(*blockers, return_exceptions=True)
        await self._operations.when_idle(*resources)
        if any(self._operations.is_closed(resource) for resource in resources):
            return
        for resource in resources:
            self._operations.mark_closed(resource)
        try:
            await closer()
        except Exception:
            logger.exception(
                "Conference channel %r could not close its %s after retaining it; "
                "the resource is leaked",
                self._channel_id,
                "+".join(resource.value for resource in resources),
            )
