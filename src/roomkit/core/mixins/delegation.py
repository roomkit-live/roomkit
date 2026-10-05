"""DelegationMixin — task delegation to child rooms."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._realtime_context import get_current_voice_session
from roomkit.core._failure_log import log_failure
from roomkit.core.exceptions import ChannelNotRegisteredError

# _persist_child_stream and _run_with_structured_result are re-exported (self-
# aliased) for the test suite, which imports them from this module.
from roomkit.core.mixins._child_execution import (
    _persist_child_stream as _persist_child_stream,
)
from roomkit.core.mixins._child_execution import (
    _run_with_structured_result as _run_with_structured_result,
)
from roomkit.core.mixins._child_execution import (
    run_agent_in_child_room,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.task_utils import await_interruptible, check_open, hold_task, shielded
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    TaskStatus,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.tasks._child_status import record_task_end
from roomkit.tasks.handback import CALLER_HANDS_BACK, bounded, hand_back, result_text
from roomkit.tasks.models import (
    DelegatedTask,
    DelegatedTaskResult,
    cancelled_task_fields,
    finished_task_fields,
    task_work,
)
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.orchestration.result import ResultTool
    from roomkit.store.base import ConversationStore
    from roomkit.tasks.base import TaskRunner
    from roomkit.telemetry.base import TelemetryProvider


_tasks_logger = logging.getLogger("roomkit.tasks")


# ---------------------------------------------------------------------------
# Hook metadata builder
# ---------------------------------------------------------------------------


def _delegation_metadata(
    *,
    task_id: str,
    child_room_id: str,
    parent_room_id: str,
    agent_id: str,
    task_input: str | None = None,
    task_status: TaskStatus | str | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    loop_end_reason: str | None = None,
) -> dict[str, Any]:
    """Build consistent metadata for delegation hooks."""
    meta: dict[str, Any] = {
        "task_id": task_id,
        "child_room_id": child_room_id,
        "parent_room_id": parent_room_id,
        "agent_id": agent_id,
    }
    if task_input is not None:
        meta["task_input"] = task_input
    if task_status is not None:
        meta["task_status"] = task_status
    if duration_ms is not None:
        meta["duration_ms"] = duration_ms
    if error is not None:
        meta["error"] = error
    if loop_end_reason is not None:
        # How a worker cut short ended its turn (RFC §23.3).
        meta["loop_end_reason"] = loop_end_reason
    return meta


class _DelegationSpan:
    """A delegation's span, ended with its task's status (RFC §23.3)."""

    def __init__(self, telemetry: TelemetryProvider, span_id: str) -> None:
        self._telemetry = telemetry
        self._span_id = span_id

    def end(self, result: DelegatedTaskResult) -> None:
        """End the span with *result*: ``ok`` completed, ``error`` failed
        (with its error), ``cancelled`` cancelled."""
        status = {TaskStatus.COMPLETED: "ok", TaskStatus.FAILED: "error"}.get(
            result.status, "cancelled"
        )
        self._telemetry.end_span(
            self._span_id,
            status=status,
            error_message=result.error if status == "error" else None,
            attributes={
                Attr.DELEGATION_STATUS: result.status,
                Attr.DURATION_MS: result.duration_ms,
            },
        )


async def _turn_outcome(
    turn: asyncio.Task[str | None],
) -> tuple[str | None, Exception | None, bool]:
    """A delegated turn's answer or failure once it ended, and whether its
    caller was cancelled after it had: a turn that ran to its end ends as it
    stands (RFC §23.3). A cancellation that cut the turn itself goes on."""
    try:
        return await turn, None, False
    except asyncio.CancelledError:
        if not turn.done() or turn.cancelled():
            raise
        # The caller was cancelled once the turn had ended.
        failure = turn.exception()
        if failure is None:
            return turn.result(), None, True
        if isinstance(failure, Exception):
            return None, failure, True
        raise
    except Exception as exc:
        return None, exc, False


def _unstarted_task_fields(cut: BaseException, context: dict[str, Any] | None) -> dict[str, Any]:
    """The outcome of a task its delegation cut before it ran: cancelled,
    or failed with what cut it."""
    if isinstance(cut, asyncio.CancelledError):
        return cancelled_task_fields(context)
    return finished_task_fields(None, cut, context)


def _result_from_handle(
    handle: DelegatedTask,
    *,
    status: TaskStatus,
    output: str | None,
    error: str | None,
    duration_ms: float,
    metadata: dict[str, Any],
) -> DelegatedTaskResult:
    """Build a result from a task handle's identity fields (id, room ids, agent)."""
    return DelegatedTaskResult(
        task_id=handle.id,
        child_room_id=handle.child_room_id,
        parent_room_id=handle.parent_room_id,
        agent_id=handle.agent_id,
        status=status,
        output=output,
        error=error,
        duration_ms=duration_ms,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# DelegationMixin
# ---------------------------------------------------------------------------


@runtime_checkable
class DelegationHost(Protocol):
    """Contract: capabilities a host class must provide for DelegationMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _task_runner: Background task execution backend.
        _hook_engine: Engine for hook execution (via :class:`HelpersMixin`).
        _telemetry: Telemetry / tracing provider (optional — mixin
            falls back to ``NoopTelemetryProvider`` when absent).

    Cross-mixin methods (provided by other mixins in the MRO):
        get_room: From :class:`RoomLifecycleMixin`.
        create_room: From :class:`RoomLifecycleMixin`.
        attach_channel: From :class:`ChannelOpsMixin`.
        deliver: From :class:`DeliverMixin`, which hands a background
            result back (:func:`~roomkit.tasks.handback.hand_back`).
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner
    _hook_engine: HookEngine
    _telemetry: TelemetryProvider | None


def _delegation_result_text(result: DelegatedTaskResult) -> str:
    """What the notified agent receives of a finished background delegation."""
    # A failed task's error is an exception's message: for the logs and
    # ON_TASK_COMPLETED, never for a model (RFC §9.3).
    outcome = {TaskStatus.COMPLETED: "completed", TaskStatus.CANCELLED: "cancelled"}.get(
        result.status, "failed"
    )
    return result_text(
        f"[Background task from {result.agent_id} {outcome}. Share the outcome with the user.]",
        bounded(task_work(result) or "No output"),
    )


class DelegationMixin(HelpersMixin):
    """Task delegation to child rooms — sync and background.

    Host contract: :class:`DelegationHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner
    _closed: bool

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see DelegationHost
    create_room: Any  # see DelegationHost
    attach_channel: Any  # see DelegationHost
    deliver: Any  # see DelegationHost

    async def delegate(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        *,
        wait: bool = False,
        context: dict[str, Any] | None = None,
        share_channels: list[str] | None = None,
        notify: str | None = None,
        on_complete: Any | None = None,
        require_structured_result: bool = False,
        max_result_retries: int = 3,
        result_tool: ResultTool | None = None,
    ) -> DelegatedTask:
        """Delegate a task to an agent in a child room.

        Creates a child room linked to *room_id*, attaches the agent and
        any shared channels, then either runs the agent inline or submits
        the task for background execution.

        Args:
            room_id: Parent room ID.
            agent_id: Channel ID of the agent to run the task.
            task: Description of what the agent should do.
            wait: If ``True``, run the agent inline and return a
                pre-completed :class:`DelegatedTask`.  If ``False``
                (default), submit as a background task.
            context: Optional context dict passed to the agent.
            share_channels: Channel IDs from the parent to share.
            notify: Channel the result is handed to when a background task
                completes (RFC §23.3): an agent receives it, bounded, as an
                instruction and answers through the room's transport; a
                transport receives it as a delivery. Defaults to *agent_id*.
            on_complete: Optional async callback ``(DelegatedTaskResult) -> None``.
            require_structured_result: Inline runs only: the agent must hand its
                work back by calling a result tool, re-prompted up to
                *max_result_retries* times; the result's ``output`` is then
                the tool's JSON payload.
            max_result_retries: How many times a turn ending without the call
                is re-prompted.
            result_tool: The tool to force, when not ``submit_result``
                (:data:`~roomkit.orchestration.result.SUBMIT_RESULT`).

        Returns:
            A :class:`DelegatedTask` handle. When *wait* is ``True``,
            the result is already set.  When ``False``, call ``.wait()``
            to block for the result, or let it run fire-and-forget.

        Raises:
            RoomNotFoundError: If the parent room doesn't exist.
            ChannelNotRegisteredError: If the agent channel isn't registered.
        """
        from uuid import uuid4

        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        # Validate: nothing starts on a closing kit (RFC §19.7.3).
        check_open(self, "delegation")
        parent_room = await self.get_room(room_id)
        if agent_id not in self._channels:
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")

        child_room_id = f"{room_id}::task-{uuid4().hex[:12]}"
        task_id = f"task-{uuid4().hex[:12]}"
        handle = DelegatedTask(
            id=task_id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task=task,
        )
        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        mode = "inline" if wait else "background"
        span = _DelegationSpan(
            telemetry,
            telemetry.start_span(
                SpanKind.DELEGATION,
                f"delegation.{mode}",
                parent_id=get_current_span(),
                room_id=room_id,
                channel_id=agent_id,
                attributes={
                    Attr.DELEGATION_TASK_ID: task_id,
                    Attr.DELEGATION_WORKER_ID: agent_id,
                    Attr.DELEGATION_CHILD_ROOM_ID: child_room_id,
                    Attr.DELEGATION_PARENT_ROOM_ID: room_id,
                    Attr.DELEGATION_MODE: mode,
                },
            ),
        )
        start = time.monotonic()
        announced = started = False
        try:
            await self._open_child_room(handle, parent_room, context, share_channels)
            announced = True
            await self._announce_task(handle)
            started = True
            if wait:
                return await self._run_inline(
                    handle,
                    context,
                    on_complete,
                    span,
                    require_structured_result=require_structured_result,
                    max_result_retries=max_result_retries,
                    result_tool=result_tool,
                )
            return await self._run_background(handle, context, notify, on_complete, span)
        except BaseException as exc:
            if not started:
                ended = _result_from_handle(
                    handle,
                    duration_ms=(time.monotonic() - start) * 1000,
                    **_unstarted_task_fields(exc, context),
                )
                await shielded(self._end_unstarted(handle, ended, on_complete, span, announced))
            raise

    async def _end_unstarted(
        self,
        handle: DelegatedTask,
        result: DelegatedTaskResult,
        on_complete: Any | None,
        span: _DelegationSpan,
        announced: bool,
    ) -> None:
        """End a task its delegation cut before it ran (RFC §23.3): once
        announced, as an inline task ends; before that, only its span."""
        if announced:
            await self._complete_inline(handle, result, on_complete, span)
        else:
            span.end(result)

    async def _open_child_room(
        self,
        handle: DelegatedTask,
        parent_room: Any,
        context: dict[str, Any] | None,
        share_channels: list[str] | None,
    ) -> None:
        """Create the task's child room, its agent and the channels shared
        into it (RFC §23.3 steps 1 to 3)."""
        room_id, child_room_id = handle.parent_room_id, handle.child_room_id
        agent_id, task = handle.agent_id, handle.task
        # Create child room — no orchestration so the parent's strategy
        # doesn't leak (e.g. Supervisor attaching itself to the child).
        # The caller may stamp a ``_child_metadata`` envelope on the parent
        # room (e.g. its owning user / tenant); copy it verbatim onto the
        # child so the delegated agent resolves the same ambient context as a
        # turn in the parent room. RoomKit chooses no keys of its own — the
        # envelope's contents are entirely the caller's. Re-stamping the
        # envelope itself lets the context cascade to nested delegations.
        # Delegation bookkeeping keys are applied last so the envelope can
        # never overwrite them.
        inherited_metadata = parent_room.metadata.get("_child_metadata") or {}
        await self.create_room(
            room_id=child_room_id,
            metadata={
                **inherited_metadata,
                "_child_metadata": inherited_metadata,
                "parent_room_id": room_id,
                "task_agent_id": agent_id,
                "task_input": task,
                "task_context": context or {},
                "task_status": "pending",
            },
            orchestration=None,
        )

        # Attach agent as intelligence
        await self.attach_channel(
            child_room_id,
            agent_id,
            category=ChannelCategory.INTELLIGENCE,
        )

        # Share channels from parent. The child binding carries the parent's
        # permissions too (RFC §7.5-6): copying category and metadata while
        # letting access/visibility/muted fall back to the attach defaults
        # would turn a read-only observer — or a muted one — into a full
        # participant in a room the integrator never configured.
        for ch_id in share_channels or []:
            parent_binding = await self._store.get_binding(room_id, ch_id)
            if parent_binding:
                await self.attach_channel(
                    child_room_id,
                    ch_id,
                    category=parent_binding.category,
                    access=parent_binding.access,
                    visibility=parent_binding.visibility,
                    muted=parent_binding.muted,
                    metadata=parent_binding.metadata,
                )

    async def _announce_task(self, handle: DelegatedTask) -> None:
        """Fire ``ON_TASK_DELEGATED`` in the parent room."""
        room_id, child_room_id = handle.parent_room_id, handle.child_room_id
        agent_id, task = handle.agent_id, handle.task
        # Fire ON_TASK_DELEGATED hook
        hook_meta = _delegation_metadata(
            task_id=handle.id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task_input=task,
        )
        hook_event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id=agent_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=f"[Task delegated to {agent_id}] {task}"),
            type=EventType.TASK_DELEGATED,
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
            metadata=hook_meta,
        )
        room_context = await self._build_context(room_id)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_TASK_DELEGATED, hook_event, room_context
        )

    async def _run_inline(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        on_complete: Any | None,
        span: _DelegationSpan,
        *,
        require_structured_result: bool = False,
        max_result_retries: int = 3,
        result_tool: ResultTool | None = None,
    ) -> DelegatedTask:
        """Run the agent inline and return a pre-completed task."""
        start = time.monotonic()
        handle.status = TaskStatus.IN_PROGRESS
        turn = self._child_turn(
            handle,
            require_structured_result=require_structured_result,
            max_result_retries=max_result_retries,
            result_tool=result_tool,
        )
        try:
            agent_response, failure, caller_cut = await _turn_outcome(turn)
        except asyncio.CancelledError:
            # A caller cancelled this delegation (a supervisor's per-task
            # timeout through asyncio.wait_for): the task ends as any task
            # does, cancelled, its completion run to its end, then the
            # cancellation goes on (RFC §23.3).
            elapsed = (time.monotonic() - start) * 1000
            cancelled = _result_from_handle(
                handle, duration_ms=elapsed, **cancelled_task_fields(context)
            )
            await shielded(self._complete_inline(handle, cancelled, on_complete, span))
            raise
        if failure is not None:
            log_failure(_tasks_logger, failure, f"Inline task {handle.id}")
        elapsed = (time.monotonic() - start) * 1000
        result = _result_from_handle(
            handle,
            duration_ms=elapsed,
            **finished_task_fields(agent_response, failure, context),
        )
        # Its work ran: it ends as it stands, whatever cancels its caller now.
        await shielded(self._complete_inline(handle, result, on_complete, span))
        if caller_cut:
            raise asyncio.CancelledError
        return handle

    def _child_turn(
        self,
        handle: DelegatedTask,
        *,
        require_structured_result: bool,
        max_result_retries: int,
        result_tool: ResultTool | None,
    ) -> asyncio.Task[str | None]:
        """The delegated turn, in a task of its own: the worker's turn sets
        its tool-loop context in a copy of the caller's, so a cut that ends
        the turn elsewhere never leaves it in the delegating call's (RFC §23.3)."""
        return asyncio.create_task(
            run_agent_in_child_room(
                self,  # ty: ignore[invalid-argument-type]
                handle.child_room_id,
                handle.task,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
                result_tool=result_tool,
            )
        )

    async def _complete_inline(
        self,
        handle: DelegatedTask,
        result: DelegatedTaskResult,
        on_complete: Any | None,
        span: _DelegationSpan,
    ) -> None:
        """End an inline delegation: its span, its child room's status,
        ON_TASK_COMPLETED, its completion callback, then its waiters. No
        proactive delivery: the caller presents the result itself."""
        span.end(result)
        await record_task_end(self, result)  # ty: ignore[invalid-argument-type]
        await self._on_delegation_complete(result)
        if on_complete:
            try:
                await on_complete(result)
            except Exception:
                _tasks_logger.exception("on_complete failed for task %s", handle.id)
        handle._set_result(result)

    async def _run_background(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        notify: str | None,
        on_complete: Any | None,
        span: _DelegationSpan,
    ) -> DelegatedTask:
        """Submit the task to the background task runner."""
        notify_channel = notify or handle.agent_id
        # Read now, inside the tool call that delegated: the result continues
        # that turn's chain (RFC §23.3), whenever it comes back, and a
        # realtime voice channel is told in the session that made the call.
        chain_depth = _current_turn_chain_depth()
        session = get_current_voice_session()
        session_id = session.id if session is not None else None

        async def _on_bg_complete(result: DelegatedTaskResult) -> None:
            span.end(result)
            await self._on_delegation_complete(result)
            if notify_channel != CALLER_HANDS_BACK:
                await self._hand_back_held(result, notify_channel, chain_depth, session_id)
            if on_complete:
                await on_complete(result)

        await self._task_runner.submit(
            self,  # ty: ignore[invalid-argument-type]
            handle,
            context=context,
            on_complete=_on_bg_complete,
        )
        return handle

    async def _on_delegation_complete(self, result: DelegatedTaskResult) -> None:
        """Fire ``ON_TASK_COMPLETED`` in the parent room for a finished delegation.

        The result reaches the notified agent in the delivered content
        (:meth:`_deliver_delegation_result`), never in the room's stored
        prompt, which is its configuration, not a turn's (RFC §23.3).
        """
        # Fire ON_TASK_COMPLETED hook with enriched metadata
        hook_meta = _delegation_metadata(
            task_id=result.task_id,
            child_room_id=result.child_room_id,
            parent_room_id=result.parent_room_id,
            agent_id=result.agent_id,
            task_status=result.status,
            duration_ms=result.duration_ms,
            error=result.error,
            loop_end_reason=result.metadata.get("loop_end_reason"),
        )
        hook_event = RoomEvent(
            room_id=result.parent_room_id,
            source=EventSource(
                channel_id=result.agent_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=result.output or result.error or ""),
            type=EventType.TASK_COMPLETED,
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
            metadata=hook_meta,
        )
        try:
            room_context = await self._build_context(result.parent_room_id)
            await self._hook_engine.run_async_hooks(
                result.parent_room_id, HookTrigger.ON_TASK_COMPLETED, hook_event, room_context
            )
        except Exception:
            _tasks_logger.exception(
                "Failed to fire ON_TASK_COMPLETED hook for task %s", result.task_id
            )

    async def _hand_back_held(
        self,
        result: DelegatedTaskResult,
        notify_channel_id: str,
        chain_depth: int,
        session_id: str | None,
    ) -> None:
        """Hand a background delegation's result back in a task the kit holds,
        so its ``close()`` cuts it as it cuts a strategy's hand-back: the turn
        it started is cancelled and the cut logged, and the task's end goes
        on (RFC §23.3 step 8)."""
        delivery = self._deliver_delegation_result(
            result, notify_channel_id, chain_depth, session_id
        )
        if await await_interruptible(hold_task(self, delivery)):  # ty: ignore[invalid-argument-type]
            _tasks_logger.info(
                "Result of task %s not handed back: the framework closed", result.task_id
            )

    async def _deliver_delegation_result(
        self,
        result: DelegatedTaskResult,
        notify_channel_id: str,
        chain_depth: int,
        session_id: str | None = None,
    ) -> None:
        """Hand a background delegation's result back to its room, at *chain_depth*.

        The depth of the turn that delegated (RFC §23.3): the notified agent's
        answer is one deeper, so a cycle of delegation, result and delegation
        again ends at ``max_chain_depth``. A task that did not complete is
        handed back whatever it left, a failure says it failed (§23.3 step 8);
        only a completed task with nothing to say is not. A realtime voice
        channel is told in *session_id*, the session that delegated. An
        inline delegation never comes here: its caller presents the result
        itself.
        """
        if result.status == TaskStatus.COMPLETED and not result.output:
            return
        try:
            await hand_back(
                self,  # ty: ignore[invalid-argument-type]
                result.parent_room_id,
                notify_channel_id,
                _delegation_result_text(result),
                chain_depth,
                session_id=session_id,
            )
        except Exception:
            _tasks_logger.exception("Delivery failed for task %s", result.task_id)
