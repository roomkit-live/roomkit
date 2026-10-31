"""DelegationMixin — task delegation to child rooms."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit._text import bounded_text, identifier, one_line, quoted
from roomkit.channels._discussion_turn import agent_handle
from roomkit.channels._realtime_context import get_current_voice_session
from roomkit.core._failure_log import log_failure
from roomkit.core.exceptions import ChannelNotRegisteredError, RoomNotFoundError

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
from roomkit.core.task_utils import (
    _finish_cleanup,
    await_interruptible,
    check_open,
    hold_task,
    shielded,
)
from roomkit.core.visibility import normal_visibility
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
from roomkit.models.response_metadata import TurnEntries
from roomkit.tasks._child_status import record_task_end
from roomkit.tasks._room_tasks import AGENT_REACH, RoomTaskRecord, TaskCancel, TaskReach
from roomkit.tasks.brief import conversation_block, worker_description
from roomkit.tasks.handback import CALLER_HANDS_BACK, bounded, hand_back, result_text
from roomkit.tasks.models import (
    DelegatedTask,
    DelegatedTaskResult,
    cancelled_task_fields,
    finished_task_fields,
    task_work,
)
from roomkit.tasks.posting import HAND_BACKS, post_result, posted_text
from roomkit.tasks.status import (
    post_task_ended,
    post_task_pending,
    room_task_lines,
)
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.tools.context import _current_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import RoomTasksLoader
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.orchestration.result import ResultTool
    from roomkit.store.base import ConversationStore
    from roomkit.tasks.base import TaskRunner
    from roomkit.telemetry.base import TelemetryProvider


_tasks_logger = logging.getLogger("roomkit.tasks")

_ROOM_TASKS_WINDOW = 500
"""Bus entries an AI channel reads for its room's tasks, as ``task_status`` does."""


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


def _cut_task_fields(
    cut: BaseException, context: dict[str, Any] | None, turns: TurnEntries | None
) -> dict[str, Any]:
    """The outcome of a task cut outside its turn's outcome (its delegation
    before it ran, or its run): cancelled, or failed with what cut it and the
    record of the turn it ran (RFC §23.3 step 6). A cancellation comes here
    only before a turn began, so it carries no record: a run cancelled from
    outside ends through ``_end_inline_cancelled`` or its runner."""
    if isinstance(cut, asyncio.CancelledError):
        return cancelled_task_fields(context)
    return finished_task_fields(None, cut, context, turns)


def _result_from_handle(
    handle: DelegatedTask,
    *,
    status: TaskStatus,
    output: str | None,
    error: str | None,
    duration_ms: float,
    metadata: dict[str, Any],
    exception: Exception | None = None,
) -> DelegatedTaskResult:
    """Build a result from a task handle's identity fields (id, room ids, agent)."""
    return DelegatedTaskResult(
        task_id=handle.id,
        child_room_id=handle.child_room_id,
        parent_room_id=handle.parent_room_id,
        agent_id=handle.agent_id,
        status=status,
        output=output,
        exception=exception,
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
        _closed: Whether the framework is closing (no delegation starts).
        _background_runs: The background work the framework holds until it
            ends, so its close cuts it (a hand-back under way included).
        _task_cancels: The ``cancel_task`` cancels under way, by task: the
            channels whose calls asked, and the cancel's outcome.
        _room_tasks: What the framework knows of the tasks it runs for
            each room: those running, how each can end, and how the latest
            ended (RFC §23.4).

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
    _closed: bool
    _background_runs: set[asyncio.Task[None]]
    _task_cancels: dict[str, TaskCancel]
    _room_tasks: RoomTaskRecord


_TASK_NAME_CHARS = 200
"""How much of what was asked a hand-back names: enough to tell one request from
another, short enough to stay a name."""


def task_name(task: str) -> str:
    """*task* as a hand-back's metadata names it: on one line, bounded (RFC §23.3
    step 8)."""
    return bounded_text(one_line(task), _TASK_NAME_CHARS)


def _hand_back_metadata(result: DelegatedTaskResult, task: str) -> dict[str, Any]:
    """The task a hand-back carries, relayed or posted, so a hook tells a task's
    result from any other event and names the task without reading its text
    (RFC §23.3 step 8)."""
    return {
        "task_id": result.task_id,
        "agent_id": result.agent_id,
        "task_status": str(result.status),
        "task": task_name(task),
    }


def _delegation_result_text(
    result: DelegatedTaskResult, task: str = "", worker: str | None = None
) -> str:
    """What the notified agent receives of a finished background delegation: its
    outcome, what was asked (*task*, so a result back after the conversation moved
    on is said for what was asked), then the worker's output. *worker* names the
    worker as the room addresses it (a discussion agent's handle); None, its
    channel id."""
    # A failed task's error is an exception's message: for the logs and
    # ON_TASK_COMPLETED, never for a model (RFC §9.3).
    outcome = {TaskStatus.COMPLETED: "completed", TaskStatus.CANCELLED: "cancelled"}.get(
        result.status, "failed"
    )
    # What was asked is quoted, the worker named by an identifier (RFC §6.4).
    asked = f" Task: {quoted(task, _TASK_NAME_CHARS)}." if task.strip() else ""
    named = identifier(worker or result.agent_id, "worker")
    return result_text(
        f"[Background task from {named} {outcome}.{asked} Share the outcome with the user.]",
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
    _background_runs: set[asyncio.Task[None]]
    _task_cancels: dict[str, TaskCancel]
    _room_tasks: RoomTaskRecord

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see DelegationHost
    create_room: Any  # see DelegationHost
    attach_channel: Any  # see DelegationHost
    deliver: Any  # see DelegationHost

    def _build_room_tasks_loader(self) -> RoomTasksLoader:
        """Build the loader an AI channel reads its room's background tasks with,
        for the turn's notes (RFC §23.4): the StatusBus's lines for the room, as
        ``task_status`` reads them."""
        kit_ref = self

        async def _load(room_id: str) -> list[dict[str, Any]]:
            bus = kit_ref.status_bus  # ty: ignore[unresolved-attribute]  # RoomKit's
            return await room_task_lines(bus, room_id, _ROOM_TASKS_WINDOW)

        return _load

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
        post_status: bool = True,
        brief: str | None = None,
        conversation: int = 0,
        hand_back: str = "relay",
        organization_id: str | None = None,
    ) -> DelegatedTask:
        """Delegate a task to an agent in a child room.

        Creates a child room linked to *room_id*, attaches the agent and
        any shared channels, then either runs the agent inline or submits
        the task for background execution.

        Args:
            room_id: Parent room ID.
            agent_id: Channel ID of the agent to run the task.
            task: What was asked: the agent reads it unless *brief* is given,
                and a hand-back, the StatusBus and a turn's notes name the
                task by it (RFC §23.3 step 4).
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
            post_status: Follow the task on :attr:`status_bus` (RFC §23.3):
                ``pending`` once announced, then ``completed`` or ``failed``.
                ``False`` for a caller that posts its tasks there itself (an
                orchestration strategy's worker run), so none shows twice.
            brief: What the agent is told instead of *task*, when it says
                more (how to work, what to report); it never reaches the
                parent room.
            conversation: How many of the parent room's last messages the
                agent is handed, those the notified channel may read, quoted
                as data (0: none). Its child room starts empty.
            hand_back: How a background task's result reaches the room:
                ``"relay"``, the notified channel is told it (step 8), or
                ``"post"``, it is posted as it is, as a message of the notified
                agent, without a turn of its model. ``"post"`` needs *notify*
                (or *agent_id*) to name an intelligence channel attached to
                the room, and a background task.
            organization_id: The organization the caller acts for (RFC
                §17.2): the parent room is read with it before anything is
                created or run.

        Returns:
            A :class:`DelegatedTask` handle. When *wait* is ``True``,
            the result is already set.  When ``False``, call ``.wait()``
            to block for the result, or let it run fire-and-forget.

        Raises:
            RoomNotFoundError: If the parent room doesn't exist, or is
                another organization's.
            ChannelNotRegisteredError: If the agent channel isn't registered.
            ValueError: An unknown *hand_back*, a ``"post"`` the notified
                channel cannot take, or a negative *conversation*.
        """
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        # Validate: nothing starts on a closing kit (RFC §19.7.3).
        check_open(self, "delegation")
        parent_room = await self.get_room(room_id, organization_id=organization_id)
        if agent_id not in self._channels:
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")
        notify_channel = notify or agent_id
        await self._check_hand_back(room_id, hand_back, notify_channel, wait=wait)
        description = worker_description(
            task, brief, await self._room_conversation(room_id, notify_channel, conversation)
        )

        child_room_id = f"{room_id}::task-{uuid4().hex[:12]}"
        task_id = f"task-{uuid4().hex[:12]}"
        handle = DelegatedTask(
            id=task_id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task=task,
            brief=brief,
            description=description,
            hand_back=hand_back,
        )
        handle._post_status = post_status
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
        started = False
        try:
            await self._open_child_room(handle, parent_room, context, share_channels)
            await self._announce_task(handle)
            if self._closed:
                # The kit's close began while the task was being set up: it
                # ends as the close ends a running one, on both doors.
                started = True
                await self._end_inline_cancelled(handle, context, on_complete, span, start)
                return handle
            if wait:
                started = True  # its run ends it, however it ends
                return await self._run_inline(
                    handle,
                    context,
                    on_complete,
                    span,
                    require_structured_result=require_structured_result,
                    max_result_retries=max_result_retries,
                    result_tool=result_tool,
                )
            # A task its runner refused never started: the except below ends it.
            return await self._run_background(handle, context, notify, on_complete, span)
        except BaseException as exc:
            if not started:
                cut = self._end_cut(handle, exc, context, on_complete, span, start)
                await shielded(cut)
            raise

    async def _check_hand_back(
        self, room_id: str, hand_back: str, notify_channel: str, *, wait: bool
    ) -> None:
        """Refuse a *hand_back* the delegation cannot keep (RFC §23.3 step 8)."""
        if hand_back not in HAND_BACKS:
            raise ValueError(f"hand_back must be one of {HAND_BACKS}, not {hand_back!r}")
        if hand_back != "post":
            return
        if wait:
            raise ValueError("hand_back='post' is for a background task: an inline one returns")
        channel = self._channels.get(notify_channel)
        binding = await self._store.get_binding(room_id, notify_channel)
        if (
            channel is None
            or channel.category != ChannelCategory.INTELLIGENCE
            or binding is None
            or binding.category != ChannelCategory.INTELLIGENCE
        ):
            raise ValueError(
                f"hand_back='post' posts as {notify_channel!r}, which must be an "
                f"intelligence channel attached to room {room_id!r}"
            )

    async def _room_conversation(self, room_id: str, reader: str, count: int) -> str | None:
        """The parent room's last *count* messages *reader* may read, set
        apart as data for the worker (RFC §23.3 step 4)."""
        if count < 0:
            raise ValueError("conversation must be 0 or more")
        if count == 0:
            return None
        context = await self._build_context(
            room_id, recent_limit=max(4 * count, 50), reads_history=True
        )
        return conversation_block(context, reader, count)

    async def _end_cut(
        self,
        handle: DelegatedTask,
        cut: BaseException,
        context: dict[str, Any] | None,
        on_complete: Any | None,
        span: _DelegationSpan,
        start: float,
        turns: TurnEntries | None = None,
    ) -> None:
        """End a task *cut* ended outside its turn's outcome, its delegation
        before it ran (its runner refusing it included) or its run (RFC
        §23.3): cancelled, or failed with *cut* and the record of the turn it
        ran (*turns*); once announced, as an inline task ends, before that,
        only its span."""
        ended = _result_from_handle(
            handle,
            duration_ms=(time.monotonic() - start) * 1000,
            **_cut_task_fields(cut, context, turns),
        )
        if handle._announced:
            await self._complete_inline(handle, ended, on_complete, span)
        else:
            span.end(ended)

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
        # The child room is the parent's organization's (RFC §17.2, §23.3 step
        # 1): what acts in it reads its tenant from the room, as in the parent.
        await self.create_room(
            room_id=child_room_id,
            organization_id=parent_room.organization_id,
            metadata={
                **inherited_metadata,
                "_child_metadata": inherited_metadata,
                "parent_room_id": room_id,
                "task_id": handle.id,
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
                    # A value stored before the grammar was enforced is read
                    # leniently, never widened (RFC §7.3).
                    visibility=normal_visibility(parent_binding.visibility),
                    muted=parent_binding.muted,
                    direction=parent_binding.direction,
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
        handle._announced = True
        self._hook_engine._start_async_hooks(
            room_id, HookTrigger.ON_TASK_DELEGATED, hook_event, room_context
        )
        if handle._post_status:
            post_task_pending(self, handle)  # ty: ignore[invalid-argument-type]

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
        """Run the agent inline, as a task the kit holds until it ends, and
        return the ended task (RFC §23.3): the kit's close cuts it as it cuts
        the background, cancelled with its completion run; the caller's own
        cancellation is raised once the task has ended."""
        handle.status = TaskStatus.IN_PROGRESS
        start = time.monotonic()
        # Recorded for its room: it ends with the turn that asked for it, or
        # is the strategy's that runs it without posting it (RFC §23.4).
        reach: TaskReach = "turn" if handle._post_status else "strategy"
        self._room_tasks.start(handle.id, handle.parent_room_id, reach)
        run = hold_task(
            self,
            self._inline_task(
                handle,
                context,
                on_complete,
                span,
                start,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
                result_tool=result_tool,
            ),
            awaited=True,
            eager=True,
        )
        try:
            await await_interruptible(run)
        finally:
            # Its completion recorded its end; a run that never reached its
            # completion is recorded failed.
            self._room_tasks.end(handle.id, str(TaskStatus.FAILED))
        return handle

    async def _inline_task(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        on_complete: Any | None,
        span: _DelegationSpan,
        start: float,
        *,
        require_structured_result: bool,
        max_result_retries: int,
        result_tool: ResultTool | None,
    ) -> None:
        """Run an inline task's turn and end it, as one run the kit holds: a
        cut from its caller or from the kit's close ends it cancelled, a raise
        outside its turn's outcome failed, its completion run to its end (RFC
        §23.3)."""
        turns: TurnEntries = {}
        turn = self._child_turn(
            handle,
            require_structured_result=require_structured_result,
            max_result_retries=max_result_retries,
            result_tool=result_tool,
            turns=turns,
        )
        try:
            agent_response, failure, caller_cut = await _turn_outcome(turn)
            if failure is not None:
                log_failure(_tasks_logger, failure, f"Inline task {handle.id}")
            result = _result_from_handle(
                handle,
                duration_ms=(time.monotonic() - start) * 1000,
                **finished_task_fields(agent_response, failure, context, turns),
            )
        except asyncio.CancelledError:
            # A cut from outside (a supervisor's per-task timeout through
            # asyncio.wait_for, the kit's close): the task ends as any task
            # does, its begun turn on it, then the cancellation goes on.
            await self._end_inline_cancelled(
                handle, context, on_complete, span, start, turns=turns
            )
            raise
        except BaseException as exc:
            # Raised outside its turn's outcome: it ends failed, once.
            cut = self._end_cut(handle, exc, context, on_complete, span, start, turns)
            await _finish_cleanup(cut)
            raise
        # Its work ran: it ends as it stands, whatever cancels its caller now.
        await _finish_cleanup(self._complete_inline(handle, result, on_complete, span))
        if caller_cut:
            raise asyncio.CancelledError

    async def _end_inline_cancelled(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        on_complete: Any | None,
        span: _DelegationSpan,
        start: float,
        *,
        turns: TurnEntries | None = None,
    ) -> None:
        """End an inline task cut before its answer: cancelled, with no
        output, the worker's turn under ``turns`` when it had begun (*turns*
        given), its completion run to its end whatever cancels it again."""
        elapsed = (time.monotonic() - start) * 1000
        worker = handle.agent_id if turns is not None else None
        fields = cancelled_task_fields(context, turns, worker=worker)
        cancelled = _result_from_handle(handle, duration_ms=elapsed, **fields)
        await _finish_cleanup(self._complete_inline(handle, cancelled, on_complete, span))

    def _child_turn(
        self,
        handle: DelegatedTask,
        *,
        require_structured_result: bool,
        max_result_retries: int,
        result_tool: ResultTool | None,
        turns: TurnEntries,
    ) -> asyncio.Task[str | None]:
        """The delegated turn, in a task of its own: the worker's turn sets
        its tool-loop context in a copy of the caller's, so a cut that ends
        the turn elsewhere never leaves it in the delegating call's (RFC §23.3)."""
        return asyncio.create_task(
            run_agent_in_child_room(
                self,  # ty: ignore[invalid-argument-type]
                handle.child_room_id,
                handle.description,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
                result_tool=result_tool,
                turns=turns,
            )
        )

    async def _complete_inline(
        self,
        handle: DelegatedTask,
        result: DelegatedTaskResult,
        on_complete: Any | None,
        span: _DelegationSpan,
    ) -> None:
        """End an inline delegation: its end for its room's ``cancel_task``,
        its span, its child room's status, ON_TASK_COMPLETED, its completion
        callback, then its waiters. No proactive delivery: the caller
        presents the result itself."""
        self._room_tasks.end(handle.id, str(result.status))
        span.end(result)
        await record_task_end(self, result)  # ty: ignore[invalid-argument-type]
        await self._on_delegation_complete(result)
        if handle._post_status:
            post_task_ended(self, result)  # ty: ignore[invalid-argument-type]
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
            # Recorded ended first: a cancel meanwhile reads it ended, whenever
            # the bus shows that end.
            self._room_tasks.end(result.task_id, str(result.status))
            await _complete(result)

        async def _complete(result: DelegatedTaskResult) -> None:
            span.end(result)
            await self._on_delegation_complete(result)
            if handle._post_status:
                post_task_ended(self, result)  # ty: ignore[invalid-argument-type]
            if notify_channel != CALLER_HANDS_BACK and not self._cancelled_by_notified(
                result, notify_channel
            ):
                if handle.hand_back == "post":
                    await self._post_held(result, notify_channel, chain_depth, handle.task)
                else:
                    await self._hand_back_held(
                        result, notify_channel, chain_depth, session_id, handle.task
                    )
            if on_complete:
                await on_complete(result)

        # A task a strategy runs without posting it is the strategy's to
        # follow, never a room's cancel_task's (RFC §23.4).
        reach: TaskReach = "tool" if handle._post_status else "strategy"
        self._room_tasks.start(handle.id, handle.parent_room_id, reach)
        # A submit that raises leaves the task to its delegation, which ends
        # it failed (RFC §23.3).
        await self._task_runner.submit(
            self,  # ty: ignore[invalid-argument-type]
            handle,
            context=context,
            on_complete=_on_bg_complete,
        )
        return handle

    async def cancel_task(self, task_id: str, *, organization_id: str | None = None) -> bool:
        """Cancel a delegated task still running in the background (RFC §23.3).

        The task ends ``cancelled``, as any task cancelled from outside:
        ``ON_TASK_COMPLETED`` fires, the StatusBus posts its end, and the
        channel its result was going to is told it was cancelled. A task that
        already ended, or that the framework's task runner does not hold (an
        inline delegation), is left as it stands. Calls cancelling it at once,
        this one and an agent's ``cancel_task`` among them, cancel it once.

        *organization_id* is the organization the caller acts for (RFC
        §17.2): only a task running for one of its rooms is cancelled;
        another organization's, as a task the runner does not hold, is left
        as it stands.

        Returns:
            ``True`` when the task was running and is now cancelled.
        """
        if organization_id is not None and not await self._task_in_scope(task_id, organization_id):
            return False
        return await self._cancel_joined(task_id, None)

    async def _task_in_scope(self, task_id: str, organization_id: str) -> bool:
        """Whether *task_id* runs for a room of *organization_id* (RFC §17.2)."""
        room_id = self._room_tasks.room_of(task_id)
        if room_id is None:
            return False
        try:
            await self.get_room(room_id, organization_id=organization_id)
        except RoomNotFoundError:
            return False
        return True

    async def _cancel_task_for(
        self,
        task_id: str,
        channel_id: str | None,
        *,
        room_id: str,
        reaches: frozenset[TaskReach] = AGENT_REACH,
    ) -> bool:
        """Cancel *task_id* for a call made in *room_id* that may cancel the
        room's tasks of *reaches* (an agent's ``cancel_task``, the host's
        console); when *channel_id*, the caller, is the one the result was
        going to, it is not handed the cancellation back, the call's answer
        having told it. Calls asking at once cancel it once, each answered as
        the first is, one asking as a cancel's end is being completed included
        (RFC §23.4)."""
        reach = self._room_tasks.reach(task_id, room_id)
        if reach in reaches or self._cancel_ending(task_id, room_id, reaches):
            return await self._cancel_joined(task_id, channel_id)
        return False

    def _cancel_ending(self, task_id: str, room_id: str, reaches: frozenset[TaskReach]) -> bool:
        """Whether *task_id*, a task of *room_id* of *reaches*, ended cancelled
        while a cancel of it is still under way, its end not yet decided whom
        it hands the cancellation back to: a call asking now is one of that
        cancel's (RFC §23.4)."""
        cancel = self._task_cancels.get(task_id)
        record = self._room_tasks
        return (
            cancel is not None
            and not cancel.sealed
            and record.ended_with(task_id, room_id) == str(TaskStatus.CANCELLED)
            and record.ended_reach(task_id, room_id) in reaches
        )

    async def _cancel_joined(self, task_id: str, channel_id: str | None) -> bool:
        """Cancel *task_id* once, whichever doors ask at once (the host's, an
        agent's ``cancel_task``), each answered as the first is; *channel_id*,
        a calling agent, is not handed the cancellation back (RFC §23.4)."""
        under_way = self._task_cancels.get(task_id)
        if under_way is None:
            return await self._cancel_first(task_id, channel_id)
        if channel_id:
            under_way.callers.add(channel_id)
        return await asyncio.shield(under_way.cancelled)

    async def _cancel_first(self, task_id: str, channel_id: str | None) -> bool:
        """Cancel *task_id* for the first call asking, its outcome kept for
        the calls asking meanwhile."""
        cancel = self._task_cancels[task_id] = TaskCancel(channel_id)
        cancelled = False
        try:
            cancelled = await self._task_runner.cancel(task_id)
        finally:
            del self._task_cancels[task_id]
            cancel.cancelled.set_result(cancelled)
        return cancelled

    def _cancelled_by_notified(self, result: DelegatedTaskResult, notify_channel: str) -> bool:
        """Whether *result* is a cancellation its notified channel asked for:
        the cancel's callers read once, the cancel sealed, so a call asking
        after this is answered as for any ended task (RFC §23.4)."""
        cancel = self._task_cancels.get(result.task_id)
        if cancel is None:
            return False
        cancel.sealed = True
        return result.status == TaskStatus.CANCELLED and notify_channel in cancel.callers

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
            self._hook_engine._start_async_hooks(
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
        task: str = "",
    ) -> None:
        """Hand a background delegation's result back in a task the kit holds,
        so its ``close()`` cuts it as it cuts a strategy's hand-back: the turn
        it started is cancelled and the cut logged, and the task's end goes
        on (RFC §23.3 step 8)."""
        delivery = self._deliver_delegation_result(
            result, notify_channel_id, chain_depth, session_id, task
        )
        if await await_interruptible(hold_task(self, delivery)):
            _tasks_logger.info(
                "Result of task %s not handed back: the framework closed", result.task_id
            )

    async def _post_held(
        self, result: DelegatedTaskResult, agent_id: str, chain_depth: int, task: str
    ) -> None:
        """Post a background delegation's result as *agent_id*'s message
        (``hand_back="post"``, RFC §23.3 step 8), in a task the kit holds, so
        its ``close()`` cuts it as it cuts a relayed hand-back."""
        text = posted_text(result, task)
        if text is None:
            return
        post = post_result(
            self,  # ty: ignore[invalid-argument-type]
            result.parent_room_id,
            agent_id,
            text,
            chain_depth=chain_depth,
            metadata=_hand_back_metadata(result, task),
        )
        try:
            if await await_interruptible(hold_task(self, post)):
                _tasks_logger.info(
                    "Result of task %s not posted: the framework closed", result.task_id
                )
        except Exception:
            _tasks_logger.exception("Posting failed for task %s", result.task_id)

    async def _deliver_delegation_result(
        self,
        result: DelegatedTaskResult,
        notify_channel_id: str,
        chain_depth: int,
        session_id: str | None = None,
        task: str = "",
    ) -> None:
        """Hand a background delegation's result back to its room, at *chain_depth*.

        The depth of the turn that delegated (RFC §23.3): the notified agent's
        answer is one deeper, so a cycle of delegation, result and delegation
        again ends at ``max_chain_depth``. A task that did not complete is
        handed back whatever it left, a failure says it failed (§23.3 step 8);
        only a completed task with nothing to say is not. A realtime voice
        channel is told in *session_id*, the session that delegated. The
        result names *task*, what was asked. An inline delegation never comes
        here: its caller presents the result itself.
        """
        if result.status == TaskStatus.COMPLETED and not result.output:
            return
        try:
            room = await self._store.get_room(result.parent_room_id)
            worker = agent_handle(room.metadata if room else None, result.agent_id)
            await hand_back(
                self,  # ty: ignore[invalid-argument-type]
                result.parent_room_id,
                notify_channel_id,
                _delegation_result_text(result, task, worker),
                chain_depth,
                session_id=session_id,
                metadata=_hand_back_metadata(result, task),
            )
        except Exception:
            _tasks_logger.exception("Delivery failed for task %s", result.task_id)
