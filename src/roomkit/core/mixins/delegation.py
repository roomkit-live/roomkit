"""DelegationMixin — task delegation to child rooms."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
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
    return meta


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
        deliver: From :class:`DeliverMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner
    _hook_engine: HookEngine
    _telemetry: TelemetryProvider | None


class DelegationMixin(HelpersMixin):
    """Task delegation to child rooms — sync and background.

    Host contract: :class:`DelegationHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner

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
            notify: Channel ID to update when the task completes
                (system prompt injection). Defaults to *agent_id*.
            on_complete: Optional async callback ``(DelegatedTaskResult) -> None``.

        Returns:
            A :class:`DelegatedTask` handle. When *wait* is ``True``,
            the result is already set.  When ``False``, call ``.wait()``
            to block for the result, or let it run fire-and-forget.

        Raises:
            RoomNotFoundError: If the parent room doesn't exist.
            ChannelNotRegisteredError: If the agent channel isn't registered.
        """
        from uuid import uuid4

        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        # Validate
        parent_room = await self.get_room(room_id)
        if agent_id not in self._channels:
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")

        child_room_id = f"{room_id}::task-{uuid4().hex[:12]}"
        task_id = f"task-{uuid4().hex[:12]}"

        # Start telemetry span
        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        mode = "inline" if wait else "background"
        span_id = telemetry.start_span(
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
        )

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

        # Share channels from parent
        for ch_id in share_channels or []:
            parent_binding = await self._store.get_binding(room_id, ch_id)
            if parent_binding:
                await self.attach_channel(
                    child_room_id,
                    ch_id,
                    category=parent_binding.category,
                    metadata=parent_binding.metadata,
                )

        # Create task handle
        handle = DelegatedTask(
            id=task_id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task=task,
        )

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

        if wait:
            result_handle = await self._run_inline(
                handle,
                context,
                notify,
                on_complete,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
            )
            telemetry.end_span(
                span_id,
                attributes={
                    Attr.DELEGATION_STATUS: result_handle.status,
                    Attr.DURATION_MS: result_handle.result.duration_ms
                    if result_handle.result
                    else 0,
                },
            )
            return result_handle

        # Background — span ends when task completes (via callback)
        return await self._run_background(handle, context, notify, on_complete, span_id, telemetry)

    async def _run_inline(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        notify: str | None,
        on_complete: Any | None,
        *,
        require_structured_result: bool = False,
        max_result_retries: int = 3,
    ) -> DelegatedTask:
        """Run the agent inline and return a pre-completed task."""
        start = time.monotonic()
        handle.status = TaskStatus.IN_PROGRESS
        agent_response: str | None = None
        error: str | None = None

        try:
            agent_response = await run_agent_in_child_room(
                self,  # ty: ignore[invalid-argument-type]
                handle.child_room_id,
                handle.task,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
            )
        except asyncio.CancelledError:
            # A caller cancelled this delegation (e.g. a supervisor's per-task
            # timeout via asyncio.wait_for). CancelledError is a BaseException, so
            # without handling it here the completion hook below never runs — and
            # consumers that close a step on ON_TASK_COMPLETED (the orchestration
            # timeline) leave it stuck on "running". Fire the completion as FAILED,
            # then propagate the cancellation.
            elapsed = (time.monotonic() - start) * 1000
            cancelled = _result_from_handle(
                handle,
                status=TaskStatus.FAILED,
                output=None,
                error="cancelled (timed out)",
                duration_ms=elapsed,
                metadata=context or {},
            )
            try:
                await self._on_delegation_complete(
                    cancelled, notify or handle.agent_id, deliver=False
                )
            except Exception:
                # Best-effort: the cancellation still propagates below. Log so a
                # failure to fire the completion (which unsticks a "running" step)
                # is visible rather than silently swallowed.
                _tasks_logger.exception(
                    "Completion hook failed for cancelled task %s (room %s)",
                    handle.id,
                    handle.child_room_id,
                )
            handle._set_result(cancelled)
            raise
        except Exception as exc:
            _tasks_logger.exception("Inline task %s failed: %s", handle.id, exc)
            error = str(exc)

        elapsed = (time.monotonic() - start) * 1000
        status = TaskStatus.COMPLETED if agent_response else TaskStatus.FAILED

        result = _result_from_handle(
            handle,
            status=status,
            output=agent_response,
            error=error,
            duration_ms=elapsed,
            metadata=context or {},
        )

        # Fire completion hooks + callbacks (skip proactive delivery for inline —
        # the caller handles presenting results directly)
        notify_channel = notify or handle.agent_id
        await self._on_delegation_complete(result, notify_channel, deliver=False)
        if on_complete:
            try:
                await on_complete(result)
            except Exception:
                _tasks_logger.exception("on_complete failed for task %s", handle.id)

        handle._set_result(result)
        return handle

    async def _run_background(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        notify: str | None,
        on_complete: Any | None,
        span_id: str,
        telemetry: Any,
    ) -> DelegatedTask:
        """Submit the task to the background task runner."""
        from roomkit.telemetry.base import Attr

        notify_channel = notify or handle.agent_id

        async def _on_bg_complete(result: DelegatedTaskResult) -> None:
            telemetry.end_span(
                span_id,
                attributes={
                    Attr.DELEGATION_STATUS: result.status,
                    Attr.DURATION_MS: result.duration_ms,
                },
            )
            await self._on_delegation_complete(result, notify_channel)
            if on_complete:
                await on_complete(result)

        await self._task_runner.submit(
            self,  # ty: ignore[invalid-argument-type]
            handle,
            context=context,
            on_complete=_on_bg_complete,
        )
        return handle

    async def _on_delegation_complete(
        self,
        result: DelegatedTaskResult,
        notify_channel_id: str,
        *,
        deliver: bool = True,
    ) -> None:
        """Handle delegation completion: inject result + fire hook + deliver."""
        # Inject result into the notified agent's system prompt
        max_delegation_prompt = 4000
        binding = await self._store.get_binding(result.parent_room_id, notify_channel_id)
        if binding:
            current_prompt = binding.metadata.get("system_prompt", "")
            appendix = (
                "\n\n--- BACKGROUND TASK COMPLETED ---\n"
                + f"Task ID: {result.task_id}\n"
                + f"Agent: {result.agent_id}\n"
                + f"Status: {result.status}\n"
                + f"Result:\n{result.output or result.error or 'No output'}\n"
                + "--- END ---\n"
            )
            new_prompt = current_prompt + appendix
            if len(new_prompt) > max_delegation_prompt:
                new_prompt = "...\n" + new_prompt[-max_delegation_prompt:]
            updated = binding.model_copy(
                update={"metadata": {**binding.metadata, "system_prompt": new_prompt}}
            )
            await self._store.update_binding(updated)

        # Fire ON_TASK_COMPLETED hook with enriched metadata
        hook_meta = _delegation_metadata(
            task_id=result.task_id,
            child_room_id=result.child_room_id,
            parent_room_id=result.parent_room_id,
            agent_id=result.agent_id,
            task_status=result.status,
            duration_ms=result.duration_ms,
            error=result.error,
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

        # Deliver result via kit.deliver() (background path only)
        if not deliver:
            return
        content = result.output or result.error
        if content:
            try:
                prompt = (
                    f"[Background task from {result.agent_id} completed. "
                    f"Share the result with the user.]"
                )
                await self.deliver(
                    result.parent_room_id,
                    prompt,
                    channel_id=notify_channel_id,
                )
            except Exception:
                _tasks_logger.exception("Delivery failed for task %s", result.task_id)
