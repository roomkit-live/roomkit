"""DelegationMixin — task delegation to child rooms."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    TaskStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.framework import RoomKit
    from roomkit.store.base import ConversationStore
    from roomkit.tasks.base import TaskRunner


_tasks_logger = logging.getLogger("roomkit.tasks")


# ---------------------------------------------------------------------------
# Shared agent execution — single code path for both sync and background
# ---------------------------------------------------------------------------


async def run_agent_in_child_room(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
) -> str | None:
    """Send a task to a child room and collect the agent's text response.

    This is the **single code path** for executing a delegated agent,
    used by both ``delegate(wait=True)`` (inline) and the background
    task runner.

    1. Stores *task_desc* as a system message in the child room.
    2. Broadcasts the event — the attached agent picks it up.
    3. Collects the response (sync or streaming) and returns it.
    """
    room = await kit.get_room(child_room_id)
    bindings = await kit.store.list_bindings(child_room_id)

    # Store the task as a message
    task_event = RoomEvent(
        room_id=child_room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body=task_desc),
    )
    task_event = await kit.store.add_event_auto_index(child_room_id, task_event)

    # Build context AFTER storing so the agent's memory provider
    # can see the task event in recent_events.
    recent = await kit.store.list_events(child_room_id, offset=0, limit=50)
    context = RoomContext(room=room, bindings=bindings, recent_events=recent)

    # Broadcast — the agent picks up the task
    router = kit._get_router()
    source_binding = ChannelBinding(
        channel_id="system",
        room_id=child_room_id,
        channel_type=ChannelType.SYSTEM,
    )
    result = await router.broadcast(task_event, source_binding, context)

    # Collect response: synchronous output first, then streaming
    for output in result.outputs.values():
        if output.responded and output.response_events:
            for resp in output.response_events:
                if isinstance(resp.content, TextContent) and resp.content.body:
                    await kit.store.add_event_auto_index(child_room_id, resp)
                    return resp.content.body

    for sr in result.streaming_responses:
        parts: list[str] = []
        async for delta in sr.stream:
            parts.append(delta)
        text = "".join(parts)
        if text:
            resp_event = RoomEvent(
                room_id=child_room_id,
                type=EventType.MESSAGE,
                source=EventSource(
                    channel_id=sr.source_channel_id,
                    channel_type=sr.source_channel_type,
                ),
                content=TextContent(body=text),
                chain_depth=task_event.chain_depth + 1,
            )
            await kit.store.add_event_auto_index(child_room_id, resp_event)
            return text

    return None


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


# ---------------------------------------------------------------------------
# DelegationMixin
# ---------------------------------------------------------------------------


class DelegationMixin(HelpersMixin):
    """Task delegation to child rooms — sync and background."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner

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
        await self.get_room(room_id)  # type: ignore[attr-defined]
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
        await self.create_room(  # type: ignore[attr-defined]
            room_id=child_room_id,
            metadata={
                "parent_room_id": room_id,
                "task_agent_id": agent_id,
                "task_input": task,
                "task_context": context or {},
                "task_status": "pending",
            },
            orchestration=None,
        )

        # Attach agent as intelligence
        await self.attach_channel(  # type: ignore[attr-defined]
            child_room_id,
            agent_id,
            category=ChannelCategory.INTELLIGENCE,
        )

        # Share channels from parent
        for ch_id in share_channels or []:
            parent_binding = await self._store.get_binding(room_id, ch_id)
            if parent_binding:
                await self.attach_channel(  # type: ignore[attr-defined]
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
            visibility="internal",
            metadata=hook_meta,
        )
        room_context = await self._build_context(room_id)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_TASK_DELEGATED, hook_event, room_context
        )

        if wait:
            result_handle = await self._run_inline(handle, context, notify, on_complete)
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
    ) -> DelegatedTask:
        """Run the agent inline and return a pre-completed task."""
        start = time.monotonic()
        handle.status = TaskStatus.IN_PROGRESS
        agent_response: str | None = None
        error: str | None = None

        try:
            agent_response = await run_agent_in_child_room(
                self,  # type: ignore[arg-type]
                handle.child_room_id,
                handle.task,
            )
        except Exception as exc:
            _tasks_logger.exception("Inline task %s failed: %s", handle.id, exc)
            error = str(exc)

        elapsed = (time.monotonic() - start) * 1000
        status = TaskStatus.COMPLETED if agent_response else TaskStatus.FAILED

        result = DelegatedTaskResult(
            task_id=handle.id,
            child_room_id=handle.child_room_id,
            parent_room_id=handle.parent_room_id,
            agent_id=handle.agent_id,
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
            self,
            handle,
            context=context,
            on_complete=_on_bg_complete,  # type: ignore[arg-type]
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
            visibility="internal",
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
                await self.deliver(  # type: ignore[attr-defined]
                    result.parent_room_id,
                    prompt,
                    channel_id=notify_channel_id,
                )
            except Exception:
                _tasks_logger.exception("Delivery failed for task %s", result.task_id)
