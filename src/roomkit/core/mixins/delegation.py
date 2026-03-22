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
from roomkit.tasks.delivery import BackgroundTaskDeliveryStrategy, TaskDeliveryContext
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
# DelegationMixin
# ---------------------------------------------------------------------------


class DelegationMixin(HelpersMixin):
    """Task delegation to child rooms — sync and background."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner
    _delivery_strategy: BackgroundTaskDeliveryStrategy | None

    _DELIVERY_UNSET: Any = object()

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
        delivery_strategy: BackgroundTaskDeliveryStrategy | None = _DELIVERY_UNSET,
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
            delivery_strategy: Per-task override for proactive delivery of the
                result.  ``None`` disables proactive delivery for this task.
                When not provided (default), the framework-level strategy is
                used.

        Returns:
            A :class:`DelegatedTask` handle. When *wait* is ``True``,
            the result is already set.  When ``False``, call ``.wait()``
            to block for the result, or let it run fire-and-forget.

        Raises:
            RoomNotFoundError: If the parent room doesn't exist.
            ChannelNotRegisteredError: If the agent channel isn't registered.
        """
        from uuid import uuid4

        # Validate
        await self.get_room(room_id)  # type: ignore[attr-defined]
        if agent_id not in self._channels:
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")

        child_room_id = f"{room_id}::task-{uuid4().hex[:12]}"

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
            id=f"task-{uuid4().hex[:12]}",
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task=task,
        )

        # Fire ON_TASK_DELEGATED hook
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
            metadata={
                "task_id": handle.id,
                "child_room_id": child_room_id,
                "agent_id": agent_id,
            },
        )
        room_context = await self._build_context(room_id)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_TASK_DELEGATED, hook_event, room_context
        )

        if wait:
            return await self._run_inline(handle, context, notify, on_complete)

        return await self._run_background(handle, context, notify, on_complete, delivery_strategy)

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

        # Fire completion hooks + callbacks
        notify_channel = notify or handle.agent_id
        await self._on_delegation_complete(result, notify_channel)
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
        delivery_strategy: BackgroundTaskDeliveryStrategy | None,
    ) -> DelegatedTask:
        """Submit the task to the background task runner."""
        notify_channel = notify or handle.agent_id
        # Resolve delivery strategy: per-task override > framework-level
        effective_strategy: BackgroundTaskDeliveryStrategy | None
        if delivery_strategy is not self._DELIVERY_UNSET:
            effective_strategy = delivery_strategy
        else:
            effective_strategy = self._delivery_strategy

        async def _on_bg_complete(result: DelegatedTaskResult) -> None:
            await self._on_delegation_complete(result, notify_channel, strategy=effective_strategy)
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
        strategy: BackgroundTaskDeliveryStrategy | None = None,
    ) -> None:
        """Handle delegation completion: update notified agent + fire hook."""
        # Inject result into the notified agent's system prompt.
        # Cap total prompt size to prevent unbounded growth from many delegations.
        max_delegation_prompt = 4000
        from roomkit.tasks.delivery import ContextOnlyDelivery

        binding = await self._store.get_binding(result.parent_room_id, notify_channel_id)
        if binding:
            current_prompt = binding.metadata.get("system_prompt", "")
            # When a proactive delivery strategy will trigger, use a passive
            # instruction to avoid the AI volunteering the result twice (once
            # from the system prompt, once from the delivery-triggered turn).
            proactive = strategy is not None and not isinstance(strategy, ContextOnlyDelivery)
            instruction = (
                "This result will be delivered in a follow-up message. "
                "Do not proactively share it until then."
                if proactive
                else "Inform the user naturally about this result."
            )
            appendix = (
                "\n\n--- BACKGROUND TASK COMPLETED ---\n"
                + f"Task ID: {result.task_id}\n"
                + f"Agent: {result.agent_id}\n"
                + f"Status: {result.status}\n"
                + f"Result:\n{result.output or result.error or 'No output'}\n"
                + "--- END ---\n"
                + instruction
            )
            new_prompt = current_prompt + appendix
            # Sliding window: keep only the tail when prompt exceeds cap
            if len(new_prompt) > max_delegation_prompt:
                new_prompt = "...\n" + new_prompt[-max_delegation_prompt:]
            updated = binding.model_copy(
                update={
                    "metadata": {
                        **binding.metadata,
                        "system_prompt": new_prompt,
                    }
                }
            )
            await self._store.update_binding(updated)

        # Fire ON_TASK_COMPLETED hook
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
            metadata={
                "task_id": result.task_id,
                "child_room_id": result.child_room_id,
                "agent_id": result.agent_id,
                "task_status": result.status,
                "duration_ms": result.duration_ms,
            },
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

        # Proactive delivery via strategy (if configured)
        if strategy is not None:
            try:
                ctx = TaskDeliveryContext(
                    kit=self,  # type: ignore[arg-type]
                    result=result,
                    notify_channel_id=notify_channel_id,
                )
                await strategy.deliver(ctx)
            except Exception:
                _tasks_logger.exception("Delivery strategy failed for task %s", result.task_id)
