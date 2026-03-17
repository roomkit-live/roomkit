"""DelegationMixin — background task delegation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.tasks.delivery import BackgroundTaskDeliveryStrategy, TaskDeliveryContext
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.store.base import ConversationStore
    from roomkit.tasks.base import TaskRunner


_tasks_logger = logging.getLogger("roomkit.tasks")


class DelegationMixin(HelpersMixin):
    """Background task delegation to child rooms."""

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
        context: dict[str, Any] | None = None,
        share_channels: list[str] | None = None,
        notify: str | None = None,
        on_complete: Any | None = None,
        delivery_strategy: BackgroundTaskDeliveryStrategy | None = _DELIVERY_UNSET,
    ) -> DelegatedTask:
        """Delegate a task to a background agent in a child room.

        Creates a child room linked to *room_id*, attaches the agent and
        any shared channels, then submits the task for background execution.

        Args:
            room_id: Parent room ID.
            agent_id: Channel ID of the agent to run the task.
            task: Description of what the agent should do.
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
            A :class:`DelegatedTask` handle. Call ``.wait()`` to block
            for the result, or let it run fire-and-forget.

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

        # Create child room
        await self.create_room(  # type: ignore[attr-defined]
            room_id=child_room_id,
            metadata={
                "parent_room_id": room_id,
                "task_agent_id": agent_id,
                "task_input": task,
                "task_context": context or {},
                "task_status": "pending",
            },
        )

        # Attach agent as intelligence
        from roomkit.models.enums import ChannelCategory

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

        # Build completion callback
        notify_channel = notify or agent_id
        # Resolve delivery strategy: per-task override > framework-level
        effective_strategy: BackgroundTaskDeliveryStrategy | None
        if delivery_strategy is not self._DELIVERY_UNSET:
            effective_strategy = delivery_strategy
        else:
            effective_strategy = self._delivery_strategy

        async def _on_complete(result: DelegatedTaskResult) -> None:
            await self._on_delegation_complete(result, notify_channel, strategy=effective_strategy)
            if on_complete:
                await on_complete(result)

        # Submit to task runner
        await self._task_runner.submit(self, handle, context=context, on_complete=_on_complete)  # type: ignore[arg-type]

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
