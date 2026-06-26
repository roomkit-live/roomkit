"""Framework-driven auto-delegate wiring for the supervisor.

Mixin for :class:`Supervisor`: wraps the supervisor's ``on_event`` (sync) or
injects a background ``delegate_workers`` tool on a ``RealtimeVoiceChannel``
(async). Host attributes are declared as annotations; they are set in
``Supervisor.__init__``.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.supervisor._common import (
    WorkerStrategy,
    _fallthrough,
    _is_subtask_room,
    logger,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _async_run_and_deliver,
    _one_pass_delegate,
    _two_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.results import _worker_roles_csv

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class _AutoDelegateInstallMixin:
    """Install framework-driven delegation (sync wrap or async voice tool)."""

    _supervisor: Agent
    _workers: list[Agent]
    _strategy: WorkerStrategy | None
    _async_delivery: bool
    _refine_task: bool
    _refine_instruction: str | None
    _share_channels: list[str]
    _max_revisions: int
    _task_timeout: float

    def _install_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Install framework-driven delegation (sync or async)."""
        if self._async_delivery:
            self._install_async_auto_delegate(kit, room_id)
        else:
            self._install_sync_auto_delegate(kit, room_id)

    def _install_sync_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Wrap supervisor's on_event — blocks until workers complete."""
        supervisor = self._supervisor
        strategy = self._strategy
        workers = self._workers
        refine = self._refine_task
        refine_instruction = self._refine_instruction
        share_channels = self._share_channels
        max_revisions = self._max_revisions
        task_timeout = self._task_timeout
        original_on_event = supervisor.on_event

        async def auto_delegate_on_event(
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            if event.source.channel_id == supervisor.channel_id:
                return ChannelOutput.empty()
            if event.source.channel_type == _ChannelType.AI:
                return ChannelOutput.empty()

            rid = context.room.id if context.room else room_id
            # Only the parent room drives delegation. Inside a child task room
            # (e.g. a supervisor review room created by the supervised loop, or
            # any delegated worker room), the supervisor must run NORMALLY —
            # otherwise the review prompt would be treated as a fresh user task
            # and re-trigger delegation, recursing without bound.
            if _is_subtask_room(rid):
                return await original_on_event(event, binding, context)

            if refine:
                return await _two_pass_delegate(
                    kit,
                    rid,
                    supervisor,
                    original_on_event,
                    event,
                    binding,
                    context,
                    strategy,
                    workers,
                    instruction=refine_instruction,
                    share_channels=share_channels,
                    max_revisions=max_revisions,
                    task_timeout=task_timeout,
                )
            return await _one_pass_delegate(
                kit,
                rid,
                supervisor,
                original_on_event,
                event,
                binding,
                context,
                strategy,
                workers,
                share_channels=share_channels,
                max_revisions=max_revisions,
                task_timeout=task_timeout,
            )

        supervisor.on_event = auto_delegate_on_event  # ty: ignore[invalid-assignment]

    def _install_async_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Inject delegate_workers tool into RealtimeVoiceChannel.

        The tool handler runs workers in the background and returns
        immediately. Results are delivered via kit.deliver().
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        strategy = self._strategy
        workers = self._workers
        share_channels = self._share_channels

        # Find the RealtimeVoiceChannel in registered channels
        voice_channel: RealtimeVoiceChannel | None = None
        for ch in kit.channels.values():
            if isinstance(ch, RealtimeVoiceChannel):
                voice_channel = ch
                break

        if voice_channel is None:
            logger.warning("async_delivery=True but no RealtimeVoiceChannel found")
            return

        # Build tool definition
        worker_roles = _worker_roles_csv(workers)
        tool_def = {
            "name": "delegate_workers",
            "description": (
                f"Delegate analysis to specialist workers ({worker_roles}). "
                f"Call when the user requests analysis, research, or investigation. "
                f"Workers run in {strategy} mode. Pass the topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The topic to analyze",
                    },
                },
                "required": ["task"],
            },
        }

        # Add tool to voice channel
        if voice_channel._tools is None:
            voice_channel._tools = []
        voice_channel._tools.append(tool_def)

        # Wrap tool handler for async delegation
        original_handler = voice_channel.tool_handler
        _running = False
        _lock = asyncio.Lock()

        async def async_tool_handler(name: str, arguments: dict[str, Any]) -> str:
            nonlocal _running

            if name != "delegate_workers":
                return await _fallthrough(original_handler, name, arguments)

            async with _lock:
                if _running:
                    return json.dumps(
                        {
                            "status": "already_running",
                            "message": "Workers are already running.",
                        }
                    )

                task_desc = arguments.get("task", "")
                _running = True

                # Launch workers inside the lock so _running and the
                # task creation are atomic. If create_task raises
                # (shutdown race), release _running so the voice
                # channel isn't permanently stuck in already_running.
                try:
                    task = asyncio.create_task(
                        _async_run_and_deliver(
                            kit=kit,
                            room_id=room_id,
                            strategy=strategy,
                            workers=workers,
                            task_desc=task_desc,
                            share_channels=share_channels,
                            on_done=_clear,
                        )
                    )
                    task.add_done_callback(log_task_exception)
                except BaseException:
                    _running = False
                    raise

            return json.dumps(
                {
                    "status": "dispatched",
                    "workers": worker_roles,
                    "message": "Workers are running. Results will be delivered when ready.",
                }
            )

        def _clear(**_: Any) -> None:
            """Accept but ignore ``success`` from _async_run_and_deliver.

            The voice path has no dedup cache to evict on failure, so
            success/failure doesn't change behaviour here.
            """
            nonlocal _running
            _running = False

        voice_channel.tool_handler = async_tool_handler
