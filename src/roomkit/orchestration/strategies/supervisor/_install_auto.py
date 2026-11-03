"""Framework-driven auto-delegate wiring for the supervisor.

Mixin for :class:`Supervisor`: takes the supervisor's turns in the room on
the room's owner (sync, RFC §10.6.1) or injects a background
``delegate_workers`` tool on a ``RealtimeVoiceChannel`` (async). Host
attributes are declared as annotations; they are set in
``Supervisor.__init__``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool, schema_tool
from roomkit.core._changes import opens_no_turn
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType as _ChannelType
from roomkit.models.event import RoomEvent
from roomkit.orchestration._background import calling_channel_id, start_background_run
from roomkit.orchestration._call_room import in_call_room
from roomkit.orchestration._installs import set_up_for_voice_room
from roomkit.orchestration.strategies.supervisor._common import (
    WorkerStrategy,
)
from roomkit.orchestration.strategies.supervisor.delegate import (
    _async_run_and_deliver,
    _one_pass_delegate,
    _two_pass_delegate,
)
from roomkit.orchestration.strategies.supervisor.results import _worker_roles_csv

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.orchestration._agents import HeldAgent


class _AutoDelegateInstallMixin:
    """Install framework-driven delegation (sync wrap or async voice tool)."""

    _supervisor: HeldAgent
    _workers: list[HeldAgent]
    _strategy: WorkerStrategy | None
    _async_delivery: bool
    _refine_task: bool
    _refine_instruction: str | None
    _share_channels: list[str]
    _max_revisions: int
    _task_timeout: float

    def _install_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Install framework-driven delegation (sync or async), once.

        The supervisor and the voice channel serve every room the strategy is
        installed in (RFC §19.7): a second room's install finds them wired,
        and each delegation runs for the room it came from.
        """
        if self._async_delivery:
            self._install_async_auto_delegate(kit, room_id)
        else:
            self._install_sync_auto_delegate(kit, room_id)

    def _install_sync_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Take the supervisor's turns in *room_id* with the delegation passes.

        Blocks until workers complete. The supervisor serves every room it is
        attached to (RFC §19.7): the passes run in the room they were installed
        in, with this install's team, and the supervisor answers as itself
        everywhere else, a room with no install included.
        """
        turns = _DelegatingTurns(
            kit,
            self._supervisor,
            self._workers,
            self._strategy,
            refine=self._refine_task,
            refine_instruction=self._refine_instruction,
            share_channels=self._share_channels,
            max_revisions=self._max_revisions,
            task_timeout=self._task_timeout,
        )
        # Kept by the room's owner, never on the shared supervisor (RFC §10.6.1).
        kit._turn_runners.take(room_id, self._supervisor.channel_id, turns.run, owner=self)

    def _install_async_auto_delegate(self, kit: RoomKit, room_id: str) -> None:
        """Serve ``delegate_workers`` in *room_id*'s realtime sessions.

        The tool runs this install's workers in the background and returns
        immediately; results are handed back to the session that made the
        call. It is set up for the room (RFC §19.7): another room's sessions
        do not declare it, and a second room's install serves its own team.
        """
        tool = schema_tool(_voice_delegate_tool(self._workers, self._strategy))
        # One server for every voice channel: one run per room, whichever
        # channel's session asked for it (RFC §19.7.3).
        server = _VoiceDelegateServer(
            kit,
            self._supervisor,
            self._workers,
            self._strategy,
            share_channels=self._share_channels,
            max_revisions=self._max_revisions,
            task_timeout=self._task_timeout,
        )
        # A strategy's tool: the channel's default call bound does not apply
        # to it (RFC §21.6).
        entry = orchestration_tool(tool, in_call_room(tool.name, server.serve), waits=True)
        set_up_for_voice_room(kit, room_id, self, lambda _channel: entry)


class _DelegatingTurns:
    """One install's framework-driven delegation: the supervisor's turns run
    the workers, then the supervisor answers with what they found."""

    def __init__(
        self,
        kit: RoomKit,
        supervisor: HeldAgent,
        workers: list[HeldAgent],
        strategy: WorkerStrategy | None,
        *,
        refine: bool,
        refine_instruction: str | None,
        share_channels: list[str],
        max_revisions: int,
        task_timeout: float,
    ) -> None:
        self._kit = kit
        self._supervisor = supervisor
        self._workers = workers
        self._strategy = strategy
        self._refine = refine
        self._refine_instruction = refine_instruction
        self._share_channels = share_channels
        self._max_revisions = max_revisions
        self._task_timeout = task_timeout

    async def run(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Take one of the supervisor's turns in the installed room."""
        supervisor = self._supervisor
        if event.source.channel_id == supervisor.channel_id:
            return ChannelOutput.empty()
        if event.source.channel_type == _ChannelType.AI:
            return ChannelOutput.empty()
        if opens_no_turn(event):
            # A notice, an edit or a delete is no request to delegate (RFC
            # §6.4, §10.3): the supervisor's own channel reads it, and takes
            # no turn.
            return await self._kit._strategy_turn(event, binding, context)
        rid = context.room.id if context.room else event.room_id
        if self._refine:
            return await _two_pass_delegate(
                self._kit,
                rid,
                supervisor,
                self._kit._strategy_turn,
                event,
                binding,
                context,
                self._strategy,
                self._workers,
                instruction=self._refine_instruction,
                share_channels=self._share_channels,
                max_revisions=self._max_revisions,
                task_timeout=self._task_timeout,
            )
        return await _one_pass_delegate(
            self._kit,
            rid,
            supervisor,
            self._kit._strategy_turn,
            event,
            binding,
            context,
            self._strategy,
            self._workers,
            share_channels=self._share_channels,
            max_revisions=self._max_revisions,
            task_timeout=self._task_timeout,
        )


def _voice_delegate_tool(
    workers: list[HeldAgent], strategy: WorkerStrategy | None
) -> dict[str, Any]:
    """The ``delegate_workers`` declaration a voice channel carries."""
    worker_roles = _worker_roles_csv(workers)
    return {
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


class _VoiceDelegateServer:
    """Serves a voice channel's ``delegate_workers``: runs the workers in the
    background for the room of the call, and answers at once."""

    def __init__(
        self,
        kit: RoomKit,
        supervisor: HeldAgent,
        workers: list[HeldAgent],
        strategy: WorkerStrategy | None,
        *,
        share_channels: list[str],
        max_revisions: int,
        task_timeout: float,
    ) -> None:
        self._kit = kit
        self._supervisor = supervisor
        self._workers = workers
        self._strategy = strategy
        self._share_channels = share_channels
        self._max_revisions = max_revisions
        self._task_timeout = task_timeout
        self._running: set[str] = set()  # rooms whose workers are running

    async def serve(self, rid: str, name: str, arguments: dict[str, Any]) -> str:
        """Answer one ``delegate_workers`` call made in room *rid*."""
        running = self._running
        if rid in running:
            return json.dumps(
                {"status": "already_running", "message": "Workers are already running."}
            )
        running.add(rid)
        # Launch in the same step as the flag, so a second call of this
        # room's cannot slip between them. If the start raises (shutdown
        # race), release the room so it isn't stuck in already_running.
        try:
            start_background_run(
                self._kit,
                _async_run_and_deliver(
                    kit=self._kit,
                    room_id=rid,
                    # The voice channel whose session made the call is told.
                    supervisor_id=calling_channel_id(),
                    supervisor=self._supervisor,
                    strategy=self._strategy,
                    workers=self._workers,
                    task_desc=arguments.get("task", ""),
                    share_channels=self._share_channels,
                    max_revisions=self._max_revisions,
                    task_timeout=self._task_timeout,
                    on_done=lambda **_: running.discard(rid),
                ),
            )
        except BaseException:
            running.discard(rid)
            raise
        return json.dumps(
            {
                "status": "dispatched",
                "workers": _worker_roles_csv(self._workers),
                "message": "Workers are running. Results will be delivered when ready.",
            }
        )
