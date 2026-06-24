"""The ``Supervisor`` orchestration strategy class.

Wires supervisor routing and one of three delegation modes. The install
helpers live in the ``_install_auto`` / ``_inject_strategy`` / ``_inject_per_worker``
mixins; the execution helpers in the sibling modules of this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_MAX_REVISIONS,
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    WorkerStrategy,
)
from roomkit.orchestration.strategies.supervisor._inject_per_worker import (
    _PerWorkerToolMixin,
)
from roomkit.orchestration.strategies.supervisor._inject_strategy import (
    _StrategyToolMixin,
)
from roomkit.orchestration.strategies.supervisor._install_auto import (
    _AutoDelegateInstallMixin,
)

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class Supervisor(
    _AutoDelegateInstallMixin, _StrategyToolMixin, _PerWorkerToolMixin, Orchestration
):
    """Supervisor orchestration strategy.

    The supervisor handles all user interaction. Workers are registered
    on the kit (so ``delegate()`` can find them) but are NOT attached
    to the parent room — they run in child rooms.

    Examples::

        # Framework-driven: auto-delegate with task refinement
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
            auto_delegate=True,
        )

        # Framework-driven: workers get raw user message
        Supervisor(
            supervisor=coordinator,
            workers=[technical, business],
            strategy="parallel",
            auto_delegate=True,
            refine_task=False,
        )

        # Tool-based: AI decides when to delegate
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
        )

        # Manual: per-worker tools, AI decides everything
        Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
        )
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: list[Agent],
        *,
        strategy: WorkerStrategy | str | None = None,
        auto_delegate: bool = False,
        async_delivery: bool = False,
        refine_task: bool = True,
        refine_instruction: str | None = None,
        delegation_message: str | None = "I'm dispatching my team to work on this.",
        wait_for_result: bool = True,
        share_channels: list[str] | None = None,
        task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
    ) -> None:
        """Initialise the supervisor strategy.

        Args:
            supervisor: The agent that handles user interaction and
                delegates tasks.
            workers: Agents that run delegated tasks in child rooms.
            strategy: Deterministic execution pattern for workers.

                - ``"sequential"``: workers run in order, each receiving
                  the previous worker's output.
                - ``"parallel"``: all workers run concurrently on the
                  same task.
                - ``None`` (default): per-worker ``delegate_to_<id>``
                  tools are injected and the AI decides when to call
                  them.

            auto_delegate: If ``True``, the framework triggers workers
                automatically — no tool needed. For sync channels (CLI),
                blocks until results are ready. For async channels
                (voice), runs in background. Requires *strategy*.

            async_delivery: If ``True``, worker delegation returns
                immediately and results are delivered back to the
                room via ``kit.deliver()`` when they complete. This
                keeps the supervisor's tool-loop clock bounded by
                its own reasoning time rather than aggregated worker
                wall-clock time. Applies to:

                - strategy-tool mode (``strategy`` set): the
                  ``delegate_workers`` tool fires background workers
                  and returns ``{"status": "dispatched", ...}``
                  immediately. Use ``check_status_bus`` to follow
                  progress.
                - voice ``auto_delegate`` mode: injects a
                  ``delegate_workers`` tool on the voice channel;
                  supervisor is not attached (voice handles UI).

                Lifecycle events (``pending`` / ``completed`` /
                ``failed``) are posted to ``kit.status_bus`` for
                every worker regardless of mode.
                If ``False`` (default), delegation blocks until
                results are ready (sync mode).

            refine_task: Controls whether the framework extracts a clean
                topic from the user's message before sending to workers.

            refine_instruction: Custom instruction for topic extraction.
                Overrides the default.

            delegation_message: Message injected into the conversation
                when workers are dispatched (async mode only). Set to
                ``None`` to disable. Default: "I'm dispatching my team
                to work on this."

            wait_for_result: When *strategy* is ``None``, controls
                whether delegation runs inline (``True``, default) or
                in the background (``False``).  Ignored when *strategy*
                is set or *auto_delegate* is ``True``.

            share_channels: Channel IDs from the parent room to share
                with every child room created during delegation.  For
                example, passing ``["system", "ws-status"]`` attaches
                those channels to each worker's child room so the
                worker can emit events visible on those channels.

            task_timeout: Per-worker delegation budget in seconds
                (default 120). Each delegated task is bounded
                individually so the chain is governed per-task rather
                than by a single global timeout that can never be
                coherent across team sizes. A worker that exceeds it is
                recorded as failed and the run continues.

            max_revisions: In supervised sequential mode, the maximum
                number of supervisor→worker rework round-trips per step
                (default 3). The supervisor validates each worker's
                output; if unacceptable it sends feedback for a rework,
                up to this many times. On exhaustion the step is
                delivered flagged as unvalidated rather than looping.
        """
        self._supervisor = supervisor
        self._workers = list(workers)
        self._strategy = WorkerStrategy(strategy) if strategy else None
        self._auto_delegate = auto_delegate
        self._async_delivery = async_delivery
        self._refine_task = refine_task
        self._refine_instruction = refine_instruction
        self._delegation_message = delegation_message
        self._wait_for_result = wait_for_result
        self._share_channels = list(share_channels) if share_channels else []
        self._task_timeout = task_timeout
        self._max_revisions = max_revisions

        if auto_delegate and self._strategy is None:
            msg = "auto_delegate=True requires strategy to be set"
            raise ValueError(msg)

    def agents(self) -> list[Agent]:
        """Return agents to attach to the room.

        ``async_delivery`` removes the supervisor from the room ONLY
        in voice ``auto_delegate`` mode, where a
        ``RealtimeVoiceChannel`` handles user interaction directly.
        In strategy-tool or per-worker-tool modes the supervisor
        still drives the conversation via its tool loop; background
        dispatch only affects worker delegation, not user routing.
        """
        if self._async_delivery and self._auto_delegate:
            return []
        return [self._supervisor]

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire supervisor routing and delegation tools."""
        # Router only needed when the supervisor agent is in the room.
        # In voice async_delivery mode (auto_delegate=True), the voice
        # channel owns routing and the supervisor is not attached.
        supervisor_in_room = not (self._async_delivery and self._auto_delegate)
        if supervisor_in_room:
            router = ConversationRouter(
                default_agent_id=self._supervisor.channel_id,
            )
            kit.hook_engine.add_room_hook(
                room_id,
                HookRegistration(
                    trigger=HookTrigger.BEFORE_BROADCAST,
                    execution=HookExecution.SYNC,
                    fn=router.as_hook(),
                    priority=-100,
                    name=f"supervisor_router_{room_id}",
                ),
            )

        # Register workers on the kit (not attached to room)
        for worker in self._workers:
            if worker.channel_id not in kit.channels:
                kit.register_channel(worker)

        # Wire delegation based on mode
        if self._auto_delegate:
            self._install_auto_delegate(kit, room_id)
        elif self._strategy is not None:
            self._inject_strategy_tool(kit, room_id)
        else:
            self._inject_per_worker_tools(kit, room_id)

        # Set initial conversation state
        room = await kit.get_room(room_id)
        initial_state = ConversationState(
            phase="supervisor",
            active_agent_id=self._supervisor.channel_id,
        )
        room = set_conversation_state(room, initial_state)
        await kit.store.update_room(room)
