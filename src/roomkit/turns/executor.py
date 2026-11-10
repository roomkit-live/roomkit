"""The turn executor: builds a channel and runs its turns (RFC §10.6).

An executor never commits: what a turn produces reaches the room through its
owner. The kit owns one in memory, which reads the channels the host
registered; a host runs others with a factory of its own.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable, Collection, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.core._turn_scope import turns_run_under
from roomkit.core.exceptions import ToolNameCollisionError, TurnLinkError
from roomkit.core.hooks import HookEngine, HookRegistration, hook_decorator
from roomkit.core.task_utils import acting_tasks, cancel_and_wait
from roomkit.models.enums import HookExecution
from roomkit.turns._error_codec import ErrorClasses, error_record
from roomkit.turns._executor_channels import ChannelFactory, ExecutorChannels
from roomkit.turns._executor_retiring import Retiring
from roomkit.turns._executor_rooms import ExecutorRooms
from roomkit.turns._executor_turn import ExecutorTurn
from roomkit.turns._hosting import Hosted
from roomkit.turns._wiring import TurnWiring
from roomkit.turns._wiring_env import TurnWiringEnv
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.wire import (
    ChannelCall,
    ChannelReleased,
    ChannelResult,
    CommittedEvent,
    RoomAttached,
    RoomDetached,
)

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import AsyncHookFn, SyncHookFn
    from roomkit.models.enums import ChannelType, EventDirection, EventType, HookTrigger
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.turns._wiring_env import WiringEnv
    from roomkit.turns.base import ExecutorLink, RoomOwner
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")


class TurnExecutor:
    """Runs the turns of the channels it declares (RFC §10.6).

    Args:
        build_channel: The host's factory: the channel for a channel id, or
            ``None`` when it has none; sync or async. The executor builds
            each channel when a turn, a channel call or a committed event
            first needs it, wires it (RFC §10.6.2) and closes it when it
            closes.
        hosts: Which channel ids this executor declares: a collection of
            ids and of prefixes ending in ``*`` (``"org-*"``; ``"*"`` for
            every id), or a predicate; every id when omitted. Only a
            collection is told to the room owners of other processes: a
            predicate, or no *hosts*, takes the turns of its own process
            alone (RFC §10.6.7).
        executor_id: The id the executor answers with; a random one when
            omitted.
        timeouts: The heartbeat bounds it keeps.
        media_resolver: The host's media resolver, given to its AI channels
            as the kit gives the ones it registers (RFC §6.4).
        attached_media: The host's attached-media reader (RFC §6.4).
        telemetry: The telemetry its hooks and channels report to.
        owns_channels: ``False`` when *build_channel* reads a registry whose
            owner wires and closes its channels (the kit's own executor).
        max_channels: How many built channels it keeps at most; no
            bound when omitted. Past it, the one used least recently that
            nothing runs on (a turn, a channel call, a committed event, a
            room's attach or detach) is let go as
            ``kit.release_channel`` lets it go: what it held for its rooms
            and the room does not keep (an ACP session) goes, and its next
            turn builds it anew (RFC §10.6.2). Unused with
            ``owns_channels=False``.
        error_classes: The host's exception classes an error raised on the
            room owner is rebuilt as here, with their subclasses, beyond
            RoomKit's and Python's built-in ones (RFC §10.6.8). Any other comes back as
            a ``RemoteTurnError``.

    Raises:
        TypeError: An entry of *error_classes* is not an ``Exception`` class.
        ValueError: An entry of *hosts* holds a ``*`` elsewhere than at its
            end, or *max_channels* is below 1.
    """

    def __init__(
        self,
        build_channel: ChannelFactory,
        *,
        hosts: Collection[str] | Callable[[str], bool] | None = None,
        executor_id: str | None = None,
        timeouts: TurnTimeouts | None = None,
        media_resolver: MediaResolver | None = None,
        attached_media: AttachedMedia | None = None,
        telemetry: TelemetryProvider | None = None,
        owns_channels: bool = True,
        error_classes: Iterable[type[Exception]] = (),
        max_channels: int | None = None,
    ) -> None:
        self.executor_id = executor_id or f"executor-{uuid4().hex[:12]}"
        self._errors = ErrorClasses(error_classes)
        self._hosted, self._hosts = _declaration(hosts)
        if max_channels is not None and max_channels < 1:
            raise ValueError(f"max_channels is at least 1, not {max_channels}")
        self._timeouts = timeouts or TurnTimeouts()
        self._turns: dict[asyncio.Task[None], ExecutorTurn] = {}
        self._closed = False
        self._hooks = HookEngine(turns=True)
        self._telemetry = telemetry
        if telemetry is not None:
            self._hooks._telemetry = telemetry
        # A hook's timeout (RFC §8.2) is emitted where the turn's events go.
        self._hooks._framework_emitter = self._emit_framework_event
        # The owner in this process sets the rooms' hooks itself (its own
        # executor), so a request's copy never overwrites them.
        self._room_hooks_from_owner = False
        self._owner: RoomOwner | None = None
        env = TurnWiringEnv(media_resolver=media_resolver, attached_media=attached_media)
        self._wiring = TurnWiring(self._hooks, env)
        self._channels = ExecutorChannels(
            build_channel,
            owns=owns_channels,
            wire=self.wire,
            telemetry=telemetry,
            bound=max_channels,
            in_use=self._in_use,
            let_go=self._retire,
        )
        self._rooms = ExecutorRooms(self.executor_id, self._channels.built, self._channels.use)
        # Builds let go of (``channel_released``, the bound), closed once the
        # turns that run on them end.
        self._retiring = Retiring()
        # What the backends serving it run once it closed (it no longer
        # covers its channels, RFC §10.6.7).
        self._closed_listeners: list[Callable[[], Awaitable[None]]] = []

    @property
    def hooks(self) -> HookEngine:
        """The hooks of the triggers a turn fires (RFC §10.6.6)."""
        return self._hooks

    @property
    def wiring(self) -> TurnWiring:
        """What the executor gives the channels it runs (RFC §10.6.2)."""
        return self._wiring

    def use_wiring(self, env: WiringEnv) -> None:
        """Wire channels from *env* in place of what the executor was given:
        what the kit gives its own executor, before any channel is wired."""
        self._wiring = TurnWiring(self._hooks, env)

    async def _emit_framework_event(self, event_type: str, **fields: Any) -> None:
        await self._wiring.emit(event_type, **fields)

    def wire(self, channel: Channel) -> None:
        """Give *channel* what its turns fire this executor's hooks through:
        the kit's own executor wires each channel the kit registers, another
        each channel its factory builds."""
        self._wiring.wire(channel)

    def hook(
        self,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        name: str = "",
        timeout: float = 30.0,
        channel_types: set[ChannelType] | None = None,
        channel_ids: set[str] | None = None,
        directions: set[EventDirection] | None = None,
        event_types: set[EventType] | None = None,
        fail_closed: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to register a hook on a trigger a turn fires
        (``ON_TOOL_CALL``, ``BEFORE_TOOL_USE``, ``ON_AI_RESPONSE``...), with
        ``RoomKit.hook``'s options. An owner's trigger raises ``ValueError``.

        The kit's own executor (``kit.turn_executor``) fires these hooks for
        the turns it runs and for the tool calls the kit's process serves
        (realtime voice, conference). An executor a host builds runs its turns
        without them for now: the kit wires the channels it registers, and an
        executor that wires its own comes with the distributed backends.
        """
        return hook_decorator(
            self._hooks.register,
            trigger,
            execution=execution,
            priority=priority,
            name=name,
            timeout=timeout,
            channel_types=channel_types,
            channel_ids=channel_ids,
            directions=directions,
            event_types=event_types,
            fail_closed=fail_closed,
        )

    def room_hook(
        self,
        name: str,
        trigger: HookTrigger,
        execution: HookExecution = HookExecution.SYNC,
        priority: int = 0,
        timeout: float = 30.0,
        *,
        fail_closed: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to define a room hook named *name* on a trigger a turn
        fires (RFC §10.6.6): it runs for a room whose owner enabled the name
        (``kit.enable_room_hook(room_id, name)``), the turn's request naming
        it. Every executor that may run the room's turns defines it; one that
        does not is never given them. A name may carry a hook per trigger:
        defining it again on a trigger it carries replaces the first.

        ``executor.room_hook("audit", HookTrigger.ON_TOOL_CALL)(fn)``
        """

        def decorator(fn: SyncHookFn | AsyncHookFn) -> SyncHookFn | AsyncHookFn:
            self._hooks.define_room_hook(
                HookRegistration(
                    trigger=trigger,
                    execution=execution,
                    fn=fn,
                    priority=priority,
                    name=name,
                    timeout=timeout,
                    fail_closed=fail_closed,
                )
            )
            return fn

        return decorator

    def remove_room_hook(self, name: str) -> bool:
        """Forget the room hooks defined under *name*."""
        return self._hooks.undefine_room_hook(name)

    def declares(self, channel_id: str) -> bool:
        """Whether this executor takes the turns of *channel_id*."""
        return not self._closed and self.hosts_channel(channel_id)

    def hosts_channel(self, channel_id: str) -> bool:
        """Whether *channel_id* is among the channels this executor was given
        (``hosts=``), whether or not it is closed. A backend reads it of the
        owner's own executor, whose live channels' turns no other executor
        takes, even once it closed (RFC §10.6.7)."""
        return self._hosts is None or self._hosts(channel_id)

    @property
    def hosted(self) -> frozenset[str] | None:
        """The channel ids and prefixes it declares when they were given as
        a collection; ``None`` for a predicate or every id (RFC §10.6.7)."""
        return self._hosted

    @property
    def timeouts(self) -> TurnTimeouts:
        """The heartbeat bounds its turns keep (RFC §10.6.8)."""
        return self._timeouts

    @property
    def closed(self) -> bool:
        """Whether the executor closed: it takes no more turns."""
        return self._closed

    def reach_owner(self, owner: RoomOwner) -> None:
        """Reach the room owner outside a turn through *owner*, which a backend
        gives the executors it serves (RFC §10.6.3): work a turn began sends
        its host verbs and publications there once the turn ended."""
        self._owner = owner

    def takes(self, channel_id: str, room_hooks: Iterable[str] = ()) -> bool:
        """Whether this executor takes a turn of *channel_id* whose room has
        the hooks named *room_hooks* (RFC §10.6.7)."""
        return self.declares(channel_id) and self._hooks.defines_room_hooks(room_hooks)

    def take(self, link: ExecutorLink) -> None:
        """Start the turn *link* brought; it runs on its own task."""
        request = link.request
        stale = self._rooms.enter(
            request.channel_id, request.room_id, request.turn_id, request.previous_turn
        )
        self._wiring.env.turn_started(request)
        turn = ExecutorTurn(
            link,
            self.executor_id,
            partial(self._turn_channel, link.request, stale),
            self._timeouts,
            self._errors,
            owner=self._owner,
            telemetry=self._telemetry,
        )
        task = asyncio.get_running_loop().create_task(
            turn.run(),
            context=contextvars.Context(),
            name=f"roomkit-turn-{link.request.turn_id}",
        )
        self._turns[task] = turn
        task.add_done_callback(self._forget)
        task.add_done_callback(partial(_let_go, link))

    def _spare_turns_run_under(self) -> None:
        """Let the turns the current close runs under run on whatever cuts
        them: a kit's close cuts every model turn at once, the one its call
        runs in included, but not the call that runs it (RFC §12.10.4)."""
        spared = set(turns_run_under())
        for turn in self._turns.values():
            if turn.turn_id in spared:
                turn.spare()

    async def close(self) -> None:
        """Take no more turns, and cut the ones running, but the turns this
        close runs under: waiting for them would wait for itself."""
        await self._close(None)

    async def _close(self, until: float | None) -> set[asyncio.Task[Any]]:
        """:meth:`close`, waiting for the turns it cut until *until* at most
        (the kit's close, RFC §12.10.4): the ones still running then,
        ignoring their cut, which it leaves."""
        self._closed = True
        self._spare_turns_run_under()
        spared = set(turns_run_under())
        acting = acting_tasks()
        running = [
            task
            for task, turn in self._turns.items()
            if turn.turn_id not in spared and task not in acting
        ]
        left: set[asyncio.Task[Any]] = set()
        if until is None:
            if running:
                await cancel_and_wait(*running, log_errors_to=logger)
        elif running:
            for task in running:
                if not task.done() and not task.cancelling():
                    task.cancel()
            remaining = max(0.0, until - asyncio.get_running_loop().time())
            done, left = await asyncio.wait(running, timeout=remaining)
            for task in done:
                if not task.cancelled() and (error := task.exception()) is not None:
                    logger.debug("Turn %s ended with %r as it closed", task.get_name(), error)
            for task in left:
                # Left running past its cut: nothing of it is heard again.
                self._turns[task].stop_watching()
        await self._channels.close()
        await self._retiring.close()
        await self._tell_closed()
        return left

    def _listen_closed(self, listener: Callable[[], Awaitable[None]]) -> None:
        """Have *listener* run once this executor closed: a backend serving
        it learns it no longer covers its channels."""
        if listener not in self._closed_listeners:
            self._closed_listeners.append(listener)

    async def _tell_closed(self) -> None:
        for listener in self._closed_listeners:
            try:
                await listener()
            except Exception:
                logger.exception("Executor %s: a listener of its close failed", self.executor_id)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Apply a channel control message of the owner's (RFC §10.6.3): a
        ``channel_call``'s result; nothing for ``room_attached``,
        ``room_detached``, ``committed_event`` and ``channel_released``.
        A room's attach or detach builds nothing: an executor holding no
        build of the channel has nothing to apply."""
        if isinstance(message, ChannelReleased):
            await self._release(message.channel_id)
            return None
        if isinstance(message, RoomDetached):
            self._wiring.env.room_left(message.room_id, message.channel_id)
        with self._channels.use(message.channel_id):
            if isinstance(message, RoomAttached | RoomDetached):
                channel = await self._channels.built(message.channel_id)
                return await self._rooms.apply(channel, message)
            return await self._apply_on_build(message)

    async def _apply_on_build(self, message: ChannelCall | CommittedEvent) -> ChannelResult | None:
        """Apply a channel call or a committed event on the channel, built
        for it when it is not: a committed event feeds the channel's memory
        (RFC §19.7.5 rule 3), which a skipped event would miss. A build made
        for a committed event alone is the first the bound lets go."""
        try:
            event = isinstance(message, CommittedEvent)
            channel = await self._channels.get(message.channel_id, recent=not event)
        except Exception as exc:
            if isinstance(message, ChannelCall):
                return ChannelResult(error=error_record(exc), executor_id=self.executor_id)
            logger.exception("The factory failed to build channel %s", message.channel_id)
            return None
        if isinstance(message, CommittedEvent):
            await _hand_event(channel, message)
            return None
        return await self._rooms.apply(channel, message)

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._turns.pop(task, None)
        self._retiring.turn_ended(task)
        # A channel the bound kept while this turn ran on it may go now.
        self._channels.shrink()

    def _in_use(self, channel: Channel) -> bool:
        """Whether a turn runs on *channel*, or still may: the bound never
        lets it go then."""
        return any(turn.may_run_on(channel) for turn in self._turns.values())

    async def _release(self, channel_id: str) -> None:
        """Let go of the channel built for *channel_id* (``channel_released``):
        the next turn builds it anew."""
        channel = await self._channels.release(channel_id)
        if channel is not None:
            self._retire(channel)

    def _retire(self, channel: Channel) -> None:
        """Close *channel*, which the executor let go of (a release, the
        bound), once the turns that run on it, or still may, end."""
        holders = {task for task, turn in self._turns.items() if turn.may_run_on(channel)}
        self._retiring.retire(channel, holders)

    async def _turn_channel(self, request: TurnRequest, stale: bool) -> Channel | None:
        """The channel of *request*'s turn, readied for its room (RFC §10.6.2)
        and its hooks: a turn whose room hooks this executor does not define
        fails rather than run without them (RFC §10.6.6)."""
        if not self._hooks.defines_room_hooks(request.room_hooks):
            raise TurnLinkError(
                "no_executor",
                f"Executor {self.executor_id} defines no room hook among {request.room_hooks}",
            )
        if not self._room_hooks_from_owner:
            # Between the room's turns, its hooks are those its last turn named.
            self._hooks.use_room_hooks(request.room_id, request.room_hooks)
        channel = await self._channels.get(request.channel_id)
        if channel is not None:
            self._refuse_clashing_owner_tools(channel, request)
            await self._rooms.prepare(channel, request, stale=stale)
        return channel

    def _refuse_clashing_owner_tools(self, channel: Channel, request: TurnRequest) -> None:
        """Refuse a turn whose owner serves a tool the channel declares too
        (RFC §10.6.5): which handler a call reaches would depend on the
        executor. The owner refuses it at setup only for a channel it holds
        live."""
        registry = getattr(channel, "_registry", None)
        if not isinstance(registry, ChannelRegistry) or not request.owner_tools:
            return
        clash = sorted({owned.tool.name for owned in request.owner_tools} & registry.own_names())
        if clash:
            raise ToolNameCollisionError(
                f"Tool {clash[0]!r} is served both by the room owner and by channel "
                f"{request.channel_id!r} on executor {self.executor_id}: rename one of them "
                "(RFC §10.6.5)"
            )


def _declaration(
    hosts: Collection[str] | Callable[[str], bool] | None,
) -> tuple[frozenset[str] | None, Callable[[str], bool] | None]:
    """What *hosts* declares (RFC §10.6.7): the ids and prefixes told to
    other processes when it is a collection, and the test of a channel id;
    ``(None, None)`` for every id."""
    if isinstance(hosts, str):
        raise TypeError("hosts is a collection of channel ids or a predicate, not one id")
    if isinstance(hosts, Collection):
        declared = Hosted(frozenset(hosts))
        return declared.entries, declared.covers
    return None, hosts


def _let_go(link: ExecutorLink, task: asyncio.Task[None]) -> None:
    """The turn's task ended: its link lets go of it, whatever its end."""
    try:
        link.turn_ended()
    except Exception:
        logger.exception("Turn %s: its link failed to let go", link.request.turn_id)


async def _hand_event(channel: Channel | None, message: CommittedEvent) -> None:
    """Hand a committed event to *channel* for its room, logging what it
    raises: the room's state is left as it is (RFC §10.6.3)."""
    if channel is None:
        return
    try:
        await channel.on_room_event(message.room_id, message.event)
    except Exception:
        logger.exception(
            "Channel %s failed on an event committed in room %s",
            message.channel_id,
            message.room_id,
        )
