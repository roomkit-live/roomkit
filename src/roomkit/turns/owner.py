"""The room owner's side of the turn backend (RFC §10.6).

:class:`TurnClient` is how the pipeline runs a turn: it builds the request,
submits it, and reads the turn back as the channel's ``ChannelOutput``. The
kit's own executor is served on first use.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.channels._ai_context import history_start
from roomkit.channels._discussion_turn import DISCUSSION_TURN, discussion_notes, discussion_turn
from roomkit.channels.declared import DeclaredChannel
from roomkit.core.task_utils import acting_tasks, close_callers
from roomkit.core.visibility import answers_reach
from roomkit.models.enums import ChannelCategory, Visibility
from roomkit.models.pending_input import PendingInputEvent
from roomkit.models.steering import Cancel as SteeringCancel
from roomkit.telemetry.context import get_current_span
from roomkit.turns._error_codec import ErrorClasses, error_record
from roomkit.turns._owner_between import answer_room_call, publish_between_turns
from roomkit.turns._owner_inputs import OwnerInputs
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns._registry_tools import OwnerToolStore, RegistryOwnerTools
from roomkit.turns._rooms import RoomRecords
from roomkit.turns.base import RoomOwner, TurnTimeouts
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.wire import (
    ChannelCall,
    CommittedEvent,
    HostResult,
    PreviousTurn,
    RoomDetached,
    TurnRequest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.steering import SteeringDirective
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.turns.base import OwnerEvents, OwnerHost, OwnerTools, TurnBackend
    from roomkit.turns.wire import (
        ChannelControl,
        ChannelResult,
        OwnerTool,
        RoomCall,
        RoomPublication,
    )


logger = logging.getLogger("roomkit.turns")


def turn_request(
    event: RoomEvent,
    binding: ChannelBinding,
    context: RoomContext,
    *,
    strategy_turn: bool = False,
    owner_tools: list[OwnerTool] | None = None,
    telemetry: TelemetryProvider | None = None,
) -> TurnRequest:
    """The request for one turn of *binding*'s channel on *event*, with the
    tools the owner serves it (*owner_tools*) and the broadcast span it runs
    under, named by *telemetry* for an executor elsewhere (RFC §10.6.9).

    A discussion's turn mark is this process's: its notes travel in the
    request, and the event goes without it (RFC §10.6.3).
    """
    in_place = discussion_turn(event) is not None
    notes = discussion_notes(event) if in_place else []
    if DISCUSSION_TURN in (event.metadata or {}):
        metadata = {k: v for k, v in event.metadata.items() if k != DISCUSSION_TURN}
        event = event.model_copy(update={"metadata": metadata})
    return TurnRequest(
        turn_id=f"turn-{uuid4().hex}",
        room_id=event.room_id,
        channel_id=binding.channel_id,
        event=event,
        binding=binding,
        context=context,
        strategy_turn=strategy_turn,
        notes=notes,
        in_place=in_place,
        owner_tools=owner_tools or [],
        trace=_trace(telemetry),
    )


def _trace(telemetry: TelemetryProvider | None) -> dict[str, str] | None:
    """The broadcast span a turn runs under, with what names it to another
    process, for an executor elsewhere to continue its trace (RFC §10.6.9);
    in memory the turn's context carries it too."""
    span_id = get_current_span()
    if not span_id:
        return None
    carrier = telemetry.inject_context(span_id) if telemetry is not None else {}
    return {**carrier, "span_id": span_id}


class TurnClient(RoomOwner):
    """Runs turns through a backend and reads them back (RFC §10.6).

    Args:
        backend: The turn backend.
        executor: An executor this client serves on the backend before its
            first turn (the kit's own), or ``None``.
        timeouts: The bounds the owner keeps.
        tools: The tools the owner serves its turns (RFC §10.6.5), or ``None``.
        host: What the owner runs for its turns' ``TurnHost`` (RFC §10.6.5),
            or ``None``.
        events: Where the owner publishes what its turns published on their
            executors (RFC §10.6.3), or ``None``.
        error_classes: The host's exception classes an error raised on an
            executor is rebuilt as here, with their subclasses, beyond
            RoomKit's and Python's built-in ones (RFC §10.6.8). Any other comes back as
            a ``RemoteTurnError``.
        telemetry: The telemetry whose spans a turn runs under, named in its
            request for an executor elsewhere to continue the trace (RFC
            §10.6.9), or ``None``.

    Raises:
        TypeError: An entry of *error_classes* is not an ``Exception`` class.
    """

    def __init__(
        self,
        backend: TurnBackend,
        *,
        executor: TurnExecutor | None = None,
        timeouts: TurnTimeouts | None = None,
        tools: OwnerTools | None = None,
        host: OwnerHost | None = None,
        events: OwnerEvents | None = None,
        error_classes: Iterable[type[Exception]] = (),
        telemetry: TelemetryProvider | None = None,
    ) -> None:
        self.backend = backend
        self.executor = executor
        self.telemetry = telemetry
        """The telemetry whose spans a turn runs under (RFC §10.6.9)."""
        self.errors = ErrorClasses(error_classes)
        """The classes this owner rebuilds an error as (RFC §10.6.8)."""
        self._timeouts = timeouts or TurnTimeouts()
        self._tools = tools
        self._host = host
        self._events = events
        self._serving = executor is None
        self._attending = False
        # Run once, as the client first reaches its backend (the kit starts
        # watching its agents' presence then, RFC §10.6.7).
        self._on_first_use: Callable[[], None] | None = None
        if executor is not None:
            # Its rooms' hooks follow this client's names, at once (RFC §10.6.6).
            executor._room_hooks_from_owner = True
        self._open: weakref.WeakSet[OwnerTurn] = weakref.WeakSet()
        # The turns this client started and that still run, in start order:
        # what steering resolves its targets from (RFC §10.6.8).
        self._started: list[OwnerTurn] = []
        # The turns being handed to an executor (a distributed backend waits
        # for its acknowledgement): one may already run there, so a steer
        # waits for its hand-over before it chooses its targets.
        self._handing: dict[str, tuple[TurnRequest, asyncio.Future[None]]] = {}
        # The owner-served calls running for this client's turns (RFC §10.6.5).
        self._calls: set[asyncio.Task[None]] = set()
        # The open human-input requests of its turns, answered here.
        self.inputs = OwnerInputs(self._timeouts.input_hold_timeout)
        # Each room's last turn of a channel submitted here (RFC §10.6.2).
        self._previous: RoomRecords[PreviousTurn] = RoomRecords()
        self._closed = False
        # The names of each room's hooks on executor triggers (RFC §10.6.6).
        self._room_hooks: dict[str, list[str]] = {}

    async def run(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        *,
        strategy_turn: bool = False,
    ) -> ChannelOutput:
        """Run one turn of *binding*'s channel on *event*: its output, its
        stream reading the executor's (RFC §10.6.4); a strategy's generation
        when *strategy_turn* (RFC §10.6.1)."""
        await self._serve_own()
        channel_id = binding.channel_id
        declared = self._tools.declared(channel_id, event.room_id) if self._tools else []
        request = turn_request(
            event,
            binding,
            context,
            strategy_turn=strategy_turn,
            owner_tools=declared,
            telemetry=self.telemetry,
        )
        request.room_hooks = list(self._room_hooks.get(request.room_id, ()))
        request.previous_turn, submitted = self._leave(
            channel_id, request.room_id, request.turn_id
        )
        submitted._record_scope(request.turn_id, unscoped=answers_reach(event) == Visibility.ALL)
        handed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._handing[request.turn_id] = (request, handed)
        try:
            request.channel_rows = await self._channel_rows(event, binding, context)
            link = await self.backend.submit(request)
        except BaseException:
            # Failed before its backend gave it to an executor, the turn is
            # not the room's previous one; after, a cancel or a timeout
            # included, its executor may have begun it (RFC §10.6.2).
            if not request.handed:
                self._unleave(channel_id, request.room_id, submitted, request.previous_turn)
            self._handed(request, handed)
            raise
        turn = OwnerTurn(
            link,
            self._timeouts,
            self.errors,
            self._tools,
            self._calls,
            self._host,
            self.inputs,
            self._events,
            accepted=partial(_took, submitted),
        )
        self._open.add(turn)
        self._started = [*(t for t in self._started if not t.over), turn]
        # Steerable from now: a steer that waited for the hand-over finds it.
        self._handed(request, handed)
        return await turn.output()

    async def _channel_rows(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> list[RoomEvent]:
        """What the turn reads of the room, for an executor elsewhere (RFC
        §10.6.3); the owner's own executor reads the store itself. A store
        that fails costs the turn its earlier calls, not the turn."""
        channel_id = binding.channel_id
        own = self.executor is not None and self.executor.hosts_channel(channel_id)
        if own or self._host is None or binding.category != ChannelCategory.INTELLIGENCE:
            return []
        try:
            return await self._host.channel_rows(
                event.room_id, channel_id, history_start(event, context)
            )
        except Exception:
            logger.warning(
                "Could not read room %s's rows of %s for its turn",
                event.room_id,
                channel_id,
                exc_info=True,
            )
            return []

    def _handed(self, request: TurnRequest, handed: asyncio.Future[None]) -> None:
        self._handing.pop(request.turn_id, None)
        handed.set_result(None)

    def enable_room_hook(self, room_id: str, name: str) -> None:
        """Name *name* among *room_id*'s hooks on executor triggers: its
        turns' requests carry it, and only an executor that defines it takes
        them (RFC §10.6.6, §10.6.7). Synchronous and trusted, reading no
        room: :meth:`RoomKit.enable_room_hook` reads the room with the
        caller's organization first (RFC §17.2), as
        :meth:`RoomKit.disable_room_hook` does before :meth:`disable_room_hook`."""
        names = self._room_hooks.setdefault(room_id, [])
        if name not in names:
            names.append(name)
        self._apply_room_hooks(room_id)

    def disable_room_hook(self, room_id: str, name: str) -> bool:
        """Stop naming *name* among *room_id*'s hooks; whether it was."""
        names = self._room_hooks.get(room_id, [])
        if name not in names:
            return False
        names.remove(name)
        if not names:
            self._room_hooks.pop(room_id, None)
        self._apply_room_hooks(room_id)
        return True

    def _apply_room_hooks(self, room_id: str) -> None:
        """The own executor fires turn triggers outside a turn too (realtime
        voice, conference, RFC §10.6.6): its room hooks follow the names at
        once, not at the room's next turn."""
        if self.executor is not None:
            self.executor.hooks.use_room_hooks(room_id, self.room_hook_names(room_id))

    def room_hook_names(self, room_id: str) -> list[str]:
        """The names of *room_id*'s hooks on executor triggers."""
        return list(self._room_hooks.get(room_id, ()))

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Send a channel control message (RFC §10.6.3): a ``channel_call``'s
        result; nothing for ``room_attached``, ``room_detached``,
        ``committed_event`` and ``channel_released``. After a
        ``room_detached``, the room's next turn of the channel names no
        previous turn, so an executor that missed the message still drops the
        room's state (RFC §10.6.3). A
        ``committed_event`` goes where the room's turns would, the room's
        hooks named.

        Raises:
            TurnLinkError: ``no_executor`` for a ``channel_call`` or a
                ``committed_event`` no executor takes.
        """
        await self._serve_own()
        if isinstance(message, RoomDetached):
            self._previous.pop(message.channel_id, message.room_id)
        if isinstance(message, CommittedEvent):
            message = message.model_copy(
                update={"room_hooks": self.room_hook_names(message.room_id)}
            )
        if not isinstance(message, ChannelCall):
            return await self.backend.control(message)
        # A channel call leaves the room's state as a turn does.
        channel_id, room_id = message.channel_id, message.room_id
        previous, made = self._leave(channel_id, room_id, message.call_id)
        call = message.model_copy(update={"previous_turn": previous})
        try:
            answer = await self.backend.control(call)
        except BaseException:
            if not call.handed:
                self._unleave(channel_id, room_id, made, previous)
            raise
        if answer is not None:
            made.executor_id = answer.executor_id
        return answer

    def _leave(
        self, channel_id: str, room_id: str, entry_id: str
    ) -> tuple[PreviousTurn | None, PreviousTurn]:
        """Record the turn or call *entry_id* as the room's last of the
        channel before it is sent, so one sent alongside names it: the record
        it replaces (a copy, as a request carries it), and its own."""
        previous = self._previous.get(channel_id, room_id)
        made = PreviousTurn(turn_id=entry_id)
        made._inherit_scopes(previous)
        self._previous.put(channel_id, room_id, made)
        return (previous.model_copy() if previous is not None else None), made

    def _unleave(
        self,
        channel_id: str,
        room_id: str,
        made: PreviousTurn,
        previous: PreviousTurn | None,
    ) -> None:
        """Take back the record of a turn or call that was never sent, unless
        a later one replaced it."""
        if self._previous.get(channel_id, room_id) is not made:
            return
        if previous is None:
            self._previous.pop(channel_id, room_id)
        else:
            self._previous.put(channel_id, room_id, previous)

    async def room_call(self, message: RoomCall) -> HostResult:
        """A host verb an executor made after its turn ended (RFC §10.6.5),
        work that starts fresh once its turn is over (RFC §8.3); refused once
        the client closed, and cancelled by its close."""
        if self._closed:
            refusal = RuntimeError("The room owner has closed")
            return HostResult(call_id=message.call_id, error=error_record(refusal))
        task = asyncio.current_task()
        if task is not None:
            self._calls.add(task)
        try:
            return await answer_room_call(self._previous, self._host, message)
        finally:
            if task is not None:
                self._calls.discard(task)

    async def room_publication(self, message: RoomPublication) -> None:
        """What a channel published after its turn ended (RFC §10.6.3); none
        once the client closed."""
        if not self._closed:
            await publish_between_turns(self._previous, self._events, message)

    async def _serve_own(self) -> None:
        """Have the backend reach this client outside a turn, and serve the
        client's own executor on it, once."""
        if not self._attending:
            self._attending = True
            if self._on_first_use is not None:
                self._on_first_use()
            await self.backend.attend(self)
        if not self._serving:
            self._serving = True
            assert self.executor is not None
            await self.backend.serve(self.executor)

    def input_request(
        self, request_id: str, *, organization_id: str | None = None
    ) -> PendingInputEvent | None:
        """The open human-input request *request_id* of a turn this client
        started (RFC §10.6.5), as ``ON_USER_INPUT_REQUIRED`` reads it: its room,
        channel, tool call and the actor of its turn. ``None`` when no turn of
        this client holds it open, or, with *organization_id*, when its room is
        another organization's (RFC §17.2)."""
        entry = self.inputs.request(request_id)
        if entry is None:
            return None
        turn, request = entry
        asked = turn.request
        if organization_id is not None and asked.context.room.organization_id != organization_id:
            return None
        return PendingInputEvent(
            pending_id=request.request_id,
            tool_name=request.tool_name,
            arguments=dict(request.arguments),
            room_id=request.room_id or asked.room_id,
            tool_call_id=request.tool_call_id or "",
            channel_id=request.channel_id or asked.channel_id,
            channel_type=asked.binding.channel_type,
            timestamp=request.created_at or asked.event.created_at,
            actor_id=request.actor_id,
        )

    async def steer(
        self,
        directive: SteeringDirective,
        channel_id: str,
        *,
        room_id: str | None = None,
        turn_id: str | None = None,
        organization_id: str | None = None,
    ) -> int:
        """Steer the running turns of *channel_id* this client started (RFC
        §21.3, §10.6.8): the turn *turn_id*; else the turns of *room_id*, all
        of them for a ``Cancel`` and the most recent for any other directive;
        else the most recent turn of the channel. How many loops it reached.
        With *organization_id*, only the turns of that organization's rooms
        (RFC §17.2).

        Raises:
            ValueError: both *room_id* and *turn_id* given.
        """
        if room_id is not None and turn_id is not None:
            raise ValueError("steer() addresses a room_id or a turn_id, not both")
        await self._handed_over(channel_id, room_id, turn_id)
        running = [
            t
            for t in self._started
            if not t.over
            and t.request.channel_id == channel_id
            and (
                organization_id is None
                or t.request.context.room.organization_id == organization_id
            )
        ]
        if turn_id is not None:
            targets = [t for t in running if t.request.turn_id == turn_id]
        elif room_id is not None:
            in_room = [t for t in running if t.request.room_id == room_id]
            targets = in_room if isinstance(directive, SteeringCancel) else in_room[-1:]
        else:
            targets = running[-1:]
        reached = await asyncio.gather(*(turn.steer(directive) for turn in targets))
        return sum(reached)

    async def _handed_over(
        self, channel_id: str, room_id: str | None, turn_id: str | None
    ) -> None:
        """Wait for the turns of *channel_id* (in *room_id*, or *turn_id*)
        still being handed to an executor: each may already run there."""
        pending = [
            handed
            for request, handed in self._handing.values()
            if request.channel_id == channel_id
            and (room_id is None or request.room_id == room_id)
            and (turn_id is None or request.turn_id == turn_id)
        ]
        if pending:
            await asyncio.wait(pending, timeout=self._timeouts.accept)

    async def close(self) -> None:
        """Close the own executor, which cuts its turns, stop holding every
        turn still open (a stream nobody read), wait for the calls its turns
        made to end (but the one running this close), then close the backend.
        Work of a turn that reaches it after this is refused."""
        self._closed = True
        if self.executor is not None:
            await self.executor.close()
        for turn in list(self._open):
            turn.abandon()
        acting = acting_tasks()
        calls = [task for task in self._calls if task not in acting]
        for task in calls:
            if not task.cancelling():
                # One already cancelled is let finish its cleanup.
                task.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        self._started.clear()
        await self.backend.close()

    async def _close(self, until: float) -> set[asyncio.Task[Any]]:
        """:meth:`close` as the kit's close runs it (RFC §12.10.4): its own
        executor's turns and the calls they made waited for until *until*,
        then the backend closed on its own budget (its cancel bound, when
        it offers a bounded close); what is still running then, which it
        leaves, for the kit to name."""
        self._closed = True
        left: set[asyncio.Task[Any]] = set()
        if self.executor is not None:
            left |= await self.executor._close(until)
        for turn in list(self._open):
            turn.abandon()
        acting = acting_tasks() | close_callers()
        calls = [task for task in self._calls if task not in acting]
        for task in calls:
            if not task.cancelling():
                # One already cancelled is let finish its cleanup.
                task.cancel()
        loop = asyncio.get_running_loop()
        if calls:
            _, pending = await asyncio.wait(calls, timeout=max(0.0, until - loop.time()))
            left |= pending
        self._started.clear()
        bounded = getattr(self.backend, "_close_within", None)
        if bounded is None:
            await self.backend.close()
        else:
            left |= await bounded(loop.time() + self._timeouts.cancel)
        return left


@runtime_checkable
class _TurnStream(Protocol):
    """A turn's stream: the turn backend's, or a turn's that waits its place
    in the room (RFC §6.4)."""

    def drop(self) -> None: ...

    def read_as_trace(self) -> None: ...


def drop_unread(stream: object) -> None:
    """Give up a turn's stream nobody will read: a stream of the turn
    backend is closed in the background; a plain generator is left to its
    collection, which closes it."""
    if isinstance(stream, _TurnStream):
        stream.drop()


def read_as_trace(stream: object) -> None:
    """Say, before reading it, that a turn's stream is read as a delegated
    task's trace, committed without crossing a hook (RFC §23.3 step 5): only
    such a turn's answer text is published (RFC §6.4). A plain generator is
    no turn's: its text is published by nobody."""
    if isinstance(stream, _TurnStream):
        stream.read_as_trace()


def drop_unread_streams(responses: Iterable[Any]) -> None:
    """Give up the streams of *responses* (each a ``StreamingResponse``) that
    a failure leaves nobody to read: none is left to the garbage collector."""
    for response in responses:
        drop_unread(response.stream)


@contextmanager
def unread_dropped_on_failure(responses: Iterable[Any]) -> Generator[None]:
    """Within it, a failure (a cancellation included) gives up the streams of
    *responses* nobody will then read; one being read at that moment, or read
    to its end, is left to its reader."""
    try:
        yield
    except BaseException:
        drop_unread_streams(responses)
        raise


def registry_turn_client(
    channels: Mapping[str, Channel],
    backend: TurnBackend | None = None,
    *,
    host: OwnerHost | None = None,
    events: OwnerEvents | None = None,
    owner_tools: OwnerToolStore | None = None,
    error_classes: Iterable[type[Exception]] = (),
) -> TurnClient:
    """A client whose own executor serves the channels of a registry (a
    kit's, a router's), on *backend* or an in-memory one, its turns' host
    verbs run by *host*, what they publish published by *events*, the tools
    orchestration set up for them kept in *owner_tools* (a store of its own
    when None), the host's *error_classes* rebuilt by both. The registry's
    owner wires and closes its channels; one it holds by its declaration
    alone is a host executor's to build (RFC §10.6.2)."""

    def runs_here(channel_id: str) -> bool:
        channel = channels.get(channel_id)
        return channel is not None and not isinstance(channel, DeclaredChannel)

    backend = backend or InMemoryTurnBackend()
    # Read once: both sides take the same classes, a generator included.
    error_classes = tuple(error_classes)
    return TurnClient(
        backend,
        executor=TurnExecutor(
            channels.get,
            hosts=runs_here,
            owns_channels=False,
            timeouts=backend.timeouts,
            error_classes=error_classes,
        ),
        tools=RegistryOwnerTools(owner_tools or OwnerToolStore(channels.get)),
        host=host,
        events=events,
        timeouts=backend.timeouts,
        error_classes=error_classes,
    )


def _took(submitted: PreviousTurn, executor_id: str) -> None:
    """Record which executor took the turn *submitted* names, as it took it."""
    submitted.executor_id = executor_id
