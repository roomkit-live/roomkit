"""One turn on its executor (RFC §10.6.3, §10.6.4, §10.6.8).

The turn runs the channel's ``on_event``, sends its output, then advances its
stream as far as the owner's pulls grant: deltas ahead of the owner's reader,
never past any other item without a later pull. The channel changes some of
what it already yielded (its response-metadata record, a call's arguments
once its gate fixed them, a call's end before the end itself is read): after
each step the turn sends what changed, before the item it read, so the
owner's copies read as the channel's own (before a delta, the record's
change is a write to one of its keys, ``_executor_changes``).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, NoReturn

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.channels._strategy_turn import strategy_turn
from roomkit.channels._tool_registry import set_turn_owner_tools
from roomkit.core._turn_scope import enter_turn
from roomkit.core.exceptions import ChannelNotRegisteredError, TurnLinkError
from roomkit.core.hooks import set_turn_room_hooks
from roomkit.telemetry.context import get_current_span, set_current_span
from roomkit.tools._turn_inputs import set_turn_inputs
from roomkit.turns._between_turns import BetweenTurns
from roomkit.turns._codec import encode_output
from roomkit.turns._error_codec import error_record
from roomkit.turns._executor_calls import ExecutorCalls
from roomkit.turns._executor_changes import ChannelChanges
from roomkit.turns._executor_inputs import ExecutorInputs
from roomkit.turns._executor_pulls import PullGate
from roomkit.turns._executor_steer import steer_turn
from roomkit.turns._frame_sender import FrameSender
from roomkit.turns._outlet import TurnOutlet, set_turn_outlet
from roomkit.turns.base import heartbeat_tick
from roomkit.turns.host import TurnHost, set_turn_host
from roomkit.turns.wire import (
    Accepted,
    Cancel,
    Close,
    End,
    Error,
    Heartbeat,
    HostResult,
    InputAnswer,
    Item,
    Output,
    Pull,
    Steer,
    SteerResult,
    ToolResult,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.channel import ChannelOutput
    from roomkit.models.event import RoomEvent
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.turns._error_codec import ErrorClasses
    from roomkit.turns.base import ExecutorLink, RoomOwner, TurnTimeouts
    from roomkit.turns.wire import Control, Frame, TurnRequest

logger = logging.getLogger("roomkit.turns")


class _UncarriedError(Exception):
    """A frame of the turn could not cross: the owner was told, the turn ends."""


async def _reaction(channel: Channel, event: RoomEvent, request: TurnRequest) -> ChannelOutput:
    """The channel's ``on_event``, told whether the turn is a strategy's
    generation (RFC §10.6.1)."""
    with strategy_turn(request.strategy_turn):
        return await channel.on_event(event, request.binding, request.context)


async def _close_stream(stream: AsyncIterator[Any]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


class ExecutorTurn:
    """Runs one turn a link brought and answers it with frames."""

    def __init__(
        self,
        link: ExecutorLink,
        executor_id: str,
        channel: Callable[[], Awaitable[Channel | None]],
        timeouts: TurnTimeouts,
        errors: ErrorClasses,
        owner: RoomOwner | None = None,
        telemetry: TelemetryProvider | None = None,
    ) -> None:
        self._link = link
        self._request = link.request
        self._telemetry = telemetry
        self._executor_id = executor_id
        self._channel = channel
        self._timeouts = timeouts
        # What the owner's pulls let the stream read (RFC §10.6.4), whether a
        # close cut the read ahead of the reader, and whether the channel's
        # stream is being closed (a cleanup no later cut cancels).
        self._pulls = PullGate()
        self._close_cut = False
        self._cleaning = False
        self._cut: str | None = None
        self._wake = asyncio.Event()
        self._heard = True
        self._work: asyncio.Task[Any] | None = None
        self._built: Channel | None = None
        self._spared = False
        # A spared turn its owner cut: the owner was told it ended, and
        # hears nothing more of it (RFC §12.10.4).
        self._let_go = False
        self._letting_go: asyncio.Future[None] | None = None
        # Its owner's controls read, and its owner's silence watched.
        self._watchers: tuple[asyncio.Task[None], ...] = ()
        self._mark: dict[str, Any] | None = None
        # Every frame goes in order, the stream's text deltas by burst.
        self._frames = FrameSender(link.send)
        # What the channel changes after yielding it, sent before what follows.
        self._changes = ChannelChanges(self._send, self._refuse)
        # The calls the turn makes to its owner: the tools it serves, and
        # the verbs of the turn's host (RFC §10.6.5).
        self._calls = ExecutorCalls(self._request, self._send_call, errors)
        # Work the turn began reaches the owner after it, where the backend
        # carries that (RFC §10.6.3, §10.6.5).
        after = (
            BetweenTurns(owner, self._request, executor_id, errors) if owner is not None else None
        )
        self._host = TurnHost(
            self._calls.host_call,
            between_turns=after.call if after is not None else None,
            errors=errors,
        )
        self._inputs = ExecutorInputs(self._send_call)
        # What the turn's channel publishes and its hooks emit (RFC §10.6.2).
        self._outlet = TurnOutlet(
            self._send_published, after=after.publish if after is not None else None
        )

    async def run(self) -> None:
        """Run the turn to its end, which the owner is always told."""
        watch = asyncio.create_task(self._read_controls())
        beat = asyncio.create_task(self._beat())
        self._watchers = (watch, beat)
        try:
            await self._send(Accepted(executor_id=self._executor_id))
            await self._turn()
        except _UncarriedError:
            pass
        finally:
            # The turn is over: its host answers no more verbs, the calls on
            # their way are answered first, and the mark its notes rode reads
            # as no mark.
            self._host._close()
            self._outlet.close()
            self._inputs.close()
            try:
                await self._settle_calls()
            finally:
                if self._mark is not None:
                    retire_mark(self._mark)
                watch.cancel()
                beat.cancel()
                await asyncio.gather(watch, beat, return_exceptions=True)
                await self._send(End())

    async def _settle_calls(self) -> None:
        """Wait for the calls still on their way. A cut turn, or one cut while
        it waits (its executor closing), fails them instead: nothing waits for
        an owner that may no longer answer."""
        task = asyncio.current_task()
        if self._cut is not None or (task is not None and task.cancelling()):
            self._calls.fail_all(TurnLinkError(self._cut or "cancelled"))
        try:
            await self._calls.settle()
        except asyncio.CancelledError:
            self._calls.fail_all(TurnLinkError(self._cut or "cancelled"))
            raise

    # -- The turn -----------------------------------------------------------

    async def _turn(self) -> None:
        """``on_event``, its output, then its stream as the owner pulls it."""
        try:
            # Built by the turn's task, never before: a task cancelled before
            # its first step (a close as the turn arrives) leaves nothing
            # unawaited.
            channel = await self._channel()
        except Exception as exc:
            await self._send(Error(error=error_record(exc)))
            return
        self._built = channel
        if self._cut is not None:
            # Cut while its channel was being built (a steering Cancel): it
            # never runs.
            return
        if channel is None:
            missing = ChannelNotRegisteredError(
                f"No channel {self._request.channel_id!r} on executor {self._executor_id}"
            )
            await self._send(Error(error=error_record(missing)))
            return
        event = self._marked_event()
        try:
            output = await self._run_work(
                _reaction(channel, event, self._request), self._link.turn_context()
            )
        except asyncio.CancelledError:
            if self._cut is None:
                raise
            return
        except Exception as exc:
            await self._send(Error(error=error_record(exc)))
            return
        self._changes.watch(output.response_metadata)
        stream = output.response_stream
        try:
            wire = encode_output(output)
        except (TypeError, ValueError) as exc:
            if stream is not None:
                await self._close(stream, contextvars.Context())
            await self._refuse("output", exc)
        self._changes.sent(wire.response_metadata)
        await self._send(Output(output=wire, streams=stream is not None))
        if stream is not None:
            await self._serve_stream(stream)

    def _marked_event(self) -> RoomEvent:
        """The request's event, carrying a mark of this process for a
        discussion's turn so the channel reads its notes here."""
        event = self._request.event
        if not self._request.in_place:
            return event
        self._mark = issue_mark(self._request.notes)
        metadata = {**(event.metadata or {}), DISCUSSION_TURN: self._mark}
        return event.model_copy(update={"metadata": metadata})

    async def _serve_stream(self, stream: AsyncIterator[Any]) -> None:
        """Advance *stream* as the owner pulls it, until its end, a close or a
        cut; a stream whose next frame cannot cross is closed. The stream is
        read by one task of the turn's work, never one task per item."""
        try:
            await self._wait_for_pull()
        except asyncio.CancelledError:
            await self._close(stream, contextvars.Context())
            raise
        # The first pull lends the stream its context (RFC §10.6.4).
        context = self._link.stream_context() or contextvars.Context()
        if self._pulls.closing or self._cut is not None:
            await self._close(stream, context)
            return
        try:
            ended = await self._run_work(self._pump(stream), context)
        except _UncarriedError:
            await self._close(stream, context)
            raise
        except asyncio.CancelledError:
            if self._close_cut:
                # A close cut the read ahead of the reader: the stream is
                # closed, the turn goes on to its end.
                await self._close(stream, context)
                return
            if self._cut is None:
                # The executor closes: the channel's stream is closed, as a
                # collected generator's would be, before the cut goes on.
                await self._close(stream, context)
                raise
            # Cut, reading or waiting for a pull: the channel's stream is
            # closed under the turn (nothing to do when the cut ended it in
            # its read), and what changed before still crosses.
            await self._close(stream, context)
            return
        if not ended:
            await self._close(stream, context)

    async def _pump(self, stream: AsyncIterator[Any]) -> bool:
        """Read *stream* as the owner's pulls allow, each item sent after what
        changed before it: ``True`` once it ended (or failed), ``False`` when
        a close or a cut stops it first."""
        while True:
            await self._wait_for_pull()
            if self._pulls.closing or self._cut is not None:
                return False
            if not await self._step(stream):
                return True

    async def _step(self, stream: AsyncIterator[Any]) -> bool:
        """Read one item and pass it on after what changed before it, a text
        delta to the burst being kept; whether the stream goes on."""
        last: Frame | None = None
        self._pulls.start_read()
        try:
            item = await anext(stream)
        except StopAsyncIteration:
            last = Item(item={"kind": "stream_end"})
        except Exception as exc:
            last = Error(error=error_record(exc))
        else:
            self._pulls.item_read(item)
            if self._pulls.closing or self._cut is not None:
                # Read, then closed: the reader asked for nothing more.
                return True
            await self._changes.report(before=item)
            if isinstance(item, str):
                await self._add_text(item)
            else:
                await self._changes.send_item(item)
            return True
        finally:
            self._pulls.end_read()
        await self._changes.report()
        await self._send(last)
        return False

    async def _close(self, stream: AsyncIterator[Any], context: contextvars.Context) -> None:
        """Close the stream as a local reader's close would."""
        self._cleaning = True
        try:
            await self._run_work(_close_stream(stream), context)
        except asyncio.CancelledError:
            if self._cut is None:
                raise
        except Exception:
            logger.exception("Closing the stream of turn %s failed", self._request.turn_id)
        finally:
            self._cleaning = False
        await self._changes.report()

    async def _run_work(self, work: Coroutine[Any, Any, Any], context: contextvars.Context) -> Any:
        """Run one piece of the turn's work as a task the owner can cancel,
        the code under it knowing it runs under this turn."""
        context.run(enter_turn, self._request.turn_id)
        context.run(set_turn_owner_tools, self._calls.owner_tools)
        context.run(set_turn_host, self._host)
        context.run(set_turn_inputs, self._inputs)
        context.run(set_turn_outlet, self._outlet)
        context.run(set_turn_room_hooks, self._request.room_id, self._request.room_hooks)
        trace = self._request.trace or {}
        span_id = trace.get("span_id")
        if span_id and context.run(get_current_span) != span_id:
            # The turn continues the broadcast's span, whichever process runs
            # it and whatever its context holds: its work reads the broadcast
            # as current, in the owner's trace (RFC §10.6.9, §15.7).
            remote = self._telemetry.extract_context(trace) if self._telemetry else None
            context.run(set_current_span, span_id, telemetry_ctx=remote)
        self._work = asyncio.get_running_loop().create_task(
            work, context=context, name=f"roomkit-turn-{self._request.turn_id}-work"
        )
        try:
            return await self._work
        finally:
            self._work = None

    async def _wait_for_pull(self) -> None:
        while not self._pulls.granted and not self._pulls.closing and self._cut is None:
            self._wake.clear()
            await self._wake.wait()

    async def _add_text(self, text: str) -> None:
        """Keep a text delta for the burst; one that cannot cross refuses
        the turn, as any item that cannot."""
        try:
            self._frames.add_text(text)
        except ValueError as exc:
            await self._refuse("stream item", exc)

    # -- The link -------------------------------------------------------------

    async def _send(self, frame: Frame) -> int:
        """Send *frame*; one that cannot cross is refused to the owner as an
        error naming it. An owner lost hears nothing more."""
        if self._cut == "owner_lost" or self._let_go:
            return 0
        try:
            return await self._frames.send(frame)
        except (TypeError, ValueError) as exc:
            if isinstance(frame, Error):
                logger.exception("Turn %s could not report its error", self._request.turn_id)
                raise _UncarriedError from exc
            await self._refuse(frame.type, exc)

    async def _send_published(self, frame: Frame) -> None:
        """Send what the turn published (an ephemeral event, a framework
        event); one that cannot cross is dropped, the turn going on."""
        if self._cut in ("owner_lost", "frame_lost"):
            return
        try:
            await self._frames.send(frame)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Turn %s: a %s cannot cross the turn backend, dropped: %s",
                self.turn_id,
                frame.type,
                exc,
            )

    async def _send_call(self, frame: Frame) -> int:
        """Send a call of the turn to its owner; an owner lost answers none."""
        if self._cut in ("owner_lost", "frame_lost"):
            raise TurnLinkError(self._cut)
        return await self._frames.send(frame)

    async def _refuse(self, what: str, exc: Exception) -> NoReturn:
        """Tell the owner the turn's *what* cannot cross the backend, and end the turn."""
        logger.warning(
            "Turn %s: its %s cannot cross the turn backend: %s", self.turn_id, what, exc
        )
        refusal = TypeError(f"The turn's {what} cannot cross the turn backend: {exc}")
        await self._send(Error(error=error_record(refusal)))
        raise _UncarriedError from exc

    async def _read_controls(self) -> None:
        """Apply the owner's control messages as they come."""
        while True:
            try:
                message = await self._link.receive()
            except TurnLinkError as lost:
                self._cut_turn("frame_lost" if lost.reason == "frame_lost" else "owner_lost")
                return
            self._heard = True
            try:
                await self._apply_control(message)
            except _UncarriedError:
                raise
            except Exception:
                # One message failing to apply never stops the turn's controls.
                logger.exception(
                    "Turn %s: a %s control failed to apply", self.turn_id, message.type
                )
            self._wake.set()

    def _cut_read(self) -> None:
        """A close found the stream reading ahead of its reader: the read is
        cut, as the reader's close would end the channel's stream."""
        if self._pulls.reading and self._cancel_work("closed"):
            self._close_cut = True

    async def _apply_control(self, message: Control) -> None:
        """Apply one control message of the owner's."""
        if isinstance(message, Pull):
            self._pulls.grant(message)
        elif isinstance(message, Close):
            if self._pulls.close():
                # A read ahead of the reader is cut at once; one it asked for
                # (a running call) finishes first (RFC §10.6.4, §12.2).
                self._cut_read()
        elif isinstance(message, Cancel):
            self._cut_turn(message.reason)
            if self._spared:
                await self._let_owner_go()
        elif isinstance(message, ToolResult | HostResult):
            self._calls.resolve(message)
        elif isinstance(message, InputAnswer):
            await self._inputs.answer(message)
        elif isinstance(message, Steer):
            loops = await steer_turn(self._built, message.directive, self._request)
            await self._send(SteerResult(steer_id=message.steer_id, loops=loops))

    async def _beat(self) -> None:
        """Tell the owner the turn runs; cut it when the owner falls silent."""
        silent = 0
        while True:
            await heartbeat_tick(self._timeouts.heartbeat_interval)
            await self._send(Heartbeat())
            silent = 0 if self._heard else silent + 1
            self._heard = False
            if silent >= self._timeouts.silent_intervals:
                logger.warning("Owner of turn %s fell silent", self._request.turn_id)
                self._cut_turn("owner_lost")
                return

    @property
    def turn_id(self) -> str:
        return self._request.turn_id

    def spare(self) -> None:
        """Let the turn's work run on whatever ends its link: a close called
        from inside this turn must not cut the turn it runs under. One cut
        already ends for its owner now (:meth:`_let_owner_go`)."""
        self._spared = True
        if self._cut is not None and not self._let_go and self._letting_go is None:
            self._letting_go = asyncio.ensure_future(self._let_owner_go())

    async def _let_owner_go(self) -> None:
        """Tell the owner the turn ended, once, and nothing after: its work
        runs the kit's close that cut the turn, and ends only once that close
        returned, which waits for the owner (RFC §12.10.4)."""
        if self._let_go:
            return
        self._let_go = True
        try:
            await self._frames.send(End())
        except Exception:  # noqa: BLE001 - an owner gone needs no end
            logger.debug("Turn %s: its owner is gone", self._request.turn_id)
        self.stop_watching()

    def stop_watching(self) -> None:
        """Watch its owner no more: an owner that let the turn go, or an
        executor closed under a turn left running (RFC §12.10.4), is no
        owner fallen silent."""
        for watcher in self._watchers:
            if watcher is not asyncio.current_task():
                watcher.cancel()
        self._heard = True
        self._watchers = ()

    def _cut_turn(self, reason: str) -> None:
        """Cut the turn: its work is cancelled once, so a later cut never
        cancels the close the first one runs (the channel's own cleanup)."""
        first = self._cut is None
        if first:
            self._cut = reason
        if reason in ("owner_lost", "frame_lost"):
            self._calls.fail_all(TurnLinkError(reason))
        if first and not self._spared:
            self._cancel_work(reason)
        self._wake.set()

    def _cancel_work(self, reason: str) -> bool:
        """Cancel the piece of the turn's work that runs, once: one already
        cancelled, or the stream's close a close started (a close then a cut,
        or a cut then a close), is let finish its cleanup, the channel's
        finalizers included. Whether it was cancelled now."""
        work = self._work
        if work is None or work.done() or work.cancelling() or self._cleaning:
            return False
        work.cancel(reason)
        return True
