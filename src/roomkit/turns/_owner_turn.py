"""One turn on its room owner (RFC §10.6.4, §10.6.8).

The owner reads the turn's frames back into the objects the pipeline reads
locally: a ``ChannelOutput``, a stream whose reads grant the executor a
window of items, the turn's one response-metadata record updated in place as
the reader reaches the item it was sent before, and call markers whose
fixed arguments and early ends are set on the owner's own copies. The calls
the turn makes to its owner run here as they arrive (RFC §10.6.5), and the
owner steers the turn's loop (RFC §10.6.8). The turn is watched from its
submission to its end: an executor silent too long has lost the turn.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.core.task_utils import acting_tasks, cancel_reason
from roomkit.models.channel import ChannelOutput
from roomkit.models.steering import Cancel as SteeringCancel
from roomkit.models.steering import SteeringDirective
from roomkit.models.streaming import ToolCallEndMarker
from roomkit.turns._codec import decode_end, decode_output
from roomkit.turns._owner_calls import OwnerCalls
from roomkit.turns._owner_inputs import OwnerInputs
from roomkit.turns._owner_publish import TurnPublisher, names_its_room
from roomkit.turns._owner_stream import StreamCopies
from roomkit.turns._remote_stream import RemoteStream
from roomkit.turns.base import heartbeat_tick
from roomkit.turns.wire import (
    Accepted,
    CallCancel,
    Cancel,
    Close,
    End,
    Ephemeral,
    Error,
    FrameworkEventFrame,
    Heartbeat,
    HostCall,
    InputClosed,
    InputRequest,
    InputTaken,
    Item,
    Output,
    Record,
    Steer,
    SteerResult,
    ToolArgs,
    ToolCall,
    ToolEnded,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.turns._error_codec import ErrorClasses
    from roomkit.turns.base import OwnerEvents, OwnerHost, OwnerTools, TurnLink, TurnTimeouts
    from roomkit.turns.wire import Control, Frame, TurnRequest

logger = logging.getLogger("roomkit.turns")


def _cancel_reason(cancelled: asyncio.CancelledError) -> str:
    """The reason a cancellation names, ``cancelled`` when it names none."""
    return cancel_reason(cancelled) or "cancelled"


def _of_the_turn(request: TurnRequest, room_id: str | None, channel_id: str | None) -> bool:
    """Whether what a turn published names its room and channel (or none)."""
    return names_its_room(request.room_id, request.channel_id, room_id, channel_id)


class OwnerTurn:
    """The owner's side of one turn, from its submission to its end.

    One task reads the turn's frames as they come: it applies at once the
    ones that change what the owner holds or ask it to act (a record, a
    call's arguments or end, a call to the owner, a steering result), and
    queues the others (the output, a stream item, an error, the end) for the
    task reading the turn. It watches the executor too: one silent too long
    has lost the turn. A call the turn makes runs in a copy of the context
    of the task that last waited on the turn (RFC §10.6.5).
    """

    def __init__(
        self,
        link: TurnLink,
        timeouts: TurnTimeouts,
        errors: ErrorClasses,
        tools: OwnerTools | None = None,
        held_calls: set[asyncio.Task[None]] | None = None,
        host: OwnerHost | None = None,
        inputs: OwnerInputs | None = None,
        events: OwnerEvents | None = None,
        accepted: Callable[[str], None] | None = None,
    ) -> None:
        self._link = link
        self._timeouts = timeouts
        # The classes the turn's errors are rebuilt as (RFC §10.6.8).
        self._errors = errors
        # The owner's copies of what the stream yielded, and its reads.
        self._copies = StreamCopies()
        # Told the executor's id as soon as it took the turn: work the turn
        # began may reach the owner before its output does (RFC §10.6.3).
        self._on_accepted = accepted
        # What the turn published on its executor, published here in order.
        self._publisher = TurnPublisher(events, link.request)
        held = held_calls if held_calls is not None else set()
        self._calls = OwnerCalls(link.request, tools, host, self._send_result, held)
        self._inputs = inputs if inputs is not None else OwnerInputs(timeouts.input_hold_timeout)
        self._accepted = False
        self.executor_id: str | None = None
        """The executor that took the turn, once it said so."""
        self._over = False
        self._main: asyncio.Queue[tuple[int, Frame] | BaseException] = asyncio.Queue()
        self._waiting: contextvars.Context | None = None
        self._steers: dict[str, asyncio.Future[int | None]] = {}
        self._steer_cut = False
        self._cut_ends: list[ToolCallEndMarker] = []
        # Whether a frame came since the last heartbeat interval, and how many
        # intervals passed with none (RFC §10.6.8).
        self._heard = False
        self._silent = 0
        # The executor's end of the turn came: what it cleaned up is done; or
        # the owner gave the turn up itself, and waits for nothing of it.
        self._ended = False
        self._cleanup_incomplete = False
        self._abandoned = False
        loop = asyncio.get_running_loop()
        self._beat = loop.create_task(self._beat_while_open())
        self._pump = loop.create_task(self._read_frames(), context=contextvars.Context())

    @property
    def request(self) -> TurnRequest:
        """The request the turn was submitted with."""
        return self._link.request

    @property
    def end_owed(self) -> bool:
        """Whether the turn is over with cleanup not known to be done:
        its executor's end is missing or its owner calls outlasted the
        cancellation bound (RFC §10.6.8)."""
        return self._over and (not self._ended or self._cleanup_incomplete) and not self._abandoned

    @property
    def over(self) -> bool:
        """Whether the turn is over for the owner."""
        return self._over

    def abandon(self) -> None:
        """Stop holding the turn now (the owner closes): later frames are
        dropped, and the executor, no longer hearing its owner, cuts it. A
        turn already over keeps what it reported: one whose end is owed stays
        owed (RFC §10.6.8)."""
        if not self._over:
            self._abandoned = True
        self._finish()

    # -- Output and stream ---------------------------------------------------

    async def output(self) -> ChannelOutput:
        """The channel's output, its stream reading this turn when it has one."""
        try:
            frame = await self._next_main()
        except asyncio.CancelledError as cancelled:
            await self._cancel(_cancel_reason(cancelled))
            raise
        if isinstance(frame, Output):
            output = self._decoded(decode_output, frame.output, self._errors)
            self._copies.record = output.response_metadata
            if frame.streams:
                output.response_stream = RemoteStream(self)
            return output
        if isinstance(frame, Error):
            self._finish()
            raise self._errors.rebuild(frame.error)
        self._finish()
        if self._steer_cut:
            # Cut by a steering Cancel before it answered (RFC §10.6.8).
            return ChannelOutput.empty()
        raise TurnLinkError("executor_lost", "The turn ended before its output")

    async def read_item(self) -> Any:
        """The stream's next item; ``StopAsyncIteration`` at its end."""
        if self._over and self._steer_cut:
            # Read after a steering Cancel ended the turn: the calls it left
            # open end cancelled first.
            self.reader_stopped()
            self._cut_ends.extend(self._copies.open_call_ends())
        if self._cut_ends:
            return self._cut_ends.pop(0)
        if self._over and self._steer_cut:
            raise StopAsyncIteration
        await self._pull()
        try:
            seq, frame = await self._next_main_with_seq()
        except asyncio.CancelledError as cancelled:
            await self._cancel(_cancel_reason(cancelled))
            raise
        if isinstance(frame, Item):
            self._decoded(self._copies.count_read, frame.item)
            self._copies.apply_records(before=seq)
            if frame.item.get("kind") == "stream_end":
                await self._drain_to_end(None)
                raise StopAsyncIteration
            return self._decoded(self._copies.item, seq, frame.item)
        if isinstance(frame, Error):
            await self._drain_to_end(None)
            raise self._errors.rebuild(frame.error)
        # The turn ended with every item read.
        self.reader_stopped()
        self._finish()
        if self._steer_cut:
            # Cut by a steering Cancel: the calls it left open end cancelled,
            # one per read, then the stream (RFC §10.6.4).
            self._cut_ends = self._copies.open_call_ends()
            if self._cut_ends:
                return self._cut_ends.pop(0)
            raise StopAsyncIteration
        raise TurnLinkError("executor_lost", "The turn ended in the middle of its stream")

    async def _pull(self) -> None:
        """Grant the executor the window again when the reader read enough of
        it; nothing once the turn is over: its last items are read from what
        came."""
        if not self._over and (pull := self._copies.pull()) is not None:
            await self._link.send(pull)

    def read_as_trace(self) -> None:
        """Its reader commits the turn as a delegated task's trace, crossing
        no hook: the turn's answer text is published from now (RFC §6.4).
        Said before the first read, so before any of that text was produced."""
        self._publisher.traced = True

    async def close_stream(self) -> None:
        """Close the stream: the executor closes the channel's, a running
        call let finish (RFC §12.2 step 13s), and its end is waited for."""
        if self._over:
            self.reader_stopped()
            return
        await self._link.send(Close())
        await self._drain_to_end(None)

    def reader_stopped(self) -> None:
        """The reader stops, however: the record is the turn's last, every
        record the executor sent applied (RFC §10.6.4)."""
        self._copies.apply_records()

    def _decoded(self, decode: Any, *args: Any) -> Any:
        """What *decode* builds of a frame's payload; a payload it cannot
        read fails the turn as an invalid frame."""
        try:
            return decode(*args)
        except (KeyError, TypeError, ValueError) as exc:
            invalid = TurnLinkError(
                "frame_invalid", f"A frame of the turn could not be read: {exc}"
            )
            self._main.put_nowait(invalid)
            self._finish()
            raise invalid from exc

    # -- Steering --------------------------------------------------------------

    async def steer(self, directive: SteeringDirective) -> int:
        """Send *directive* to the turn's loop: how many loops it reached. A
        channel without steering takes a ``Cancel`` as the turn's cancel,
        which counts as one (RFC §10.6.8)."""
        if self._over:
            return 0
        steer_id = uuid4().hex
        reached: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self._steers[steer_id] = reached
        try:
            await self._link.send(Steer(steer_id=steer_id, directive=directive))
            loops = await reached
        finally:
            self._steers.pop(steer_id, None)
        if loops is not None:
            return loops
        if not isinstance(directive, SteeringCancel) or self._over:
            return 0
        # Read to its end by whoever reads it, as after a close.
        self._steer_cut = True
        await self._send_best_effort(Cancel(reason=directive.reason or "steered"))
        return 1

    # -- Frames ---------------------------------------------------------------

    async def _next_main(self) -> Frame:
        _, frame = await self._next_main_with_seq()
        return frame

    async def _next_main_with_seq(self) -> tuple[int, Frame]:
        """The next output, item, error or end, waited for by the current
        task, whose context a call arriving meanwhile copies."""
        self._waiting = contextvars.copy_context()
        got = await self._main.get()
        if isinstance(got, BaseException):
            # Every later reader learns it too.
            self._main.put_nowait(got)
            raise got
        return got

    async def _read_frames(self) -> None:
        """Read every frame of the turn until its end, applying those that
        change what the owner holds, queueing the others; watch the executor."""
        try:
            while not self._over:
                # The beat counts the silent intervals: no timer per frame.
                seq, frame = await self._link.receive()
                self._heard = True
                if isinstance(frame, Ephemeral | FrameworkEventFrame):
                    self._publisher.publish_later(frame, self._waiting)
                    continue
                if self._apply(seq, frame):
                    continue
                self._main.put_nowait((seq, frame))
                if isinstance(frame, End):
                    self._ended = True
                    self._finish()
                    return
        except TurnLinkError as lost:
            self._main.put_nowait(lost)
        except Exception as exc:
            logger.exception("Turn %s: its frames could not be read", self._link.request.turn_id)
            broken = TurnLinkError("frame_invalid", f"The turn's frames could not be read: {exc}")
            broken.__cause__ = exc
            self._main.put_nowait(broken)
            await self._send_best_effort(Cancel(reason="frame_invalid"))
        finally:
            # However the reading ends, the turn is over for the owner.
            self._finish()

    def _apply(self, seq: int, frame: Frame) -> bool:
        """Apply a frame that changes what the owner holds or asks it to act;
        whether it was one."""
        if isinstance(frame, Accepted | Heartbeat):
            self._accepted = True
            if isinstance(frame, Accepted):
                self.executor_id = frame.executor_id
                if self._on_accepted is not None:
                    self._on_accepted(frame.executor_id)
        elif isinstance(frame, Record):
            self._copies.record_sent(seq, frame.record)
        elif isinstance(frame, ToolArgs):
            self._copies.args_fixed(frame.start, frame.arguments)
        elif isinstance(frame, ToolEnded):
            self._copies.ended_early(frame.start, self._decoded(decode_end, frame.end))
        elif isinstance(frame, ToolCall | HostCall):
            # The call holds the reentrancy of the task waiting on the turn.
            self._calls.start(frame, self._waiting)
        elif isinstance(frame, CallCancel):
            self._calls.withdraw(frame.call_id)
        elif isinstance(frame, InputRequest):
            self._inputs.opened(self, frame)
        elif isinstance(frame, InputClosed):
            self._inputs.closed(self, frame.request_id)
        elif isinstance(frame, InputTaken):
            self._inputs.taken(self, frame.request_id, frame.taken)
        elif isinstance(frame, SteerResult):
            if (reached := self._steers.get(frame.steer_id)) is not None and not reached.done():
                reached.set_result(frame.loops)
        else:
            return False
        return True

    def _silence_limit(self) -> int:
        """How many silent intervals lose the turn: the accept bound until an
        executor took it, the heartbeat bound after."""
        if self._accepted:
            return self._timeouts.silent_intervals
        return max(1, round(self._timeouts.accept / self._timeouts.heartbeat_interval))

    async def _lost(self) -> None:
        """The executor fell silent: the turn fails, its reader learning why."""
        reason = "executor_lost" if self._link.taken or self._accepted else "no_executor"
        logger.warning("Turn %s failed: %s", self._link.request.turn_id, reason)
        await self._send_best_effort(Cancel(reason=reason))
        self._main.put_nowait(TurnLinkError(reason))
        self._finish()

    # -- Ending ---------------------------------------------------------------

    async def _cancel(self, reason: str) -> None:
        """Cut the turn and wait for its end within the cancel bound; a
        second cancellation does not cut the wait short."""
        if self._abandoned:
            self.reader_stopped()
            return
        if not self._over:
            await self._send_best_effort(Cancel(reason=reason))
        drain = asyncio.get_running_loop().create_task(self._drain_to_end(self._timeouts.cancel))
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

    async def _drain_to_end(self, bound: float | None) -> None:
        """Read the turn's last frames until its end; past *bound*, report
        incomplete drainage and drop the rest."""
        try:
            async with asyncio.timeout(bound):
                try:
                    while not self._over or not self._main.empty():
                        frame = await self._next_main()
                        if isinstance(frame, End):
                            break
                except TurnLinkError:
                    pass
                finally:
                    self._finish()
                # End confirms the executor's cleanup, not the handlers it
                # withdrew on this owner. Their inline tasks end before the
                # caller continues, within the same cancellation budget.
                if bound is not None:
                    await self._calls.drain()
        except TimeoutError:
            self._cleanup_incomplete = True
            logger.warning(
                "Turn %s did not end within %ss of its cancellation: drainage incomplete",
                self._link.request.turn_id,
                bound,
            )
        finally:
            self.reader_stopped()
            self._finish()

    def _finish(self) -> None:
        """The turn is over for the owner: stop beating and reading, cancel
        its calls, answer its steering, drop later frames."""
        if self._over:
            return
        self._over = True
        # What the turn published is still published; nothing is added after.
        self._publisher.close()
        self._beat.cancel()
        if self._pump not in acting_tasks():
            self._pump.cancel()
        self._calls.cancel_all()
        self._inputs.turn_ended(self)
        for reached in self._steers.values():
            if not reached.done():
                reached.set_result(0)
        # A reader still waiting learns the turn is gone.
        self._main.put_nowait(TurnLinkError("backend_closed"))
        self._link.release()

    async def _send_result(self, message: Control) -> None:
        """Answer a call of the turn, while the turn is still the owner's."""
        if not self._over:
            await self._link.send(message)

    async def send_control(self, message: Control) -> None:
        """Send *message* to the turn's executor while the turn is the owner's."""
        await self._send_result(message)

    async def _send_best_effort(self, message: Control) -> None:
        try:
            await self._link.send(message)
        except Exception:
            logger.debug("Could not send %s to turn %s", message.type, self._link.request.turn_id)

    async def _beat_while_open(self) -> None:
        """Tell the executor its owner still holds the turn, each heartbeat
        interval, and lose the turn once the executor was silent for the
        limit's intervals."""
        while not self._over:
            await heartbeat_tick(self._timeouts.heartbeat_interval)
            if self._over:
                return
            # Once an executor took the turn, its beat outside the messages
            # counts too: a turn busy with its CPU still beats.
            heard = self._heard or (self._accepted and self._link.beat_heard())
            self._silent = 0 if heard else self._silent + 1
            self._heard = False
            if self._silent >= self._silence_limit():
                await self._lost()
                return
            await self._send_best_effort(Heartbeat())
