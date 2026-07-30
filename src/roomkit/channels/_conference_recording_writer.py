"""Everything one conference track's recording asks of the recorder.

The frame callback is not the place for any of it. ``ConferenceBackend._emit``
awaits its subscribers one after another, so a container opened, a frame encoded
or a file closed where the frame arrives makes one track's storage latency into
every participant's delivery latency — the lane isolation rule of RFC section
12.10.4, which section 12.10.8 states for the recorder in its own terms.

Queueing the writes alone would not settle it. ``MediaRecorder`` is synchronous
throughout, and opening a recording blocks for as long as the storage takes
exactly as writing to it does: a file created on the delivery path stalls the
conference whether it is created once per track or once per frame, and a
container closed on the event loop blocks that loop just as thoroughly a few
microseconds later. So the whole of one recording's life happens here — the
open, the writes, the close — each on a worker thread, all driven by one task.

That single task is also the promise RFC section 12.11 asks for in return for
leaving the loop's thread: the calls belonging to one handle are ordered and
never overlap. Made structural rather than something to be careful about, since
there is only ever one of them in flight.

The announcement is the exception, and deliberately: it runs integrator code,
and integrator code reached from the task that owns the recording can close the
channel — which comes back here to cancel and await the very task it is running
on. So it is announced on a task of its own, and every wait for it knows how to
recognise a close that came from inside it.

What it costs is that a recording is open a moment after the frame that decided
to record it, and a bound on what may accumulate in between: a track whose
storage cannot keep up loses audio, and the loss is counted where an integrator
can read it rather than left to be discovered in the file.

Every wait here is bounded, including the close. A recorder that will not
finalize cannot be allowed to hold the teardown, because a teardown held open is
a bot left in a conference — and what a bound leaves running is not forgotten
either: it is kept, and the recorder is not released until it returns.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_backlog import TrackBacklog

if TYPE_CHECKING:
    from roomkit.recorder.base import (
        MediaRecorder,
        MediaRecordingConfig,
        MediaRecordingHandle,
        MediaRecordingResult,
        RecordingTrack,
    )

logger = logging.getLogger("roomkit.channels.conference")

_WRITE_FAILURE_LOG_INTERVAL = 100
"""How often a failing write is logged, in failed batches.

A recorder that raises usually raises for every batch after it — the disk is
still full — so a line each would bury the rest of the meeting's log.
"""

_ids = itertools.count()

_announcing: ContextVar[frozenset[int]] = ContextVar(
    "roomkit_conference_recording_announce", default=frozenset()
)
"""Recordings whose opening announcement the current context is running inside.

A close reached from an ``ON_RECORDING_STARTED`` handler — a disclosure policy
that will not be recorded is the realistic one — arrives here while the
announcement is still on the stack, and waiting for it would be waiting for the
caller to finish being the caller. Carried on a ContextVar rather than compared
against the running task for the same reason the conference's activity marker
is: the hook engine dispatches handlers onto tasks of their own, and a task
inherits a copy of the context that created it.
"""


@dataclass(frozen=True)
class QueuedFrame:
    """One frame waiting to be written, timed when it arrived.

    The timestamp is taken in the callback rather than at the write: a
    recording is a timeline, and stamping frames as the writer gets to them
    would encode the writer's own lateness into the media.
    """

    data: bytes
    timestamp_ms: float


class TrackWriter:
    """One track's recording, from the file open to the close, off the loop.

    Constructed with what the recording is to be and started with :meth:`start`,
    which is where the recorder is first asked for anything. The open runs on
    the writer's own task, so the frame that decided to record does not wait for
    a container to be created; frames that arrive meanwhile queue behind it and
    are written in order once it exists.

    Frames enter through :meth:`submit`, which never blocks and never awaits.

    Backpressure: the backlog is bounded and drops its *oldest* frame when it
    is full, which leaves a gap in the recording rather than an ever-growing
    queue in memory and an ever-growing lag behind the conference. Frames a
    failed write lost are counted the same way, and both are readable on
    :attr:`dropped`.
    """

    def __init__(
        self,
        *,
        recorder: MediaRecorder,
        config: MediaRecordingConfig,
        track: RecordingTrack,
        room_id: str,
        max_queued_frames: int,
    ) -> None:
        self._id = next(_ids)
        self._recorder = recorder
        self._config = config
        self._track = track
        self._room_id = room_id
        self._handle: MediaRecordingHandle | None = None
        self._refused = False
        self._backlog: TrackBacklog[QueuedFrame] = TrackBacklog(
            maxsize=max_queued_frames, on_overflow=self._report_overflow
        )
        self._write_failures = 0
        # The open, and the batch currently in a worker thread. Kept because a
        # thread cannot be cancelled: see :meth:`aclose`.
        self._opening: asyncio.Future[MediaRecordingHandle] | None = None
        self._writing: asyncio.Future[None] | None = None
        self._writing_frames = 0
        self._on_opened: Callable[[], Awaitable[None]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._announcement: asyncio.Task[None] | None = None
        # Every call currently executing inside the recorder, whether this
        # object is still waiting for it or gave up on it. A thread runs to
        # completion whatever the task does, so the recorder is busy either way
        # — and it is not released while any of them is (RFC section 12.11).
        # See :meth:`settle` and :attr:`unsettled`.
        self._in_recorder: set[asyncio.Future[Any]] = set()

    def start(self, on_opened: Callable[[], Awaitable[None]] | None = None) -> None:
        """Open the recording, and go on writing whatever is submitted to it.

        ``on_opened`` is awaited once the recorder has accepted the recording,
        and not at all if it never does: it is how the announcement of a
        recording that started gets made from somewhere that can await, which
        the frame callback cannot.
        """
        self._on_opened = on_opened
        self._task = asyncio.create_task(self._run())

    @property
    def handle(self) -> MediaRecordingHandle | None:
        """The recording, once the recorder has opened it."""
        return self._handle

    @property
    def refused(self) -> bool:
        """Whether the recorder would not open this recording at all.

        What tells a caller to stop submitting. The failure belongs to this
        recording for good — the realistic causes, a path that cannot be
        written or a container the codec will not open, do not clear between two
        frames — so it is not retried here. A re-published track is a new
        subscription and gets a new writer.
        """
        return self._refused

    @property
    def dropped(self) -> int:
        """Frames this recording never wrote, evicted or lost to a failed write."""
        return self._backlog.dropped

    @property
    def unsettled(self) -> bool:
        """Whether any call of this writer's is running inside the recorder.

        Running, not abandoned. The distinction was the defect: a finalization
        the framework is still *waiting* for is as much a call inside the
        recorder as one it gave up on, and reading only the abandoned ones let
        a concurrent close conclude that nothing was running and release the
        provider into the middle of it. What a caller needs to know is whether
        the recorder is busy, and by whom does not come into it.

        Which is also what decides whether the writer has to be held on to at
        all: a recording that closed cleanly answers no and is finished with,
        where keeping every one a long meeting ever opened would grow with the
        number of tracks it has seen.
        """
        return any(not call.done() for call in self._in_recorder)

    def _enter_recorder[T](self, call: asyncio.Future[T]) -> asyncio.Future[T]:
        """Note that a call is now executing inside the recorder.

        Every call goes through here, not only the ones a budget later gives up
        on: what makes releasing the provider unsafe is that a call is running,
        and whether this object is still waiting for it does not change that.
        It leaves the set when it returns, which is the only moment it stops
        being true.
        """
        self._in_recorder.add(call)
        call.add_done_callback(self._in_recorder.discard)
        return call

    def submit(self, data: bytes, timestamp_ms: float) -> None:
        """Queue one frame for writing. Never blocks, never awaits."""
        self._backlog.submit(QueuedFrame(data=data, timestamp_ms=timestamp_ms))

    async def drain(self) -> None:
        """Wait until every queued frame has been written or given up on.

        Which is also a wait for the recording to have opened, since nothing is
        taken off the backlog before that.
        """
        await self._backlog.join()

    async def aclose(self, *, timeout: float) -> MediaRecordingResult | None:
        """Write what is queued, then close the recording. Says where it went.

        Flushing first is what keeps a finalized container honest: closing over
        frames that were still in flight ends the recording early and says
        nothing about it. The budget is what keeps a recorder that has stopped
        draining from holding the teardown — and with it the bot — in the
        conference. What the budget could not write is loss, and is counted as
        such.

        A call already handed to a worker thread is the one thing cancelling
        cannot take back: the thread runs to completion whatever the task does.
        So the open and the batch in flight are settled separately, before
        anything else is asked of the same handle — the recorder was promised
        the calls for one recording never overlap (RFC section 12.11).

        Returns ``None`` where there is no recording to report: one the recorder
        refused to open, one it would not close, or one whose open or close was
        still running when the budget ran out.

        Re-entrant, because it has to be: a close reached from an
        ``ON_RECORDING_STARTED`` handler runs while this recording's
        announcement is still on the stack, and the announcement is not
        something such a call can wait for — see :meth:`_settle_announcement`.

        The whole of it is one owned operation rather than a sequence of them.
        Its stages hand the recorder one call after another — the open, the last
        batch, the finalization — and anything watching a *snapshot* of what is
        running can see a stage end, conclude the recorder is idle, and be wrong
        the moment the next one starts. What is true for as long as this runs is
        that this recording is being closed, so that is what is tracked; and
        owning it means a caller that gives up on the wait does not leave the
        stages behind unattributed.
        """
        return await asyncio.shield(
            self._enter_recorder(asyncio.ensure_future(self._aclose(timeout)))
        )

    async def _aclose(self, timeout: float) -> MediaRecordingResult | None:
        """Run the stages of a close, in order. See :meth:`aclose`."""
        await self._settle_announcement(timeout)
        try:
            await asyncio.wait_for(self._backlog.join(), timeout)
        except TimeoutError:
            self._give_up_on_queued(timeout)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._settle_open(timeout)
        await self._settle_in_flight(timeout)
        return await self._finalize(timeout)

    async def settle(self, *, timeout: float) -> bool:
        """Wait until nothing of this writer's is running in the recorder.

        What :meth:`aclose` could not wait for is still running inside the
        recorder, on a thread nothing can cancel. The recording is finished as
        far as the conference is concerned — the bot has left, the teardown is
        over — but the *provider* must not be released while a call is still in
        it, which for a native muxer is a crash rather than an error (RFC
        section 12.11). So this is the wait that belongs to closing the
        recorder, not to closing a recording, and it is bounded like every other
        one here: what it does not cover is named, and the recorder is left
        unreleased rather than released underneath it.

        Quiescence rather than a snapshot, and that is the whole of it. What is
        in flight changes *because* something finished: a close moves from the
        open to the finalization, so waiting for the set as it stood and
        returning is how a caller concludes the recorder is idle at the exact
        moment the next call enters it. So it is re-read after every wait, until
        there is nothing left or the budget is gone.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            pending = [call for call in self._in_recorder if not call.done()]
            if not pending:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.error(
                    "%d call(s) to the recorder for track %s have not returned %.1fs after "
                    "the recording was closed. The recorder is not being released: a "
                    "provider freed while a call is running in it is a crash rather than an "
                    "error",
                    len(pending),
                    self._track.id,
                    timeout,
                )
                return False
            await asyncio.wait(pending, timeout=remaining)

    # -------------------------------------------------------------------------
    # The writer's own task
    # -------------------------------------------------------------------------

    async def _run(self) -> None:
        """Open the recording, then write what arrives in the order it arrived.

        The announcement is started rather than awaited. It runs integrator
        code, and this task owns the recording: a handler that closes the
        channel would come back to cancel and await the task it is standing on,
        which is a task awaiting itself. Off to the side, the writing also
        carries on rather than queueing behind whatever the handler does.
        """
        if not await self._open():
            return
        self._announce()
        while True:
            batch = [await self._backlog.get()]
            batch.extend(self._backlog.take_ready())
            self._writing_frames = len(batch)
            self._writing = self._enter_recorder(
                asyncio.ensure_future(asyncio.to_thread(self._write, batch))
            )
            try:
                await asyncio.shield(self._writing)
            except Exception:
                # The batch is lost with the frame that failed — a recorder
                # that raised is not in a state to be told the rest of it — but
                # the writer is not: a disk that empties, a codec that refuses
                # one frame, and the recording goes on.
                self._backlog.discard(len(batch))
                self._report_write_failure(len(batch))
            finally:
                # On cancellation too: a flush waiting on join() must not be
                # left waiting for a batch nothing will report.
                self._backlog.task_done(len(batch))

    async def _open(self) -> bool:
        """Ask the recorder to open the recording. Says whether there is one.

        Shielded, so a close arriving mid-open does not leave a thread creating
        a container this object has stopped holding: the task ends, the future
        does not, and :meth:`aclose` collects the handle from it.
        """
        self._opening = self._enter_recorder(
            asyncio.ensure_future(asyncio.to_thread(self._open_blocking))
        )
        try:
            self._handle = await asyncio.shield(self._opening)
        except Exception:
            self._refuse()
            return False
        return True

    def _open_blocking(self) -> MediaRecordingHandle:
        """Open the recording and declare its track. Runs on a worker thread."""
        handle = self._recorder.on_recording_start(self._config)
        handle.room_id = self._room_id
        try:
            self._recorder.on_track_added(handle, self._track)
        except Exception:
            # The recorder accepted the recording and then refused the track it
            # was opened for. Left alone it is a container held open for a track
            # that will never arrive, and closing it here is what keeps that
            # call on the same thread as the open it undoes.
            self._close_unused(handle)
            raise
        return handle

    def _close_unused(self, handle: MediaRecordingHandle) -> None:
        """Close a recording that was opened and never carried its track."""
        try:
            self._recorder.on_recording_stop(handle)
        except Exception:
            logger.exception(
                "Could not close conference recording %s, opened for track %s and never used",
                handle.id,
                self._track.id,
            )

    def _refuse(self) -> None:
        """Give up on a recording the recorder would not open.

        The frames queued for it are dropped without being counted as loss: a
        recording that never existed has no hole in it, and reporting one would
        put a number against a file an integrator will not find.
        """
        self._refused = True
        pending = self._backlog.take_ready()
        self._backlog.task_done(len(pending))
        logger.exception(
            "Could not open a conference recording for track %s (participant=%s, room=%s). "
            "The track is not being recorded; it is still being transcribed",
            self._track.id,
            self._track.participant_id,
            self._room_id,
        )

    def _announce(self) -> None:
        """Start saying that the recording is open, on a task of its own."""
        if self._on_opened is None:
            return
        self._announcement = asyncio.create_task(self._announcing())

    async def _announcing(self) -> None:
        """Say that the recording is open, without letting that end the writing.

        The callback runs integrator code, and an exception from it belongs to
        the announcement rather than to the recording: a hook that raised is no
        reason to stop writing the file it was told about.

        Marked for the whole of it, so a close this announcement drives can tell
        that it is downstream of the thing it would otherwise wait for.
        """
        if self._on_opened is None:  # pragma: no cover — checked by the caller
            return
        token = _announcing.set(_announcing.get() | {self._id})
        try:
            await self._on_opened()
        except Exception:
            logger.exception(
                "Error announcing conference recording %s (track=%s)",
                self._handle_id(),
                self._track.id,
            )
        finally:
            _announcing.reset(token)

    async def _settle_announcement(self, timeout: float) -> None:
        """Let the opening announcement finish before the recording is closed.

        So that a recording is reported as started before it is reported as
        stopped, which is the order the two events describe and the only one an
        integrator accumulating them by track can act on.

        Except where this call *is* that announcement's own doing. A handler
        that closes the channel is ordinary code — a disclosure policy that will
        not be recorded ends the meeting — and waiting there would be waiting
        for the caller to finish being the caller. It goes ahead instead, and
        says so: the start and the end of this one recording arrive in the order
        the handler chose rather than the order they happened.
        """
        announcement = self._announcement
        if announcement is None or announcement.done():
            return
        if self._id in _announcing.get():
            logger.info(
                "Conference recording %s (track=%s) is being closed from inside its own "
                "opening announcement; the announcement cannot be waited for here, so its "
                "end may be reported before its start",
                self._handle_id(),
                self._track.id,
            )
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(announcement), timeout)

    def _write(self, batch: list[QueuedFrame]) -> None:
        """Hand a batch to the recorder. Runs on a worker thread."""
        handle = self._handle
        if handle is None:  # pragma: no cover — the loop starts after the open
            return
        for frame in batch:
            self._recorder.on_data(handle, self._track, frame.data, frame.timestamp_ms)

    # -------------------------------------------------------------------------
    # Closing
    # -------------------------------------------------------------------------

    def _give_up_on_queued(self, timeout: float) -> None:
        """Count and announce the frames the flush budget did not reach."""
        pending = len(self._backlog.take_ready())
        # Reported done as well as counted lost. Taking frames out without
        # reporting them leaves the backlog permanently mid-flight, and the
        # next thing to wait on it — a drain, a second close — waits for a
        # batch that no longer exists.
        self._backlog.task_done(pending)
        self._backlog.discard(pending)
        logger.warning(
            "Conference recording %s (track=%s) did not finish writing within %.1fs; "
            "%d queued frame(s) are lost and the recording is being closed without them",
            self._handle_id(),
            self._track.id,
            timeout,
            pending,
        )

    async def _settle_open(self, timeout: float) -> None:
        """Wait for an open still running in a worker thread, and take its handle.

        The recording cannot be closed before it exists, so this is waited for
        rather than raced — but on the same budget as everything else here,
        because a teardown held open is a bot left in a conference and that is
        the worse of the two failures. What the budget does not cover is handled
        by :meth:`_abandon_open` rather than by finalizing something that is
        still being created, which would be the second overlapping call RFC
        section 12.11 does not admit.
        """
        if self._opening is None or self._handle is not None or self._refused:
            return
        try:
            self._handle = await asyncio.wait_for(asyncio.shield(self._opening), timeout)
        except TimeoutError:
            self._abandon_open(timeout)
        except Exception:
            self._refuse()

    def _abandon_open(self, timeout: float) -> None:
        """Stop waiting for an open that has not returned, and close it later.

        A thread cannot be cancelled, so the container is still being created
        and something still has to close it. Not this call: it is what a
        teardown is waiting on. So the closing is put on a task of its own,
        which :meth:`settle` is the wait for — the recorder is not released
        while that is still going — and the recording is reported as one with no
        result, because nothing here knows where it was written.
        """
        self._enter_recorder(asyncio.ensure_future(self._close_when_opened()))
        logger.error(
            "Opening the conference recording of track %s is still running after %.1fs. The "
            "teardown is going ahead without it: where it was written is not being reported, "
            "and it is closed as soon as the recorder finishes opening it",
            self._track.id,
            timeout,
        )

    async def _close_when_opened(self) -> None:
        """Close a recording whose open outlasted the teardown that gave up on it."""
        assert self._opening is not None  # noqa: S101 — only reached from _abandon_open
        try:
            handle = await asyncio.shield(self._opening)
        except Exception:
            return
        await asyncio.to_thread(self._close_unused, handle)

    async def _settle_in_flight(self, timeout: float) -> None:
        """Wait for a write already running in a worker thread to return."""
        if self._writing is None or self._writing.done():
            return
        try:
            # Shielded, so the wait does not cancel what it is waiting for —
            # cancelling would return here while the thread carried on writing,
            # which is the overlap this exists to prevent.
            await asyncio.wait_for(asyncio.shield(self._writing), timeout)
        except TimeoutError:
            # The one overlapping call RFC section 12.11 admits, and the report
            # it obliges: a write past its budget cannot be taken back, and
            # waiting for it is a bot left in the conference. Kept, so that the
            # recorder is not released while it runs.
            # Already tracked, and it stays tracked: giving up on the wait does
            # not take the call out of the recorder.
            logger.error(
                "A write to conference recording %s (track=%s) is still running after %.1fs. "
                "The recording is being closed while it runs: a teardown held open is a bot "
                "left in the conference, which is the worse of the two",
                self._handle_id(),
                self._track.id,
                timeout,
            )
        except Exception:
            # Counted here rather than in the run loop: the loop was cancelled
            # at the await that would have caught this, so nothing else is
            # going to notice these frames never landed.
            self._backlog.discard(self._writing_frames)
            logger.warning(
                "The last %d frame(s) of conference recording %s (track=%s) failed to write "
                "as it was closing; the recording is finalized without them",
                self._writing_frames,
                self._handle_id(),
                self._track.id,
            )

    async def _finalize(self, timeout: float) -> MediaRecordingResult | None:
        """Flush the track and close the container, on a worker thread.

        Bounded, like every other wait here, and for the reason that outranks
        the rest: the teardown waits for this before the bot leaves the
        conference, so a muxer that will not finalize — a codec draining a
        deadlocked encoder, a network filesystem that stopped answering — keeps
        the bot sitting in the meeting for as long as it takes. There is no
        length of time for which that is the better failure.

        What the budget gives up on is not forgotten. The thread runs on, so the
        call is kept and :meth:`settle` waits for it before the recorder is
        released; where the recording went goes unreported, because the recorder
        never got as far as saying.
        """
        handle, self._handle = self._handle, None
        if handle is None:
            return None
        closing = self._enter_recorder(
            asyncio.ensure_future(asyncio.to_thread(self._finalize_blocking, handle))
        )
        try:
            return await asyncio.wait_for(asyncio.shield(closing), timeout)
        except TimeoutError:
            # Already tracked, and it stays tracked until it returns.
            logger.error(
                "Closing conference recording %s (track=%s) is still running after %.1fs. The "
                "teardown is going ahead without it: where it was written is not being "
                "reported, and the recorder is not released until the call returns",
                handle.id,
                self._track.id,
                timeout,
            )
            return None

    def _finalize_blocking(self, handle: MediaRecordingHandle) -> MediaRecordingResult | None:
        """End the track and close the recording. Runs on a worker thread."""
        try:
            self._recorder.on_track_removed(handle, self._track)
        except Exception:
            logger.exception(
                "Could not flush track %s of conference recording %s; closing the recording "
                "anyway",
                self._track.id,
                handle.id,
            )
        try:
            return self._recorder.on_recording_stop(handle)
        except Exception:
            logger.exception(
                "Could not close conference recording %s (track=%s, room=%s). Where it was "
                "written is not being announced, because the recorder never said",
                handle.id,
                self._track.id,
                self._room_id,
            )
            return None

    # -------------------------------------------------------------------------
    # Saying what happened
    # -------------------------------------------------------------------------

    def _handle_id(self) -> str:
        """The recording's id, or what to call it before it has one."""
        return self._handle.id if self._handle is not None else f"opening for {self._track.id}"

    def _report_overflow(self, dropped: int) -> None:
        logger.warning(
            "Conference recording %s is behind: dropped %d frame(s) of track %s",
            self._handle_id(),
            dropped,
            self._track.id,
        )

    def _report_write_failure(self, frames: int) -> None:
        """Say a write failed, on the first and then rarely."""
        self._write_failures += 1
        if self._write_failures % _WRITE_FAILURE_LOG_INTERVAL == 1:
            logger.exception(
                "Could not write %d frame(s) to conference recording %s (track=%s); this is "
                "failure %d, the recording continues with that audio missing",
                frames,
                self._handle_id(),
                self._track.id,
                self._write_failures,
            )
