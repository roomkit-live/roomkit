"""A teardown whose steps fail — RFC §12.10.4 step 5, §12.10.7, §17.7.

The teardown used to be one happy sequence: the first exception skipped
everything behind it, and the `finally` declared the bot gone regardless. A
`leave()` the SFU refused therefore left the bot sitting in the meeting while
`info()` reported the conference unattended — a disclosure problem (§17.7 wants
the bot's presence observable *at any time*) before it is an operational one.
An `on_recording_stop` that raised on the first track reached even further: the
remaining recordings stayed open and the `leave()` behind them never ran, so a
full disk was enough to strand the bot.

What these tests hold down is that the destructive steps are independent, that
the room keeps saying what is true — the bot is still in there, and why — and
that `conference_ended` is only ever announced for a departure that happened.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    ConferenceCloseError,
    ConferenceRecordingConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import BotSession, ConferenceGrants
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.recorder.base import MediaRecordingHandle, MediaRecordingResult, RecordingTrack
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conference.lane_audio import drain_recordings, say, speech_frame
from tests.conference.test_conference_races import _settle, _until

ROOM = "room-1"


class SFUHeldTheBotError(RuntimeError):
    """What an SFU that will not release the bot raises through the backend.

    A control call that timed out, a session the SFU no longer recognises and a
    gateway that is simply down all look the same from the channel: `leave()`
    raises and the bot is, as far as anything here knows, still in the meeting.
    """


class _Ended:
    """The `conference_ended` announcements a kit made, in order."""

    def __init__(self, kit: RoomKit) -> None:
        self.sessions: list[str] = []

        @kit.on("conference_ended")
        async def _record(event: Any) -> None:
            self.sessions.append(event.data["bot_session_id"])


async def _conference(
    *,
    backend: MockConferenceBackend | None = None,
    **channel_kwargs: Any,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = backend or MockConferenceBackend()
    # A need is what arms the lazy join (RMK-75). Departures presuppose a
    # session, so a recognizer stands in unless the test brought a need of
    # its own.
    if not {"stt", "tts", "recording"} & channel_kwargs.keys():
        channel_kwargs["stt"] = MockSTTProvider()
    channel = ConferenceChannel("conf", backend=backend, **channel_kwargs)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


async def _with_a_bot_in_it(
    **channel_kwargs: Any,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend, BotSession]:
    """A conference the bot has joined, on the first participant's arrival."""
    kit, channel, backend = await _conference(**channel_kwargs)
    await backend.simulate_participant_joined(ROOM, "p-alice")
    bot = channel._room(ROOM).bot
    assert bot is not None, "the bot never joined, so there is no departure to test"
    return kit, channel, backend, bot


def _room_info(channel: ConferenceChannel) -> dict[str, Any]:
    return channel.info()["rooms"][ROOM]


class TestALeaveTheSFURefuses:
    """The state the ghost bot came from: `leave()` raises on detach."""

    async def test_the_bot_is_still_reported_present(self) -> None:
        """`forget_leaving` ran in a `finally`, so the room forgot a session it
        had not managed to remove. `bot_present` is what a disclosure obligation
        is answered with, and it said no while the bot was listening."""
        _, channel, backend, bot = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("the SFU will not release the session"))

        await channel.on_room_detached(ROOM)

        info = _room_info(channel)
        assert info["bot_present"] is True
        assert info["leaving_session_ids"] == [bot.id]
        assert info["detaching"] is True

    async def test_the_reason_is_reported_with_it(self) -> None:
        """Present is half the answer. Nothing else says a leave was refused, so
        `info()` carries what went wrong rather than only that something did."""
        _, channel, backend, bot = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("the SFU will not release the session"))

        await channel.on_room_detached(ROOM)

        assert _room_info(channel)["leave_failed"] == {
            bot.id: "SFUHeldTheBotError: the SFU will not release the session"
        }

    async def test_the_bot_really_is_still_in_the_conference(self) -> None:
        """What the report above is a report *of*: the mock keeps the session
        because the call that would have removed it failed."""
        _, channel, backend, bot = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert [session.id for session in backend.bots] == [bot.id]

    async def test_no_end_is_announced(self) -> None:
        """RFC §12.10.7 puts `conference_ended` at `leave()` completing. It did
        not complete, and an end announced here asserts a departure that never
        happened."""
        kit, channel, backend, _ = await _with_a_bot_in_it()
        ended = _Ended(kit)
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert ended.sessions == []

    async def test_a_normal_leave_still_ends_the_conference(self) -> None:
        """The other side of the rule, so the guard above cannot be satisfied by
        never announcing anything."""
        kit, channel, _, bot = await _with_a_bot_in_it()
        ended = _Ended(kit)

        await channel.on_room_detached(ROOM)

        assert ended.sessions == [bot.id]
        assert channel.info()["rooms"] == {}

    async def test_the_detach_itself_still_completes(self) -> None:
        """`on_room_detached` is awaited by `detach_channel` between removing the
        binding and announcing the detach — an exception here would cost the room
        its ON_CHANNEL_DETACHED and leave the framework's own bookkeeping half
        done."""
        kit, _, backend, _ = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("held"))
        detached: list[RoomEvent] = []

        @kit.hook(HookTrigger.ON_CHANNEL_DETACHED, execution=HookExecution.ASYNC)
        async def _seen(event: RoomEvent, context: RoomContext) -> None:
            detached.append(event)

        assert await kit.detach_channel(ROOM, "conf") is True
        assert len(detached) == 1
        assert await kit.store.get_binding(ROOM, "conf") is None


class TestTheOtherStepsStillHappen:
    """A failing step is one step. The rest of the teardown owes the room the
    same cleanup it always did."""

    async def test_the_lanes_are_closed(self) -> None:
        _, channel, backend, _ = await _with_a_bot_in_it(stt=MockSTTProvider())
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())
        assert channel.active_lanes != {}
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert channel.active_lanes == {}

    async def test_the_recordings_are_finalized(self) -> None:
        recorder = MockMediaRecorder()
        _, channel, backend, _ = await _with_a_bot_in_it(
            recorder=recorder, recording=ConferenceRecordingConfig()
        )
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert [result.id for result in recorder.results] == [
            handle.id for handle in recorder.handles
        ]

    async def test_a_lane_that_will_not_close_does_not_keep_the_bot_in(self) -> None:
        """The order the sequence ran in put the lanes in front of the bot's
        departure, so a stage that raised on close stranded the bot behind it."""
        _, channel, backend, bot = await _with_a_bot_in_it(stt=MockSTTProvider())
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())

        async def _refuse() -> None:
            raise RuntimeError("the pipeline stage will not release its state")

        channel.active_lanes[track.id].aclose = _refuse  # type: ignore[method-assign]

        await channel.on_room_detached(ROOM)

        assert backend.bots == [], "a lane that would not close kept the bot in the conference"
        assert channel.info()["rooms"] == {}
        assert bot.id not in channel._room(ROOM).leaving


class TestAnUnsubscribeTheSFURefuses:
    """A track stops being consumed on two paths — it was unpublished, or the
    binding stopped permitting collection — and both ask the backend to stop
    delivering before closing what consumed it.

    That order is right, and only the second half is the channel's to
    guarantee. The subscription is forgotten first, so a failing
    ``unsubscribe_track`` that skipped the teardown behind it left the lane, its
    pipeline stage state and the track's recording alive with nothing able to
    find them again — the room no longer listed the subscription they were
    reached through, so the detach could not close them either.
    """

    async def test_an_unpublished_track_still_loses_its_lane(self) -> None:
        _, channel, backend, _ = await _with_a_bot_in_it(stt=MockSTTProvider())
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())
        assert channel.active_lanes != {}
        backend.fail("unsubscribe_track", SFUHeldTheBotError("the SFU will not unsubscribe"))

        await backend.simulate_track_unpublished(track.id)

        assert channel.active_lanes == {}

    async def test_an_unpublished_track_is_still_announced(self) -> None:
        """RFC §12.10.4 step 4 asks for all three of unsubscribe, teardown and
        hooks. The track is gone whatever the backend makes of being told.
        """
        kit, _, backend, _ = await _with_a_bot_in_it(stt=MockSTTProvider())
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())
        fired: list[str] = []

        @kit.hook(
            HookTrigger.ON_CONFERENCE_TRACK_UNPUBLISHED,
            execution=HookExecution.ASYNC,
            name="observer",
        )
        async def _observe(event: RoomEvent, context: RoomContext) -> None:
            fired.append(event.content.data["track_id"])  # type: ignore[union-attr]

        backend.fail("unsubscribe_track", SFUHeldTheBotError("held"))
        await backend.simulate_track_unpublished(track.id)

        assert fired == [track.id]

    async def test_a_closed_binding_still_loses_every_lane(self) -> None:
        """The other path, where one refusal used to take the rest of the list
        with it: the remaining tracks were never unsubscribed and none of them
        were torn down.
        """
        kit, channel, backend, _ = await _with_a_bot_in_it(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())
        assert set(channel.active_lanes) == {alice.id, bob.id}
        backend.fail("unsubscribe_track", SFUHeldTheBotError("held"))

        await kit.mute(ROOM, "conf")
        await _until(lambda: channel.active_lanes == {})

        assert channel.active_lanes == {}
        assert channel._room(ROOM).subscribed == {}


class _RecorderWithAFullDisk(MockMediaRecorder):
    """A recorder that cannot close one named participant's recording.

    The realistic shape of it: the media is written frame by frame while the
    meeting runs, and the failure lands on the call that finalizes a container
    at the end — one recording's problem, not the conference's.

    Named rather than counted, because a room's recordings are finalized
    concurrently: which handle reaches ``on_recording_stop`` first is not the
    framework's to promise (RFC §12.11 orders calls per handle and says nothing
    across them), so "the n-th one fails" would be a test asserting on the
    scheduler.
    """

    def __init__(self, *, fails_for: str) -> None:
        super().__init__()
        self._fails_for = fails_for
        self._doomed: set[str] = set()

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        # Where the participant is knowable: `on_recording_stop` is handed a
        # handle and nothing else, and this runs immediately before it.
        if track.participant_id == self._fails_for:
            self._doomed.add(handle.id)
        super().on_track_removed(handle, track)

    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        if handle.id in self._doomed:
            raise OSError("no space left on device")
        return super().on_recording_stop(handle)


class TestARecordingThatWillNotClose:
    """`finish()` was a list comprehension: the first exception left every
    recording behind it open and skipped the `leave()` that followed."""

    async def test_the_other_tracks_are_still_finalized(self) -> None:
        recorder = _RecorderWithAFullDisk(fails_for="p-alice")
        _, channel, backend, _ = await _with_a_bot_in_it(
            recorder=recorder, recording=ConferenceRecordingConfig()
        )
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())
        await drain_recordings(channel)
        assert len(recorder.handles) == 2

        await channel.on_room_detached(ROOM)

        assert len(recorder.results) == 1, "the second recording was never closed"

    async def test_the_bot_still_leaves(self) -> None:
        """The consequence that mattered: a disk that filled at the end of a
        meeting left the framework's bot sitting in it."""
        kit, channel, backend, bot = await _with_a_bot_in_it(
            recorder=_RecorderWithAFullDisk(fails_for="p-alice"),
            recording=ConferenceRecordingConfig(),
        )
        ended = _Ended(kit)
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())

        await channel.on_room_detached(ROOM)

        assert backend.bots == []
        assert ended.sessions == [bot.id]

    async def test_what_could_be_closed_is_still_announced(self) -> None:
        """A recording nothing managed to close has no location to report, so it
        is dropped rather than announced with whatever the recorder last said."""
        recorder = _RecorderWithAFullDisk(fails_for="p-alice")
        kit, channel, backend, _ = await _with_a_bot_in_it(
            recorder=recorder, recording=ConferenceRecordingConfig()
        )
        stopped: list[str] = []

        @kit.on("recording_stopped")
        async def _seen(event: Any) -> None:
            stopped.append(event.data["track_id"])

        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())

        await channel.on_room_detached(ROOM)

        assert stopped == [bob.id]


class TestClosingTheRoomEvictsTheBot:
    """`close_room_on_detach` is the integrator asking for the conference to be
    destroyed. RFC §12.10.4 step 5 makes it a MUST "whether or not a bot ever
    joined" — and it is the one compensation a refused `leave()` has."""

    async def test_the_room_is_destroyed_even_when_the_bot_could_not_leave(self) -> None:
        _, channel, backend, _ = await _with_a_bot_in_it(close_room_on_detach=True)
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert ROOM not in backend.rooms

    async def test_the_session_is_settled_by_the_eviction(self) -> None:
        """Destroying the room is blunt and thorough: the conference no longer
        exists, so nobody is in it. That is when the departure became true."""
        kit, channel, backend, bot = await _with_a_bot_in_it(close_room_on_detach=True)
        ended = _Ended(kit)
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)

        assert channel.info()["rooms"] == {}
        assert ended.sessions == [bot.id], "the end was never announced for an evicted bot"

    async def test_a_room_that_cannot_be_closed_either_keeps_the_session(self) -> None:
        """No compensation, no announcement. What is left is a bot an operator
        has to remove, and the room saying so."""
        kit, channel, backend, bot = await _with_a_bot_in_it(close_room_on_detach=True)
        ended = _Ended(kit)
        backend.fail("leave", SFUHeldTheBotError("held"))
        backend.fail("close_room", SFUHeldTheBotError("the SFU is unreachable"))

        await channel.on_room_detached(ROOM)

        assert ended.sessions == []
        assert _room_info(channel)["leave_failed"] == {bot.id: "SFUHeldTheBotError: held"}


class TestClosingTheChannel:
    """A close is the last moment anything can be done about a stuck session —
    and it used to be the moment the channel forgot it (`leaving.clear()`)."""

    async def test_the_close_takes_out_a_session_the_detach_could_not(self) -> None:
        kit, channel, backend, bot = await _with_a_bot_in_it()
        ended = _Ended(kit)
        backend.fail("leave", SFUHeldTheBotError("held"), times=1)

        await channel.on_room_detached(ROOM)
        assert backend.bots != []

        await channel.close()

        assert backend.bots == []
        assert channel.info()["rooms"] == {}
        assert ended.sessions == [bot.id], "the end a detach owed the session never arrived"

    async def test_a_session_that_stays_stuck_is_not_forgotten(self) -> None:
        """The bug the retry cannot fix, and must not paper over: a closed
        channel that reports the meeting unattended is how a bot goes on
        listening to a room nobody is watching. And a close that could not
        remove a session is a close that failed — it says so with an
        exception naming the session, not with a log the caller never reads.
        """
        _, channel, backend, bot = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("held"))

        await channel.on_room_detached(ROOM)
        with pytest.raises(ConferenceCloseError) as failure:
            await channel.close()

        assert bot.id in str(failure.value)
        info = _room_info(channel)
        assert info["bot_present"] is True
        assert info["leave_failed"] == {bot.id: "SFUHeldTheBotError: held"}
        assert [session.id for session in backend.bots] == [bot.id]

    async def test_the_close_finishes_whatever_the_leave_did(self) -> None:
        """The backend still has to be closed: a channel that stopped halfway
        through its own close leaks the transport as well as the bot. The
        failure is raised at the very end, once every step has run."""
        _, channel, backend, _ = await _with_a_bot_in_it()
        backend.fail("leave", SFUHeldTheBotError("held"))

        with pytest.raises(ConferenceCloseError):
            await channel.close()

        assert [call.method for call in backend.calls if call.method == "close"] == ["close"]

    async def test_a_live_bot_gets_no_end_from_a_close(self) -> None:
        """Closing a channel is not a detach and has never announced ends for
        the conferences it was still in. The retry above is a detach's own
        announcement arriving late, not a new one."""
        kit, channel, backend, _ = await _with_a_bot_in_it()
        ended = _Ended(kit)

        await channel.close()

        assert backend.bots == []
        assert ended.sessions == []


class _SlowJoinBackend(MockConferenceBackend):
    """Holds a join open. ``joining`` is set once it is actually suspended."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.joining = asyncio.Event()

    async def join_as_bot(
        self, room_id: str, identity: str, grants: ConferenceGrants
    ) -> BotSession:
        self.joining.set()
        await self.gate.wait()
        return await super().join_as_bot(room_id, identity, grants)


class TestAJoinAbandonedMidFlight:
    """A join the detach overtook leaves the session with the SFU and nowhere in
    the room's books — so a `leave()` that fails there stranded a bot no part of
    the channel could account for."""

    async def _abandon_a_join(
        self, kit: RoomKit, channel: ConferenceChannel, backend: _SlowJoinBackend
    ) -> None:
        """Detach while ``join_as_bot`` is suspended, and let both finish.

        Both on their own tasks: the join holds the room's lock and the detach
        queues on it, so a detach awaited inline would be waiting for a join
        that is waiting for this test.
        """
        generation = channel._room(ROOM).generation
        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await _until(lambda: channel._room(ROOM).generation != generation)
        backend.gate.set()
        await asyncio.gather(joining, detaching)
        await _settle(channel)

    async def test_the_abandoned_session_is_reported(self) -> None:
        backend = _SlowJoinBackend()
        kit, channel, _ = await _conference(backend=backend)
        backend.fail("leave", SFUHeldTheBotError("held"))

        await self._abandon_a_join(kit, channel, backend)

        assert backend.bots != [], "there is no stranded session, so nothing to report"
        info = _room_info(channel)
        assert info["bot_present"] is True
        assert list(info["leave_failed"]) == [backend.bots[0].id]

    async def test_it_is_never_announced_as_an_end(self) -> None:
        """It was never announced as a start either: the detach got to the
        generation check before `conference_started` did. So the close takes the
        session out, and says nothing about a conference nobody was told about.
        """
        backend = _SlowJoinBackend()
        kit, channel, _ = await _conference(backend=backend)
        ended = _Ended(kit)
        backend.fail("leave", SFUHeldTheBotError("held"), times=1)

        await self._abandon_a_join(kit, channel, backend)
        await channel.close()

        assert backend.bots == [], "the close never retried the stranded session"
        assert ended.sessions == []


class TestADeferredTeardown:
    """The same guarantees when the destroying half runs on its own task —
    a detach from inside a `conference_started` handler."""

    async def test_the_session_survives_a_failed_leave(self) -> None:
        kit, channel, backend = await _conference()
        ended = _Ended(kit)

        @kit.on("conference_started")
        async def _detach_from_inside(event: Any) -> None:
            backend.fail("leave", SFUHeldTheBotError("held"))
            await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _settle(channel)

        info = _room_info(channel)
        assert info["bot_present"] is True
        assert list(info["leave_failed"]) == [backend.bots[0].id]
        assert ended.sessions == []

    async def test_the_lanes_are_still_closed(self) -> None:
        kit, channel, backend = await _conference(stt=MockSTTProvider())
        detached = asyncio.Event()

        @kit.on("conference_started")
        async def _detach_from_inside(event: Any) -> None:
            backend.fail("leave", SFUHeldTheBotError("held"))
            await kit.detach_channel(ROOM, "conf")
            detached.set()

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track, speech=1, silence=1)
        await asyncio.wait_for(detached.wait(), timeout=5.0)
        await _settle(channel)

        assert channel.active_lanes == {}
