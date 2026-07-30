"""What the conference channel promises across an await it does not control.

Every defect here is the same shape: the channel reads the world, awaits
something it cannot cancel — a backend's network call, an integrator's hook, a
synthesizer's next chunk — and then acts on what it read before. The suite
passed without these because nothing in it ever detached at the moment the
channel was mid-step; each test below is that moment, held open on purpose.

They are separated by *what* moved underneath the await: the room, the track,
the lane the callback is running on, the attachment itself, the bot's own voice,
and the roster.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from roomkit import MockConferenceBackend, RoomKit
from roomkit.channels import _conference_activity as activity_module
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceInterruptionConfig
from roomkit.core.exceptions import (
    ConferenceCloseError,
    ParticipantNotAdmittedError,
    RoomKitError,
    RoomNotAttachedError,
)
from roomkit.core.locks import InMemoryLockManager
from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    ParticipantStatus,
)
from roomkit.models.hook import HookResult
from roomkit.models.identity import IdentityResult
from roomkit.models.participant import Participant
from roomkit.store.memory import InMemoryStore
from roomkit.voice.base import AudioChunk
from roomkit.voice.interruption import InterruptionStrategy
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider
from tests.conference.lane_audio import say, speech_frame

ROOM = "room-1"
OTHER = "room-2"


async def _until(predicate, *, timeout: float = 5.0) -> None:
    """Wait until a predicate holds, rather than towards when it might.

    A conference race test has to run the channel up to a precise moment and
    hold it there. Sleeping towards that moment makes the test a guess about
    how fast the machine is: on a loaded CI it can pass without the window ever
    opening, which is the one outcome worse than failing.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0)


async def _settle(channel: ConferenceChannel) -> None:
    """Wait for a detach that was deferred onto its own task.

    A detach triggered from inside an announcement cannot finish inline, so it
    returns and destroys afterwards. Tests that assert on what the backend was
    told have to wait for that, and waiting on the tasks themselves keeps them
    off sleeps that a loaded machine can outrun.
    """
    while channel._teardowns:
        await asyncio.wait(list(channel._teardowns), timeout=5.0)


async def _kit(
    backend: MockConferenceBackend | None = None,
    *,
    rooms: tuple[str, ...] = (ROOM,),
    resolver: IdentityResolver | None = None,
    **kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = backend or MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, **kwargs)  # type: ignore[arg-type]
    kit = RoomKit(identity_resolver=resolver)
    kit.register_channel(channel)
    for room_id in rooms:
        await kit.create_room(room_id)
        await kit.attach_channel(room_id, "conf")
    return kit, channel, backend


class _SlowJoinBackend(MockConferenceBackend):
    """Holds a join open. ``joining`` is set once it is actually suspended."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.joining = asyncio.Event()
        self.slow_room: str | None = None

    async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
        if room_id == self.slow_room:
            self.joining.set()
            await self.gate.wait()
        return await super().join_as_bot(room_id, identity, grants)


class _CountingResolver(IdentityResolver):
    """Answers nothing, and counts how many times it was asked."""

    def __init__(self) -> None:
        self.asked = 0

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        self.asked += 1
        return IdentityResult(status=IdentificationStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# The room moves: a detach lands while the join is being announced
# ---------------------------------------------------------------------------


class TestDetachDuringJoin:
    """``_ensure_bot`` releases the join lock before announcing the conference,
    so integrator code cannot hold every other room's joins behind it. A detach
    takes the lock the moment it is free — which puts it exactly between the
    bot being registered and the conference being announced.
    """

    async def test_the_start_is_never_announced_after_the_end(self) -> None:
        kit, _, backend = await _kit()
        events: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            events.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _detach_mid_join(event: object, ctx: object) -> None:
            await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _until(lambda: "conference_ended" in events)

        conference = [e for e in events if e.startswith("conference_")]
        assert conference.index("conference_started") < conference.index("conference_ended")

    async def test_no_participant_is_left_active_on_a_detached_room(self) -> None:
        """The roster is written after the bot is in, and bringing the bot in
        runs integrator code. A record written after the detach outlives the
        channel's presence and makes the roster lie to everything reading it.
        """
        kit, channel, backend = await _kit()

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _detach_mid_join(event: object, ctx: object) -> None:
            await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _until(lambda: not channel._room(ROOM).attached)
        await _settle(channel)

        assert await kit.store.list_participants(ROOM) == []

    async def _detach_inside_the_join(
        self,
        kit: RoomKit,
        channel: ConferenceChannel,
        backend: _SlowJoinBackend,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Detach while ``join_as_bot`` is suspended, and let both finish."""
        backend.slow_room = ROOM
        generation = channel._room(ROOM).generation
        joining = asyncio.create_task(
            backend.simulate_participant_joined(ROOM, "p-alice", metadata=metadata)
        )
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        # The generation is bumped before the detach queues on the join lock, so
        # this is the moment the join is holding a room that has moved on.
        await _until(lambda: channel._room(ROOM).generation != generation)
        backend.gate.set()
        await asyncio.gather(joining, detaching)
        await _settle(channel)

    async def test_a_detach_inside_the_backend_call_records_nothing(self) -> None:
        """An arrival is recorded even when the bot could not be brought in —
        but not when the reason is that this channel has left the conference the
        arrival belongs to. Both arrive as an exception out of the join, and only
        the second may write nothing.
        """
        backend = _SlowJoinBackend()
        kit, channel, _ = await _kit(backend)

        await self._detach_inside_the_join(kit, channel, backend)

        assert await kit.store.list_participants(ROOM) == []
        assert backend.bots == []

    async def test_a_detach_inside_the_backend_call_is_not_a_failed_join(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Which is what the two have to be told apart *for*. Not writing is
        settled twice over — the transaction re-reads the generation as well —
        so what a channel that could not tell them apart would get wrong is
        everything either side of the write: an ordinary detach reported as an
        SFU failure, with a stack trace, and a resolver asked to identify an
        arrival into a conference the channel is no longer in.
        """
        backend = _SlowJoinBackend()
        resolver = _CountingResolver()
        kit, channel, _ = await _kit(backend, resolver=resolver)

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await self._detach_inside_the_join(
                kit, channel, backend, metadata={"sip.phoneNumber": "+15551234"}
            )

        assert [r for r in caplog.records if r.exc_info] == []
        assert resolver.asked == 0

    async def test_an_observer_never_finishes_the_end_before_the_start(self) -> None:
        """Checking the generation before emitting does not settle this: the
        emission is itself an await into integrator code, and a detach landing
        inside it lets the end reach observers while the start is still being
        delivered. What settles it is the teardown waiting.
        """
        kit, channel, backend = await _kit()
        observed: list[str] = []
        gate = asyncio.Event()
        entered = asyncio.Event()

        @kit.on("conference_started")
        async def _slow_start(event: object) -> None:
            entered.set()
            await gate.wait()
            observed.append("started")

        @kit.on("conference_ended")
        async def _end(event: object) -> None:
            observed.append("ended")

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(joining, detaching, return_exceptions=True)
        await _settle(channel)

        assert observed == ["started", "ended"]

    async def test_a_started_handler_may_detach_the_channel(self) -> None:
        """The teardown waits for the announcement — except when the
        announcement is what triggered it. An integrator detaching from a
        ``conference_started`` handler is ordinary code, and must not hang.
        """
        kit, channel, backend = await _kit()
        seen: list[str] = []

        @kit.on("conference_started")
        async def _detach_from_within(event: object) -> None:
            seen.append("started")
            await kit.detach_channel(ROOM, "conf")
            seen.append("detached")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)

        assert seen == ["started", "detached"]
        assert not channel._room(ROOM).attached

    async def test_a_reentrant_detach_finishes_once_the_announcement_does(self) -> None:
        """Deferred, not skipped. The detach returns while it is still nested
        inside the announcement — it cannot wait there — and the destroying
        happens the moment that announcement is done.
        """
        kit, channel, backend = await _kit()

        @kit.on("conference_started")
        async def _detach_from_within(event: object) -> None:
            await kit.detach_channel(ROOM, "conf")
            # Still inside the announcement: the bot is out of the channel's
            # books, but has not left the conference yet.
            assert backend.bots != []

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await _settle(channel)

        assert backend.bots == []

    async def test_a_reentrant_detach_does_not_overtake_the_announcement(self) -> None:
        """Skipping the wait to avoid the deadlock would put the end in front of
        the announcement's remaining observers — the inversion the drain exists
        to prevent, reintroduced by the escape from it.
        """
        kit, channel, backend = await _kit()
        seen: list[str] = []

        @kit.on("conference_started")
        async def _first(event: object) -> None:
            seen.append("started-detacher")
            await kit.detach_channel(ROOM, "conf")

        @kit.on("conference_started")
        async def _second(event: object) -> None:
            seen.append("started-observer")

        @kit.on("conference_ended")
        async def _ended(event: object) -> None:
            seen.append("ended")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await _settle(channel)

        assert seen == ["started-detacher", "started-observer", "ended"]

    async def test_no_session_hook_fires_for_a_conference_already_over(self) -> None:
        """ON_SESSION_STARTED hands out the bot session. Firing it after the bot
        has left gives a greeting somewhere to speak that no longer exists.
        """
        kit, channel, backend = await _kit()
        fired: list[str] = []

        @kit.on("conference_started")
        async def _detach_from_within(event: object) -> None:
            await kit.detach_channel(ROOM, "conf")

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _session(event: object, ctx: object) -> None:
            fired.append("session")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await _settle(channel)

        assert fired == []

    async def test_no_session_hook_fires_once_the_end_has_been_announced(self) -> None:
        """The check above stands where the room is read, and the dispatch is an
        await of its own. A detach landing between the two announced the end and
        then handed a greeting the session that had just left with it.
        """
        kit, channel, backend = await _kit()
        order: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            order.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        entered = asyncio.Event()
        gate = asyncio.Event()
        fire = channel._fire_session_started

        async def held(room_id: str, bot: object) -> None:
            entered.set()
            await gate.wait()
            await fire(room_id, bot)  # type: ignore[arg-type]

        channel._fire_session_started = held  # type: ignore[method-assign, assignment]

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _session(event: object, ctx: object) -> None:
            order.append("session")

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(joining, detaching, return_exceptions=True)
        await _settle(channel)

        assert "conference_ended" in order
        if "session" in order:
            assert order.index("session") < order.index("conference_ended"), order

    async def test_the_arrival_is_recorded_and_announced_as_one_step(self) -> None:
        """The roster write, the hook and the event are three awaits. A detach
        landing between them announced a participant joining a conference
        observers had already been told ended.
        """
        kit, _, backend = await _kit()
        events: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            events.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        gate = asyncio.Event()
        entered = asyncio.Event()
        original_add = kit.store.add_participant

        async def slow_add(participant: object) -> object:
            entered.set()
            await gate.wait()
            return await original_add(participant)  # type: ignore[arg-type]

        kit.store.add_participant = slow_add  # type: ignore[method-assign, assignment]

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(joining, detaching, return_exceptions=True)
        kit.store.add_participant = original_add  # type: ignore[method-assign]

        conference = [e for e in events if e.startswith("conference_")]
        assert conference.index("conference_participant_joined") < conference.index(
            "conference_ended"
        )

    async def test_the_bot_does_not_stay_behind(self) -> None:
        kit, channel, backend = await _kit()

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _detach_mid_join(event: object, ctx: object) -> None:
            await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await _until(lambda: not channel._room(ROOM).attached)
        await _settle(channel)

        assert backend.bots == []
        assert channel.info()["rooms"] == {}


# ---------------------------------------------------------------------------
# One room's slow join must not decide when another room may connect
# ---------------------------------------------------------------------------


class TestPerRoomJoinLock:
    async def test_a_slow_room_does_not_block_another_room(self) -> None:
        """The join lock is held across the backend's network call. One lock for
        the whole channel would make a single unresponsive conference stall
        every other room's joins behind it.
        """
        backend = _SlowJoinBackend()
        kit, _, _ = await _kit(backend, rooms=(ROOM, OTHER))
        backend.slow_room = ROOM

        stuck = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)
        assert backend.bots == []

        await asyncio.wait_for(
            backend.simulate_participant_joined(OTHER, "p-bob"),
            timeout=1.0,
        )
        assert [bot.room_id for bot in backend.bots] == [OTHER]

        backend.gate.set()
        await stuck
        assert sorted(bot.room_id for bot in backend.bots) == [ROOM, OTHER]
        del kit

    async def test_reattaching_does_not_mint_a_second_lock(self) -> None:
        """The lock outlives the attachment, like the generation beside it.

        Dropping a room's lock on detach would let a re-attach create a new one
        while a join is still queued on the old — two joins for one room, each
        holding a different lock, publishing the AI on two tracks.
        """
        kit, channel, backend = await _kit()
        first = channel._room(ROOM).lock

        await kit.detach_channel(ROOM, "conf")
        await kit.attach_channel(ROOM, "conf")

        assert channel._room(ROOM).lock is first

    async def test_reattaching_still_brings_a_single_bot_in(self) -> None:
        kit, _, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")

        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-bob")

        assert len(backend.bots) == 1


# ---------------------------------------------------------------------------
# The track moves: it is unpublished while the subscription is in flight
# ---------------------------------------------------------------------------


class _SlowSubscribeBackend(MockConferenceBackend):
    """Holds a subscription open. ``subscribing`` marks the open window."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.subscribing = asyncio.Event()
        self.slow_track: str | None = None

    async def subscribe_track(self, bot, track_id):  # type: ignore[no-untyped-def]
        if track_id == self.slow_track:
            self.subscribing.set()
            await self.gate.wait()
        return await super().subscribe_track(bot, track_id)


class TestUnpublishDuringSubscribe:
    """The unpublish callback can only close a lane that already exists, so a
    track that goes away while its subscription is in flight would otherwise
    leave behind a subscription and a lane for something nobody is publishing —
    and a lane owns a task and pipeline stage state nobody will release.
    """

    @pytest.fixture
    async def raced(self) -> tuple[ConferenceChannel, MockConferenceBackend]:
        backend = _SlowSubscribeBackend()
        _, channel, _ = await _kit(backend, stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        backend.slow_track = "tr-slow"

        published = asyncio.create_task(
            backend.simulate_track_published(ROOM, "p-alice", track_id="tr-slow")
        )
        await asyncio.wait_for(backend.subscribing.wait(), timeout=5.0)
        await backend.simulate_track_unpublished("tr-slow")
        backend.gate.set()
        await published
        return channel, backend

    async def test_no_lane_is_left_for_a_track_that_went_away(
        self, raced: tuple[ConferenceChannel, MockConferenceBackend]
    ) -> None:
        channel, _ = raced

        assert "tr-slow" not in channel.active_lanes

    async def test_the_subscription_is_undone(
        self, raced: tuple[ConferenceChannel, MockConferenceBackend]
    ) -> None:
        _, backend = raced

        assert "tr-slow" not in backend.subscriptions

    async def test_an_unpublish_during_the_publish_hooks_is_not_overtaken(self) -> None:
        """The subscription is not the callback's first await. It runs the
        track-published hooks and brings the bot in first, and reading the
        track's generation only at the subscription would adopt an unpublish
        that landed in either of those as the starting state.
        """
        backend = MockConferenceBackend()
        kit, channel, _ = await _kit(backend, stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        gate = asyncio.Event()
        entered = asyncio.Event()

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_PUBLISHED, execution=HookExecution.ASYNC)
        async def _slow_hook(event: object, ctx: object) -> None:
            entered.set()
            await gate.wait()

        publishing = asyncio.create_task(
            backend.simulate_track_published(ROOM, "p-alice", track_id="tr-x")
        )
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await backend.simulate_track_unpublished("tr-x")
        gate.set()
        await publishing

        assert "tr-x" not in channel.active_lanes
        assert "tr-x" not in backend.subscriptions

    async def test_a_track_that_stays_still_gets_its_lane(self) -> None:
        """The guard must not cost the ordinary case its lane."""
        backend = _SlowSubscribeBackend()
        _, channel, _ = await _kit(backend, stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")

        track = await backend.simulate_track_published(ROOM, "p-alice")

        assert track.id in channel.active_lanes
        assert track.id in backend.subscriptions


# ---------------------------------------------------------------------------
# The lane's own task: a detach triggered by the work it would destroy
# ---------------------------------------------------------------------------


class TestDetachFromInsideALane:
    """A lane's callbacks run integrator code on the lane's own task, and a
    handler that detaches the channel is ordinary — a keyword that ends the
    meeting, a policy that stops it when someone talks over the bot.

    The teardown that came of it closed the lane it was standing on: it
    cancelled the task, then awaited it, so the cancellation reached the gather
    the teardown was running inside and took the teardown down with it. What
    the suite saw was a hang and a ``RecursionError`` out of ``Task.cancel``;
    what the conference saw was a bot nobody left.
    """

    async def test_a_transcription_handler_may_detach_the_channel(self) -> None:
        kit, channel, backend = await _kit(stt=MockSTTProvider(transcripts=["au revoir"]))
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def _detach_on_keyword(payload: object, ctx: object) -> None:
            seen.append("hook")
            await kit.detach_channel(ROOM, "conf")
            seen.append("detached")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await asyncio.wait_for(_until(lambda: seen == ["hook", "detached"]), timeout=5.0)
        await _settle(channel)

        assert backend.bots == []
        assert channel.info()["rooms"] == {}

    async def test_the_transcript_does_not_arrive_after_the_end(self) -> None:
        """The detach returns before its destructive half, so the handler
        carries on and the text it just allowed goes on to be delivered. Into a
        room the channel has left, and after observers were told the conference
        was over.
        """
        kit, channel, backend = await _kit(stt=MockSTTProvider(transcripts=["au revoir"]))

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def _detach_on_keyword(payload: object, ctx: object) -> None:
            await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await asyncio.wait_for(_until(lambda: not channel._room(ROOM).attached), timeout=5.0)
        await _settle(channel)

        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "au revoir"] == []

    async def test_a_barge_in_handler_may_detach_the_channel(self) -> None:
        """The same graph of dependencies, reached from the other callback: the
        lane reports speech, the bot is interrupted, and ON_BARGE_IN runs on
        that same task.
        """
        kit, channel, backend = await _kit(
            stt=MockSTTProvider(transcripts=["stop"]),
            tts=_ChunkedTTS(40),
            interruption=ConferenceInterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE),
        )
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
        async def _detach_on_barge_in(payload: object, ctx: object) -> None:
            seen.append("barge-in")
            await kit.detach_channel(ROOM, "conf")
            seen.append("detached")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        speaking = asyncio.create_task(channel._voice.speak(ROOM, "je parle encore"))
        await _until(lambda: len(backend.published_audio) >= 1)
        await say(backend, track)
        await asyncio.wait_for(_until(lambda: seen == ["barge-in", "detached"]), timeout=5.0)
        await asyncio.gather(speaking, return_exceptions=True)
        await _settle(channel)

        assert backend.bots == []


# ---------------------------------------------------------------------------
# The attachment moves: a re-attach lands inside a deferred detach
# ---------------------------------------------------------------------------


class _SlowLeaveBackend(MockConferenceBackend):
    """Holds the bot's departure open, which is where a teardown spends its
    last await — and so the window a re-attach lands in.

    ``hold_first_only`` holds one departure and lets the rest through, which is
    how a room comes to be leaving twice over: the first teardown is still
    inside ``leave()`` when a second one runs to completion behind it.
    """

    def __init__(self, *, hold_first_only: bool = False) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.leaving = asyncio.Event()
        self.slow = False
        self.hold_first_only = hold_first_only
        self.held: str | None = None

    async def leave(self, bot):  # type: ignore[no-untyped-def]
        if self.hold_first_only and self.held is None:
            self.held = bot.id
            self.leaving.set()
            await self.gate.wait()
        elif self.slow:
            self.leaving.set()
            await self.gate.wait()
        return await super().leave(bot)


class TestReattachDuringTeardown:
    """A deferred detach returns before it has destroyed anything, so an
    integrator can re-attach in the gap — ``detach()`` then ``attach()`` are two
    calls that both appear to have completed.

    What lands afterwards belongs to the attachment it closed. Left unbound to
    it, the teardown announced an end after the next start, reset the track
    generations the new attachment was already using, and — with
    ``close_room_on_detach`` — deleted the conference room the re-attach had
    just created, leaving the channel attached to nothing.
    """

    async def test_the_new_conference_survives_the_old_detach(self) -> None:
        """The re-entrant case: detach and re-attach from one handler. The
        teardown cannot be made to wait for a re-attach nested inside the very
        announcement it is waiting for, so what protects the new attachment is
        the generation the teardown carries, not the wait.
        """
        kit, channel, backend = await _kit(close_room_on_detach=True)
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _cycle(event: object) -> None:
            if done.is_set():
                return
            await kit.detach_channel(ROOM, "conf")
            await kit.attach_channel(ROOM, "conf")
            done.set()

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await _settle(channel)

        assert [rid for rid, room in channel._rooms.items() if room.attached] == [ROOM]
        assert ROOM in backend.rooms, "the old teardown destroyed the new conference room"

    async def test_the_end_names_the_session_that_ended(self) -> None:
        """Where the wait would be circular the end genuinely does arrive after
        the next start, and no ordering can be promised there. What can be is
        that an observer is able to tell *which* conference ended: an
        unattributed `conference_ended` between two starts is unreadable, and
        the start it belongs to has been naming its session all along.
        """
        backend = _SlowLeaveBackend()
        kit, channel, _ = await _kit(backend)
        started: list[str] = []
        ended: list[str] = []
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _cycle(event) -> None:  # type: ignore[no-untyped-def]
            started.append(event.data["bot_session_id"])
            if done.is_set():
                return
            backend.slow = True
            await kit.detach_channel(ROOM, "conf")
            await kit.attach_channel(ROOM, "conf")
            done.set()
            await backend.simulate_participant_joined(ROOM, "p-bob")

        @kit.on("conference_ended")
        async def _end(event) -> None:  # type: ignore[no-untyped-def]
            ended.append(event.data.get("bot_session_id"))

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)
        backend.gate.set()
        await _settle(channel)

        assert len(started) == 2, started
        assert ended == [started[0]], "the end does not say which conference it ended"

    async def test_the_current_bot_is_the_one_reported(self) -> None:
        """``info()`` answers "is this meeting being transcribed, and by whom".
        Reading the leaving session first made it answer for a bot on its way
        out while a live one was sitting in the same conference — and report the
        room as detaching when it had just been attached.
        """
        backend = _SlowLeaveBackend()
        kit, channel, _ = await _kit(backend)
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _cycle(event: object) -> None:
            if done.is_set():
                return
            backend.slow = True
            await kit.detach_channel(ROOM, "conf")
            await kit.attach_channel(ROOM, "conf")
            done.set()
            await backend.simulate_participant_joined(ROOM, "p-bob")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)

        current = channel._room(ROOM).bot
        assert current is not None
        room_info = channel.info()["rooms"][ROOM]

        assert room_info["bot_session_id"] == current.id
        assert room_info["detaching"] is False, "a room that is attached is not detaching"
        assert room_info["leaving_session_ids"] == sorted(channel._room(ROOM).leaving)

        backend.gate.set()
        await _settle(channel)

    async def test_a_room_leaving_twice_over_reports_both(self) -> None:
        """One departure recorded per room lost the older of two. A teardown
        held open in ``leave()`` is still running when a re-attach brings a
        second bot in and a second detach sends it out behind the first: the
        second overwrote the first, then removed the entry on its way out, and
        ``info()`` dropped a room with a bot still sitting in the conference.
        """
        backend = _SlowLeaveBackend(hold_first_only=True)
        kit, channel, _ = await _kit(backend)
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _cycle(event: object) -> None:
            if done.is_set():
                return
            await kit.detach_channel(ROOM, "conf")
            await kit.attach_channel(ROOM, "conf")
            done.set()
            await backend.simulate_participant_joined(ROOM, "p-bob")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)
        await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=5.0)

        assert backend.bots != [], "the held departure was let through after all"
        room_info = channel.info()["rooms"].get(ROOM)
        assert room_info is not None, "the room vanished with a bot still in the conference"
        assert room_info["bot_present"] is True
        assert [bot.id for bot in backend.bots] == room_info["leaving_session_ids"]

        backend.gate.set()
        await _settle(channel)

        assert backend.bots == []
        assert channel.info()["rooms"] == {}

    async def test_a_departing_session_is_not_reported_as_the_current_bot(self) -> None:
        """``bot_session_id`` names the bot an integrator would act on. Falling
        back to the departing session made an ordinary detach report the same id
        under both keys, and the field then meant nothing.
        """
        backend = _SlowLeaveBackend()
        kit, channel, _ = await _kit(backend)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        backend.slow = True
        leaving_id = backend.bots[0].id

        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)
        room_info = channel.info()["rooms"][ROOM]

        assert room_info["bot_session_id"] is None
        assert room_info["leaving_session_ids"] == [leaving_id]
        assert room_info["bot_present"] is True, "a bot still in the conference is present"
        assert room_info["detaching"] is True

        backend.gate.set()
        await asyncio.wait_for(detaching, timeout=5.0)

    async def test_the_end_is_announced_before_the_next_start(self) -> None:
        """And when the re-attach is *not* nested inside it, the attachment
        waits: a conference that ended and one that started are reported in the
        order they happened, rather than interleaved by which teardown task the
        loop got to first.
        """
        backend = _SlowLeaveBackend()
        kit, channel, _ = await _kit(backend)
        order: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            if name.startswith("conference_start") or name == "conference_ended":
                order.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        @kit.on("conference_started")
        async def _detach_once(event: object) -> None:
            if len(order) > 1:
                return
            backend.slow = True
            await kit.detach_channel(ROOM, "conf")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)

        async def reattach() -> None:
            await kit.attach_channel(ROOM, "conf")
            await backend.simulate_participant_joined(ROOM, "p-bob")

        reattaching = asyncio.create_task(reattach())
        # The teardown is held inside `leave()`, so nothing can advance it. Any
        # progress the re-attach makes here it makes by overtaking a detach that
        # has not finished.
        for _ in range(50):
            await asyncio.sleep(0)
        assert not channel._room(ROOM).attached, "the re-attach overtook the previous detach"

        backend.gate.set()
        await asyncio.wait_for(reattaching, timeout=5.0)
        await _settle(channel)

        assert order == ["conference_started", "conference_ended", "conference_started"]

    async def test_the_new_attachment_keeps_its_track_generations(self) -> None:
        """The track generations are keyed by room, and the teardown cleared
        them by name. A subscription in flight for the *new* attachment then
        compared itself against a counter that had been reset underneath it.
        """
        kit, channel, backend = await _kit(stt=MockSTTProvider())
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _cycle(event: object) -> None:
            if done.is_set():
                return
            await kit.detach_channel(ROOM, "conf")
            await kit.attach_channel(ROOM, "conf")
            done.set()

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_track_unpublished(track.id)
        epochs = dict(channel._room(ROOM).track_epochs)
        await _settle(channel)

        assert channel._room(ROOM).track_epochs == epochs


class _HeldCloseRoomBackend(MockConferenceBackend):
    """Holds ``close_room`` open, so a teardown can be caught inside it.

    The window the drain cannot cover: a teardown that has read the generation,
    found it current, and is now in the backend destroying the conference room.
    Everything before that point the next attach can be made to wait for; this
    is the part where waiting has a deadline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closing = asyncio.Event()
        self.gate = asyncio.Event()

    async def close_room(self, room_id: str) -> None:
        self.closing.set()
        await self.gate.wait()
        await super().close_room(room_id)


class TestReattachWhileTheRoomIsBeingDestroyed:
    """The one window a deferred teardown leaves open, and the one the drain
    cannot close: the old ``close_room`` is *already running* when the next
    attach gives up waiting for it.

    Read outside the create/destroy lock, the generation is read before the
    re-attach creates the room and acted on afterwards — so the teardown
    destroyed a conference that was live, with its participants in it, and left
    a channel attached to nothing. Serialising the two is what makes the answer
    usable: either the destroy finishes and the create follows it, or the create
    goes first and the destroy sees a generation that is no longer its own.
    """

    async def _detached_from_inside_an_announcement(
        self, backend: _HeldCloseRoomBackend
    ) -> tuple[RoomKit, ConferenceChannel]:
        """Attach, join, and detach from the announcement — a deferred teardown."""
        kit, channel, _ = await _kit(backend, close_room_on_detach=True)

        @kit.on("conference_started")
        async def _detach_once(event: object) -> None:
            if not channel._room(ROOM).attached:
                return
            await kit.detach_channel(ROOM, "conf")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        await asyncio.wait_for(backend.closing.wait(), timeout=5.0)
        return kit, channel

    async def test_the_attach_is_refused_rather_than_handed_a_doomed_room(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conference created here is one the running ``close_room`` destroys
        the moment it returns. There is no answer that is both immediate and
        true, so the attach says so instead of returning a binding to a room
        that is about to stop existing.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _HeldCloseRoomBackend()
        kit, channel = await self._detached_from_inside_an_announcement(backend)

        with pytest.raises(RoomNotAttachedError):
            await asyncio.wait_for(kit.attach_channel(ROOM, "conf"), timeout=5.0)

        assert await kit.list_bindings(ROOM) == []
        assert not channel._room(ROOM).attached

        backend.gate.set()
        await _settle(channel)

        # The invariant the whole thing is about: what the channel says it is
        # attached to and what exists on the SFU do not disagree. The defect
        # left `attached` true against a room the old teardown had deleted.
        assert not channel._room(ROOM).attached
        assert ROOM not in backend.rooms

    async def test_the_room_is_created_again_once_the_destroy_has_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And the refusal is not the end of it: the ordering is destroy, then
        create, and an attach that arrives after the teardown has let go gets a
        conference that lasts.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _HeldCloseRoomBackend()
        kit, channel = await self._detached_from_inside_an_announcement(backend)
        backend.gate.set()
        await _settle(channel)

        await asyncio.wait_for(kit.attach_channel(ROOM, "conf"), timeout=5.0)

        assert ROOM in backend.rooms
        assert channel._room(ROOM).attached

    async def test_a_teardown_that_arrives_late_leaves_the_new_room_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other order. The re-attach gets the lock first, so the teardown
        reads the generation on the far side of a conference that has since been
        created — and does not destroy it.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _SlowLeaveBackend()
        kit, channel, _ = await _kit(backend, close_room_on_detach=True)

        @kit.on("conference_started")
        async def _detach_once(event: object) -> None:
            if not channel._room(ROOM).attached:
                return
            backend.slow = True
            await kit.detach_channel(ROOM, "conf")

        await asyncio.wait_for(backend.simulate_participant_joined(ROOM, "p-alice"), timeout=5.0)
        # Held in `leave()`, which is ahead of the destroy: the re-attach gives
        # up waiting for the teardown and creates the room while the old one is
        # still on its way to closing it.
        await asyncio.wait_for(backend.leaving.wait(), timeout=5.0)
        await asyncio.wait_for(kit.attach_channel(ROOM, "conf"), timeout=5.0)
        backend.gate.set()
        await _settle(channel)

        assert ROOM in backend.rooms, "the old teardown destroyed the new conference room"
        assert channel._room(ROOM).attached


# ---------------------------------------------------------------------------
# A credential outlives the check that allowed it
# ---------------------------------------------------------------------------


class _ShieldedMintBackend(MockConferenceBackend):
    """Shields its mint, so cancelling the call does not stop it minting.

    An SDK that runs its request under ``asyncio.shield`` — or on a connection
    pool task of its own — behaves exactly like this: the caller is released
    with a CancelledError while the credential goes on being created.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.minting = asyncio.Event()
        self.issued: list[str] = []

    async def mint_access(self, room_id, participant_id, grants):  # type: ignore[no-untyped-def]
        return await asyncio.shield(self._mint(room_id, participant_id, grants))

    async def _mint(self, room_id, participant_id, grants):  # type: ignore[no-untyped-def]
        self.minting.set()
        await self.gate.wait()
        access = await super().mint_access(room_id, participant_id, grants)
        self.issued.append(participant_id)
        return access


class TestMintDuringDetach:
    """``mint_access`` reads the room, consults the roster, and then mints. The
    roster is an await, and a detach landing in it left the check describing a
    room the channel had already left.

    It is the one refusal here that cannot be made good afterwards: a token has
    been handed out, the SFU will honour it, and there is no revocation in the
    backend contract. So it is not enough to notice — the mint has to be
    something the detach cannot land in the middle of, and cannot simply
    outlive either.
    """

    @staticmethod
    def _hold(target: object, name: str, *, nth: int = 1) -> tuple[asyncio.Event, asyncio.Event]:
        """Suspend one call on the way in, and hand back its two signals.

        ``nth`` picks which call to hold, for the methods a single operation
        makes more than once: ``mint_access`` consults the roster twice — once
        to decide whether to start, once to decide whether to hand the answer
        on — and holding the first says nothing about the second.
        """
        entered = asyncio.Event()
        gate = asyncio.Event()
        original = getattr(target, name)
        calls = 0

        async def held(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls != nth:
                return await original(*args, **kwargs)
            entered.set()
            await gate.wait()
            return await original(*args, **kwargs)

        setattr(target, name, held)
        return entered, gate

    async def test_no_credential_is_minted_for_a_room_the_channel_is_leaving(self) -> None:
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        entered, gate = self._hold(channel._roster, "standing")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        # Released on the detach having closed admission, not on a turn of the
        # loop: the window this is about is the one that opens exactly there,
        # and a sleep(0) only happens to land in it.
        await _until(lambda: not channel._room(ROOM).attached)
        gate.set()
        minted, _ = await asyncio.gather(minting, detaching, return_exceptions=True)

        assert isinstance(minted, RoomNotAttachedError), f"minted anyway: {minted!r}"
        assert not [call for call in backend.calls if call.method == "mint_access"]

    async def test_a_detach_does_not_overtake_a_mint_in_flight(self) -> None:
        """The other half of the same guarantee. A mint that has passed its
        check is work describing a live conference, so the teardown waits for it
        rather than leaving the conference under a credential it has just
        issued.
        """
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        order: list[str] = []

        entered = asyncio.Event()
        gate = asyncio.Event()
        original = backend.mint_access

        async def held(room_id, participant_id, grants):  # type: ignore[no-untyped-def]
            entered.set()
            await gate.wait()
            order.append("minted")
            return await original(room_id, participant_id, grants)

        backend.mint_access = held  # type: ignore[method-assign, assignment]

        @kit.on("conference_ended")
        async def _ended(event: object) -> None:
            order.append("ended")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await _until(lambda: not channel._room(ROOM).attached)
        gate.set()
        await asyncio.gather(minting, detaching, return_exceptions=True)
        await _settle(channel)

        assert order == ["minted", "ended"]

    async def test_a_mint_that_outlasts_the_drain_is_taken_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The drain is bounded — it waits on code the channel does not own — and
        everything else it protects degrades gracefully past the deadline: a
        chunk lands late, an event arrives out of order. A credential does not
        degrade. It is valid for as long as it says it is, against a conference
        the framework has left.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        entered, gate = self._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=5.0)

        assert backend.bots == [], "the detach did not go ahead"

        gate.set()
        minted = (await asyncio.gather(minting, return_exceptions=True))[0]

        assert isinstance(minted, RoomNotAttachedError), f"minted anyway: {minted!r}"

    async def test_a_mint_that_shrugged_off_the_cancellation_still_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cancelling reaches backends that let cancellation through. One that
        shields its network call finishes and mints, and the framework never
        learns of it — so the refusal must not claim the credential was never
        issued, and the warning that tells an operator where to look for it must
        be emitted on this ending too, not only on the one where the answer came
        back.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _ShieldedMintBackend()
        kit, channel, _ = await _kit(backend)
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(backend.minting.wait(), timeout=5.0)
        with caplog.at_level(logging.WARNING, logger="roomkit.channels.conference"):
            await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=5.0)
            minted = (await asyncio.gather(minting, return_exceptions=True))[0]

        backend.gate.set()
        await _until(lambda: backend.issued != [])

        assert isinstance(minted, RoomNotAttachedError), f"minted anyway: {minted!r}"
        assert "no credential was issued" not in str(minted), str(minted)
        assert "may have issued one" in str(minted), str(minted)
        warned = [r for r in caplog.records if "revoking" in r.getMessage()]
        assert warned, "a credential the backend may hold went unreported"

    async def test_a_mint_that_outlasts_a_close_is_taken_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``close()`` shuts the backend on the same bounded budget, so the same
        request would come back against a backend that is gone — and hand out
        its credential regardless.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        entered, gate = self._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await asyncio.wait_for(channel.close(), timeout=5.0)

        gate.set()
        minted = (await asyncio.gather(minting, return_exceptions=True))[0]

        assert isinstance(minted, RoomNotAttachedError), f"minted anyway: {minted!r}"


class TestBanDuringMint:
    """The other precondition a mint outlives, and the one whose whole purpose
    is to be raced: a ban.

    The room is re-read when the backend answers, and the participant was not —
    so ``remove_member(..., BANNED)`` landing inside ``mint_access()`` described
    the roster as it was before the ban and the credential was handed over
    anyway. Banning someone is exactly what an operator does *while* that
    someone is trying to get in, and the SFU honours what it minted.
    """

    async def test_a_ban_that_lands_mid_mint_withholds_the_credential(self) -> None:
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        entered, gate = TestMintDuringDetach._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-mallory"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)
        gate.set()
        minted = (await asyncio.gather(minting, return_exceptions=True))[0]

        assert isinstance(minted, ParticipantNotAdmittedError), f"minted anyway: {minted!r}"

    async def test_the_room_is_told_a_credential_may_exist_anyway(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Withholding is not revocation and does not claim to be: the backend
        answered, so a credential may be on the SFU's books that only an
        operator can take back.
        """
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        entered, gate = TestMintDuringDetach._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-mallory"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        with caplog.at_level(logging.WARNING, logger="roomkit.channels.conference"):
            gate.set()
            await asyncio.gather(minting, return_exceptions=True)

        assert "may have issued one" in caplog.text
        assert "p-mallory" in caplog.text

    async def test_a_ban_cannot_commit_inside_the_admission_decision(self) -> None:
        """Two reads cannot make this atomic, so the decision joins the queue.

        A store answers about the moment the query reached it: a ban committed
        while the read was in flight comes back as ``ACTIVE``, and re-reading
        only moves the window. What closes it is being in the same queue as the
        writer — the framework's per-room lock, which ``remove_member()`` holds
        while it writes.

        Which is what this holds down, from the writer's side: while the
        decision is being made, a ban cannot complete. So it lands either
        wholly before it — and is seen — or wholly after, against a credential
        that was legitimate when it was issued.
        """
        kit, channel, _ = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        # The second lookup is the decision's own; the first only decides
        # whether there is anything to mint.
        entered, gate = TestMintDuringDetach._hold(channel._roster, "standing", nth=2)

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        banning = asyncio.create_task(
            kit.remove_member(ROOM, "p-alice", status=ParticipantStatus.BANNED)
        )
        # Nothing here can advance the decision, so any progress the ban makes
        # it makes by overtaking it.
        for _ in range(50):
            await asyncio.sleep(0)
        assert not banning.done(), "the ban landed inside the admission decision"

        gate.set()
        minted = await asyncio.wait_for(minting, timeout=5.0)
        await asyncio.wait_for(banning, timeout=5.0)

        assert minted.token
        assert (await kit.store.get_participant(ROOM, "p-alice")).status is (  # type: ignore[union-attr]
            ParticipantStatus.BANNED
        )

    async def test_a_ban_the_decision_waited_behind_is_seen(self) -> None:
        """The other order, and the one that matters: the ban got the lock
        first, so the decision reads a roster it has already been written to.
        """
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        entered, gate = TestMintDuringDetach._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-mallory"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)
        gate.set()
        minted = (await asyncio.gather(minting, return_exceptions=True))[0]

        assert isinstance(minted, ParticipantNotAdmittedError), f"minted anyway: {minted!r}"

    async def test_a_participant_left_mid_mint_is_still_admitted(self) -> None:
        """The re-read is a status check, not a freeze: leaving is not a
        refusal, and a dropped connection asking for a second credential is the
        ordinary case.
        """
        kit, channel, backend = await _kit()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        entered, gate = TestMintDuringDetach._hold(backend, "mint_access")

        minting = asyncio.create_task(channel.mint_access(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await kit.remove_member(ROOM, "p-alice")
        gate.set()

        assert (await asyncio.wait_for(minting, timeout=5.0)).token


# ---------------------------------------------------------------------------
# The bot's own voice: a detach lands mid-utterance
# ---------------------------------------------------------------------------


class _SlowPublishBackend(MockConferenceBackend):
    """Holds a publication open, so a detach can land inside one.

    ``publishing`` is set once the loop is actually inside the call, so a test
    can wait for the window rather than sleep towards it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.publishing = asyncio.Event()
        self.slow = False

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        if self.slow:
            self.publishing.set()
            await self.gate.wait()
        return await super().publish_audio(bot, chunk)


class _ChunkedTTS(TTSProvider):
    """Synthesizes in several chunks, awaiting between them — which is the
    window a detach lands in.
    """

    def __init__(self, chunks: int = 4) -> None:
        self._chunks = chunks

    @property
    def name(self) -> str:
        return "chunked"

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        return AudioChunk(data=b"\x00" * 320, format="pcm_s16le", sample_rate=16000)

    async def synthesize_stream(self, text: str, voice: str | None = None):  # type: ignore[no-untyped-def]
        for i in range(self._chunks):
            await asyncio.sleep(0.01)
            yield AudioChunk(
                data=b"\x00" * 320,
                format="pcm_s16le",
                sample_rate=16000,
                is_final=i == self._chunks - 1,
            )

    async def close(self) -> None:
        return None


class TestPlaybackOutlivesTheRoom:
    async def test_detach_stops_the_bot_speaking(self) -> None:
        """Dropping the playback entry does not stop anything on its own: the
        synthesis loop holds its own reference to the playback and to the bot
        session, so it would go on publishing into a conference the channel has
        left.
        """
        kit, channel, backend = await _kit(tts=_ChunkedTTS(4))
        await backend.simulate_participant_joined(ROOM, "p-alice")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "hello there"))
        await _until(lambda: len(backend.published_audio) >= 1)
        published_before = len(backend.published_audio)

        await kit.detach_channel(ROOM, "conf")
        await speaking

        assert len(backend.published_audio) == published_before

    async def test_after_tts_does_not_fire_for_a_room_that_was_left(self) -> None:
        """AFTER_TTS pairs with BEFORE_TTS for an utterance that happened. A
        detach is not an utterance ending, and firing a room's hooks after
        detaching from it is what detaching was supposed to stop.
        """
        kit, channel, backend = await _kit(tts=_ChunkedTTS(4))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        fired: list[str] = []

        @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
        async def _after(event: object, ctx: object) -> None:
            fired.append("after_tts")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "hello there"))
        await _until(lambda: len(backend.published_audio) >= 1)
        await kit.detach_channel(ROOM, "conf")
        await speaking
        await asyncio.sleep(0)

        assert fired == []

    async def test_close_stops_the_bot_speaking(self) -> None:
        kit, channel, backend = await _kit(tts=_ChunkedTTS(4))
        await backend.simulate_participant_joined(ROOM, "p-alice")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "hello there"))
        await _until(lambda: len(backend.published_audio) >= 1)
        published_before = len(backend.published_audio)

        await channel.close()
        await speaking

        assert len(backend.published_audio) == published_before
        del kit

    async def test_no_chunk_reaches_the_conference_after_the_bot_left(self) -> None:
        """Reading the abandoned flag before publishing leaves the publication
        itself exposed: a detach landing inside it takes the bot out while that
        chunk is on its way to the very session it names.
        """
        backend = _SlowPublishBackend()
        kit, channel, _ = await _kit(backend, tts=_ChunkedTTS(4))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        backend.slow = True

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "hello"))
        await asyncio.wait_for(backend.publishing.wait(), timeout=5.0)

        async def release() -> None:
            # Let the teardown reach its wait — `_leaving` is set on the way in
            # — then let the publication finish. The ordering under test is what
            # happens between those two.
            await _until(lambda: bool(channel._room(ROOM).leaving))
            backend.gate.set()

        releasing = asyncio.create_task(release())
        await kit.detach_channel(ROOM, "conf")
        await asyncio.gather(speaking, releasing)

        methods = [call.method for call in backend.calls]
        assert "leave" in methods
        assert "publish_audio" not in methods[methods.index("leave") :]

    async def test_every_concurrent_playback_is_abandoned(self) -> None:
        """A room asked to say two things holds one utterance and queues the
        other, and the detach has to reach both. The one speaking is in a
        publishing loop; the one queued is parked on the room's floor, holding
        the bot session it will publish into as soon as the first releases.
        Reaching only the first leaves that answer to speak into a room the
        channel has left.
        """
        backend = MockConferenceBackend()
        kit, channel, _ = await _kit(backend, tts=_ChunkedTTS(6))
        await backend.simulate_participant_joined(ROOM, "p-alice")

        first = asyncio.create_task(channel._voice.speak(ROOM, "one"))
        await _until(lambda: len(backend.published_audio) >= 1)
        second = asyncio.create_task(channel._voice.speak(ROOM, "two"))
        await _until(lambda: len(channel._voice._speaking[ROOM].playbacks) == 2)
        await _until(lambda: len(backend.published_audio) >= 3)

        await kit.detach_channel(ROOM, "conf")
        await asyncio.gather(first, second, return_exceptions=True)

        # A chunk already in flight is allowed to land — the teardown waits for
        # it rather than pulling the session out from under it. What must not
        # happen is either playback carrying on afterwards.
        methods = [call.method for call in backend.calls]
        assert "leave" in methods
        assert "publish_audio" not in methods[methods.index("leave") :]
        assert len(backend.published_audio) < 6  # not even the first loop finished

    async def test_an_undisturbed_utterance_still_completes(self) -> None:
        """The guard must not cost the ordinary case its audio or its hook."""
        kit, channel, backend = await _kit(tts=_ChunkedTTS(3))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        fired: list[str] = []

        @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
        async def _after(event: object, ctx: object) -> None:
            fired.append("after_tts")

        await channel._voice.speak(ROOM, "hello there")
        await _until(lambda: bool(fired))

        assert len(backend.published_audio) == 3
        assert fired == ["after_tts"]


# ---------------------------------------------------------------------------
# Everything the detach must get in front of, not only the paths it started with
# ---------------------------------------------------------------------------


class TestTeardownOrder:
    async def test_a_departure_is_announced_before_the_end(self) -> None:
        """The departure path has the same three-await shape as the arrival and
        was left outside the transaction, so a detach could announce the end
        between the roster write and the event.
        """
        kit, _, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "alice")
        events: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            events.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        entered = asyncio.Event()
        release = asyncio.Event()
        original_get = kit.store.get_participant

        async def slow_get(room_id: str, pid: str) -> object:
            entered.set()
            await release.wait()
            return await original_get(room_id, pid)

        kit.store.get_participant = slow_get  # type: ignore[method-assign, assignment]

        leaving = asyncio.create_task(backend.simulate_participant_left(ROOM, "alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(leaving, detaching, return_exceptions=True)
        kit.store.get_participant = original_get  # type: ignore[method-assign]

        conference = [e for e in events if e.startswith("conference_")]
        assert conference.index("conference_participant_left") < conference.index(
            "conference_ended"
        )

    async def test_a_transcription_does_not_land_inside_the_end(self) -> None:
        """The lanes were closed after the end was announced, and the collection
        gate read an absent binding as open — which is what a detached room has.
        A recognizer suspended over the detach passed both.
        """
        backend = MockConferenceBackend()
        kit, channel, _ = await _kit(backend, stt=MockSTTProvider(transcripts=["hello there"]))
        observed: list[str] = []
        entered = asyncio.Event()
        release = asyncio.Event()

        @kit.on("conference_ended")
        async def _ended(event: object) -> None:
            observed.append("ended-start")
            entered.set()
            await release.wait()
            observed.append("ended-finish")

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def _transcribed(event: object, ctx: object) -> HookResult:
            observed.append("transcription")
            return HookResult(allowed=True)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = asyncio.create_task(say(backend, track))
        await asyncio.sleep(0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        release.set()
        await asyncio.gather(speaking, detaching, return_exceptions=True)

        assert "transcription" not in observed[observed.index("ended-start") :]
        assert channel._room(ROOM).may_collect() is False

    async def test_a_detached_room_never_permits_collection(self) -> None:
        """An absent binding means "attached, binding not yet seen" — which is
        open. A detached room has no binding either, and must not read as open.
        """
        kit, channel, _ = await _kit()
        assert channel._room(ROOM).may_collect() is True

        await kit.detach_channel(ROOM, "conf")

        assert channel._room(ROOM).may_collect() is False
        assert channel._attached_room("never-attached") is None


class TestDisclosureDuringTeardown:
    async def test_the_bot_stays_observable_until_it_has_left(self) -> None:
        """RFC 17.7 asks for the bot's presence to be observable at any time.
        The detach takes it off the channel's books first and out of the
        conference last, and reporting it absent in between tells an integrator
        the meeting is unattended while the bot is still sitting in it.
        """
        backend = _SlowPublishBackend()
        kit, channel, _ = await _kit(backend, tts=_ChunkedTTS(4))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        backend.slow = True

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "hello"))
        await asyncio.wait_for(backend.publishing.wait(), timeout=5.0)
        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        await _until(lambda: bool(channel._room(ROOM).leaving))

        room_info = channel.info()["rooms"].get(ROOM)
        assert room_info is not None, "the room vanished from info() while the bot was still in it"
        assert room_info["bot_present"] is True
        assert room_info["detaching"] is True
        assert backend.bots != []

        backend.gate.set()
        await asyncio.gather(speaking, detaching, return_exceptions=True)

        assert ROOM not in channel.info()["rooms"]
        assert backend.bots == []


# ---------------------------------------------------------------------------
# Closing is a detach of every room at once, and owes them the same order
# ---------------------------------------------------------------------------


class TestCloseDuringJoin:
    """``close()`` cleared the attachment set without touching the generations,
    so a join suspended in the backend resumed, found its own token unchanged
    because nothing had bumped it, and registered a bot on a closed channel.
    """

    @pytest.fixture
    async def closed(self) -> tuple[ConferenceChannel, _SlowJoinBackend, list[str]]:
        backend = _SlowJoinBackend()
        backend.slow_room = ROOM
        kit, channel, _ = await _kit(backend)
        events: list[str] = []
        original = kit._emit_framework_event

        async def spy(name: str, **kw: object) -> None:
            events.append(name)
            await original(name, **kw)  # type: ignore[arg-type]

        kit._emit_framework_event = spy  # type: ignore[method-assign, assignment]

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)

        async def release() -> None:
            # close() closes admission on every room on its way in and only
            # then waits for the join to let go of the room lock. Releasing on
            # that signal is what lets the wait be observed rather than timed
            # out.
            await _until(lambda: not any(room.attached for room in channel._rooms.values()))
            backend.gate.set()

        releasing = asyncio.create_task(release())
        await channel.close()
        await asyncio.gather(joining, releasing, return_exceptions=True)
        return channel, backend, events

    async def test_the_channel_holds_no_bot(
        self, closed: tuple[ConferenceChannel, _SlowJoinBackend, list[str]]
    ) -> None:
        channel, _, _ = closed

        assert all(room.bot is None for room in channel._rooms.values())

    async def test_the_late_join_leaves_the_conference(
        self, closed: tuple[ConferenceChannel, _SlowJoinBackend, list[str]]
    ) -> None:
        _, backend, _ = closed

        assert backend.bots == []

    async def test_no_conference_is_announced_after_the_close(
        self, closed: tuple[ConferenceChannel, _SlowJoinBackend, list[str]]
    ) -> None:
        _, _, events = closed

        assert [e for e in events if e.startswith("conference_")] == []


class TestCloseOutlastsItsBudget:
    """Every wait a close makes is bounded, because the work it waits for ends
    in code the channel does not own. Which means the budget can pass with a
    join still in flight — and the close went on to shut the backend, leaving
    that join to call ``leave()`` on it.
    """

    async def test_no_leave_reaches_a_closed_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _SlowJoinBackend()
        backend.slow_room = ROOM
        _, channel, _ = await _kit(backend)

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)
        await channel.close()
        closed_at = len(backend.calls)

        backend.gate.set()
        await asyncio.gather(joining, return_exceptions=True)

        # `join_as_bot` is the call that was already in flight arriving at its
        # own end, not a new one. What must not be there is what the channel
        # would have gone on to do with the session it came back with.
        after = [call.method for call in backend.calls[closed_at:] if call.method != "join_as_bot"]
        assert after == [], f"the backend was called after it was closed: {after}"

    async def test_every_room_settles_on_one_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The budget was spent per room, so a channel serving twenty
        conferences paid twenty times over to close — and closing is not where a
        channel is allowed to be slow.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.3)
        rooms = ("r-1", "r-2", "r-3", "r-4")

        class _NeverJoins(MockConferenceBackend):
            async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
                await asyncio.Event().wait()

        backend = _NeverJoins()
        _, channel, _ = await _kit(backend, rooms=rooms)
        joins = [
            asyncio.create_task(backend.simulate_participant_joined(room, "p-alice"))
            for room in rooms
        ]
        await _until(lambda: all(room.lock.locked() for room in channel._rooms.values()))

        loop = asyncio.get_running_loop()
        started = loop.time()
        await channel._settle_joins()
        elapsed = loop.time() - started

        for join in joins:
            join.cancel()
        await asyncio.gather(*joins, return_exceptions=True)

        assert elapsed < 0.3 * len(rooms), f"{elapsed:.2f}s for {len(rooms)} rooms"


class _HeldPublishBackend(MockConferenceBackend):
    """Holds a publication open, and records the order of what it was told."""

    def __init__(self) -> None:
        super().__init__()
        self.publishing = asyncio.Event()
        self.gate = asyncio.Event()
        self.order: list[str] = []

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        if chunk.data:
            self.publishing.set()
            await self.gate.wait()
        self.order.append("publish_audio")
        return await super().publish_audio(bot, chunk)

    async def leave(self, bot):  # type: ignore[no-untyped-def]
        self.order.append("leave")
        return await super().leave(bot)


class _OneWordTTS(TTSProvider):
    """One chunk, not marked final, so the utterance owes a boundary."""

    @property
    def default_voice(self) -> str:
        return "word"

    async def synthesize(self, text, *, voice=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def synthesize_stream(self, text, *, voice=None):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        yield AudioChunk(data=text.encode(), sample_rate=16000, is_final=False)


class TestAPublicationOutlivesItsCaller:
    """A publication is owned by the channel, so it survives its caller being
    cancelled — and the room must go on holding it until it settles.

    Registered through the caller's own activity block, it did not: the caller's
    registration ended when the caller was cancelled, so the teardown drained a
    room it believed quiet and took the bot out from under a chunk that was
    still on its way to that very session.
    """

    async def test_a_detach_does_not_leave_before_a_publication_settles(self) -> None:
        backend = _HeldPublishBackend()
        kit, channel, _ = await _kit(backend, tts=_OneWordTTS())

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha"))
        await asyncio.wait_for(backend.publishing.wait(), timeout=5.0)
        speaking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking

        detaching = asyncio.create_task(kit.detach_channel(ROOM, "conf"))
        # Nothing here can advance the publication, so any progress the detach
        # makes towards `leave()` it makes by overtaking it.
        for _ in range(50):
            await asyncio.sleep(0)
        assert "leave" not in backend.order, "the bot left with a chunk still in flight"

        backend.gate.set()
        await asyncio.wait_for(detaching, timeout=5.0)
        await _settle(channel)

        assert backend.order.index("publish_audio") < backend.order.index("leave")


class _ClosingStore(InMemoryStore):
    """A store that becomes unusable when it is closed, as a real pool does.

    ``InMemoryStore`` has nothing to release, so its ``close()`` is a no-op and a
    test written against it cannot tell waiting from failing. This one records
    every use that arrives after the release, which is the thing that must not
    happen.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.used_after_close: list[str] = []
        self.recorded: list[Participant] = []

    async def close(self) -> None:
        self.closed = True

    def _check(self, call: str) -> None:
        if self.closed:
            self.used_after_close.append(call)
            raise RuntimeError(f"store is closed: {call}")

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        self._check("get_participant")
        return await super().get_participant(room_id, participant_id)

    async def add_participant(self, participant: Participant) -> Participant:
        self._check("add_participant")
        self.recorded.append(participant)
        return await super().add_participant(participant)

    async def update_participant(self, participant: Participant) -> Participant:
        self._check("update_participant")
        return await super().update_participant(participant)


class _ClosingLocks(InMemoryLockManager):
    """A lock manager that refuses to be entered once released."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.used_after_close: list[str] = []

    async def close(self) -> None:
        self.closed = True

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        if self.closed:
            self.used_after_close.append(room_id)
            raise RuntimeError(f"lock manager is closed: {room_id}")
        async with super().locked(room_id):
            yield


def _hold_roster(channel: ConferenceChannel, name: str) -> tuple[asyncio.Event, asyncio.Event]:
    """Suspend a roster write after its check and before its store call."""
    entered = asyncio.Event()
    gate = asyncio.Event()
    original = getattr(channel._roster, name)

    async def held(*args: object, **kwargs: object) -> object:
        entered.set()
        await gate.wait()
        return await original(*args, **kwargs)

    setattr(channel._roster, name, held)
    return entered, gate


class _HeldAcquisitionLocks(_ClosingLocks):
    """Suspends one acquisition mid-flight, as a distant lock service might.

    Armed with ``hold_next``; the suspended caller is announced on ``entered``
    and resumes when ``gate`` is set. The suspension happens *inside*
    ``locked()``'s entry — the operation has reached the lock manager, the way
    a ``pg_advisory_lock`` call has reached its pool — which is what the
    framework's lease must therefore cover.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hold_next = False
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        if self.hold_next:
            self.hold_next = False
            self.entered.set()
            await self.gate.wait()
        async with super().locked(room_id):
            yield


class TestARosterWriteOutlivingTheChannel:
    """A roster write is ordered against ``remove_member()`` and against a
    detach, which hold the same room lock. ``close()`` holds nothing.

    So a write that had passed its ``attached`` check resumed after the channel
    had finished closing and added an ACTIVE participant to a room it had left —
    and after ``RoomKit.close()``, into a store that had been closed underneath
    it.
    """

    async def test_an_arrival_is_written_before_the_channel_closes(self) -> None:
        kit, channel, backend = await _kit()
        entered, gate = _hold_roster(channel, "record")

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(channel.close())
        for _ in range(50):
            await asyncio.sleep(0)
        assert not closing.done(), "the channel closed with a roster write in flight"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)
        await kit.close()

        assert [p.id for p in await kit.store.list_participants(ROOM)] == ["p-alice"]

    async def test_a_departure_is_written_before_the_channel_closes(self) -> None:
        kit, channel, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        entered, gate = _hold_roster(channel, "mark_left")

        leaving = asyncio.create_task(backend.simulate_participant_left(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(channel.close())
        for _ in range(50):
            await asyncio.sleep(0)
        assert not closing.done(), "the channel closed with a roster write in flight"

        gate.set()
        await asyncio.wait_for(asyncio.gather(leaving, closing), timeout=5.0)
        await kit.close()

        participant = await kit.store.get_participant(ROOM, "p-alice")
        assert participant is not None
        assert participant.status is ParticipantStatus.LEFT

    async def test_a_resolver_suspended_across_the_close_writes_nothing(self) -> None:
        """The barrier covers the whole callback, not the write at the end of it.

        Identity resolution is an await into integrator code and it comes first,
        so a barrier that only registered the write photographed an empty
        registry while a resolver was still suspended — and the callback came
        back to a store and a room lock ``RoomKit.close()`` had released.
        """
        resolving = asyncio.Event()
        gate = asyncio.Event()

        class _SlowResolver(IdentityResolver):
            async def resolve(self, channel_type, address, organization_id=None):  # type: ignore[no-untyped-def]
                resolving.set()
                await gate.wait()
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        kit, channel, backend = await _kit(resolver=_SlowResolver())

        arriving = asyncio.create_task(
            backend.simulate_participant_joined(
                ROOM, "p-alice", metadata={"phone_number": "+15551234567"}
            )
        )
        await asyncio.wait_for(resolving.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        for _ in range(50):
            await asyncio.sleep(0)
        assert not closing.done(), "the framework closed with a callback in flight"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

    async def test_a_wedged_store_does_not_hold_the_bot_in_the_conference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The barrier waits for the bookkeeping *after* the media is released.

        Waiting for it first made a slow store into a bot left in a meeting,
        potentially listening — the one failure this module ranks above every
        other. The bot leaves and the backend closes regardless of what the store
        is doing; what the store is doing only holds up the shutdown itself.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        kit, channel, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-bob")
        assert backend.bots != []
        entered, gate = _hold_roster(channel, "mark_left")

        leaving = asyncio.create_task(backend.simulate_participant_left(ROOM, "p-bob"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        await _until(lambda: backend.bots == [] and channel._backend_closed)

        assert not closing.done(), "the close returned with a write inside the store"

        gate.set()
        await asyncio.wait_for(asyncio.gather(leaving, closing), timeout=5.0)

    async def test_the_store_is_not_released_under_a_write_it_is_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserted against resources that actually become unusable.

        ``InMemoryStore.close()`` is a no-op and ``InMemoryLockManager.close()``
        too, so a test built on them cannot tell a shutdown that waits from one
        that closes the store and lets the write fail — it normalises the very
        thing it should refuse. These two raise once closed, which is what a
        connection pool does.
        """
        budget = 0.05
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", budget)
        store = _ClosingStore()
        locks = _ClosingLocks()
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit(store=store, lock_manager=locks)
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        entered, gate = _hold_roster(channel, "record")

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        # Well past the budget: a bounded wait would have given up by now and
        # gone on to release the store, which is what this refuses to do.
        await asyncio.sleep(budget * 4)
        assert not closing.done(), "the close returned while the store was running a write"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert store.used_after_close == [], f"store used after close: {store.used_after_close}"
        assert locks.used_after_close == [], f"locks used after close: {locks.used_after_close}"
        assert [p.id for p in store.recorded] == ["p-alice"]

    async def test_one_budget_covers_every_callback_it_waits_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The budget is for the step, not for each thing inside it.

        A property test rather than a regression: the close it was reported
        against spent a budget on each of two registries, and there is only one
        budgeted wait now, so the double spend is gone by construction rather
        than by a fix. This pins what remains — several callbacks in flight cost
        one deadline between them, not one each.
        """
        budget = 0.1
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", budget)
        resolving = asyncio.Event()
        gate = asyncio.Event()
        seen = 0

        class _StuckResolver(IdentityResolver):
            async def resolve(self, channel_type, address, organization_id=None):  # type: ignore[no-untyped-def]
                nonlocal seen
                seen += 1
                if seen >= 3:
                    resolving.set()
                await gate.wait()
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        kit, _, backend = await _kit(resolver=_StuckResolver())
        arriving = [
            asyncio.create_task(
                backend.simulate_participant_joined(
                    ROOM, f"p-{index}", metadata={"phone_number": f"+1555000000{index}"}
                )
            )
            for index in range(3)
        ]
        await asyncio.wait_for(resolving.wait(), timeout=5.0)

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait_for(kit.close(), timeout=5.0)
        elapsed = loop.time() - started

        assert elapsed < budget * 3, f"the budget was paid per callback: {elapsed:.3f}s"

        gate.set()
        await asyncio.wait_for(asyncio.gather(*arriving), timeout=5.0)

    async def test_a_write_that_returns_in_time_still_lands(self) -> None:
        """And the budget is not the common case: a store that answers is waited
        for, and the arrival it carried is on the roster.
        """
        kit, channel, backend = await _kit()
        entered, gate = _hold_roster(channel, "record")

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        for _ in range(50):
            await asyncio.sleep(0)
        assert not closing.done(), "the framework closed with a write inside the store"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert [p.id for p in await kit.store.list_participants(ROOM)] == ["p-alice"]

    async def test_a_store_failure_during_the_close_is_not_relabelled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The catch around the room lock covered the write as well.

        So anything the store raised after a close had begun came back as an
        INFO saying the write had been waiting for a room — which is untrue and
        one severity too quiet. What the store raises belongs on the path
        everything else the backend's callbacks raise takes.
        """
        kit, channel, backend = await _kit()
        entered = asyncio.Event()
        gate = asyncio.Event()

        async def refuse(*args: object, **kwargs: object) -> None:
            entered.set()
            await gate.wait()
            raise RuntimeError("the store went away")

        channel._roster.record = refuse  # type: ignore[method-assign]

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        for _ in range(50):
            await asyncio.sleep(0)

        with caplog.at_level(logging.ERROR):
            gate.set()
            await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert "the store went away" in caplog.text
        assert "abandoned a roster write" not in caplog.text

    async def test_a_callback_the_budget_gave_up_on_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before the store has it, the work *is* the channel's to stop.

        A callback suspended in an identity resolver is integrator code, so the
        wait for it is bounded — and past the barrier it must touch nothing at
        all. Asserted on the *room lock* rather than on the roster, because the
        lock is reached first: ``RoomKit.close()`` releases the lock manager
        right after the channels, and with a ``PostgresAdvisoryLockManager``
        acquiring one afterwards is a closed pool rather than a no-op.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        resolving = asyncio.Event()
        gate = asyncio.Event()

        class _StuckResolver(IdentityResolver):
            async def resolve(self, channel_type, address, organization_id=None):  # type: ignore[no-untyped-def]
                resolving.set()
                await gate.wait()
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        kit, _, backend = await _kit(resolver=_StuckResolver())
        closed = asyncio.Event()
        used_after_close: list[str] = []
        original = kit.lock_manager.locked

        def spy(room_id: str):  # type: ignore[no-untyped-def]
            if closed.is_set():
                used_after_close.append(room_id)
            return original(room_id)

        kit.lock_manager.locked = spy  # type: ignore[method-assign]

        arriving = asyncio.create_task(
            backend.simulate_participant_joined(
                ROOM, "p-alice", metadata={"phone_number": "+15551234567"}
            )
        )
        await asyncio.wait_for(resolving.wait(), timeout=5.0)
        await asyncio.wait_for(kit.close(), timeout=5.0)
        closed.set()
        gate.set()
        await asyncio.wait_for(arriving, timeout=5.0)

        assert used_after_close == [], "the callback took the room lock after it was released"
        assert await kit.store.list_participants(ROOM) == []

    async def test_a_framework_close_does_not_outrun_one_either(self) -> None:
        """``RoomKit.close()`` closes the channels and then the store, so a write
        it did not wait for reaches a storage backend that has been released.
        """
        kit, channel, backend = await _kit()
        entered, gate = _hold_roster(channel, "record")

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        for _ in range(50):
            await asyncio.sleep(0)
        assert not closing.done(), "the framework closed with a roster write in flight"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

    async def test_a_wedged_store_does_not_hold_the_channel_close_either(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The channel's own ``close()`` is bounded even against a wedged store.

        The framework closes channels in sequence, so a ``close()`` that waits
        for the store without a deadline is not spending its own time — it is
        holding every channel behind it in its conference (RFC 12.10.4). The
        write the store already has is not lost by returning: it sits under the
        framework's resource lease, and the store outlives the channel.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        kit, channel, backend = await _kit()
        entered, gate = _hold_roster(channel, "record")

        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await asyncio.wait_for(channel.close(), timeout=2.0)

        assert backend.bots == [], "the bot stayed in the conference"
        assert channel._backend_closed is True

        gate.set()
        await asyncio.wait_for(arriving, timeout=5.0)
        assert [p.id for p in await kit.store.list_participants(ROOM)] == ["p-alice"]
        await asyncio.wait_for(kit.close(), timeout=5.0)


class TestOneChannelsStoreIsNotAnothersMedia:
    """The framework closes channels in sequence and owns what they share.

    So the wait for a write the store already has belongs to the framework,
    after every channel's media is released — inside one channel's ``close()``
    it held the channels behind it in their conferences: first backend closed,
    second backend never reached, second bot still in its meeting, for as long
    as the store cared to take (RFC 12.10.4).
    """

    @staticmethod
    async def _two_channels(
        store: InMemoryStore | None = None,
        locks: InMemoryLockManager | None = None,
    ) -> tuple[RoomKit, tuple[ConferenceChannel, ...], tuple[MockConferenceBackend, ...]]:
        backends = (MockConferenceBackend(), MockConferenceBackend())
        channels = (
            ConferenceChannel("conf-a", backend=backends[0]),
            ConferenceChannel("conf-b", backend=backends[1]),
        )
        kit = RoomKit(store=store, lock_manager=locks)
        for channel, room_id in zip(channels, (ROOM, OTHER), strict=True):
            kit.register_channel(channel)
            await kit.create_room(room_id)
            await kit.attach_channel(room_id, channel.channel_id)
        return kit, channels, backends

    async def test_a_wedged_store_in_one_channel_frees_the_others_bot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both bots leave and both backends close while the store is wedged —
        and the store is still not released under the write it is running."""
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        store = _ClosingStore()
        locks = _ClosingLocks()
        kit, channels, backends = await self._two_channels(store, locks)
        await backends[1].simulate_participant_joined(OTHER, "p-bob")
        assert backends[1].bots != []
        entered, gate = _hold_roster(channels[0], "record")

        arriving = asyncio.create_task(backends[0].simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        await _until(
            lambda: (
                backends[0].bots == []
                and backends[1].bots == []
                and channels[0]._backend_closed
                and channels[1]._backend_closed
            )
        )

        assert not closing.done(), "the close returned while the store was running a write"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert store.used_after_close == [], f"store used after close: {store.used_after_close}"
        assert locks.used_after_close == [], f"locks used after close: {locks.used_after_close}"
        assert [p.id for p in store.recorded] == ["p-bob", "p-alice"]

    async def test_a_close_that_fails_does_not_strand_the_next_channels_bot(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Channels close in sequence, so one raising must not stop the rest —
        and must not be reported as a success either. The channel that failed
        may still be holding its media, so the caller is told, once everything
        that could be closed has been.
        """
        store = _ClosingStore()
        locks = _ClosingLocks()
        kit, channels, backends = await self._two_channels(store, locks)
        await backends[1].simulate_participant_joined(OTHER, "p-bob")
        assert backends[1].bots != []

        async def refuse() -> None:
            raise RuntimeError("this channel will not close")

        channels[0].close = refuse  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR), pytest.raises(ExceptionGroup) as failure:
            await asyncio.wait_for(kit.close(), timeout=5.0)

        assert backends[1].bots == [], "the second channel's bot was stranded"
        assert channels[1]._backend_closed is True
        assert "failed to close" in caplog.text
        assert [str(error) for error in failure.value.exceptions] == [
            "this channel will not close"
        ]
        # Raised after the whole shutdown, not instead of it: the shared
        # resources were genuinely released first.
        assert store.closed is True, "the store was not released before the raise"
        assert locks.closed is True, "the lock manager was not released before the raise"

    async def test_the_lock_manager_outlives_an_acquisition_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An acquisition is already an operation the lock manager is running.

        On an advisory-lock backend it is a call holding a pool connection, so
        a lease that only started once the lock was *held* left the framework
        free to close the pool underneath a caller still queued on it — which
        is where the acquisition then resumed.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        store = _ClosingStore()
        locks = _HeldAcquisitionLocks()
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit(store=store, lock_manager=locks)
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        locks.hold_next = True
        arriving = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.wait_for(locks.entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        await _until(lambda: backend.bots == [] and channel._backend_closed)
        # Well past every bounded budget: a close that was going to give up on
        # the acquisition has done so by now.
        await asyncio.sleep(0.25)

        assert not closing.done(), "the close returned with an acquisition still queued"

        locks.gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert locks.used_after_close == [], f"locks used after close: {locks.used_after_close}"
        assert store.used_after_close == [], f"store used after close: {store.used_after_close}"
        # The barrier was down by the time the lock was acquired: nothing landed.
        assert store.recorded == []

    async def test_the_store_outlives_a_read_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read the store is running is an operation like any other.

        The arrival path asks the roster whether it knows the participant
        before resolving them, and that read used to run outside any lease —
        so a close landing while it was suspended released the store, and the
        read resumed inside a resource that no longer existed.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)

        class _InstantResolver(IdentityResolver):
            async def resolve(self, channel_type, address, organization_id=None):  # type: ignore[no-untyped-def]
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        store = _ClosingStore()
        locks = _ClosingLocks()
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit(store=store, lock_manager=locks, identity_resolver=_InstantResolver())
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        entered = asyncio.Event()
        gate = asyncio.Event()
        original = store.get_participant

        async def held(room_id: str, participant_id: str) -> Participant | None:
            entered.set()
            await gate.wait()
            return await original(room_id, participant_id)

        store.get_participant = held  # type: ignore[method-assign]

        arriving = asyncio.create_task(
            backend.simulate_participant_joined(
                ROOM, "p-alice", metadata={"phone_number": "+15551234567"}
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        closing = asyncio.create_task(kit.close())
        await _until(lambda: backend.bots == [] and channel._backend_closed)
        # Well past every bounded budget: a close that was going to give up on
        # the read has done so by now.
        await asyncio.sleep(0.25)

        assert not closing.done(), "the close returned with a read inside the store"

        gate.set()
        await asyncio.wait_for(asyncio.gather(arriving, closing), timeout=5.0)

        assert store.used_after_close == [], f"store used after close: {store.used_after_close}"


class _AbandonedJoinBackend(MockConferenceBackend):
    """Suspends a join past every closing budget, then fails it.

    The shape of an SFU whose connection died while the framework had already
    given up waiting: the join holds no lease — it is suspended in backend
    code, which never runs under one — so the shutdown sees nothing to wait
    for, completes, and only then does the join resume in error.
    """

    def __init__(self) -> None:
        super().__init__()
        self.joining = asyncio.Event()
        self.gate = asyncio.Event()

    async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
        self.joining.set()
        await self.gate.wait()
        raise RuntimeError("the SFU connection died while the join was queued")


class TestACallbackResumingAfterTheShutdown:
    """A callback can suspend before its first operation on the store.

    Nothing then holds a lease, so the shutdown's final wait sees an empty
    registry, completes, and releases the store and the lock manager — and
    the callback resumes afterwards. Its join failure is swallowed by design
    (an arrival is recorded even without a bot), so without a barrier re-read
    it marched straight into an identity lookup: a *new* lease, granted after
    the wait had concluded, on a store already released.
    """

    async def test_a_join_abandoned_by_the_budget_reads_and_writes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)

        class _InstantResolver(IdentityResolver):
            async def resolve(self, channel_type, address, organization_id=None):  # type: ignore[no-untyped-def]
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        store = _ClosingStore()
        locks = _ClosingLocks()
        backend = _AbandonedJoinBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit(store=store, lock_manager=locks, identity_resolver=_InstantResolver())
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        arriving = asyncio.create_task(
            backend.simulate_participant_joined(
                ROOM, "p-alice", metadata={"phone_number": "+15551234567"}
            )
        )
        await asyncio.wait_for(backend.joining.wait(), timeout=5.0)
        # Nothing holds a lease — the join is suspended in backend code — so
        # the close completes and releases both resources.
        await asyncio.wait_for(kit.close(), timeout=5.0)
        assert store.closed is True
        assert locks.closed is True

        backend.gate.set()
        await asyncio.wait_for(arriving, timeout=5.0)

        assert store.used_after_close == [], f"store used after close: {store.used_after_close}"
        assert locks.used_after_close == [], f"locks used after close: {locks.used_after_close}"
        assert store.recorded == []

    async def test_no_lease_is_granted_once_the_shutdown_has_sealed(self) -> None:
        """The registry is sealed the moment the final wait concludes.

        A lease granted later would be registered onto a registry nothing will
        read again, over a resource already being released — a use-after-free
        with a registration on it. It is refused with an error that says what
        happened instead.
        """
        kit = RoomKit()
        await kit.close()
        with pytest.raises(RoomKitError), kit._resource_lease():
            pass  # pragma: no cover — the lease refuses before the body runs


class TestTheMediaPlaneIsBoundedToo:
    """The media calls are the channel's to bound, like everything else.

    ``leave()``, ``backend.close()`` and a lane's recogniser all end in code
    the channel does not own, and the framework closes channels in sequence —
    so a media call that never returns held every channel behind it in its
    conference, which is the exact failure the budgets exist to prevent. And
    a session the budget could not remove is a *failed* close, said with an
    exception rather than summarised into a log.
    """

    @staticmethod
    async def _two_kits(
        backend_a: MockConferenceBackend, backend_b: MockConferenceBackend
    ) -> tuple[RoomKit, tuple[ConferenceChannel, ConferenceChannel]]:
        channels = (
            ConferenceChannel("conf-a", backend=backend_a),
            ConferenceChannel("conf-b", backend=backend_b),
        )
        kit = RoomKit()
        for channel, room_id in zip(channels, (ROOM, OTHER), strict=True):
            kit.register_channel(channel)
            await kit.create_room(room_id)
            await kit.attach_channel(room_id, channel.channel_id)
        return kit, channels

    async def test_a_leave_that_never_returns_frees_the_next_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewer's reproduction: a suspended leave() used to hold
        RoomKit.close() before the second conference, both bots connected,
        the second backend open — indefinitely."""
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(activity_module, "CANCEL_GRACE_S", 0.05)
        backend_a = _SlowLeaveBackend()
        backend_a.slow = True
        backend_b = MockConferenceBackend()
        kit, channels = await self._two_kits(backend_a, backend_b)
        await backend_a.simulate_participant_joined(ROOM, "p-alice")
        await backend_b.simulate_participant_joined(OTHER, "p-bob")
        assert backend_a.bots != [] and backend_b.bots != []

        with pytest.raises(ExceptionGroup) as failure:
            await asyncio.wait_for(kit.close(), timeout=5.0)

        assert backend_b.bots == [], "the second channel's bot was held by the first's leave"
        assert channels[1]._backend_closed is True
        assert isinstance(failure.value.exceptions[0], ConferenceCloseError)
        # The session the budget could not remove is still on the books, not
        # forgotten: info() goes on reporting it.
        assert channels[0].info()["rooms"][ROOM]["bot_present"] is True

    async def test_a_leave_the_backend_refuses_is_a_failed_close(self) -> None:
        """A clean return over a bot still in a meeting was the report of the
        one thing the roster exists to never misstate."""
        backend_a = MockConferenceBackend()
        backend_b = MockConferenceBackend()
        kit, channels = await self._two_kits(backend_a, backend_b)
        await backend_a.simulate_participant_joined(ROOM, "p-alice")
        backend_a.fail("leave", RuntimeError("the SFU refused"))

        with pytest.raises(ExceptionGroup) as failure:
            await asyncio.wait_for(kit.close(), timeout=5.0)

        assert backend_b.bots == [] and channels[1]._backend_closed is True
        assert isinstance(failure.value.exceptions[0], ConferenceCloseError)
        assert backend_a.bots != [], "the bot did leave; the raise was for nothing"

    async def test_a_lane_that_shrugs_off_cancellation_does_not_hold_the_close(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An STT that swallows CancelledError used to block the lane's close
        indefinitely. It now costs one grace period, the lane is abandoned and
        reported, and its stage state is released only when the task truly
        ends — never underneath it."""
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(activity_module, "CANCEL_GRACE_S", 0.05)
        kit, channel, backend = await _kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        lane = channel._lanes[track.id]
        stuck = asyncio.Event()
        surrender = asyncio.Event()

        async def stubborn(frame: object) -> None:
            stuck.set()
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:  # noqa: PERF203 — the point of the test
                    if surrender.is_set():
                        raise
                    continue

        lane._process = stubborn  # type: ignore[method-assign]
        lane.submit(speech_frame())
        await asyncio.wait_for(stuck.wait(), timeout=5.0)
        runaway = lane._task
        assert runaway is not None

        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(kit.close(), timeout=5.0)

        assert "did not stop within" in caplog.text
        # The provider finally gives in, so the loop can close cleanly; this
        # is also the moment the abandoned lane's deferred release runs.
        surrender.set()
        runaway.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(runaway, timeout=5.0)
