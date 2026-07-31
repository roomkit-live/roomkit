"""Hot-plugging intelligence into a running conference channel (RMK-76).

RFC §12.10.4: the configuration first need is read from is not fixed at
construction. Plugging a need is a first need — the occupancy probe is re-run,
an occupied conference is joined at once, and the tracks already published are
subscribed — and unplugging the last need takes the bot out: the channel is
pure transport again, exactly as if it had been constructed that way. The
grants follow the configuration, in place when the backend can change a
connected session's (BOT_GRANT_UPDATE) and by re-joining when it cannot.

The round trip is the contract this file exists for: a meeting can begin
purely human, gain its notetaker when the host asks for one, and lose it
again the same way — without the channel being rebuilt around either.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    ConferenceCapability,
    ConferenceGrants,
    ConferenceRecordingConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.test_conference_channel import _join_settled
from tests.conference.test_conference_outbound import _WordTTS

ROOM = "room-1"


def _calls(backend: MockConferenceBackend, method: str) -> list:
    return [c for c in backend.calls if c.method == method]


async def _channel(
    backend: MockConferenceBackend, **kwargs: object
) -> tuple[RoomKit, ConferenceChannel]:
    channel = ConferenceChannel("conf", backend=backend, **kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel


async def _occupied(backend: MockConferenceBackend, channel: ConferenceChannel) -> object:
    """A participant in the meeting with an audio track published."""
    await backend.simulate_participant_joined(ROOM, "p-alice")
    track = await backend.simulate_track_published(ROOM, "p-alice")
    await _join_settled(channel)
    return track


class TestPlugIsAFirstNeed:
    async def test_plugging_stt_into_an_occupied_conference_joins_and_subscribes(self) -> None:
        """The DoD case: the meeting is transcribed from the plug forward.

        The probe is re-run at the plug, the occupied conference is joined,
        and the track published before anything consumed it is subscribed
        retroactively, with its lane open (RFC §12.10.4).
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)
        track = await _occupied(backend, channel)
        assert backend.bots == []

        await channel.plug_stt(MockSTTProvider())

        assert len(backend.bots) == 1
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]
        assert track.id in channel.active_lanes  # type: ignore[attr-defined]
        info = channel.info()
        assert info["stt_configured"] is True
        assert info["rooms"][ROOM]["stt_active"] is True

    async def test_plugging_into_an_empty_conference_stays_lazy(self) -> None:
        """The probe is made — the plug restores its justification — and an
        empty answer starts no join; the ordinary triggers stand ready.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)

        await channel.plug_stt(MockSTTProvider())

        assert _calls(backend, "list_participants")
        assert backend.bots == []

        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)

        assert len(backend.bots) == 1

    async def test_a_plug_survives_the_probe_failing(self) -> None:
        """The join a plug starts follows the discipline of every trigger:
        its failure does not fail the plug — the configuration stands and
        the lazy join remains (RFC §12.10.4).
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)
        backend.fail("list_participants", RuntimeError("SFU unreachable"))

        await channel.plug_stt(MockSTTProvider())

        assert channel.info()["stt_configured"] is True
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)
        assert len(backend.bots) == 1

    async def test_plugging_recording_is_a_first_need_too(self) -> None:
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)
        track = await _occupied(backend, channel)

        await channel.plug_recording(ConferenceRecordingConfig(), recorder=MockMediaRecorder())

        assert len(backend.bots) == 1
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]
        info = channel.info()
        assert info["recording_configured"] is True
        assert info["rooms"][ROOM]["recording_active"] is True

    async def test_plugging_stt_opens_lanes_for_already_subscribed_tracks(self) -> None:
        """A recording channel subscribed its tracks with no lanes — nothing
        transcribed them. Plugging stt must open a lane for each of them, not
        only for tracks subscribed after the plug: the meeting is transcribed
        from the plug forward (RFC §12.10.4), whoever subscribed first.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(
            backend, recording=ConferenceRecordingConfig(), recorder=MockMediaRecorder()
        )
        track = await _occupied(backend, channel)
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]
        assert channel.active_lanes == {}

        await channel.plug_stt(MockSTTProvider())

        assert track.id in channel.active_lanes  # type: ignore[attr-defined]
        assert channel.info()["rooms"][ROOM]["stt_active"] is True


class TestGrantsFollowTheConfiguration:
    async def test_a_capable_backend_widens_the_grants_in_place(self) -> None:
        """BOT_GRANT_UPDATE: same session, same connection — the SFU changes
        what it may do (RFC §12.10.3), and no re-join cuts the event bridge.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(backend, stt=MockSTTProvider())
        await _occupied(backend, channel)
        bot = backend.bots[0]
        assert backend.bot_grants[bot.id].publish_audio is False

        await channel.plug_tts(MockTTSProvider())

        assert backend.bot_grants[bot.id].publish_audio is True
        assert len(_calls(backend, "join_as_bot")) == 1
        assert not _calls(backend, "leave")

    async def test_an_incapable_backend_forces_a_rejoin(self) -> None:
        """A session the SFU will not re-permission can only be replaced: a
        leave and a join, and the new session's grants carry the voice. The
        tracks are re-subscribed on the replacement session.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend, stt=MockSTTProvider())
        track = await _occupied(backend, channel)
        old_bot = backend.bots[0]

        await channel.plug_tts(MockTTSProvider())

        assert len(_calls(backend, "leave")) == 1
        joins = _calls(backend, "join_as_bot")
        assert len(joins) == 2
        assert joins[-1].args["grants"].publish_audio is True
        assert len(backend.bots) == 1
        assert backend.bots[0].id != old_bot.id
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]

    async def test_an_unplug_narrows_in_place_when_the_backend_can(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(backend, stt=MockSTTProvider(), tts=MockTTSProvider())
        await _occupied(backend, channel)
        bot = backend.bots[0]
        assert backend.bot_grants[bot.id].publish_audio is True

        await channel.unplug_tts()

        assert backend.bot_grants[bot.id].publish_audio is False
        assert len(_calls(backend, "join_as_bot")) == 1
        assert not _calls(backend, "leave")

    async def test_a_narrowing_the_backend_cannot_apply_is_left_standing(self) -> None:
        """An unused privilege against a cut in the event bridge is the trade
        RFC §12.10.4 settles for continuity: no re-join to remove a
        permission nobody uses.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend, stt=MockSTTProvider(), tts=MockTTSProvider())
        await _occupied(backend, channel)

        await channel.unplug_tts()

        assert len(_calls(backend, "join_as_bot")) == 1
        assert not _calls(backend, "leave")
        assert len(backend.bots) == 1

    async def test_explicit_grants_are_never_rewritten(self) -> None:
        """The caller who set ``bot_grants`` took coverage on themselves, and
        that holds at the plug exactly as at construction (RFC §12.10.4).
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend,
            stt=MockSTTProvider(),
            bot_grants=ConferenceGrants.for_bot(speaks=True, listens=True),
        )
        await _occupied(backend, channel)

        await channel.plug_tts(MockTTSProvider())

        assert not _calls(backend, "update_bot_grants")
        assert len(_calls(backend, "join_as_bot")) == 1


class TestSetBotGrants:
    """Runtime ownership of explicit grants (RMK-79, RFC §12.10.4).

    The plugs never rewrite an explicit ``bot_grants`` — which makes the set
    owned, not immutable. ``set_bot_grants()`` is the owner speaking: an
    instruction applied in full, in place where the backend can, by the
    announced re-join where it cannot — and always by the re-join for a
    concealment, because no SFU can un-tell clients about a participant they
    were told of (verified live against LiveKit, 2026-07-31).
    """

    async def test_a_hidden_bot_is_revealed_in_place(self) -> None:
        """The observer who reveals itself: same session, no leave — the SFU
        announces the newly visible participant to connected clients itself.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        await _occupied(backend, channel)
        bot = backend.bots[0]
        assert backend.bot_grants[bot.id].hidden is True

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        assert backend.bot_grants[bot.id].hidden is False
        assert len(_calls(backend, "update_bot_grants")) == 1
        assert not _calls(backend, "leave")
        assert backend.bots[0].id == bot.id

    async def test_an_incapable_backend_rejoins_to_apply_the_instruction(self) -> None:
        """Where a plug's alignment would weigh continuity, the setter obeys:
        a session the SFU will not re-permission is replaced, each half
        announced as the session event it is (RFC §12.10.4).
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        track = await _occupied(backend, channel)
        old_bot = backend.bots[0]
        announced: list[str] = []

        @kit.on("conference_ended")
        async def _ended(event: object) -> None:
            announced.append("ended")

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            announced.append("started")

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        assert len(_calls(backend, "leave")) == 1
        joins = _calls(backend, "join_as_bot")
        assert len(joins) == 2
        assert joins[-1].args["grants"].hidden is False
        assert backend.bots[0].id != old_bot.id
        assert announced == ["ended", "started"]
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]
        assert track.id in channel.active_lanes

    async def test_concealing_a_visible_bot_rejoins_even_when_capable(self) -> None:
        """Visibility does not move symmetrically: a reveal propagates in
        place, but no interface un-tells a client — a session re-hidden in
        place stays on every roster that saw it. The announced leave is the
        one retraction every backend delivers (RFC §12.10.4).
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.for_bot(listens=True)
        )
        await _occupied(backend, channel)
        old_bot = backend.bots[0]

        await channel.set_bot_grants(ConferenceGrants.observer())

        assert not _calls(backend, "update_bot_grants")
        assert len(_calls(backend, "leave")) == 1
        assert backend.bots[0].id != old_bot.id
        assert backend.bot_grants[backend.bots[0].id].hidden is True

    async def test_none_returns_the_channel_to_derivation(self) -> None:
        """The round trip: from there the grants follow the configuration in
        force, exactly as on a channel that never had an explicit set.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        await _occupied(backend, channel)
        bot = backend.bots[0]

        await channel.set_bot_grants(None)

        held = backend.bot_grants[bot.id]
        assert held == ConferenceGrants.for_bot(listens=True)
        assert backend.bots[0].id == bot.id

        # Back in the derived regime, the plugs' own alignment applies again.
        await channel.plug_tts(MockTTSProvider())
        assert backend.bot_grants[bot.id].publish_audio is True

    async def test_a_set_that_does_not_cover_the_needs_is_accepted(self) -> None:
        """The construction-time bargain carries forward: the caller keeps
        coverage on themselves, so ``subscribe`` withdrawn under a plugged
        recognizer is accepted, not refused — and the next plug still does
        not rewrite it (RFC §12.10.4).
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(backend, stt=MockSTTProvider())
        await _occupied(backend, channel)
        bot = backend.bots[0]
        uncovering = ConferenceGrants(
            publish_audio=False,
            publish_video=False,
            publish_screen_share=False,
            subscribe=False,
        )

        await channel.set_bot_grants(uncovering)

        assert backend.bot_grants[bot.id].subscribe is False

        await channel.plug_tts(MockTTSProvider())
        assert backend.bot_grants[bot.id] == uncovering

    async def test_a_withdrawal_is_honoured_where_an_unplug_would_leave_it(self) -> None:
        """On a backend that cannot re-permission, a plug's narrowing is left
        standing — an unused privilege traded for continuity. The setter's is
        not: the caller asked, so the session is replaced.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(
            backend,
            stt=MockSTTProvider(),
            bot_grants=ConferenceGrants.for_bot(speaks=True, listens=True),
        )
        await _occupied(backend, channel)

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        assert len(_calls(backend, "leave")) == 1
        joins = _calls(backend, "join_as_bot")
        assert len(joins) == 2
        assert joins[-1].args["grants"].publish_audio is False

    async def test_a_set_with_no_session_only_stores(self) -> None:
        """A grant set creates no need: on a pure-transport channel it
        changes what the next session would be allowed, and joins nothing.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)
        await _occupied(backend, channel)
        assert backend.bots == []

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        assert backend.bots == []
        assert not _calls(backend, "update_bot_grants")
        assert channel.info()["bot_hidden"] is False

    async def test_the_instruction_covers_a_join_in_flight(self) -> None:
        """A lazy join reads the grants before the setter lands, and would
        seat a session on the older set with nothing left to correct it.
        The instruction waits for the seat and covers it, rather than
        skipping a room whose session was milliseconds away.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        backend.delay("join_as_bot", 0.2)
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        # Let the spawned join take the room's lock and enter the delayed
        # backend call, so the setter genuinely lands mid-join.
        await asyncio.sleep(0.05)

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        await _join_settled(channel)
        assert len(backend.bots) == 1
        assert backend.bot_grants[backend.bots[0].id].hidden is False

    async def test_an_update_failure_falls_back_to_the_rejoin(self) -> None:
        """The instruction is applied whatever the in-place attempt did: a
        failed ``update_bot_grants`` costs the re-join, not the change.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        await _occupied(backend, channel)
        old_bot = backend.bots[0]
        backend.fail("update_bot_grants", RuntimeError("SFU refused"))

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        assert len(_calls(backend, "leave")) == 1
        assert backend.bots[0].id != old_bot.id
        assert backend.bot_grants[backend.bots[0].id].hidden is False

    async def test_info_answers_before_and_after(self) -> None:
        """§17.7: the disclosure surface reports the status in force on the
        session, and says beforehand whether a change re-joins.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        await _occupied(backend, channel)
        info = channel.info()
        assert info["bot_grant_update_in_place"] is True
        assert info["bot_hidden"] is True
        assert info["rooms"][ROOM]["bot_hidden"] is True

        await channel.set_bot_grants(ConferenceGrants.for_bot(listens=True))

        info = channel.info()
        assert info["bot_hidden"] is False
        assert info["rooms"][ROOM]["bot_hidden"] is False

        incapable = MockConferenceBackend()
        _, plain = await _channel(incapable, stt=MockSTTProvider())
        assert plain.info()["bot_grant_update_in_place"] is False

    async def test_the_change_is_announced_once(self) -> None:
        """A connected session's effective grants changed — the room can
        observe it (RFC §12.10.7). A set that changes nothing emits nothing.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        kit, channel = await _channel(
            backend, stt=MockSTTProvider(), bot_grants=ConferenceGrants.observer()
        )
        await _occupied(backend, channel)
        bot = backend.bots[0]
        observed: list[dict] = []

        @kit.on("conference_bot_grants_changed")
        async def _changed(event) -> None:  # noqa: ANN001
            observed.append(event.data)

        revealed = ConferenceGrants.for_bot(listens=True)
        await channel.set_bot_grants(revealed)
        await channel.set_bot_grants(revealed)

        assert len(observed) == 1
        assert observed[0]["bot_session_id"] == bot.id
        assert observed[0]["hidden"] is False


class TestUnplugingTheLastNeed:
    async def test_the_bot_leaves_and_the_channel_stands_down(self) -> None:
        """A session kept past the last need is the silent observer §17.7
        refuses: the bot leaves, ``conference_ended`` is announced, and the
        channel is pure transport again — the next mint starts no join.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend, stt=MockSTTProvider())
        track = await _occupied(backend, channel)
        assert len(backend.bots) == 1
        ended: list[str] = []

        @kit.on("conference_ended")
        async def _ended(event: object) -> None:
            ended.append("ended")

        await channel.unplug_stt()

        assert backend.bots == []
        assert _calls(backend, "leave")
        assert track.id not in backend.subscriptions  # type: ignore[attr-defined]
        assert channel.active_lanes == {}
        assert ended == ["ended"]
        info = channel.info()
        assert info["stt_configured"] is False
        assert info["rooms"][ROOM]["bot_present"] is False
        assert info["rooms"][ROOM]["stt_active"] is False

        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)
        assert backend.bots == []

    async def test_the_round_trip_transport_intelligent_transport(self) -> None:
        """The DoD's aller-retour: one channel serves the whole lifecycle."""
        backend = MockConferenceBackend()
        kit, channel = await _channel(backend)
        track = await _occupied(backend, channel)
        assert backend.bots == []

        await channel.plug_stt(MockSTTProvider())
        assert len(backend.bots) == 1
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]

        await channel.unplug_stt()
        assert backend.bots == []
        assert channel.active_lanes == {}

        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)
        assert backend.bots == []

    async def test_unpluging_stt_keeps_the_bot_for_recording(self) -> None:
        """Recording still consumes the tracks: the bot stays, the
        subscription stays, and only the lanes — recognition's alone — close.
        """
        backend = MockConferenceBackend()
        kit, channel = await _channel(
            backend,
            stt=MockSTTProvider(),
            recording=ConferenceRecordingConfig(),
            recorder=MockMediaRecorder(),
        )
        track = await _occupied(backend, channel)

        await channel.unplug_stt()

        assert len(backend.bots) == 1
        assert not _calls(backend, "leave")
        assert track.id in backend.subscriptions  # type: ignore[attr-defined]
        assert channel.active_lanes == {}
        info = channel.info()
        assert info["stt_configured"] is False
        assert info["recording_configured"] is True
        assert info["rooms"][ROOM]["stt_active"] is False
        assert info["rooms"][ROOM]["recording_active"] is True


class TestVoiceUnplug:
    async def test_unplug_tts_ends_the_utterance_and_the_turn(self) -> None:
        """An utterance in flight is ended the way a barge-in ends one —
        ``stop_playback`` and a terminal chunk — because the conference is
        live and the bot stays in it (RFC §12.10.4).
        """
        backend = MockConferenceBackend()
        tts = _WordTTS()
        kit, channel = await _channel(backend, stt=MockSTTProvider(), tts=tts)
        await _occupied(backend, channel)
        bot = backend.bots[0]

        tts.gated = True
        speak = asyncio.create_task(channel._voice.speak(ROOM, "alpha beta gamma"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)

        unplug = asyncio.create_task(channel.unplug_tts())
        await asyncio.sleep(0)
        tts.release.set()
        await asyncio.wait_for(unplug, timeout=5.0)
        await asyncio.wait_for(speak, timeout=5.0)

        assert bot.id in backend.playback_stops
        utterance = backend.utterances_for(bot)[-1]
        assert utterance.complete
        assert len(backend.bots) == 1
        assert channel._voice.tts is None


class TestRefusals:
    async def test_e2ee_refuses_a_plugged_stt(self) -> None:
        """The constructor's refusal holds identically at the plug."""
        backend = MockConferenceBackend(capabilities=ConferenceCapability.E2EE)
        channel = ConferenceChannel("conf", backend=backend, e2ee=True)

        with pytest.raises(ValueError, match="encrypted"):
            await channel.plug_stt(MockSTTProvider())

        assert channel.info()["stt_configured"] is False

    async def test_e2ee_refuses_a_plugged_recording(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.E2EE)
        channel = ConferenceChannel("conf", backend=backend, e2ee=True)

        with pytest.raises(ValueError, match="encrypted"):
            await channel.plug_recording(ConferenceRecordingConfig(), recorder=MockMediaRecorder())

        assert channel.info()["recording_configured"] is False

    async def test_an_occupied_slot_refuses_a_second_plug(self) -> None:
        """No silent replacement: a swap is a teardown and a rebuild, and the
        observation gap belongs in the open — unplug, then plug.
        """
        backend = MockConferenceBackend()
        channel = ConferenceChannel(
            "conf",
            backend=backend,
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            recording=ConferenceRecordingConfig(),
            recorder=MockMediaRecorder(),
        )

        with pytest.raises(ValueError, match="unplug_stt"):
            await channel.plug_stt(MockSTTProvider())
        with pytest.raises(ValueError, match="unplug_tts"):
            await channel.plug_tts(MockTTSProvider())
        with pytest.raises(ValueError, match="unplug_recording"):
            await channel.plug_recording(ConferenceRecordingConfig(), recorder=MockMediaRecorder())

    async def test_unpluging_an_empty_slot_is_a_no_op(self) -> None:
        """The state the caller asked for already holds; arriving second is
        not a fault.
        """
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)

        await channel.unplug_stt()
        await channel.unplug_tts()
        await channel.unplug_recording()

        assert backend.calls == []


class TestUnplugedProviderOwnership:
    async def test_the_channel_closes_what_it_owns(self) -> None:
        """``close_providers=True`` (the default) is one ownership rule for
        the whole lifecycle: an unplug closes the provider it retires.
        """

        class _ClosableSTT(MockSTTProvider):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        backend = MockConferenceBackend()
        stt = _ClosableSTT()
        recorder = MockMediaRecorder()
        kit, channel = await _channel(
            backend, stt=stt, recording=ConferenceRecordingConfig(), recorder=recorder
        )

        await channel.unplug_stt()
        await channel.unplug_recording()

        assert stt.closed is True
        assert recorder.closed is True

    async def test_a_shared_provider_is_left_to_its_owner(self) -> None:
        class _ClosableSTT(MockSTTProvider):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        backend = MockConferenceBackend()
        stt = _ClosableSTT()
        recorder = MockMediaRecorder()
        kit, channel = await _channel(
            backend,
            stt=stt,
            recording=ConferenceRecordingConfig(),
            recorder=recorder,
            close_providers=False,
        )

        await channel.unplug_stt()
        await channel.unplug_recording()

        assert stt.closed is False
        assert recorder.closed is False
