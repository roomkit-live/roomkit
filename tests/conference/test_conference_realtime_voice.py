"""The provider's voice on the bot track (RFC §12.10.12, §12.10.4, §12.10.5).

A response is an utterance like any other: floor, chunks, terminal ``is_final``
— on a natural end and on a barge-in — and abandonment without a boundary when
the channel leaves. The barge-in itself stays the lanes': the per-lane VAD is
the interruption sensor, the scope is enforced on it, and a landed one reaches
the provider as a cancellation.
"""

from __future__ import annotations

from roomkit import (
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
)
from tests.conference.lane_audio import say
from tests.conference.test_conference_races import _settle
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until


class TestUtterances:
    async def test_a_response_publishes_and_closes_on_the_bot_track(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"bonjour-audio")
        await provider.simulate_response_end(session)
        await until(lambda: any(u.complete for u in backend.utterances_for(bot)))

        (utterance,) = backend.utterances_for(bot)
        assert utterance.data == b"bonjour-audio"
        assert utterance.complete, "the response ends on an is_final boundary"
        assert backend.published_audio[0].sample_rate == 24000

    async def test_audio_without_a_response_start_still_speaks(self) -> None:
        """The ABC never promises on_response_start; audio with no open
        response opens one rather than being dropped."""
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_audio(session, b"unannounced")
        await provider.simulate_response_end(session)
        await until(lambda: any(u.complete for u in backend.utterances_for(bot)))

        assert backend.utterances_for(bot)[0].data == b"unannounced"

    async def test_two_responses_never_interleave(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"first")
        await provider.simulate_response_end(session)
        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"second")
        await provider.simulate_response_end(session)
        await until(lambda: sum(1 for u in backend.utterances_for(bot) if u.complete) == 2)

        first, second = backend.utterances_for(bot)
        assert first.data == b"first"
        assert second.data == b"second"

    async def test_a_response_the_provider_never_closed_yields_the_floor(self) -> None:
        """A second response_start ends an unclosed first response rather than
        queueing behind a pump that would wait forever."""
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"first")
        # No response_end: the provider moved on.
        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"second")
        await provider.simulate_response_end(session)
        await until(lambda: sum(1 for u in backend.utterances_for(bot) if u.complete) == 2)


class TestBargeIn:
    async def test_a_barge_in_stops_playback_and_cancels_the_provider(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"a-long-answer")
        await until(lambda: bool(backend.published_audio))

        await say(backend, track, silence=0)
        await until(lambda: bool(backend.playback_stops))

        assert backend.playback_stops == [bot.id]
        await until(lambda: any(call.method == "interrupt" for call in provider.calls))
        # The terminal chunk still goes out: an interrupted utterance is closed.
        await until(lambda: any(u.complete for u in backend.utterances_for(bot)))

    async def test_audio_arriving_after_the_barge_in_is_dropped(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"a-long-answer")
        await until(lambda: bool(backend.published_audio))
        await say(backend, track, silence=0)
        await until(lambda: any(u.complete for u in backend.utterances_for(bot)))
        published = len(backend.published_audio)

        await provider.simulate_audio(session, b"trailing-delta")
        await provider.simulate_response_end(session)
        await _settle(channel)

        assert len(backend.published_audio) == published
        assert sum(1 for u in backend.utterances_for(bot) if u.complete) == 1

    async def test_scope_none_lets_the_provider_finish(self) -> None:
        _, channel, backend, provider = await realtime_kit(
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.NONE),
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"the-presentation")
        await until(lambda: bool(backend.published_audio))
        await say(backend, track)

        assert backend.playback_stops == []
        assert all(call.method != "interrupt" for call in provider.calls)

        await provider.simulate_response_end(session)
        await until(lambda: any(u.complete for u in backend.utterances_for(bot)))

    async def test_an_allowlist_excludes_the_unlisted(self) -> None:
        _, channel, backend, provider = await realtime_kit(
            interruption=ConferenceInterruptionConfig(
                scope=ConferenceInterruptionScope.ALLOWLIST,
                allowlist=["p-moderator"],
            ),
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"the-answer")
        await until(lambda: bool(backend.published_audio))
        await say(backend, track)

        assert backend.playback_stops == []
        assert all(call.method != "interrupt" for call in provider.calls)


class TestLeaving:
    async def test_a_detach_abandons_the_utterance_without_a_boundary(self) -> None:
        kit, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await provider.simulate_response_start(session)
        await provider.simulate_audio(session, b"cut-mid-sentence")
        await until(lambda: bool(backend.published_audio))

        await kit.detach_channel(ROOM, "conf")
        await _settle(channel)

        assert not any(u.complete for u in backend.utterances_for(bot)), (
            "a session on its way out is owed no terminal chunk (RFC 12.10.4)"
        )
        assert any(call.method == "disconnect" for call in provider.calls)
