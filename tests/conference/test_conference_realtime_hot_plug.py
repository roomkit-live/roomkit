"""Hot-plugging the speech-to-speech provider (RFC §12.10.4, §12.10.12).

The realtime slot follows every rule the other needs follow — first need,
occupied-slot refusal, idempotent unplug, last-need retirement — plus two of
its own: mutual exclusion with the synthesizer, and lanes shared with the
recognizer that survive whichever of the two unplugs first.
"""

from __future__ import annotations

import pytest

from roomkit import (
    ConferenceCapability,
    ConferenceRealtimeConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.lane_audio import say
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until


async def _transport_kit(
    backend: MockConferenceBackend | None = None,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = backend or MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, **channel_kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


class TestPlug:
    async def test_a_plug_joins_the_occupied_conference_and_hears_it(self) -> None:
        _, channel, backend = await _transport_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        assert backend.bots == [], "pure transport joins nothing"

        provider = MockRealtimeProvider()
        await channel.plug_realtime(ConferenceRealtimeConfig(provider=provider))

        assert backend.bots, "the plug is a first need: the occupied room is joined"
        assert track.id in backend.subscriptions
        assert track.id in channel.active_lanes
        bot = backend.bots[0]
        assert backend.bot_grants[bot.id].publish_audio, "a voice needs to speak"

        await say(backend, track)
        await until(lambda: bool(provider.sent_audio))

    async def test_an_occupied_slot_is_refused(self) -> None:
        _, channel, _, _ = await realtime_kit()

        with pytest.raises(ValueError, match="already plugged"):
            await channel.plug_realtime(ConferenceRealtimeConfig(provider=MockRealtimeProvider()))

    async def test_the_synthesizer_and_the_provider_exclude_each_other(self) -> None:
        _, channel, _ = await _transport_kit(tts=MockTTSProvider())
        with pytest.raises(ValueError, match="mutually exclusive"):
            await channel.plug_realtime(ConferenceRealtimeConfig(provider=MockRealtimeProvider()))

        _, channel, _, _ = await realtime_kit()
        with pytest.raises(ValueError, match="unplug_realtime"):
            await channel.plug_tts(MockTTSProvider())

    async def test_grants_widen_in_place_where_the_backend_can(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.BOT_GRANT_UPDATE)
        _, channel, backend = await _transport_kit(backend, stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")
        bot = backend.bots[0]
        assert not backend.bot_grants[bot.id].publish_audio, "a listener does not speak"

        await channel.plug_realtime(ConferenceRealtimeConfig(provider=MockRealtimeProvider()))

        assert backend.bots == [bot], "the session survived the widening"
        assert backend.bot_grants[bot.id].publish_audio


class TestUnplug:
    async def test_an_empty_slot_unplugs_as_a_no_op(self) -> None:
        _, channel, _ = await _transport_kit()

        await channel.unplug_realtime()

    async def test_unplugging_the_last_need_takes_the_bot_out(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        assert backend.bots

        await channel.unplug_realtime()

        assert backend.bots == [], "a bot with no function leaves (RFC 12.10.4)"
        assert track.id not in backend.subscriptions
        assert channel.active_lanes == {}
        assert channel._pipeline is None
        assert any(call.method == "disconnect" for call in provider.calls)
        assert any(call.method == "close" for call in provider.calls)

    async def test_the_provider_survives_when_the_channel_does_not_own_it(self) -> None:
        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(provider=provider, close_providers=False)
        await channel._realtime.ensure_session(ROOM)

        await channel.unplug_realtime()

        assert all(call.method != "close" for call in provider.calls)

    async def test_the_lanes_survive_the_unplug_that_is_not_theirs_alone(self) -> None:
        """stt and realtime share the lanes; whichever unplugs first leaves
        them for the other (RFC 12.10.12)."""
        _, channel, backend, _ = await realtime_kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        assert track.id in channel.active_lanes

        await channel.unplug_realtime()

        assert track.id in channel.active_lanes, "the recognizer still consumes them"
        assert channel._pipeline is not None
        assert backend.bots, "recognition is still a need"

    async def test_unplug_stt_leaves_the_lanes_to_the_realtime_mix(self) -> None:
        _, channel, backend, _ = await realtime_kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        assert track.id in channel.active_lanes

        await channel.unplug_stt()

        assert track.id in channel.active_lanes, "the mix still consumes them"
        assert channel._pipeline is not None
        assert backend.bots, "speech-to-speech is still a need"

    async def test_the_round_trip_restores_the_composition(self) -> None:
        _, channel, backend, _ = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await channel.unplug_realtime()
        assert backend.bots == []

        replacement = MockRealtimeProvider()
        await channel.plug_realtime(ConferenceRealtimeConfig(provider=replacement))

        assert backend.bots, "the plug re-ran the occupancy probe"
        assert track.id in channel.active_lanes
        await say(backend, track)
        await until(lambda: bool(replacement.sent_audio))

    async def test_info_follows_the_configuration_in_force(self) -> None:
        _, channel, _, _ = await realtime_kit()
        assert channel.info()["realtime_configured"] is True

        await channel.unplug_realtime()

        info = channel.info()
        assert info["realtime_configured"] is False
        assert info["realtime_provider"] is None
