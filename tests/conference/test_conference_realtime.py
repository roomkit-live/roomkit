"""Speech-to-speech composition: sessions, transcripts, tools (RFC §12.10.12).

The provider hears a mix and speaks on the bot track. These tests cover the
boundary contracts around that: what a configuration refuses, when a session
is established and what its failure costs, whose words are kept, and how a
tool call is answered. The voice path — floor, barge-in, terminal chunks —
has its own file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "room-1"


class _Source(Channel):
    """A channel that only exists to originate events of a given type."""

    def __init__(self, channel_id: str, channel_type: ChannelType) -> None:
        super().__init__(channel_id)
        self._type = channel_type

    @property
    def channel_type(self) -> ChannelType:
        return self._type

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self._type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Wait until a predicate holds, rather than towards when it might."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0)


async def realtime_kit(
    *,
    provider: MockRealtimeProvider | None = None,
    config: ConferenceRealtimeConfig | None = None,
    backend: MockConferenceBackend | None = None,
    source_type: ChannelType = ChannelType.AI,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend, MockRealtimeProvider]:
    provider = provider or MockRealtimeProvider()
    config = config or ConferenceRealtimeConfig(provider=provider)
    backend = backend or MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, realtime=config, **channel_kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src", source_type))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    return kit, channel, backend, provider


class _RefusingProvider(MockRealtimeProvider):
    """Connects never: the provider is down."""

    def __init__(self) -> None:
        super().__init__()
        self.connect_attempts = 0

    async def connect(self, session, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.connect_attempts += 1
        raise RuntimeError("provider down")


class TestConfigurationRefusals:
    async def test_tts_and_realtime_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                tts=MockTTSProvider(),
                realtime=ConferenceRealtimeConfig(provider=MockRealtimeProvider()),
            )

    async def test_an_e2ee_conference_refuses_a_realtime_provider(self) -> None:
        with pytest.raises(ValueError, match="encrypted"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                realtime=ConferenceRealtimeConfig(provider=MockRealtimeProvider()),
                e2ee=True,
            )

    async def test_tools_without_a_handler_are_refused(self) -> None:
        with pytest.raises(ValueError, match="tool_handler"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                realtime=ConferenceRealtimeConfig(
                    provider=MockRealtimeProvider(),
                    tools=[{"name": "lookup"}],
                ),
            )


class TestSessionLifecycle:
    async def test_nothing_connects_before_a_need(self) -> None:
        _, channel, _, provider = await realtime_kit()

        assert channel._realtime.session_for(ROOM) is None
        assert all(call.method != "connect" for call in provider.calls)

    async def test_the_session_carries_the_configuration(self) -> None:
        provider = MockRealtimeProvider()
        _, channel, backend, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider,
                system_prompt="Be brief.",
                voice="verse",
                input_sample_rate=24000,
                server_vad=False,
            ),
        )

        session = await channel._realtime.ensure_session(ROOM)

        assert session is not None
        assert session.participant_id == "roomkit"
        connect = next(call for call in provider.calls if call.method == "connect")
        assert connect.args["system_prompt"] == "Be brief."
        assert connect.args["voice"] == "verse"
        assert connect.args["input_sample_rate"] == 24000
        assert connect.args["server_vad"] is False
        assert backend.bots, "the session's first need joins the bot"

    async def test_the_mixed_path_connects_and_feeds_the_provider(self) -> None:
        """End to end: a participant speaks, the lanes feed the mixer, the
        mixer establishes the session and the provider hears the window."""
        from tests.conference.lane_audio import say

        _, channel, backend, provider = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, track)
        await until(lambda: bool(provider.sent_audio))

        assert channel._realtime.session_for(ROOM) is not None
        assert sum(1 for call in provider.calls if call.method == "connect") == 1

    async def test_a_connect_failure_fails_nothing_and_cools_down(self) -> None:
        provider = _RefusingProvider()
        _, channel, backend, _ = await realtime_kit(
            provider=provider, config=ConferenceRealtimeConfig(provider=provider)
        )

        assert await channel._realtime.ensure_session(ROOM) is None
        assert await channel._realtime.ensure_session(ROOM) is None

        assert provider.connect_attempts == 1, "the cooldown holds retries off"
        assert backend.bots, "the join itself stood: only the provider failed"

    async def test_a_detach_disconnects_the_session(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await kit.detach_channel(ROOM, "conf")
        from tests.conference.test_conference_races import _settle

        await _settle(channel)

        assert any(call.method == "disconnect" for call in provider.calls)
        assert channel._realtime.session_for(ROOM) is None

    async def test_a_lost_bot_session_takes_the_provider_session_with_it(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await backend.simulate_bot_disconnected(bot)

        assert channel._realtime.session_for(ROOM) is None
        await until(lambda: any(call.method == "disconnect" for call in provider.calls))


class TestTranscription:
    async def test_user_transcriptions_are_discarded(self) -> None:
        """The provider heard a mix; its user-side transcript names nobody and
        is not stored (RFC 12.10.12) — the lanes' STT is the attributed path."""
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "who said this?", role="user")

        events = await kit.store.list_events(ROOM)
        assert all("who said this?" not in str(event.content) for event in events)

    async def test_assistant_finals_become_room_events(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonjour la salle.", role="assistant")

        events = await kit.store.list_events(ROOM)
        spoken = [e for e in events if isinstance(e.content, TextContent)]
        assert [e.content.body for e in spoken] == ["Bonjour la salle."]
        (event,) = spoken
        assert event.source.channel_type is ChannelType.CONFERENCE
        assert event.source.participant_id is None
        assert event.metadata["role"] == "assistant"

    async def test_assistant_finals_are_not_respoken_or_reinjected(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonjour la salle.", role="assistant")

        assert backend.published_audio == []
        assert provider.injected_texts == []

    async def test_assistant_partials_are_not_stored(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonj", role="assistant", is_final=False)

        events = await kit.store.list_events(ROOM)
        assert all("Bonj" not in str(event.content) for event in events)


class TestDeliver:
    async def test_an_ai_text_event_is_injected_not_synthesized(self) -> None:
        kit, channel, backend, provider = await realtime_kit()

        await kit.send_event(ROOM, "src", TextContent(body="résume la réunion"))

        assert [(text, role) for _, text, role in provider.injected_texts] == [
            ("résume la réunion", "system")
        ]
        assert backend.published_audio == []

    async def test_non_ai_text_is_not_injected_by_default(self) -> None:
        kit, _, _, provider = await realtime_kit(source_type=ChannelType.SMS)

        await kit.send_event(ROOM, "src", TextContent(body="un SMS qui passe"))

        assert provider.injected_texts == []

    async def test_speak_text_events_injects_non_ai_text(self) -> None:
        kit, _, _, provider = await realtime_kit(
            source_type=ChannelType.SMS, speak_text_events=True
        )

        await kit.send_event(ROOM, "src", TextContent(body="un SMS qui passe"))

        assert [text for _, text, _ in provider.injected_texts] == ["un SMS qui passe"]


class TestToolCalls:
    async def test_a_tool_call_is_answered_through_the_handler(self) -> None:
        seen: list[tuple[str, str, dict[str, object]]] = []

        async def handler(room_id: str, name: str, arguments: dict) -> str:  # type: ignore[type-arg]
            seen.append((room_id, name, arguments))
            return '{"weather": "sunny"}'

        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider,
                tools=[{"name": "get_weather"}],
                tool_handler=handler,
            ),
        )
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "get_weather", {"city": "QC"})
        await until(lambda: bool(provider.tool_results))

        assert seen == [(ROOM, "get_weather", {"city": "QC"})]
        assert provider.tool_results == [(session.id, "call-1", '{"weather": "sunny"}')]

    async def test_a_failing_handler_submits_an_error_result(self) -> None:
        async def handler(room_id: str, name: str, arguments: dict) -> str:  # type: ignore[type-arg]
            raise RuntimeError("backend down")

        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider, tools=[{"name": "x"}], tool_handler=handler
            ),
        )
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "x", {})
        await until(lambda: bool(provider.tool_results))

        (_, call_id, result) = provider.tool_results[0]
        assert call_id == "call-1"
        assert "error" in result and "backend down" in result

    async def test_a_call_with_no_handler_configured_still_gets_an_answer(self) -> None:
        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(provider=provider)
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "surprise", {})
        await until(lambda: bool(provider.tool_results))

        (_, _, result) = provider.tool_results[0]
        assert "no handler" in result


class TestDisclosure:
    async def test_info_reports_the_composition(self) -> None:
        _, channel, _, _ = await realtime_kit()

        info = channel.info()

        assert info["realtime_configured"] is True
        assert info["realtime_provider"] == "MockRealtimeProvider"
        assert info["rooms"][ROOM]["realtime_active"] is False

    async def test_a_connected_session_reads_active(self) -> None:
        _, channel, _, _ = await realtime_kit()
        await channel._realtime.ensure_session(ROOM)

        assert channel.info()["rooms"][ROOM]["realtime_active"] is True
