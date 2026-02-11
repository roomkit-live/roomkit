"""Tests for ChannelBinding access/muted enforcement on voice audio paths."""

from __future__ import annotations

import asyncio

from roomkit import (
    MockSTTProvider,
    MockTTSProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import Access
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# ---------------------------------------------------------------------------
# VoiceChannel — binding-gated audio
# ---------------------------------------------------------------------------


def _speech_events(audio: bytes = b"\x01\x00" * 80) -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


class TestVoiceChannelBindingGate:
    """VoiceChannel drops inbound audio when binding forbids writes."""

    async def _setup(
        self, *, access: Access = Access.READ_WRITE, muted: bool = False
    ) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, MockSTTProvider, str]:
        stt = MockSTTProvider(transcripts=["Hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        vad = MockVADProvider(events=_speech_events())
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(voice=backend)
        channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1", access=access)
        if muted:
            await kit.mute(room.id, "voice-1")

        await kit.connect_voice(room.id, "user-1", "voice-1")

        return kit, channel, backend, stt, room.id

    async def test_read_write_allows_audio(self) -> None:
        kit, channel, backend, stt, room_id = await self._setup(access=Access.READ_WRITE)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        # STT should have been called (audio reached the pipeline)
        assert len(stt.calls) >= 1
        await kit.close()

    async def test_read_only_drops_audio(self) -> None:
        kit, channel, backend, stt, room_id = await self._setup(access=Access.READ_ONLY)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        # STT should NOT have been called (audio gated)
        assert len(stt.calls) == 0
        await kit.close()

    async def test_access_none_drops_audio(self) -> None:
        kit, channel, backend, stt, room_id = await self._setup(access=Access.NONE)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        assert len(stt.calls) == 0
        await kit.close()

    async def test_muted_drops_audio(self) -> None:
        kit, channel, backend, stt, room_id = await self._setup(muted=True)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        assert len(stt.calls) == 0
        await kit.close()

    async def test_write_only_allows_audio(self) -> None:
        """WRITE_ONLY means the channel CAN send inbound audio."""
        kit, channel, backend, stt, room_id = await self._setup(access=Access.WRITE_ONLY)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        assert len(stt.calls) >= 1
        await kit.close()

    async def test_dynamic_mute_stops_audio(self) -> None:
        """Muting via framework after session start stops audio."""
        kit, channel, backend, stt, room_id = await self._setup(access=Access.READ_WRITE)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        # Mute dynamically
        await kit.mute(room_id, "voice-1")

        # Reset VAD events so pipeline can process new frames
        vad_provider = channel._pipeline_config.vad  # type: ignore[union-attr]
        vad_provider._events = _speech_events()
        vad_provider._index = 0

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        # Audio should have been gated after mute
        assert len(stt.calls) == 0
        await kit.close()

    async def test_dynamic_unmute_resumes_audio(self) -> None:
        """Unmuting via framework after session start resumes audio."""
        kit, channel, backend, stt, room_id = await self._setup(muted=True)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        # Unmute dynamically
        await kit.unmute(room_id, "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        # Audio should flow again
        assert len(stt.calls) >= 1
        await kit.close()

    async def test_dynamic_set_access_read_only_stops_audio(self) -> None:
        """set_access(READ_ONLY) after session start stops audio."""
        kit, channel, backend, stt, room_id = await self._setup(access=Access.READ_WRITE)
        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None

        # Change access dynamically
        await kit.set_access(room_id, "voice-1", Access.READ_ONLY)

        # Reset VAD events
        vad_provider = channel._pipeline_config.vad  # type: ignore[union-attr]
        vad_provider._events = _speech_events()
        vad_provider._index = 0

        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))
        await asyncio.sleep(0.15)

        assert len(stt.calls) == 0
        await kit.close()


# ---------------------------------------------------------------------------
# RealtimeVoiceChannel — binding-gated audio
# ---------------------------------------------------------------------------


class TestRealtimeVoiceChannelBindingGate:
    """RealtimeVoiceChannel drops client audio when binding forbids writes."""

    async def _setup(
        self,
        *,
        access: Access = Access.READ_WRITE,
        muted: bool = False,
        mute_on_tool_call: bool = False,
    ) -> tuple[
        RoomKit,
        RealtimeVoiceChannel,
        MockRealtimeProvider,
        MockRealtimeTransport,
        str,
    ]:
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-voice-1",
            provider=provider,
            transport=transport,
            system_prompt="Test agent",
            mute_on_tool_call=mute_on_tool_call,
        )

        kit = RoomKit()
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-voice-1", access=access)
        if muted:
            await kit.mute(room.id, "rt-voice-1")

        await channel.start_session(room.id, "user-1", "fake-ws")
        return kit, channel, provider, transport, room.id

    async def test_read_write_allows_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(access=Access.READ_WRITE)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await transport.simulate_client_audio(session, b"client-audio")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 1
        await kit.close()

    async def test_read_only_drops_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(access=Access.READ_ONLY)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await transport.simulate_client_audio(session, b"client-audio")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 0
        await kit.close()

    async def test_access_none_drops_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(access=Access.NONE)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await transport.simulate_client_audio(session, b"client-audio")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 0
        await kit.close()

    async def test_muted_drops_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(muted=True)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await transport.simulate_client_audio(session, b"client-audio")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 0
        await kit.close()

    async def test_write_only_allows_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(access=Access.WRITE_ONLY)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await transport.simulate_client_audio(session, b"client-audio")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 1
        await kit.close()

    async def test_dynamic_mute_stops_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup()
        sessions = list(channel._sessions.values())
        session = sessions[0]

        # Verify audio works initially
        await transport.simulate_client_audio(session, b"audio-before")
        await asyncio.sleep(0.05)
        assert len(provider.sent_audio) == 1

        # Mute dynamically
        await kit.mute(room_id, "rt-voice-1")

        await transport.simulate_client_audio(session, b"audio-after")
        await asyncio.sleep(0.05)

        # Should still be 1 (second audio was gated)
        assert len(provider.sent_audio) == 1
        await kit.close()

    async def test_dynamic_unmute_resumes_audio(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup(muted=True)
        sessions = list(channel._sessions.values())
        session = sessions[0]

        # Verify audio is gated
        await transport.simulate_client_audio(session, b"audio-before")
        await asyncio.sleep(0.05)
        assert len(provider.sent_audio) == 0

        # Unmute dynamically
        await kit.unmute(room_id, "rt-voice-1")

        await transport.simulate_client_audio(session, b"audio-after")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 1
        await kit.close()

    async def test_dynamic_set_access_read_only(self) -> None:
        kit, channel, provider, transport, room_id = await self._setup()
        sessions = list(channel._sessions.values())
        session = sessions[0]

        await kit.set_access(room_id, "rt-voice-1", Access.READ_ONLY)

        await transport.simulate_client_audio(session, b"audio-gated")
        await asyncio.sleep(0.05)

        assert len(provider.sent_audio) == 0
        await kit.close()


# ---------------------------------------------------------------------------
# mute_on_tool_call flag
# ---------------------------------------------------------------------------


class TestMuteOnToolCall:
    """Test mute_on_tool_call=True/False behaviour."""

    async def test_mute_on_tool_call_true_mutes_transport(self) -> None:
        """When mute_on_tool_call=True, transport is muted during tool execution."""
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=transport,
            system_prompt="Test",
            mute_on_tool_call=True,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-1")
        session = await channel.start_session(room.id, "user-1", "fake-ws")

        await provider.simulate_tool_call(session, "call-1", "get_data", {})
        await asyncio.sleep(0.1)

        # Transport should have been muted and then unmuted
        mute_calls = [c for c in transport.calls if c.method == "set_input_muted"]
        assert len(mute_calls) == 2
        assert mute_calls[0].args["muted"] is True
        assert mute_calls[1].args["muted"] is False
        await kit.close()

    async def test_mute_on_tool_call_false_skips_mute(self) -> None:
        """Default (mute_on_tool_call=False) does NOT mute transport."""
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=transport,
            system_prompt="Test",
            # mute_on_tool_call defaults to False
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-1")
        session = await channel.start_session(room.id, "user-1", "fake-ws")

        await provider.simulate_tool_call(session, "call-1", "get_data", {})
        await asyncio.sleep(0.1)

        # Transport should NOT have been muted
        mute_calls = [c for c in transport.calls if c.method == "set_input_muted"]
        assert len(mute_calls) == 0
        await kit.close()
