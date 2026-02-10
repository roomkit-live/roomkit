"""Tests for VoiceChannel.say() and VoiceChannel.play()."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from roomkit import (
    MockTTSProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceBackendNotConfiguredError,
    VoiceChannel,
    VoiceNotConfiguredError,
)
from roomkit.models.enums import HookTrigger
from roomkit.models.hook import HookResult
from roomkit.voice.base import AudioChunk, VoiceSession


def _make_session(
    session_id: str = "sess-1",
    room_id: str = "room-1",
    channel_id: str = "voice-1",
) -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id=room_id,
        participant_id="user-1",
        channel_id=channel_id,
    )


async def _chunks(*words: str) -> AsyncIterator[AudioChunk]:
    for i, word in enumerate(words):
        yield AudioChunk(
            data=f"audio-{word}".encode(),
            sample_rate=16000,
            is_final=(i == len(words) - 1),
        )


class TestSay:
    async def test_say_synthesizes_and_sends_audio(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "Hello!")

        # TTS was called
        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "Hello!"

        # Audio was sent to backend
        assert len(backend.sent_audio) == 1

        # Transcription was sent
        assert len(backend.sent_transcriptions) == 1
        assert backend.sent_transcriptions[0] == (session.id, "Hello!", "assistant")

    async def test_say_passes_voice_parameter(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "Hi", voice="custom-voice")

        assert tts.calls[0]["voice"] == "custom-voice"

    async def test_say_raises_without_tts(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        with pytest.raises(VoiceNotConfiguredError, match="No TTS provider"):
            await channel.say(session, "Hello")

    async def test_say_raises_without_backend(self) -> None:
        tts = MockTTSProvider()
        channel = VoiceChannel("voice-1", tts=tts)

        session = _make_session()

        with pytest.raises(VoiceBackendNotConfiguredError, match="No voice backend"):
            await channel.say(session, "Hello")

    async def test_say_registers_playback_state(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        # Playback state should be set during send, then cleared by _finish_playback
        # We can check the playback was registered by verifying _finish_playback was scheduled
        await channel.say(session, "Hello")

        # After say() returns, _finish_playback is scheduled (async drain delay).
        # The playback state was registered during execution.
        # Verify by checking scheduled tasks exist.
        assert len(channel._scheduled_tasks) >= 1

    async def test_say_fires_before_tts_hook(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        kit = RoomKit(tts=tts, voice=backend)
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        hook_calls: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def on_before_tts(event, context):
            hook_calls.append(event)

        await channel.say(session, "Greetings")

        assert "Greetings" in hook_calls

    async def test_say_fires_after_tts_hook(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        kit = RoomKit(tts=tts, voice=backend)
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        hook_calls: list[str] = []

        @kit.hook(HookTrigger.AFTER_TTS)
        async def on_after_tts(event, context):
            hook_calls.append(event)

        await channel.say(session, "Farewell")

        assert "Farewell" in hook_calls

    async def test_say_before_tts_hook_can_block(self) -> None:
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        kit = RoomKit(tts=tts, voice=backend)
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def block_tts(event, context):
            return HookResult(action="block", reason="not allowed")

        await channel.say(session, "Blocked text")

        # TTS should not have been called
        assert len(tts.calls) == 0
        assert len(backend.sent_audio) == 0

    async def test_say_without_framework_skips_hooks(self) -> None:
        """say() works fine without framework — just no hooks fired."""
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", tts=tts, backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.say(session, "No hooks")

        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "No hooks"


class TestPlay:
    async def test_play_sends_audio_chunks(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _chunks("hello", "world"))

        assert len(backend.sent_audio) == 1

    async def test_play_sends_raw_bytes(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"raw-audio-data")

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0] == (session.id, b"raw-audio-data")

    async def test_play_with_text_sends_transcription(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"audio-data", text="Hello there")

        assert len(backend.sent_transcriptions) == 1
        assert backend.sent_transcriptions[0] == (session.id, "Hello there", "assistant")

    async def test_play_without_text_skips_transcription(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"audio-data")

        assert len(backend.sent_transcriptions) == 0

    async def test_play_raises_without_backend(self) -> None:
        channel = VoiceChannel("voice-1")

        session = _make_session()

        with pytest.raises(VoiceBackendNotConfiguredError, match="No voice backend"):
            await channel.play(session, b"audio-data")

    async def test_play_does_not_require_tts(self) -> None:
        """play() works without a TTS provider — only backend is needed."""
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)  # no tts=...

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"pre-rendered")

        assert len(backend.sent_audio) == 1

    async def test_play_registers_playback_state(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"audio-data")

        # _finish_playback was scheduled
        assert len(channel._scheduled_tasks) >= 1

    async def test_play_wraps_through_pipeline(self) -> None:
        from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider

        backend = MockVoiceBackend()
        vad = MockVADProvider()
        config = AudioPipelineConfig(vad=vad)
        channel = VoiceChannel("voice-1", backend=backend, pipeline=config)

        session = await backend.connect("room-1", "user-1", "voice-1")

        # play() with an async iterator should wrap through pipeline outbound
        await channel.play(session, _chunks("test"))

        assert len(backend.sent_audio) == 1

    async def test_play_bytes_skip_pipeline(self) -> None:
        """Raw bytes are not wrapped through the pipeline (no framing info)."""
        from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider

        backend = MockVoiceBackend()
        vad = MockVADProvider()
        config = AudioPipelineConfig(vad=vad)
        channel = VoiceChannel("voice-1", backend=backend, pipeline=config)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, b"raw-bytes")

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0] == (session.id, b"raw-bytes")
