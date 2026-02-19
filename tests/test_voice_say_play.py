"""Tests for VoiceChannel.say() and VoiceChannel.play()."""

from __future__ import annotations

import io
import wave

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
from roomkit.voice.base import VoiceSession


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


def _wav_bytes(
    *,
    num_frames: int = 160,
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    """Build a valid in-memory WAV file (16-bit PCM mono by default)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames * channels)
    return buf.getvalue()


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
    async def test_play_wav_bytes(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _wav_bytes())

        assert len(backend.sent_audio) == 1
        # Backend receives raw PCM, not WAV container
        _, pcm = backend.sent_audio[0]
        assert not pcm.startswith(b"RIFF")

    async def test_play_wav_file_path(self, tmp_path) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(_wav_bytes())

        await channel.play(session, str(wav_path))

        assert len(backend.sent_audio) == 1

    async def test_play_wav_pathlib(self, tmp_path) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(_wav_bytes())

        await channel.play(session, wav_path)

        assert len(backend.sent_audio) == 1

    async def test_play_with_text_sends_transcription(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _wav_bytes(), text="Hello there")

        assert len(backend.sent_transcriptions) == 1
        assert backend.sent_transcriptions[0] == (session.id, "Hello there", "assistant")

    async def test_play_without_text_skips_transcription(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _wav_bytes())

        assert len(backend.sent_transcriptions) == 0

    async def test_play_raises_without_backend(self) -> None:
        channel = VoiceChannel("voice-1")

        session = _make_session()

        with pytest.raises(VoiceBackendNotConfiguredError, match="No voice backend"):
            await channel.play(session, _wav_bytes())

    async def test_play_does_not_require_tts(self) -> None:
        """play() works without a TTS provider — only backend is needed."""
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)  # no tts=...

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _wav_bytes())

        assert len(backend.sent_audio) == 1

    async def test_play_registers_playback_state(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        await channel.play(session, _wav_bytes())

        # _finish_playback was scheduled
        assert len(channel._scheduled_tasks) >= 1

    async def test_play_rejects_stereo_wav(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        with pytest.raises(ValueError, match="mono"):
            await channel.play(session, _wav_bytes(channels=2))

    async def test_play_rejects_8bit_wav(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        with pytest.raises(ValueError, match="16-bit"):
            await channel.play(session, _wav_bytes(sampwidth=1))

    async def test_play_rejects_non_wav_bytes(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        session = await backend.connect("room-1", "user-1", "voice-1")

        with pytest.raises((wave.Error, EOFError)):
            await channel.play(session, b"not-a-wav-file")
