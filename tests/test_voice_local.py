"""Tests for LocalAudioBackend."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSessionState


def _mock_sounddevice() -> MagicMock:
    """Create a mock sounddevice module."""
    sd = MagicMock()
    sd.RawInputStream = MagicMock
    sd.CallbackStop = type("CallbackStop", (Exception,), {})
    sd.play = MagicMock()
    sd.wait = MagicMock()
    sd.stop = MagicMock()
    return sd


def _make_backend(**kwargs):
    """Create a LocalAudioBackend with mocked sounddevice."""
    sd = _mock_sounddevice()
    with patch.dict("sys.modules", {"sounddevice": sd}):
        from roomkit.voice.backends.local import LocalAudioBackend

        backend = LocalAudioBackend(**kwargs)
        backend._sd = sd
        return backend, sd


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestLocalAudioBackendProperties:
    def test_name(self) -> None:
        backend, _ = _make_backend()
        assert backend.name == "LocalAudio"

    def test_capabilities_include_interruption(self) -> None:
        backend, _ = _make_backend()
        assert VoiceCapability.INTERRUPTION in backend.capabilities

    def test_custom_sample_rates(self) -> None:
        backend, _ = _make_backend(input_sample_rate=48000, output_sample_rate=44100)
        assert backend._input_sample_rate == 48000
        assert backend._output_sample_rate == 44100

    def test_custom_block_duration(self) -> None:
        backend, _ = _make_backend(block_duration_ms=40)
        assert backend._block_duration_ms == 40

    def test_custom_devices(self) -> None:
        backend, _ = _make_backend(input_device=1, output_device="speakers")
        assert backend._input_device == 1
        assert backend._output_device == "speakers"


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestLocalAudioSessionManagement:
    async def test_connect_creates_session(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.channel_id == "voice-1"
        assert session.state == VoiceSessionState.ACTIVE

    async def test_connect_includes_metadata(self) -> None:
        backend, _ = _make_backend(input_sample_rate=48000)
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.metadata["input_sample_rate"] == 48000
        assert session.metadata["backend"] == "local_audio"

    async def test_connect_merges_custom_metadata(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1", metadata={"lang": "fr"})
        assert session.metadata["lang"] == "fr"
        assert "input_sample_rate" in session.metadata

    async def test_disconnect_ends_session(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)
        assert session.state == VoiceSessionState.ENDED
        assert backend.get_session(session.id) is None

    async def test_get_session(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        found = backend.get_session(session.id)
        assert found is not None
        assert found.id == session.id

    async def test_get_session_not_found(self) -> None:
        backend, _ = _make_backend()
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions_by_room(self) -> None:
        backend, _ = _make_backend()
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-1", "user-2", "voice-1")
        await backend.connect("room-2", "user-3", "voice-1")

        assert len(backend.list_sessions("room-1")) == 2
        assert len(backend.list_sessions("room-2")) == 1

    async def test_close_disconnects_all(self) -> None:
        backend, _ = _make_backend()
        s1 = await backend.connect("room-1", "user-1", "voice-1")
        s2 = await backend.connect("room-1", "user-2", "voice-1")
        await backend.close()
        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestLocalAudioCallbacks:
    async def test_on_audio_received_registers(self) -> None:
        backend, _ = _make_backend()
        cb = MagicMock()
        backend.on_audio_received(cb)
        assert backend._audio_received_callback is cb

    async def test_on_barge_in_registers(self) -> None:
        backend, _ = _make_backend()
        cb = MagicMock()
        backend.on_barge_in(cb)
        assert cb in backend._barge_in_callbacks


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------


class TestLocalAudioMicCapture:
    async def test_start_listening_creates_stream(self) -> None:
        backend, sd = _make_backend()

        mock_stream = MagicMock()
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)

        sd.RawInputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        assert session.id in backend._input_streams

    async def test_start_listening_uses_correct_params(self) -> None:
        backend, sd = _make_backend(
            input_sample_rate=48000,
            channels=2,
            block_duration_ms=40,
            input_device=3,
        )

        mock_stream = MagicMock()
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)

        call_kwargs = sd.RawInputStream.call_args[1]
        assert call_kwargs["samplerate"] == 48000
        assert call_kwargs["channels"] == 2
        assert call_kwargs["blocksize"] == 48000 * 40 // 1000  # 1920
        assert call_kwargs["dtype"] == "int16"
        assert call_kwargs["device"] == 3

    async def test_stop_listening_closes_stream(self) -> None:
        backend, sd = _make_backend()

        mock_stream = MagicMock()
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.stop_listening(session)

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert session.id not in backend._input_streams

    async def test_stop_listening_noop_if_not_listening(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        # Should not raise
        await backend.stop_listening(session)

    async def test_start_listening_twice_is_noop(self) -> None:
        backend, sd = _make_backend()

        mock_stream = MagicMock()
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.start_listening(session)  # Second call is no-op

        # Only one stream created
        assert sd.RawInputStream.call_count == 1

    async def test_stop_listening_closes_even_if_stop_raises(self) -> None:
        """stream.close() must be called even if stream.stop() raises."""
        backend, sd = _make_backend()

        mock_stream = MagicMock()
        mock_stream.stop.side_effect = RuntimeError("PortAudio error")
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.stop_listening(session)

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert session.id not in backend._input_streams

    async def test_disconnect_stops_listening(self) -> None:
        backend, sd = _make_backend()

        mock_stream = MagicMock()
        sd.RawInputStream = MagicMock(return_value=mock_stream)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.disconnect(session)

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    async def test_mic_callback_fires_on_audio_received(self) -> None:
        """The sounddevice callback should invoke on_audio_received."""
        backend, sd = _make_backend()

        captured_callback = None

        def fake_raw_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs["callback"]
            return MagicMock()

        sd.RawInputStream = fake_raw_input_stream

        received_frames: list[AudioFrame] = []

        def on_audio(session, frame):
            received_frames.append(frame)

        backend.on_audio_received(on_audio)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)

        assert captured_callback is not None

        # Simulate sounddevice calling the callback from the audio thread
        # (in tests we're in the same thread, call_soon_threadsafe dispatches)
        pcm_data = b"\x00\x01" * 160  # 160 samples of 16-bit audio
        captured_callback(pcm_data, 160, None, None)

        # Allow event loop to process call_soon_threadsafe
        await asyncio.sleep(0.01)

        assert len(received_frames) == 1
        assert received_frames[0].data == pcm_data
        assert received_frames[0].sample_rate == 16000


# ---------------------------------------------------------------------------
# Speaker playback
# ---------------------------------------------------------------------------


class TestLocalAudioSpeakerPlayback:
    async def test_send_audio_bytes(self) -> None:
        backend, sd = _make_backend()

        pytest.importorskip("numpy")

        session = await backend.connect("room-1", "user-1", "voice-1")

        # 4 bytes = 2 samples of PCM-16 LE
        pcm = b"\x00\x01\x00\x02"
        await backend.send_audio(session, pcm)

        sd.play.assert_called_once()
        sd.wait.assert_called_once()

    async def test_send_audio_stream(self) -> None:
        backend, sd = _make_backend()

        captured_callback = None
        mock_stream = MagicMock()

        def fake_raw_output_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs["callback"]
            return mock_stream

        sd.RawOutputStream = fake_raw_output_stream

        session = await backend.connect("room-1", "user-1", "voice-1")

        async def audio_gen():
            yield AudioChunk(data=b"\x00\x01\x00\x02")
            yield AudioChunk(data=b"\x00\x03\x00\x04")

        # Run send_audio in background so we can simulate the callback.
        send_task = asyncio.create_task(backend.send_audio(session, audio_gen()))
        await asyncio.sleep(0.05)  # let _consume pull all chunks

        # Simulate PortAudio callback draining the buffer.
        assert captured_callback is not None
        outdata = bytearray(1024)
        with pytest.raises(sd.CallbackStop):
            captured_callback(outdata, 512, None, None)

        await send_task

        mock_stream.start.assert_called_once()
        mock_stream.abort.assert_called_once()
        mock_stream.close.assert_called_once()

    async def test_is_playing_tracks_state(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert backend.is_playing(session) is False

        backend._playing_sessions.add(session.id)
        assert backend.is_playing(session) is True

    async def test_cancel_audio(self) -> None:
        backend, sd = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Nothing playing
        result = await backend.cancel_audio(session)
        assert result is False

        # Mark as playing (no output stream) → falls back to sd.stop()
        backend._playing_sessions.add(session.id)
        result = await backend.cancel_audio(session)
        assert result is True
        assert backend.is_playing(session) is False
        sd.stop.assert_called_once()

    async def test_cancel_audio_cancels_playback_task(self) -> None:
        backend, sd = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        mock_task = MagicMock()
        backend._playing_sessions.add(session.id)
        backend._output_streams[session.id] = MagicMock()
        backend._playback_tasks[session.id] = mock_task

        result = await backend.cancel_audio(session)
        assert result is True
        mock_task.cancel.assert_called_once()
        assert session.id not in backend._playback_tasks


# ---------------------------------------------------------------------------
# Transcription logging
# ---------------------------------------------------------------------------


class TestLocalAudioTranscription:
    async def test_send_transcription_does_not_raise(self) -> None:
        backend, _ = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        # Should just log, no crash
        await backend.send_transcription(session, "Hello", "user")
        await backend.send_transcription(session, "Hi there!", "assistant")


# ---------------------------------------------------------------------------
# Lazy loader
# ---------------------------------------------------------------------------


class TestLocalAudioLazyLoader:
    def test_get_local_audio_backend_returns_class(self) -> None:
        sd = _mock_sounddevice()
        with patch.dict("sys.modules", {"sounddevice": sd}):
            from roomkit.voice import get_local_audio_backend
            from roomkit.voice.backends.local import LocalAudioBackend

            cls = get_local_audio_backend()
            assert cls is LocalAudioBackend
