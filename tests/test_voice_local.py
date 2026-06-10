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
        # Normal completion uses stop() (graceful drain), not abort().
        mock_stream.stop.assert_called_once()
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


# ---------------------------------------------------------------------------
# Realtime speaker prebuffer (priming state machine)
# ---------------------------------------------------------------------------

_BLOCK = 960  # 20ms @ 24kHz mono int16
_PCM = b"\x01\x00"


def _drain_block(backend) -> bytes:
    """Invoke the realtime speaker callback for one 20ms block."""
    out = bytearray(_BLOCK)
    backend._rt_speaker_callback(out, _BLOCK // 2, None, None)
    return bytes(out)


async def _rt_backend(**kwargs):
    """LocalAudioBackend in realtime mode (accepted session, mocked stream)."""
    backend, _ = _make_backend(output_sample_rate=24000, block_duration_ms=20, **kwargs)
    session = await backend.connect("room-1", "user-1", "voice-1")
    await backend.accept(session, None)
    return backend, session


class TestRealtimePrebuffer:
    """Default prebuffer: 120ms = 5760 bytes; one block = 960 bytes."""

    async def test_priming_holds_silence_below_prebuffer(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (2880 // 2))  # 60ms < 120ms
        assert _drain_block(backend) == b"\x00" * _BLOCK
        assert backend._rt_buffered_bytes == 2880  # nothing consumed

    async def test_priming_releases_at_prebuffer_threshold(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (5760 // 2))  # exactly 120ms
        # Drain starts in the same callback as the release — no wasted block.
        assert _drain_block(backend) == _PCM * (_BLOCK // 2)
        assert backend._rt_buffered_bytes == 5760 - _BLOCK

    async def test_short_response_drains_on_end_of_response(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * 400)  # 800B, far below prebuffer
        assert _drain_block(backend) == b"\x00" * _BLOCK  # still priming
        backend.end_of_response(session)
        out = _drain_block(backend)
        assert out[:800] == _PCM * 400
        assert out[800:] == b"\x00" * (_BLOCK - 800)
        assert backend.rt_underruns == 0  # clean end, not starvation
        assert backend._rt_response_complete is False  # consumed by the drain

    async def test_end_of_response_noop_while_draining(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (5760 * 2 // 2))
        assert _drain_block(backend) == _PCM * (_BLOCK // 2)
        backend.end_of_response(session)
        # Mid-drain the flag changes nothing — audio keeps flowing normally.
        assert _drain_block(backend) == _PCM * (_BLOCK // 2)

    async def test_underrun_counts_and_reprimes(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (5760 // 2))
        for _ in range(6):  # 5760 / 960 = 6 full blocks
            _drain_block(backend)
        assert backend.rt_underruns == 0
        _drain_block(backend)  # starved mid-response (no end_of_response)
        assert backend.rt_underruns == 1
        # Re-primed: a sub-prebuffer append stays silent again.
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        assert _drain_block(backend) == b"\x00" * _BLOCK

    async def test_clean_end_no_underrun(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (5760 // 2))
        backend.end_of_response(session)
        for _ in range(7):  # 6 audio blocks + 1 exhaustion block
            _drain_block(backend)
        assert backend.rt_underruns == 0

    async def test_interrupt_during_priming_discards_partial(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (2880 // 2))
        backend.interrupt(session)
        assert backend._rt_buffered_bytes == 0
        assert _drain_block(backend) == b"\x00" * _BLOCK
        # First append clears the interrupt flag but must re-prime from zero.
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        assert backend._rt_interrupted is False
        assert _drain_block(backend) == b"\x00" * _BLOCK
        assert backend._rt_buffered_bytes == _BLOCK

    async def test_stale_end_of_response_ignored_after_interrupt(self) -> None:
        backend, session = await _rt_backend()
        backend.interrupt(session)
        backend.end_of_response(session)  # Gemini fires response_end on barge-in
        assert backend._rt_response_complete is False
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        # Without the guard, the stale EOR would release this audio early.
        assert _drain_block(backend) == b"\x00" * _BLOCK

    async def test_prime_idle_valve_flushes_partial_buffer(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * 400)  # 800B, EOR never arrives
        for _ in range(5):  # _rt_prime_max_idle_blocks = 100ms / 20ms = 5
            assert _drain_block(backend) == b"\x00" * _BLOCK
        out = _drain_block(backend)  # valve fires after ~100ms of priming
        assert out[:800] == _PCM * 400

    async def test_accept_after_disconnect_replays(self) -> None:
        backend, session = await _rt_backend()
        await backend.disconnect(session)
        session2 = await backend.connect("room-1", "user-2", "voice-1")
        await backend.accept(session2, None)
        await backend.send_audio(session2, _PCM * (_BLOCK // 2))
        # _rt_closing used to persist across sessions and drop everything.
        assert backend._rt_buffered_bytes == _BLOCK

    async def test_priming_idle_releases_half_duplex_mute(self) -> None:
        backend, session = await _rt_backend()  # mute_mic_during_playback=True
        await backend.send_audio(session, _PCM * (5760 // 2))
        backend.end_of_response(session)
        for _ in range(8):  # full drain + idle priming callback
            _drain_block(backend)
        assert backend._playing_sessions == set()

    async def test_prebuffer_zero_plays_first_byte(self) -> None:
        backend, session = await _rt_backend(rt_prebuffer_ms=0)
        await backend.send_audio(session, _PCM * 100)  # 200B, tiny
        out = _drain_block(backend)
        assert out[:200] == _PCM * 100  # legacy play-immediately behavior


class TestContinuousPlayedCallbacks:
    """Played callbacks fire on every block — silence included.

    The playback-time AEC reference is wired through on_audio_played;
    skipping silent blocks desyncs the reference timeline from the real
    speaker output and forces AEC3 delay re-estimation at every gap.
    """

    async def test_played_callbacks_fire_on_idle_silence(self) -> None:
        backend, _session = await _rt_backend()
        played: list[bytes] = []
        backend.on_audio_played(lambda s, f: played.append(f.data))
        assert _drain_block(backend) == b"\x00" * _BLOCK  # idle, nothing queued
        assert played == [b"\x00" * _BLOCK]

    async def test_played_callbacks_fire_during_priming_and_interrupt(self) -> None:
        backend, session = await _rt_backend()
        played: list[bytes] = []
        backend.on_audio_played(lambda s, f: played.append(f.data))
        await backend.send_audio(session, _PCM * (2880 // 2))  # sub-prebuffer
        _drain_block(backend)  # priming hold
        backend.interrupt(session)
        _drain_block(backend)  # interrupted mute
        assert played == [b"\x00" * _BLOCK, b"\x00" * _BLOCK]

    async def test_transport_aec_reference_still_skips_silence(self) -> None:
        aec = MagicMock()
        backend, _ = _make_backend(
            input_sample_rate=24000,
            output_sample_rate=24000,
            block_duration_ms=20,
            aec=aec,
        )
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.accept(session, None)
        _drain_block(backend)  # idle silence — transport-level policy unchanged
        aec.feed_reference.assert_not_called()
        await backend.send_audio(session, _PCM * (5760 // 2))
        _drain_block(backend)  # real audio
        aec.feed_reference.assert_called()
