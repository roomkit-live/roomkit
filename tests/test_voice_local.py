"""Tests for LocalAudioBackend."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSessionState
from roomkit.voice.capture import MockCaptureSource


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

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("input_sample_rate", 0),
            ("output_sample_rate", 0),
            ("channels", 0),
            ("block_duration_ms", 0),
            ("rt_prebuffer_ms", -1),
        ],
    )
    def test_invalid_audio_geometry_fails_fast(self, name: str, value: int) -> None:
        with pytest.raises(ValueError):
            _make_backend(**{name: value})


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

_BLOCK = 960  # 20ms @ 24kHz mono int16
_PCM = b"\x01\x00"


def _drain_block(backend) -> bytes:
    """Play one 20 ms block: the speaker's PortAudio callback."""
    out = bytearray(_BLOCK)
    backend._speaker._callback(out, _BLOCK // 2, None, None)
    return bytes(out)


def _speaker_backend(*, latency: float = 0.0, **kwargs):
    """A VoiceChannel-mode backend (24 kHz, 20 ms blocks) whose speaker streams
    are mocks; each records the arguments it was opened with."""
    backend, sd = _make_backend(output_sample_rate=24000, block_duration_ms=20, **kwargs)
    streams: list[MagicMock] = []

    def raw_output_stream(**stream_kwargs):
        stream = MagicMock(latency=latency)
        stream.kwargs = stream_kwargs
        streams.append(stream)
        return stream

    sd.RawOutputStream = raw_output_stream
    return backend, streams


async def _play(backend, session, audio, *, blocks: int = 50) -> None:
    """send_audio() while the speaker plays its blocks, until it returns."""
    task = asyncio.create_task(backend.send_audio(session, audio))
    for _ in range(blocks):
        for _ in range(3):
            await asyncio.sleep(0)
        if task.done():
            break
        _drain_block(backend)
    await asyncio.wait_for(task, 1)


async def _response(nbytes: int = 1200):
    yield AudioChunk(data=_PCM * (nbytes // 2), sample_rate=24000)


async def _endless():
    while True:
        yield AudioChunk(data=_PCM * (_BLOCK // 2), sample_rate=24000)
        await asyncio.sleep(0)


class TestLocalAudioSpeakerPlayback:
    """VoiceChannel mode: every response plays on one persistent stream (RMK-551)."""

    async def test_responses_share_one_output_stream(self) -> None:
        """A stream per response restarts the echo delay an AEC tracks."""
        backend, streams = _speaker_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        await _play(backend, session, _response())
        await _play(backend, session, _response())

        assert len(streams) == 1
        streams[0].start.assert_called_once()
        streams[0].stop.assert_not_called()
        streams[0].abort.assert_not_called()
        assert streams[0].kwargs["blocksize"] == 480  # fixed 20 ms blocks

    async def test_send_audio_returns_once_the_response_has_played(self) -> None:
        backend, _ = _speaker_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        played: list[int] = []
        backend.on_audio_played(lambda _s, frame: played.append(frame.metadata["played_bytes"]))

        task = asyncio.create_task(backend.send_audio(session, _response(1200)))
        await asyncio.sleep(0.01)
        assert not task.done()  # queued, not played
        _drain_block(backend)  # 960 bytes
        assert not task.done()
        _drain_block(backend)  # the last 240: drained
        await asyncio.wait_for(task, 1)

        assert played == [960, 240]
        assert backend.is_playing(session) is False

    async def test_raw_bytes_play_on_the_same_stream_with_their_reference(self) -> None:
        aec = MagicMock()
        aec.stream_delay_ms = 80
        backend, streams = _speaker_backend(input_sample_rate=24000, aec=aec)
        session = await backend.connect("room-1", "user-1", "voice-1")

        await _play(backend, session, _PCM * 600)

        backend._sd.play.assert_not_called()
        assert len(streams) == 1
        reference = b"".join(c.args[0].data for c in aec.feed_reference.call_args_list)
        assert reference.startswith(_PCM * 600)

    async def test_the_aec_runs_from_capture_to_the_end_of_the_session(self) -> None:
        """Never paused between responses: resumed, it came back a block out of
        step and missed the next response's echo (RMK-551)."""
        aec = MagicMock()
        backend, streams = _speaker_backend(input_sample_rate=24000, aec=aec)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        assert len(streams) == 1  # the speaker opens with capture
        assert session.id in backend._aec_active_sessions

        for _ in range(2):
            await _play(backend, session, _response())
            for _ in range(100):  # 2 s between responses: silence, as reference
                _drain_block(backend)

        assert len(streams) == 1
        assert [c.args for c in aec.set_stream_active.call_args_list] == [(session.id, True)]
        assert aec.feed_reference.call_count >= 200
        await backend.disconnect(session)
        assert aec.set_stream_active.call_args_list[-1].args == (session.id, False)

    async def test_a_cut_leaves_the_aec_running(self) -> None:
        aec = MagicMock()
        backend, _ = _speaker_backend(input_sample_rate=24000, aec=aec, rt_prebuffer_ms=0)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        task = asyncio.create_task(backend.send_audio(session, _endless()))
        await asyncio.sleep(0.01)
        _drain_block(backend)

        assert await backend.cancel_audio(session) is True
        await asyncio.wait_for(task, 1)
        assert backend._speaker.buffered_bytes == 0
        fed = aec.feed_reference.call_count
        for _ in range(50):
            assert _drain_block(backend) == b"\x00" * _BLOCK

        assert aec.feed_reference.call_count == fed + 50
        assert session.id in backend._aec_active_sessions
        assert all(c.args[1] is True for c in aec.set_stream_active.call_args_list)

    async def test_muted_capture_still_runs_through_the_aec(self) -> None:
        """The canceller's capture timeline never pauses; the frame is dropped after."""
        aec = MagicMock()
        aec.process.side_effect = lambda frame, stream: frame
        backend, _ = _speaker_backend(input_sample_rate=24000, aec=aec)
        delivered: list[AudioFrame] = []
        backend.on_audio_received(lambda _s, frame: delivered.append(frame))
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        backend.set_input_muted(session, True)
        handle = backend._make_frame_handler(session)

        handle(AudioFrame(data=_PCM * 480, sample_rate=24000))
        await asyncio.sleep(0)

        aec.process.assert_called_once()
        assert delivered == []

    async def test_stop_listening_stops_the_aec(self) -> None:
        aec = MagicMock()
        backend, _ = _speaker_backend(input_sample_rate=24000, aec=aec)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)

        await backend.stop_listening(session)

        assert session.id not in backend._aec_active_sessions
        assert aec.set_stream_active.call_args_list[-1].args == (session.id, False)

    async def test_streamed_audio_waits_for_room_rather_than_dropping(self) -> None:
        backend, _ = _speaker_backend(rt_prebuffer_ms=0)
        backend._speaker._max_bytes = _BLOCK
        session = await backend.connect("room-1", "user-1", "voice-1")
        played: list[int] = []
        backend.on_audio_played(lambda _s, frame: played.append(frame.metadata["played_bytes"]))

        task = asyncio.create_task(backend.send_audio(session, _response(3 * _BLOCK)))
        for _ in range(20):
            await asyncio.sleep(0.03)  # the queue is retried every block (20 ms)
            if task.done():
                break
            _drain_block(backend)
        await asyncio.wait_for(task, 1)

        assert sum(played) == 3 * _BLOCK

    async def test_played_frames_of_a_streamed_response(self) -> None:
        """Only while it plays; the channel, not the backend, ends its AEC."""
        backend, _ = _speaker_backend()  # no AEC: half-duplex
        session = await backend.connect("room-1", "user-1", "voice-1")
        frames: list[AudioFrame] = []
        backend.on_audio_played(lambda _s, frame: frames.append(frame))

        await _play(backend, session, _response())
        count = len(frames)
        _drain_block(backend)  # idle

        assert len(frames) == count
        assert sum(f.metadata["played_bytes"] for f in frames) == 1200
        assert all(f.metadata["playback_ended"] is False for f in frames)
        assert all(f.metadata["capture_paused"] for f in frames)

    async def test_disconnect_releases_a_streamed_response(self) -> None:
        backend, _ = _speaker_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        task = asyncio.create_task(backend.send_audio(session, _endless()))
        await asyncio.sleep(0.01)

        await backend.disconnect(session)

        await asyncio.wait_for(task, 1)

    async def test_a_device_that_stops_releases_the_playback(self, caplog) -> None:
        backend, streams = _speaker_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        task = asyncio.create_task(backend.send_audio(session, _response()))
        await asyncio.sleep(0.01)

        streams[0].kwargs["finished_callback"]()  # PortAudio stopped the stream

        await asyncio.wait_for(task, 1)
        assert "stopped by the audio device" in caplog.text
        await _play(backend, session, _response())
        assert len(streams) == 2  # the next response opens a new one

    async def test_a_replaced_streams_late_stop_is_ignored(self) -> None:
        backend, streams = _speaker_backend()
        backend._speaker.open()
        late_stop = streams[0].kwargs["finished_callback"]
        backend._speaker.close()
        backend._speaker.open()

        late_stop()  # PortAudio's thread, after the stream was replaced

        assert backend._speaker.is_open

    async def test_an_output_underflow_is_logged(self, caplog) -> None:
        backend, _ = _speaker_backend()
        with caplog.at_level(logging.WARNING, logger="roomkit.voice.local"):
            backend._speaker._callback(bytearray(_BLOCK), _BLOCK // 2, None, "output underflow")
        assert "output underflow" in caplog.text

    async def test_the_aec_delay_is_seeded_from_both_streams_at_capture_start(self) -> None:
        aec = MagicMock()
        aec.stream_delay_ms = 0
        backend, _ = _speaker_backend(latency=0.0181, aec=aec)
        backend._sd.RawInputStream = lambda **kwargs: MagicMock(latency=0.0378)
        session = await backend.connect("room-1", "user-1", "voice-1")

        await backend.start_listening(session)  # the speaker opens with capture

        aec.set_stream_delay_ms.assert_called_once_with(56)

    def test_with_an_aec_the_mic_stays_open_by_default(self) -> None:
        def mutes(**kwargs) -> bool:
            return _make_backend(**kwargs)[0]._mute_mic_during_playback

        assert mutes() is True
        assert mutes(aec=MagicMock()) is False
        assert mutes(aec=MagicMock(), mute_mic_during_playback=True) is True
        assert mutes(mute_mic_during_playback=False) is False

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

        backend._playing_sessions.add(session.id)
        backend._speaker.append(_PCM * 100)
        result = await backend.cancel_audio(session)
        assert result is True
        assert backend.is_playing(session) is False
        assert backend._speaker.buffered_bytes == 0  # what was queued never plays
        sd.stop.assert_not_called()

    async def test_cancel_audio_cancels_playback_task(self) -> None:
        backend, sd = _make_backend()
        session = await backend.connect("room-1", "user-1", "voice-1")

        mock_task = MagicMock()
        backend._playing_sessions.add(session.id)
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
        assert backend._speaker.buffered_bytes == 2880  # nothing consumed

    async def test_priming_releases_at_prebuffer_threshold(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * (5760 // 2))  # exactly 120ms
        # Drain starts in the same callback as the release — no wasted block.
        assert _drain_block(backend) == _PCM * (_BLOCK // 2)
        assert backend._speaker.buffered_bytes == 5760 - _BLOCK

    async def test_short_response_drains_on_end_of_response(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * 400)  # 800B, far below prebuffer
        assert _drain_block(backend) == b"\x00" * _BLOCK  # still priming
        backend.end_of_response(session)
        out = _drain_block(backend)
        assert out[:800] == _PCM * 400
        assert out[800:] == b"\x00" * (_BLOCK - 800)
        assert backend.rt_underruns == 0  # clean end, not starvation
        assert backend._speaker._response_complete is False  # consumed by the drain

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
        assert backend._speaker.buffered_bytes == 0
        assert _drain_block(backend) == b"\x00" * _BLOCK
        # First append clears the interrupt flag but must re-prime from zero.
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        assert backend._speaker._interrupted is False
        assert _drain_block(backend) == b"\x00" * _BLOCK
        assert backend._speaker.buffered_bytes == _BLOCK

    async def test_interrupt_leaves_the_aec_running(self) -> None:
        """A cut response: the device and the room still sound, and the next
        response comes; the canceller keeps its timeline and its filter."""
        aec = MagicMock()
        backend, session = await _rt_backend(
            input_sample_rate=24000,
            rt_prebuffer_ms=0,
            aec=aec,
            mute_mic_during_playback=False,
        )
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        _drain_block(backend)
        assert session.id in backend._aec_active_sessions

        backend.interrupt(session)
        for _ in range(50):
            _drain_block(backend)

        assert session.id in backend._aec_active_sessions
        assert all(c.args[1] is True for c in aec.set_stream_active.call_args_list)
        aec.reset.assert_not_called()

    async def test_stale_end_of_response_ignored_after_interrupt(self) -> None:
        backend, session = await _rt_backend()
        backend.interrupt(session)
        backend.end_of_response(session)  # Gemini fires response_end on barge-in
        assert backend._speaker._response_complete is False
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        # Without the guard, the stale EOR would release this audio early.
        assert _drain_block(backend) == b"\x00" * _BLOCK

    async def test_prime_idle_valve_flushes_partial_buffer(self) -> None:
        backend, session = await _rt_backend()
        await backend.send_audio(session, _PCM * 400)  # 800B, EOR never arrives
        for _ in range(5):  # the idle valve: 100ms / 20ms = 5 blocks
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
        assert backend._speaker.buffered_bytes == _BLOCK

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
        assert out[:200] == _PCM * 100  # no priming: plays from the first byte

    async def test_realtime_speaker_buffer_is_bounded(self) -> None:
        backend, session = await _rt_backend()
        backend._speaker._max_bytes = 4

        await backend.send_audio(session, b"123456")

        assert backend._speaker.buffered_bytes == 4
        assert list(backend._speaker._queue) == [b"1234"]
        assert backend._rt_dropped_bytes == 2

    async def test_failed_aec_activation_is_retried(self) -> None:
        aec = MagicMock()
        aec.set_stream_active.side_effect = [RuntimeError("transient"), None]
        backend, session = await _rt_backend(
            input_sample_rate=24000,
            rt_prebuffer_ms=0,
            aec=aec,
            mute_mic_during_playback=False,
        )
        assert session.id not in backend._aec_active_sessions  # failed at capture start

        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        _drain_block(backend)  # retried when a response plays

        assert session.id in backend._aec_active_sessions
        assert aec.set_stream_active.call_count == 2


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

    async def test_played_callback_reports_provider_bytes_not_filler_silence(self) -> None:
        backend, session = await _rt_backend(rt_prebuffer_ms=0)
        frames: list[AudioFrame] = []
        backend.on_audio_played(lambda _session, frame: frames.append(frame))
        await backend.send_audio(session, _PCM * 100)  # 200 provider bytes

        _drain_block(backend)

        assert frames[0].metadata["played_bytes"] == 200
        assert frames[0].data[200:] == b"\x00" * (_BLOCK - 200)

    async def test_played_callbacks_fire_during_priming_and_interrupt(self) -> None:
        backend, session = await _rt_backend()
        played: list[bytes] = []
        backend.on_audio_played(lambda s, f: played.append(f.data))
        await backend.send_audio(session, _PCM * (2880 // 2))  # sub-prebuffer
        _drain_block(backend)  # priming hold
        backend.interrupt(session)
        _drain_block(backend)  # interrupted mute
        assert played == [b"\x00" * _BLOCK, b"\x00" * _BLOCK]

    async def test_empty_response_still_signals_playback_end(self) -> None:
        """A response_start without audio must not leave pipeline AEC active."""
        backend, session = await _rt_backend()
        ended: list[bool] = []
        backend.on_audio_played(
            lambda _session, frame: ended.append(bool(frame.metadata.get("playback_ended")))
        )

        backend.end_of_response(session)
        _drain_block(backend)

        assert ended == [True]

    async def test_transport_aec_reference_is_every_block_from_capture_start(self) -> None:
        aec = MagicMock()
        backend, _ = _make_backend(
            input_sample_rate=24000,
            output_sample_rate=24000,
            block_duration_ms=20,
            rt_prebuffer_ms=0,
            aec=aec,
            mute_mic_during_playback=False,
        )
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.accept(session, None)
        _drain_block(backend)  # idle silence before any response: capture advances
        assert aec.feed_reference.call_count == 1
        await backend.send_audio(session, _PCM * (_BLOCK // 2))
        _drain_block(backend)  # real audio
        assert aec.feed_reference.call_count == 2
        aec.set_stream_active.assert_called_with(session.id, True)

        _drain_block(backend)  # mid-response underrun
        assert aec.feed_reference.call_count == 3
        silence_frame = aec.feed_reference.call_args_list[-1].args[0]
        assert silence_frame.data == b"\x00" * _BLOCK

        backend.end_of_response(session)
        for _ in range(50):
            _drain_block(backend)

        aec.reset.assert_not_called()
        assert all(c.args[1] is True for c in aec.set_stream_active.call_args_list)


# ---------------------------------------------------------------------------
# Shared capture source (RFC 12.12)
# ---------------------------------------------------------------------------


def _make_backend_with_source(source, **kwargs):
    """Create a LocalAudioBackend bound to a shared capture source."""
    sd = _mock_sounddevice()
    with patch.dict("sys.modules", {"sounddevice": sd}):
        from roomkit.voice.backends.local import LocalAudioBackend

        backend = LocalAudioBackend(source=source, **kwargs)
        backend._sd = sd
        return backend, sd


class TestSharedCaptureSource:
    async def test_no_device_stream_is_opened(self) -> None:
        """The source owns the device; the backend only subscribes to it."""
        source = MockCaptureSource()
        source.start()
        backend, sd = _make_backend_with_source(source)
        opened = []
        sd.RawInputStream = lambda **kw: opened.append(kw) or MagicMock()

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)

        assert opened == []
        assert backend._input_streams == {}
        assert session.id in backend._subscriptions

    async def test_frames_reach_on_audio_received(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[AudioFrame] = []
        backend.on_audio_received(lambda s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        sent = source.feed_blocks(2)
        await asyncio.sleep(0.01)

        assert [f.data for f in received] == sent

    async def test_the_backlog_mark_is_read_from_session_metadata(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[AudioFrame] = []
        backend.on_audio_received(lambda s, f: received.append(f))

        mark = source.mark()
        phrase = source.feed_blocks(3, fill=10)

        session = await backend.connect(
            "room-1", "user-1", "voice-1", metadata={"capture_since": mark}
        )
        await backend.start_listening(session)
        await asyncio.sleep(0.01)

        assert [f.data for f in received] == phrase

    async def test_mute_still_applies_per_session(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[AudioFrame] = []
        backend.on_audio_received(lambda s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        backend.set_input_muted(session, True)
        source.feed_blocks(2)
        await asyncio.sleep(0.01)

        assert received == []

        backend.set_input_muted(session, False)
        sent = source.feed_blocks(1, fill=5)
        await asyncio.sleep(0.01)
        assert [f.data for f in received] == sent

    async def test_gating_still_applies_per_session(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[AudioFrame] = []
        backend.on_audio_received(lambda s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        backend.set_input_gated(session, True)
        source.feed_blocks(2)
        await asyncio.sleep(0.01)

        assert received == []

    async def test_aec_still_processes_each_frame(self) -> None:
        source = MockCaptureSource()
        source.start()
        aec = MagicMock()
        aec.process.side_effect = lambda frame, sid: frame
        backend, _ = _make_backend_with_source(source, aec=aec)

        backend.on_audio_received(lambda s, f: None)
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        source.feed_blocks(2)
        await asyncio.sleep(0.01)

        assert aec.process.call_count == 2
        assert aec.process.call_args.args[1] == session.id

    async def test_stop_listening_unsubscribes_without_stopping_the_source(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[AudioFrame] = []
        backend.on_audio_received(lambda s, f: received.append(f))

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.stop_listening(session)
        source.feed_blocks(2)
        await asyncio.sleep(0.01)

        assert received == []
        assert backend._subscriptions == {}
        assert source.started is True

    async def test_closing_the_backend_leaves_the_source_running(self) -> None:
        """The source's lifecycle belongs to whoever created it."""
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.close()

        assert source.started is True

    async def test_starting_twice_is_refused(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.start_listening(session)
        await backend.start_listening(session)

        assert len(backend._subscriptions) == 1

    async def test_two_sessions_share_one_source(self) -> None:
        source = MockCaptureSource()
        source.start()
        backend, _ = _make_backend_with_source(source)

        received: list[str] = []
        backend.on_audio_received(lambda s, f: received.append(s.id))

        first = await backend.connect("room-1", "user-1", "voice-1")
        second = await backend.connect("room-2", "user-2", "voice-1")
        await backend.start_listening(first)
        await backend.start_listening(second)
        source.feed_blocks(1)
        await asyncio.sleep(0.01)

        assert sorted(received) == sorted([first.id, second.id])

    def test_the_source_owns_the_input_format(self) -> None:
        source = MockCaptureSource(sample_rate=48000, channels=2, block_duration_ms=10)
        backend, _ = _make_backend_with_source(source)

        assert backend._input_sample_rate == 48000
        assert backend._channels == 2
        assert backend._block_duration_ms == 10

    def test_a_contradicting_format_argument_is_rejected(self) -> None:
        source = MockCaptureSource(sample_rate=48000)
        with pytest.raises(ValueError, match="conflicts with the capture source"):
            _make_backend_with_source(source, input_sample_rate=16000)

    def test_a_matching_format_argument_is_accepted(self) -> None:
        source = MockCaptureSource(sample_rate=48000)
        backend, _ = _make_backend_with_source(source, input_sample_rate=48000)
        assert backend._input_sample_rate == 48000

    def test_without_a_source_the_defaults_are_unchanged(self) -> None:
        backend, _ = _make_backend()
        assert backend._input_sample_rate == 16000
        assert backend._channels == 1
        assert backend._block_duration_ms == 20
        assert backend._source is None


class TestLocalAudioTTSStreamFailure:
    """A failure of the TTS stream reaches the caller; the device's own stays here (RMK-448)."""

    async def test_a_failing_stream_reaches_the_caller(self) -> None:
        backend, sd = _make_backend()
        sd.RawOutputStream = lambda **kwargs: MagicMock(latency=0.0)
        session = await backend.connect("room-1", "user-1", "voice-1")

        async def audio_gen():
            yield AudioChunk(data=b"\x00\x01\x00\x02")
            raise RuntimeError("tts down")

        with pytest.raises(RuntimeError, match="tts down"):
            await backend.send_audio(session, audio_gen())

        assert backend._speaker.buffered_bytes == 0  # what was queued never plays
        assert backend.is_playing(session) is False

    async def test_a_device_failure_is_still_absorbed(self, caplog) -> None:
        backend, sd = _make_backend()
        sd.RawOutputStream = MagicMock(side_effect=OSError("no output device"))
        session = await backend.connect("room-1", "user-1", "voice-1")

        async def audio_gen():
            yield AudioChunk(data=b"\x00\x01\x00\x02")

        await backend.send_audio(session, audio_gen())

        assert "no output device" in caplog.text
