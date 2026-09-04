"""ScenarioVoiceBackend: cadenced playback and per-session capture."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.testing import ScenarioVoiceBackend, read_wav, tone, write_wav


async def _session(backend: ScenarioVoiceBackend) -> VoiceSession:
    return await backend.connect("room-1", "user-1", "voice-1")


class TestPlay:
    async def test_fast_playback_delivers_every_frame(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        received: list[tuple[str, AudioFrame]] = []
        backend.on_audio_received(lambda s, f: received.append((s.id, f)))

        sent = await backend.play(session, tone(1000), realtime=False)

        assert sent == 50 == len(received)
        assert {sid for sid, _ in received} == {session.id}
        assert {len(f.data) for _, f in received} == {640}
        assert received[-1][1].timestamp_ms == 980.0

    @pytest.mark.wallclock
    async def test_fast_playback_does_not_pace(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        loop = asyncio.get_running_loop()

        started = loop.time()
        await backend.play(session, tone(1000), realtime=False)

        assert loop.time() - started < 0.1

    async def test_realtime_playback_paces_one_frame_per_frame_ms(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        loop = asyncio.get_running_loop()

        started = loop.time()
        sent = await backend.play(session, tone(200))

        assert sent == 10
        # Ten frames are due 20 ms apart, anchored on the start: the last one
        # cannot be delivered before 200 ms have elapsed.
        assert loop.time() - started >= 0.19

    async def test_plays_a_wav_file(self, tmp_path) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        path = write_wav(tmp_path / "caller.wav", tone(100))

        assert await backend.play(session, path, realtime=False) == 5

    async def test_frame_ms_is_configurable(self) -> None:
        backend = ScenarioVoiceBackend(frame_ms=10)
        session = await _session(backend)
        sizes: list[int] = []
        backend.on_audio_received(lambda _s, f: sizes.append(len(f.data)))

        assert backend.frame_ms == 10
        assert await backend.play(session, tone(1000), realtime=False) == 100
        assert set(sizes) == {320}

    def test_a_non_positive_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame_ms"):
            ScenarioVoiceBackend(frame_ms=0)


async def _chunks(*data: bytes, sample_rate: int = 16000):
    for item in data:
        yield AudioChunk(data=item, sample_rate=sample_rate)


class TestCapture:
    async def test_streamed_chunks_are_captured_with_their_format(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)

        await backend.send_audio(
            session, _chunks(b"\x01\x02" * 4, b"\x03\x04" * 4, sample_rate=24000)
        )

        clip = backend.captured(session)
        assert clip.data == b"\x01\x02" * 4 + b"\x03\x04" * 4
        assert clip.sample_rate == 24000
        # The mock's own bookkeeping is intact.
        assert backend.sent_audio == [(session.id, clip.data)]

    async def test_raw_bytes_use_the_declared_rate(self) -> None:
        backend = ScenarioVoiceBackend(sample_rate=8000)
        session = await _session(backend)

        await backend.send_audio(session, b"\x00\x01" * 8)

        assert backend.captured(session).sample_rate == 8000

    async def test_sync_sends_are_captured_too(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)

        backend.send_audio_sync(session, AudioChunk(data=b"\x05\x06" * 2, sample_rate=16000))

        assert backend.captured(session).data == b"\x05\x06" * 2

    async def test_write_capture_round_trips(self, tmp_path) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        await backend.send_audio(session, _chunks(tone(50).data))

        path = backend.write_capture(session, tmp_path / "bot.wav")

        assert read_wav(path) == backend.captured(session)
        assert backend.captured(session).duration_ms == 50

    async def test_captures_are_per_session(self) -> None:
        backend = ScenarioVoiceBackend()
        one = await _session(backend)
        two = await backend.connect("room-1", "user-2", "voice-1")

        await backend.send_audio(one, b"\x01\x01")

        assert backend.captured(one).data == b"\x01\x01"
        assert backend.captured(two).data == b""
        backend.clear_capture(one)
        assert backend.captured(one).data == b""

    async def test_a_format_change_mid_capture_is_refused(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        await backend.send_audio(session, _chunks(b"\x00\x00", sample_rate=16000))

        with pytest.raises(ValueError, match="16000 Hz"):
            await backend.send_audio(session, _chunks(b"\x00\x00", sample_rate=24000))

    async def test_is_playing_while_the_bot_sends(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        seen: list[bool] = []

        async def chunks():
            seen.append(backend.is_playing(session))
            yield AudioChunk(data=b"\x00\x00")

        assert backend.is_playing(session) is False
        await backend.send_audio(session, chunks())

        assert seen == [True]
        assert backend.is_playing(session) is False

    async def test_a_held_playing_flag_survives_a_send(self) -> None:
        backend = ScenarioVoiceBackend()
        session = await _session(backend)
        backend.start_playing(session)

        await backend.send_audio(session, b"\x00\x00")

        assert backend.is_playing(session) is True

    def test_declares_the_capabilities_it_is_told(self) -> None:
        backend = ScenarioVoiceBackend(
            capabilities=VoiceCapability.INTERRUPTION | VoiceCapability.NATIVE_AEC
        )

        assert backend.name == "ScenarioVoiceBackend"
        assert VoiceCapability.NATIVE_AEC in backend.capabilities
        assert VoiceCapability.BARGE_IN not in backend.capabilities
