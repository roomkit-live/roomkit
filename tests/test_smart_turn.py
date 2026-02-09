"""Tests for SmartTurnDetector and audio threading through VoiceChannel."""

from __future__ import annotations

import asyncio
import struct

import pytest

from roomkit.voice.pipeline.turn.base import TurnContext, TurnDecision

np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")


# ---------------------------------------------------------------------------
# Unit tests: SmartTurnDetector helpers & config validation
# ---------------------------------------------------------------------------


class TestSmartTurnConfig:
    def test_empty_model_path_raises(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig, SmartTurnDetector

        with pytest.raises(ValueError, match="model_path"):
            SmartTurnDetector(SmartTurnConfig(model_path=""))

    def test_default_config_values(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig

        cfg = SmartTurnConfig(model_path="/tmp/model.onnx")
        assert cfg.threshold == 0.5
        assert cfg.num_threads == 1
        assert cfg.provider == "cpu"
        assert cfg.fallback_on_no_audio is True


class TestSmartTurnNoAudioFallback:
    """Test behaviour when audio_bytes is None or empty."""

    def _make_detector(self, *, fallback: bool = True):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig, SmartTurnDetector

        return SmartTurnDetector(
            SmartTurnConfig(model_path="/tmp/model.onnx", fallback_on_no_audio=fallback)
        )

    def test_no_audio_fallback_true(self):
        det = self._make_detector(fallback=True)
        ctx = TurnContext(transcript="hello", audio_bytes=None)
        decision = det.evaluate(ctx)
        assert decision.is_complete is True
        assert decision.confidence == 0.0
        assert "no audio" in (decision.reason or "")

    def test_no_audio_fallback_false(self):
        det = self._make_detector(fallback=False)
        ctx = TurnContext(transcript="hello", audio_bytes=None)
        decision = det.evaluate(ctx)
        assert decision.is_complete is False
        assert decision.confidence == 0.0

    def test_empty_audio_treated_as_no_audio(self):
        det = self._make_detector(fallback=True)
        ctx = TurnContext(transcript="hello", audio_bytes=b"")
        decision = det.evaluate(ctx)
        assert decision.is_complete is True
        assert "no audio" in (decision.reason or "")


class TestPCMConversion:
    """Test int16 PCM -> float32 conversion."""

    def test_silence_converts_to_zero(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        pcm = b"\x00\x00" * 100  # 100 zero samples
        result = SmartTurnDetector._pcm_int16_to_float32(pcm)
        assert result.shape == (100,)
        assert np.allclose(result, 0.0)

    def test_max_positive(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        pcm = struct.pack("<h", 32767)
        result = SmartTurnDetector._pcm_int16_to_float32(pcm)
        assert abs(result[0] - (32767 / 32768.0)) < 1e-5

    def test_max_negative(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        pcm = struct.pack("<h", -32768)
        result = SmartTurnDetector._pcm_int16_to_float32(pcm)
        assert abs(result[0] - (-1.0)) < 1e-5


class TestTruncateOrPad:
    """Test audio truncation and padding."""

    def test_pad_short_audio(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = SmartTurnDetector._truncate_or_pad(samples, 6)
        assert result.shape == (6,)
        # First 3 should be zero-pad, last 3 should be the original
        np.testing.assert_array_equal(result[:3], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[3:], [1.0, 2.0, 3.0])

    def test_truncate_long_audio(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = SmartTurnDetector._truncate_or_pad(samples, 3)
        assert result.shape == (3,)
        # Should keep the LAST 3 samples
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_exact_length_unchanged(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnDetector

        samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = SmartTurnDetector._truncate_or_pad(samples, 3)
        np.testing.assert_array_equal(result, samples)


class TestSmartTurnDetectorName:
    def test_name(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig, SmartTurnDetector

        det = SmartTurnDetector(SmartTurnConfig(model_path="/tmp/model.onnx"))
        assert det.name == "SmartTurnDetector"

    def test_close_clears_session(self):
        from roomkit.voice.pipeline.turn.smart_turn import SmartTurnConfig, SmartTurnDetector

        det = SmartTurnDetector(SmartTurnConfig(model_path="/tmp/model.onnx"))
        det._session = "something"
        det._feature_extractor = "something"
        det.close()
        assert det._session is None
        assert det._feature_extractor is None


# ---------------------------------------------------------------------------
# Integration tests: audio threading through VoiceChannel
# ---------------------------------------------------------------------------


class _MockSTT:
    """Minimal mock STT that returns preconfigured transcripts."""

    name = "mock_stt"

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = transcripts
        self._index = 0

    async def transcribe(self, frame):
        from roomkit.voice.base import TranscriptionResult

        if self._index < len(self._transcripts):
            text = self._transcripts[self._index]
            self._index += 1
            return TranscriptionResult(text=text)
        return TranscriptionResult(text="")

    async def close(self) -> None:
        pass


class _MockBackend:
    """Minimal mock backend for wiring."""

    name = "mock_backend"

    from roomkit.voice.base import VoiceCapability

    _caps = VoiceCapability.NONE
    _sessions: dict = {}
    _audio_cbs: list = []

    @property
    def capabilities(self):
        return self._caps

    @property
    def feeds_aec_reference(self):
        return False

    def on_audio_received(self, cb):
        self._audio_cbs.append(cb)

    def on_barge_in(self, cb):
        pass

    async def send_transcription(self, session, text, role):
        pass

    def get_session(self, sid):
        return None

    def list_sessions(self, room_id):
        return []

    async def cancel_audio(self, session):
        pass

    async def close(self):
        pass

    async def simulate_audio(self, session, frame):
        for cb in self._audio_cbs:
            cb(session, frame)


class TestAudioThreading:
    """Verify that audio_bytes from SPEECH_END reaches TurnContext."""

    async def test_audio_bytes_reaches_turn_context(self):
        from unittest.mock import AsyncMock

        from roomkit.channels.voice import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.audio_frame import AudioFrame
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.pipeline.config import AudioPipelineConfig
        from roomkit.voice.pipeline.turn.mock import MockTurnDetector
        from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
        from roomkit.voice.pipeline.vad.mock import MockVADProvider

        detector = MockTurnDetector(decisions=[TurnDecision(is_complete=True, confidence=0.9)])
        # Audio that the pipeline will return at SPEECH_END
        speech_audio = b"\x01\x02" * 100
        vad = MockVADProvider(
            events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=speech_audio)]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["hello"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="hello")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await asyncio.sleep(0.15)

        # The MockTurnDetector records evaluations — check audio_bytes was set
        assert len(detector.evaluations) == 1
        ctx = detector.evaluations[0]
        assert ctx.audio_bytes is not None
        assert len(ctx.audio_bytes) > 0

    async def test_audio_accumulates_across_incomplete_turns(self):
        from unittest.mock import AsyncMock

        from roomkit.channels.voice import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.audio_frame import AudioFrame
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.pipeline.config import AudioPipelineConfig
        from roomkit.voice.pipeline.turn.mock import MockTurnDetector
        from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
        from roomkit.voice.pipeline.vad.mock import MockVADProvider

        audio1 = b"\x01\x00" * 50
        audio2 = b"\x02\x00" * 60

        detector = MockTurnDetector(
            decisions=[
                TurnDecision(is_complete=False, confidence=0.4),
                TurnDecision(is_complete=True, confidence=0.9),
            ]
        )
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio1),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio2),
            ]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["hello", "world"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            side_effect=[
                AsyncMock(allowed=True, event="hello"),
                AsyncMock(allowed=True, event="world"),
            ]
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # First utterance — incomplete
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await asyncio.sleep(0.15)

        assert len(detector.evaluations) == 1
        ctx1 = detector.evaluations[0]
        first_audio_len = len(ctx1.audio_bytes) if ctx1.audio_bytes else 0
        assert first_audio_len > 0

        # Second utterance — complete, audio should be accumulated
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))
        await asyncio.sleep(0.15)

        assert len(detector.evaluations) == 2
        ctx2 = detector.evaluations[1]
        assert ctx2.audio_bytes is not None
        # Accumulated audio should be longer than the first segment alone
        assert len(ctx2.audio_bytes) > first_audio_len

    async def test_pending_audio_cleared_on_completion(self):
        from unittest.mock import AsyncMock

        from roomkit.channels.voice import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.audio_frame import AudioFrame
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.pipeline.config import AudioPipelineConfig
        from roomkit.voice.pipeline.turn.mock import MockTurnDetector
        from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
        from roomkit.voice.pipeline.vad.mock import MockVADProvider

        detector = MockTurnDetector(decisions=[TurnDecision(is_complete=True, confidence=0.9)])
        vad = MockVADProvider(
            events=[VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 50)]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["hello"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="hello")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await asyncio.sleep(0.15)

        # After a complete turn, _pending_audio should be cleared
        assert session.id not in channel._pending_audio

    async def test_pending_audio_cleared_on_unbind(self):
        from unittest.mock import AsyncMock

        from roomkit.channels.voice import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceSession
        from roomkit.voice.pipeline.config import AudioPipelineConfig
        from roomkit.voice.pipeline.turn.mock import MockTurnDetector
        from roomkit.voice.pipeline.vad.mock import MockVADProvider

        detector = MockTurnDetector(decisions=[])
        vad = MockVADProvider(events=[])
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        backend = _MockBackend()

        channel = VoiceChannel("ch1", backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # Manually seed some pending audio
        channel._pending_audio["s1"] = bytearray(b"\x01\x02\x03")

        channel.unbind_session(session)

        assert "s1" not in channel._pending_audio
        assert "s1" not in channel._pending_turns
