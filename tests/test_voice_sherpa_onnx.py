"""Tests for sherpa-onnx STT and TTS providers."""

from __future__ import annotations

import asyncio
import importlib
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.base import AudioChunk, TranscriptionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm_s16le(samples: list[float]) -> bytes:
    """Encode float32 samples as PCM S16LE bytes (test helper)."""
    int_samples = [int(max(-1.0, min(1.0, s)) * 32767) for s in samples]
    return struct.pack(f"<{len(int_samples)}h", *int_samples)


def _mock_sherpa_module() -> MagicMock:
    """Create a MagicMock that stands in for the sherpa_onnx module."""
    mod = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# STT tests
# ---------------------------------------------------------------------------


class TestSherpaOnnxSTTProvider:
    def _make_provider(self, sherpa_mock: MagicMock, **config_kwargs: Any) -> Any:
        with patch.dict("sys.modules", {"sherpa_onnx": sherpa_mock}):
            import roomkit.voice.stt.sherpa_onnx as stt_mod

            importlib.reload(stt_mod)
            from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider

            cfg = SherpaOnnxSTTConfig(**config_kwargs)
            provider = SherpaOnnxSTTProvider(cfg)
            # Replace the internal sherpa reference with our mock
            provider._sherpa = sherpa_mock
            return provider

    @pytest.mark.asyncio
    async def test_batch_transcribe_transducer(self) -> None:
        sherpa = _mock_sherpa_module()
        stream_mock = MagicMock()
        recognizer_mock = MagicMock()
        recognizer_mock.create_stream.return_value = stream_mock
        # is_ready returns True once then False to exit the decode loop
        recognizer_mock.is_ready.side_effect = [True, False]
        recognizer_mock.get_result.return_value = "  hello world  "
        sherpa.OnlineRecognizer.from_transducer.return_value = recognizer_mock

        provider = self._make_provider(
            sherpa,
            mode="transducer",
            tokens="tokens.txt",
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            joiner="joiner.onnx",
        )

        audio = AudioChunk(data=_make_pcm_s16le([0.5, -0.3, 0.1]), sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "hello world"
        recognizer_mock.create_stream.assert_called_once()
        stream_mock.input_finished.assert_called_once()
        recognizer_mock.decode_stream.assert_called_once_with(stream_mock)

    @pytest.mark.asyncio
    async def test_batch_transcribe_whisper(self) -> None:
        sherpa = _mock_sherpa_module()
        stream_mock = MagicMock()
        stream_mock.result.text = " whisper result "
        recognizer_mock = MagicMock()
        recognizer_mock.create_stream.return_value = stream_mock
        sherpa.OfflineRecognizer.from_whisper.return_value = recognizer_mock

        provider = self._make_provider(
            sherpa,
            mode="whisper",
            tokens="tokens.txt",
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            language="en",
        )

        audio = AudioChunk(data=_make_pcm_s16le([0.1, 0.2]), sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "whisper result"
        sherpa.OfflineRecognizer.from_whisper.assert_called_once()
        recognizer_mock.decode_stream.assert_called_once_with(stream_mock)

    @pytest.mark.asyncio
    async def test_streaming_transducer(self) -> None:
        sherpa = _mock_sherpa_module()
        stream_mock = MagicMock()
        recognizer_mock = MagicMock()
        recognizer_mock.create_stream.return_value = stream_mock

        # Simulate: first chunk returns partial, second chunk returns final at endpoint,
        # third call (flush on is_final) returns empty after reset.
        # get_result returns strings (str() is called on them in the provider)
        recognizer_mock.get_result.side_effect = [" hello ", " hello world ", ""]
        recognizer_mock.is_endpoint.side_effect = [False, True]
        recognizer_mock.is_ready.return_value = False  # no decode loop needed
        sherpa.OnlineRecognizer.from_transducer.return_value = recognizer_mock

        provider = self._make_provider(
            sherpa,
            mode="transducer",
            tokens="tokens.txt",
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            joiner="joiner.onnx",
        )

        async def _audio_stream() -> Any:
            yield AudioChunk(data=_make_pcm_s16le([0.1, 0.2]), sample_rate=16000)
            yield AudioChunk(data=_make_pcm_s16le([0.3, 0.4]), sample_rate=16000, is_final=True)

        results: list[TranscriptionResult] = []
        async for result in provider.transcribe_stream(_audio_stream()):
            results.append(result)

        assert len(results) >= 1
        # Should have at least a partial or final result
        assert any(r.text for r in results)

        # Verify endpoint config was passed to from_transducer
        call_kwargs = sherpa.OnlineRecognizer.from_transducer.call_args
        assert call_kwargs[1]["enable_endpoint_detection"] is True
        assert call_kwargs[1]["rule1_min_trailing_silence"] == 2.4
        assert call_kwargs[1]["rule2_min_trailing_silence"] == 1.2
        assert call_kwargs[1]["rule3_min_utterance_length"] == 20.0

    @pytest.mark.asyncio
    async def test_vad_driven_finalize_yields_final(self) -> None:
        """When VAD drives the lifecycle (endpoint never fires), _finalize
        still yields is_final=True with the recognised text."""
        sherpa = _mock_sherpa_module()
        stream_mock = MagicMock()
        recognizer_mock = MagicMock()
        recognizer_mock.create_stream.return_value = stream_mock

        # Simulate: partial "hello" on first chunk, same text on finalize.
        # is_endpoint always returns False (VAD drives the lifecycle).
        recognizer_mock.get_result.side_effect = [" hello ", " hello "]
        recognizer_mock.is_endpoint.return_value = False
        recognizer_mock.is_ready.return_value = False
        sherpa.OnlineRecognizer.from_transducer.return_value = recognizer_mock

        provider = self._make_provider(
            sherpa,
            mode="transducer",
            tokens="tokens.txt",
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            joiner="joiner.onnx",
        )

        async def _audio_stream() -> Any:
            yield AudioChunk(data=_make_pcm_s16le([0.1, 0.2]), sample_rate=16000)
            # Sentinel: VAD signals end-of-speech
            yield AudioChunk(data=b"", sample_rate=16000, is_final=True)

        results: list[TranscriptionResult] = []
        async for result in provider.transcribe_stream(_audio_stream()):
            results.append(result)

        # Should have partial + final
        partials = [r for r in results if not r.is_final]
        finals = [r for r in results if r.is_final]
        assert len(partials) >= 1
        assert partials[0].text == "hello"
        assert len(finals) == 1
        assert finals[0].text == "hello"

    @pytest.mark.asyncio
    async def test_whisper_streaming_raises(self) -> None:
        sherpa = _mock_sherpa_module()
        provider = self._make_provider(
            sherpa,
            mode="whisper",
            tokens="tokens.txt",
            encoder="encoder.onnx",
            decoder="decoder.onnx",
        )

        async def _audio_stream() -> Any:
            yield AudioChunk(data=b"\x00\x00", sample_rate=16000)

        with pytest.raises(ValueError, match="Streaming transcription is not supported"):
            async for _ in provider.transcribe_stream(_audio_stream()):
                pass

    def test_pcm_conversion(self) -> None:
        from roomkit.voice.stt.sherpa_onnx import _pcm_s16le_to_float32

        # Silence (0x0000) should give 0.0
        pcm = struct.pack("<1h", 0)
        result = _pcm_s16le_to_float32(pcm)
        assert len(result) == 1
        assert abs(result[0]) < 1e-6

        # Max positive
        pcm = struct.pack("<1h", 32767)
        result = _pcm_s16le_to_float32(pcm)
        assert abs(result[0] - 1.0) < 0.001

        # Max negative
        pcm = struct.pack("<1h", -32768)
        result = _pcm_s16le_to_float32(pcm)
        assert abs(result[0] - (-1.0)) < 0.001

    def test_url_audio_raises(self) -> None:
        sherpa = _mock_sherpa_module()
        recognizer_mock = MagicMock()
        sherpa.OfflineRecognizer.from_transducer.return_value = recognizer_mock

        provider = self._make_provider(
            sherpa,
            mode="transducer",
            tokens="t.txt",
            encoder="e.onnx",
            decoder="d.onnx",
            joiner="j.onnx",
        )

        audio_content = SimpleNamespace(url="https://example.com/audio.wav")

        with pytest.raises(ValueError, match="does not support URL-based"):
            asyncio.get_event_loop().run_until_complete(provider.transcribe(audio_content))

    def test_import_error(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": None}):
            import roomkit.voice.stt.sherpa_onnx as stt_mod

            importlib.reload(stt_mod)

            from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider

            with pytest.raises(ImportError, match="sherpa-onnx is required"):
                SherpaOnnxSTTProvider(SherpaOnnxSTTConfig())


# ---------------------------------------------------------------------------
# TTS tests
# ---------------------------------------------------------------------------


class TestSherpaOnnxTTSProvider:
    def _make_provider(self, sherpa_mock: MagicMock, **config_kwargs: Any) -> Any:
        with patch.dict("sys.modules", {"sherpa_onnx": sherpa_mock}):
            import roomkit.voice.tts.sherpa_onnx as tts_mod

            importlib.reload(tts_mod)
            from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

            cfg = SherpaOnnxTTSConfig(**config_kwargs)
            provider = SherpaOnnxTTSProvider(cfg)
            provider._sherpa = sherpa_mock
            return provider

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio_content(self) -> None:
        sherpa = _mock_sherpa_module()
        tts_mock = MagicMock()
        audio_result = SimpleNamespace(samples=[0.1, -0.2, 0.3], sample_rate=22050)
        tts_mock.generate.return_value = audio_result
        sherpa.OfflineTts.return_value = tts_mock

        provider = self._make_provider(
            sherpa,
            model="model.onnx",
            tokens="tokens.txt",
        )

        result = await provider.synthesize("Hello world")

        assert result.type == "audio"
        assert result.url.startswith("data:audio/wav;base64,")
        assert result.mime_type == "audio/wav"
        assert result.transcript == "Hello world"
        assert result.duration_seconds is not None
        tts_mock.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_override(self) -> None:
        sherpa = _mock_sherpa_module()
        tts_mock = MagicMock()
        audio_result = SimpleNamespace(samples=[0.0], sample_rate=22050)
        tts_mock.generate.return_value = audio_result
        sherpa.OfflineTts.return_value = tts_mock

        provider = self._make_provider(
            sherpa,
            model="model.onnx",
            tokens="tokens.txt",
            speaker_id=0,
        )

        await provider.synthesize("Hi", voice="5")

        call_kwargs = tts_mock.generate.call_args
        assert call_kwargs[1]["sid"] == 5

    def test_default_voice(self) -> None:
        sherpa = _mock_sherpa_module()
        provider = self._make_provider(
            sherpa,
            model="model.onnx",
            tokens="tokens.txt",
            speaker_id=3,
        )
        assert provider.default_voice == "3"

    @pytest.mark.asyncio
    async def test_streaming_yields_chunks(self) -> None:
        sherpa = _mock_sherpa_module()
        tts_mock = MagicMock()

        def fake_generate(
            text: str,
            sid: int = 0,
            speed: float = 1.0,
            callback: Any = None,
        ) -> None:
            if callback:
                callback([0.1, 0.2], 0.5)
                callback([0.3, 0.4], 1.0)

        tts_mock.generate.side_effect = fake_generate
        sherpa.OfflineTts.return_value = tts_mock

        provider = self._make_provider(
            sherpa,
            model="model.onnx",
            tokens="tokens.txt",
        )

        chunks: list[AudioChunk] = []
        async for chunk in provider.synthesize_stream("Hello"):
            chunks.append(chunk)

        # Should have at least the callback chunks + final marker
        assert len(chunks) >= 2
        assert chunks[-1].is_final is True
        # Non-final chunks should have data
        for c in chunks[:-1]:
            assert len(c.data) > 0

    def test_float32_to_pcm_conversion(self) -> None:
        from roomkit.voice.tts.sherpa_onnx import _float32_to_pcm_s16le

        pcm = _float32_to_pcm_s16le([0.0])
        assert len(pcm) == 2
        assert struct.unpack("<1h", pcm)[0] == 0

        pcm = _float32_to_pcm_s16le([1.0])
        assert struct.unpack("<1h", pcm)[0] == 32767

        pcm = _float32_to_pcm_s16le([-1.0])
        assert struct.unpack("<1h", pcm)[0] == -32767

    def test_wav_header(self) -> None:
        from roomkit.voice.tts.sherpa_onnx import _wrap_wav

        pcm = _make_pcm_s16le([0.5, -0.5])
        wav = _wrap_wav(pcm, sample_rate=16000)

        # WAV starts with RIFF
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        # Data is at the end
        assert wav[-len(pcm) :] == pcm

    def test_import_error(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": None}):
            import roomkit.voice.tts.sherpa_onnx as tts_mod

            importlib.reload(tts_mod)

            from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

            with pytest.raises(ImportError, match="sherpa-onnx is required"):
                SherpaOnnxTTSProvider(SherpaOnnxTTSConfig())


# ---------------------------------------------------------------------------
# AudioContent data: URL fix
# ---------------------------------------------------------------------------


class TestAudioContentDataURL:
    def test_data_url_accepted(self) -> None:
        from roomkit.models.event import AudioContent

        ac = AudioContent(url="data:audio/wav;base64,AAAA", mime_type="audio/wav")
        assert ac.url.startswith("data:")

    def test_invalid_url_rejected(self) -> None:
        from roomkit.models.event import AudioContent

        with pytest.raises(ValueError, match="URL must start with"):
            AudioContent(url="ftp://example.com/audio.wav", mime_type="audio/wav")

    def test_http_url_still_works(self) -> None:
        from roomkit.models.event import AudioContent

        ac = AudioContent(url="https://example.com/audio.wav", mime_type="audio/wav")
        assert ac.url == "https://example.com/audio.wav"
