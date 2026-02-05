"""sherpa-onnx speech-to-text provider."""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class SherpaOnnxSTTConfig:
    """Configuration for the sherpa-onnx STT provider.

    Attributes:
        mode: Recognition mode — ``"transducer"`` or ``"whisper"``.
        tokens: Path to ``tokens.txt``.
        encoder: Path to encoder ``.onnx`` model.
        decoder: Path to decoder ``.onnx`` model.
        joiner: Path to joiner ``.onnx`` model (transducer only).
        language: Language code (Whisper only).
        sample_rate: Expected audio sample rate.
        num_threads: Number of CPU threads for inference.
        provider_type: ONNX execution provider (``"cpu"`` or ``"cuda"``).
    """

    mode: str = "transducer"
    tokens: str = ""
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    language: str = "en"
    sample_rate: int = 16000
    num_threads: int = 2
    provider_type: str = "cpu"


def _pcm_s16le_to_float32(data: bytes) -> list[float]:
    """Convert PCM signed 16-bit little-endian bytes to float32 list in [-1, 1]."""
    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    return [s / 32768.0 for s in samples]


class SherpaOnnxSTTProvider(STTProvider):
    """Speech-to-text provider using sherpa-onnx.

    Supports transducer models for both streaming and batch recognition,
    and Whisper models for batch recognition only.
    """

    def __init__(self, config: SherpaOnnxSTTConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxSTTProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc
        self._config = config
        self._sherpa = __import__("sherpa_onnx")
        self._online_recognizer: Any = None
        self._offline_recognizer: Any = None

    @property
    def name(self) -> str:
        return "SherpaOnnxSTT"

    def _get_online_recognizer(self) -> Any:
        """Lazily create an OnlineRecognizer for streaming (transducer only)."""
        if self._online_recognizer is None:
            cfg = self._config
            self._online_recognizer = self._sherpa.OnlineRecognizer.from_transducer(
                tokens=cfg.tokens,
                encoder=cfg.encoder,
                decoder=cfg.decoder,
                joiner=cfg.joiner,
                num_threads=cfg.num_threads,
                sample_rate=cfg.sample_rate,
                feature_dim=80,
                provider=cfg.provider_type,
            )
        return self._online_recognizer

    def _get_offline_recognizer(self) -> Any:
        """Lazily create an OfflineRecognizer for batch transcription."""
        if self._offline_recognizer is None:
            cfg = self._config
            if cfg.mode == "whisper":
                self._offline_recognizer = self._sherpa.OfflineRecognizer.from_whisper(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                    tokens=cfg.tokens,
                    language=cfg.language,
                    num_threads=cfg.num_threads,
                    provider=cfg.provider_type,
                )
            else:
                self._offline_recognizer = self._sherpa.OfflineRecognizer.from_transducer(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                    joiner=cfg.joiner,
                    tokens=cfg.tokens,
                    num_threads=cfg.num_threads,
                    sample_rate=cfg.sample_rate,
                    feature_dim=80,
                    provider=cfg.provider_type,
                )
        return self._offline_recognizer

    async def transcribe(self, audio: AudioContent | AudioChunk) -> str:
        """Transcribe complete audio using the offline recognizer.

        Args:
            audio: Audio content or raw audio chunk (PCM S16LE expected).

        Returns:
            Transcribed text.
        """
        recognizer = self._get_offline_recognizer()

        if hasattr(audio, "url"):
            raise ValueError(
                "SherpaOnnxSTTProvider does not support URL-based AudioContent. "
                "Provide raw AudioChunk data instead."
            )

        samples = _pcm_s16le_to_float32(audio.data)
        sample_rate = getattr(audio, "sample_rate", self._config.sample_rate)

        def _run() -> str:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode(stream)
            return stream.result.text.strip()

        return await asyncio.to_thread(_run)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results using OnlineRecognizer.

        Only supported for transducer mode. Whisper mode raises ValueError.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptionResult with partial and final transcripts.
        """
        if self._config.mode == "whisper":
            raise ValueError(
                "Streaming transcription is not supported for Whisper mode. "
                "Use transcribe() for batch recognition instead."
            )

        recognizer = self._get_online_recognizer()
        stream = recognizer.create_stream()
        last_text = ""

        async for chunk in audio_stream:
            if chunk.data:
                samples = _pcm_s16le_to_float32(chunk.data)
                sample_rate = chunk.sample_rate or self._config.sample_rate

                def _feed_and_decode(
                    s: Any = stream, sr: int = sample_rate, sa: list[float] = samples
                ) -> None:
                    s.accept_waveform(sr, sa)
                    while recognizer.is_ready(s):
                        recognizer.decode(s)

                await asyncio.to_thread(_feed_and_decode)

                text = recognizer.get_result(stream).text.strip()
                if text and text != last_text:
                    is_endpoint = recognizer.is_endpoint(stream)
                    yield TranscriptionResult(
                        text=text,
                        is_final=is_endpoint,
                    )
                    if is_endpoint:
                        recognizer.reset(stream)
                    last_text = text if not is_endpoint else ""

            if chunk.is_final:
                # Flush any remaining text
                text = recognizer.get_result(stream).text.strip()
                if text and text != last_text:
                    yield TranscriptionResult(text=text, is_final=True)
                break
