"""Speaker diarization provider using sherpa-onnx speaker embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.diarization.base import DiarizationProvider, DiarizationResult

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


@dataclass
class SherpaOnnxDiarizationConfig:
    """Configuration for the sherpa-onnx speaker diarization provider."""

    model: str
    """Path to the ONNX speaker embedding model."""

    num_threads: int = 1
    provider: str = "cpu"
    search_threshold: float = 0.5
    """Cosine similarity threshold for speaker identification."""

    min_speech_ms: int = 500
    """Minimum accumulated speech (ms) before extracting an embedding."""


class SherpaOnnxDiarizationProvider(DiarizationProvider):
    """Speaker diarization using sherpa-onnx SpeakerEmbeddingExtractor.

    Accumulates PCM frames during speech (VAD active) and extracts a speaker
    embedding on speech boundaries (SPEECH_END) or when the buffer exceeds 2 s.
    """

    def __init__(self, config: SherpaOnnxDiarizationConfig) -> None:
        import sherpa_onnx

        self._config = config
        extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=config.model,
            num_threads=config.num_threads,
            provider=config.provider,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config=extractor_config)
        self._manager = sherpa_onnx.SpeakerEmbeddingManager(
            self._extractor.dim,
        )
        self._speech_buffer = bytearray()
        self._sample_rate = config.sample_rate if hasattr(config, "sample_rate") else 16000
        self._in_speech = False
        self._last_speaker_id = ""
        self._enrolled_embeddings: dict[str, list[float]] = {}  # for debug scoring

    @property
    def name(self) -> str:
        return "SherpaOnnxDiarizationProvider"

    def process(self, frame: AudioFrame) -> DiarizationResult | None:
        """Accumulate speech frames and identify speaker on boundaries."""
        metadata = frame.metadata or {}
        is_speech = metadata.get("vad_is_speech", False)
        is_speech_end = metadata.get("vad_speech_end", False)

        if is_speech:
            self._in_speech = True
            self._speech_buffer.extend(frame.data)

        buffer_ms = len(self._speech_buffer) * 1000 // (self._sample_rate * 2)  # 16-bit PCM

        should_extract = (is_speech_end and buffer_ms >= self._config.min_speech_ms) or (
            self._in_speech and buffer_ms >= 2000
        )

        if not should_extract:
            if is_speech_end:
                self._speech_buffer.clear()
                self._in_speech = False
            return None

        result = self._identify(bytes(self._speech_buffer), self._sample_rate)
        self._speech_buffer.clear()
        self._in_speech = False
        return result

    def _identify(self, pcm_bytes: bytes, sample_rate: int) -> DiarizationResult | None:
        """Extract embedding from PCM and search for a matching speaker."""
        import array

        samples = array.array("h")
        samples.frombytes(pcm_bytes)
        float_samples = [s / 32768.0 for s in samples]

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=float_samples)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            return None

        embedding = self._extractor.compute(stream)

        # Log scoring for debugging
        all_speakers = self._manager.all_speakers
        if all_speakers:
            scores = {}
            for spk in all_speakers:
                if spk in self._enrolled_embeddings:
                    ref = self._enrolled_embeddings[spk]
                    dot = sum(a * b for a, b in zip(embedding, ref, strict=True))
                    norm_a = sum(a * a for a in embedding) ** 0.5
                    norm_b = sum(b * b for b in ref) ** 0.5
                    scores[spk] = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            logger.info(
                "Diarization: threshold=%.2f scores=%s",
                self._config.search_threshold,
                {k: round(v, 3) for k, v in scores.items()},
            )
        else:
            logger.warning("Diarization: no enrolled speakers!")

        name = self._manager.search(embedding, threshold=self._config.search_threshold)

        if not name:
            is_new = True
            speaker_id = "unknown"
            confidence = 0.0
        else:
            is_new = name != self._last_speaker_id
            speaker_id = name
            confidence = 1.0  # Above threshold = confident match

        self._last_speaker_id = speaker_id
        return DiarizationResult(
            speaker_id=speaker_id,
            confidence=confidence,
            is_new_speaker=is_new,
        )

    def enroll_speaker(self, name: str, embedding: list[float]) -> bool:
        """Register a single speaker embedding under *name*."""
        ok: bool = self._manager.add(name, embedding)
        if ok:
            self._enrolled_embeddings[name] = embedding
        return ok

    def register_speaker(self, name: str, embeddings: list[list[float]]) -> bool:
        """Register a speaker with multiple embeddings.

        Computes the centroid (average) embedding and registers it via
        ``add``.  This is equivalent to what the C++ ``Register`` method
        does internally when it is available.
        """
        if len(embeddings) == 1:
            avg = embeddings[0]
        else:
            dim = len(embeddings[0])
            n = len(embeddings)
            avg = [sum(e[i] for e in embeddings) / n for i in range(dim)]
        ok: bool = self._manager.add(name, avg)
        if ok:
            self._enrolled_embeddings[name] = avg
        return ok

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker by name."""
        result: bool = self._manager.remove(name)
        return result

    def extract_embedding(self, pcm_bytes: bytes, sample_rate: int) -> list[float]:
        """Extract a speaker embedding from raw PCM audio."""
        import array

        samples = array.array("h")
        samples.frombytes(pcm_bytes)
        float_samples = [s / 32768.0 for s in samples]

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=float_samples)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            raise ValueError("Audio too short to extract speaker embedding")

        return list(self._extractor.compute(stream))

    def reset(self) -> None:
        self._speech_buffer.clear()
        self._in_speech = False
        self._last_speaker_id = ""

    def close(self) -> None:
        self._speech_buffer.clear()
