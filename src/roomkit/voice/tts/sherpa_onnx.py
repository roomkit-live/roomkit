"""sherpa-onnx text-to-speech provider."""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class SherpaOnnxTTSConfig:
    """Configuration for the sherpa-onnx TTS provider.

    Attributes:
        model: Path to VITS/Piper ``.onnx`` model.
        tokens: Path to ``tokens.txt``.
        data_dir: Path to espeak-ng data directory (Piper models).
        lexicon: Path to optional lexicon file.
        speaker_id: Speaker ID for multi-speaker models.
        speed: Speech speed multiplier (1.0 = normal).
        sample_rate: Output sample rate (usually determined by the model).
        num_threads: Number of CPU threads for inference.
        provider_type: ONNX execution provider (``"cpu"`` or ``"cuda"``).
    """

    model: str = ""
    tokens: str = ""
    data_dir: str = ""
    lexicon: str = ""
    speaker_id: int = 0
    speed: float = 1.0
    sample_rate: int = 22050
    num_threads: int = 2
    provider_type: str = "cpu"


def _float32_to_pcm_s16le(samples: list[float]) -> bytes:
    """Convert float32 samples in [-1, 1] to PCM signed 16-bit LE bytes."""
    clamped = [max(-1.0, min(1.0, s)) for s in samples]
    int_samples = [int(s * 32767) for s in clamped]
    return struct.pack(f"<{len(int_samples)}h", *int_samples)


def _wrap_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM S16LE data in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    # RIFF header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


class SherpaOnnxTTSProvider(TTSProvider):
    """Text-to-speech provider using sherpa-onnx with VITS/Piper models."""

    def __init__(self, config: SherpaOnnxTTSConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxTTSProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc
        self._config = config
        self._sherpa = __import__("sherpa_onnx")
        self._tts: Any = None

    @property
    def name(self) -> str:
        return "SherpaOnnxTTS"

    @property
    def default_voice(self) -> str:
        return str(self._config.speaker_id)

    def _get_tts(self) -> Any:
        """Lazily create the OfflineTts engine."""
        if self._tts is None:
            cfg = self._config
            vits_config = self._sherpa.OfflineTtsVitsModelConfig(
                model=cfg.model,
                tokens=cfg.tokens,
                data_dir=cfg.data_dir,
                lexicon=cfg.lexicon,
            )
            model_config = self._sherpa.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=cfg.num_threads,
                provider=cfg.provider_type,
            )
            tts_config = self._sherpa.OfflineTtsConfig(model=model_config)
            self._tts = self._sherpa.OfflineTts(tts_config)
        return self._tts

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Speaker ID as string (uses default if not specified).

        Returns:
            AudioContent with a ``data:`` URL containing WAV audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        tts = self._get_tts()
        sid = int(voice) if voice is not None else self._config.speaker_id
        speed = self._config.speed

        def _run() -> Any:
            return tts.generate(text, sid=sid, speed=speed)

        audio = await asyncio.to_thread(_run)

        pcm_data = _float32_to_pcm_s16le(list(audio.samples))
        wav_data = _wrap_wav(pcm_data, audio.sample_rate)
        data_url = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode()}"

        # Estimate duration from samples
        duration = len(audio.samples) / audio.sample_rate if audio.sample_rate else None

        return AudioContentModel(
            url=data_url,
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks using a callback bridge.

        Args:
            text: Text to synthesize.
            voice: Speaker ID as string (uses default if not specified).

        Yields:
            AudioChunk with PCM S16LE audio data.
        """
        tts = self._get_tts()
        sid = int(voice) if voice is not None else self._config.speaker_id
        speed = self._config.speed

        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _callback(samples: list[float], progress: float) -> int:
            """Called by sherpa-onnx from the executor thread."""
            pcm = _float32_to_pcm_s16le(samples)
            chunk = AudioChunk(
                data=pcm,
                sample_rate=self._config.sample_rate,
                format="pcm_s16le",
                is_final=False,
            )
            loop.call_soon_threadsafe(queue.put_nowait, chunk)
            return 1  # continue generation

        def _run() -> None:
            tts.generate(text, sid=sid, speed=speed, callback=_callback)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        task = asyncio.get_running_loop().run_in_executor(None, _run)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await task

        # Yield final marker
        yield AudioChunk(
            data=b"",
            sample_rate=self._config.sample_rate,
            format="pcm_s16le",
            is_final=True,
        )
