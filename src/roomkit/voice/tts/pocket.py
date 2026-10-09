"""Pocket TTS text-to-speech provider: a 100M-parameter local model, on CPU or GPU.

`Pocket TTS <https://github.com/kyutai-labs/pocket-tts>`_ (Kyutai / Gradium,
MIT code) streams 24 kHz speech in 80 ms chunks. It speaks English, French,
German, Portuguese, Italian, Spanish and, since pocket-tts 3.3, Dutch, one
language per loaded model, and clones a voice from a short clip. It runs
faster than real time on two CPU cores; a
CUDA GPU also works (Kyutai does not support it officially, but the model is a
plain ``torch`` module). Install with ``pip install roomkit[pocket-tts]``.

The licence of each pre-made voice is listed on
https://huggingface.co/kyutai/tts-voices.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts._thread_stream import iterate_in_thread
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.pocket")

SAMPLE_RATE = 24000


@dataclass
class PocketTTSConfig:
    """Configuration for :class:`PocketTTSProvider`.

    Attributes:
        language: Pocket TTS model to load: ``english``, ``french``,
            ``german``, ``portuguese``, ``italian``, ``spanish``, ``dutch``
            (pocket-tts 3.3+), each but ``english`` also as a larger ``_24l``
            variant (``french_24l``) that sounds better and is about three
            times slower.
        voices: Named voices; the first one is the default. Each value is a
            pre-made voice name (``alba``, ``estelle``...), a local audio clip
            to clone, an ``hf://`` path, or a ``.safetensors`` voice state
            exported by ``pocket-tts export-voice --language <same language>``
            (the fastest to load).
        device: ``"cpu"`` or a CUDA device (``"cuda"``, ``"cuda:1"``).
        quantize: int8 dynamic quantization, CPU only.
        temperature: Sampling temperature; ``None`` keeps the model's default.
    """

    language: str = "english"
    voices: dict[str, str] = field(default_factory=lambda: {"alba": "alba"})
    device: str = "cpu"
    quantize: bool = False
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.voices:
            raise ValueError("PocketTTSConfig.voices needs at least one voice")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError(f"device must be 'cpu' or a CUDA device, not '{self.device}'")
        if self.quantize and self.device != "cpu":
            raise ValueError("quantize=True only works on CPU")


class PocketTTSProvider(TTSProvider):
    """Kyutai Pocket TTS, streaming from a model held in this process."""

    def __init__(self, config: PocketTTSConfig | None = None) -> None:
        self._config = config or PocketTTSConfig()
        self._model: _PocketModel | None = None
        # One generation at a time: the model is not thread-safe.
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "PocketTTS"

    @property
    def default_voice(self) -> str:
        return next(iter(self._config.voices))

    async def warmup(self) -> None:
        """Load the model and every voice state."""
        async with self._lock:
            await asyncio.to_thread(self._load)

    def _load(self) -> _PocketModel:
        if self._model is None:
            self._model = _PocketModel(self._config)
        return self._model

    def _voice_name(self, voice: str | None) -> str:
        name = voice or self.default_voice
        if name not in self._config.voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._config.voices)}")
        return name

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as 16-bit PCM chunks at 24 kHz, then a final marker."""
        name = self._voice_name(voice)
        if text.strip():
            async with self._lock:
                model = await asyncio.to_thread(self._load)
                cancel = threading.Event()
                # aclosing: a barge-in closing this stream must stop the model
                # thread before the lock is released.
                frames = model.generate(text, name, cancel)
                async with contextlib.aclosing(iterate_in_thread(frames, cancel)) as pcms:
                    async for pcm in pcms:
                        yield AudioChunk(data=pcm, sample_rate=SAMPLE_RATE)
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize *text* as a WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        async with self._lock:
            self._model = None


class _PocketModel:
    """The real model and its voice states, on the configured device."""

    def __init__(self, config: PocketTTSConfig) -> None:
        try:
            import torch
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise ImportError(
                "pocket-tts is required for PocketTTSProvider. "
                "Install it with: pip install roomkit[pocket-tts]"
            ) from exc
        self._torch = torch
        logger.info("Loading Pocket TTS '%s' on %s", config.language, config.device)
        self._model = TTSModel.load_model(
            language=config.language, temp=config.temperature, quantize=config.quantize
        )
        if config.device != "cpu":
            self._model.to(config.device)
        if self._model.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Pocket TTS returned {self._model.sample_rate} Hz, not 24000")
        self._states = {
            name: self._model.get_state_for_audio_prompt(source)
            for name, source in config.voices.items()
        }

    def generate(self, text: str, voice: str, cancel: threading.Event) -> Generator[bytes]:
        torch = self._torch
        # copy_state (the default) keeps the voice state reusable across replies.
        for chunk in self._model.generate_audio_stream(self._states[voice], text, stop=cancel):
            samples = chunk.detach().float().reshape(-1).clamp(-1.0, 1.0)
            yield (samples * 32767).to(torch.int16).cpu().numpy().tobytes()
