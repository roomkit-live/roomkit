"""Qwen3-TTS text-to-speech provider with zero-shot voice cloning."""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class VoiceCloneConfig:
    """Configuration for a voice clone reference.

    Attributes:
        ref_audio: Path to reference audio file (3s+ of clean speech).
        ref_text: Transcript of the reference audio.
    """

    ref_audio: str
    ref_text: str


@dataclass
class Qwen3TTSConfig:
    """Configuration for the Qwen3-TTS provider.

    Attributes:
        model_id: HuggingFace model ID.
        device_map: Torch device mapping (e.g. ``"auto"``, ``"cuda:0"``).
        dtype: Model dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        attn_implementation: Attention implementation (e.g. ``"flash_attention_2"``).
        language: Default language for synthesis.
        voices: Named voice clones mapped to reference audio configs.
        x_vector_only_mode: Use speaker embedding only (faster, lower quality).
        max_new_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling.
        temperature: Sampling temperature.
        repetition_penalty: Repetition penalty factor.
    """

    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    language: str = "English"
    voices: dict[str, VoiceCloneConfig] = field(default_factory=dict)
    x_vector_only_mode: bool = False
    max_new_tokens: int = 4096
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 0.6
    repetition_penalty: float = 1.05


def _numpy_to_pcm_s16le(samples: Any) -> bytes:
    """Convert a numpy float32 array in [-1, 1] to PCM signed 16-bit LE bytes."""
    import numpy as np

    arr = np.clip(samples, -1.0, 1.0)
    int_samples = (arr * 32767).astype(np.int16)
    return bytes(int_samples.tobytes())


def _wrap_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM S16LE data in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
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


class Qwen3TTSProvider(TTSProvider):
    """Text-to-speech provider using Qwen3-TTS with zero-shot voice cloning."""

    def __init__(self, config: Qwen3TTSConfig) -> None:
        try:
            import qwen_tts  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "qwen-tts is required for Qwen3TTSProvider. "
                "Install it with: pip install roomkit[qwen-tts]"
            ) from exc
        self._config = config
        self._model: Any = None
        self._cached_prompts: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "Qwen3TTS"

    @property
    def default_voice(self) -> str | None:
        voices = list(self._config.voices.keys())
        return voices[0] if voices else None

    def _resolve_voice(self, voice: str | None) -> VoiceCloneConfig:
        """Resolve a voice name to its VoiceCloneConfig."""
        voices = self._config.voices
        if not voices:
            raise ValueError(
                "No voices configured. Add at least one voice to Qwen3TTSConfig.voices."
            )
        if voice is None:
            return next(iter(voices.values()))
        if voice not in voices:
            raise ValueError(f"Voice '{voice}' not found. Available: {list(voices.keys())}")
        return voices[voice]

    def _load_model(self) -> Any:
        """Load the Qwen3-TTS model (synchronous, meant to run in a thread)."""
        if self._model is not None:
            return self._model

        from qwen_tts import Qwen3TTSModel

        cfg = self._config
        logger.info(
            "Loading Qwen3-TTS model: model_id=%s, device_map=%s, dtype=%s",
            cfg.model_id,
            cfg.device_map,
            cfg.dtype,
        )

        import torch

        dtype_map: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(cfg.dtype)
        if resolved_dtype is None:
            raise ValueError(
                f"Unsupported dtype '{cfg.dtype}'. Choose from: {list(dtype_map.keys())}"
            )

        kwargs: dict[str, Any] = {
            "device_map": cfg.device_map,
            "dtype": resolved_dtype,
        }
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        self._model = Qwen3TTSModel.from_pretrained(cfg.model_id, **kwargs)
        logger.info("Qwen3-TTS model loaded successfully")
        return self._model

    def _build_prompts(self) -> None:
        """Pre-build voice clone prompts for all configured voices."""
        model = self._model
        if model is None:
            return

        for voice_name, voice_cfg in self._config.voices.items():
            if voice_name not in self._cached_prompts:
                logger.info("Building voice clone prompt for '%s'", voice_name)
                prompt = model.create_voice_clone_prompt(
                    ref_audio=voice_cfg.ref_audio,
                    ref_text=voice_cfg.ref_text,
                    x_vector_only_mode=self._config.x_vector_only_mode,
                )
                self._cached_prompts[voice_name] = prompt
                logger.info("Voice clone prompt cached for '%s'", voice_name)

    async def warmup(self) -> None:
        """Pre-load the model and build voice clone prompts."""
        await asyncio.to_thread(self._load_model)
        await asyncio.to_thread(self._build_prompts)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio using voice cloning.

        Args:
            text: Text to synthesize.
            voice: Voice name from configured voices (uses first voice if not specified).

        Returns:
            AudioContent with a ``data:`` URL containing WAV audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        voice_cfg = self._resolve_voice(voice)
        voice_name = voice or self.default_voice
        model = self._load_model()

        # Use cached prompt or build one
        clone_prompt = self._cached_prompts.get(voice_name) if voice_name else None
        if clone_prompt is None:
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=voice_cfg.ref_audio,
                ref_text=voice_cfg.ref_text,
                x_vector_only_mode=self._config.x_vector_only_mode,
            )
            if voice_name is not None:
                self._cached_prompts[voice_name] = clone_prompt

        cfg = self._config

        def _run() -> tuple[Any, int]:
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=clone_prompt,
                language=cfg.language,
                max_new_tokens=cfg.max_new_tokens,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
                repetition_penalty=cfg.repetition_penalty,
            )
            return wavs, sr

        wavs_list, sample_rate = await asyncio.to_thread(_run)

        # wavs_list is List[np.ndarray]; take the first (single text input)
        samples = wavs_list[0]
        pcm_data = _numpy_to_pcm_s16le(samples)
        wav_data = _wrap_wav(pcm_data, sample_rate)
        data_url = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode()}"

        duration = len(samples) / sample_rate if sample_rate else None

        return AudioContentModel(
            url=data_url,
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from synthesized speech.

        Generates the full audio in a thread, then yields fixed-size PCM chunks
        for progressive playback. The qwen_tts SDK does not expose a streaming
        API for local inference, so chunking happens post-generation.

        Args:
            text: Text to synthesize.
            voice: Voice name from configured voices (uses first voice if not specified).

        Yields:
            AudioChunk with PCM S16LE audio data.
        """
        voice_cfg = self._resolve_voice(voice)
        voice_name = voice or self.default_voice
        model = self._load_model()

        clone_prompt = self._cached_prompts.get(voice_name) if voice_name else None
        if clone_prompt is None:
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=voice_cfg.ref_audio,
                ref_text=voice_cfg.ref_text,
                x_vector_only_mode=self._config.x_vector_only_mode,
            )
            if voice_name is not None:
                self._cached_prompts[voice_name] = clone_prompt

        cfg = self._config

        def _run() -> tuple[Any, int]:
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=clone_prompt,
                language=cfg.language,
                max_new_tokens=cfg.max_new_tokens,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
                repetition_penalty=cfg.repetition_penalty,
            )
            return wavs, sr

        wavs_list, sample_rate = await asyncio.to_thread(_run)

        samples = wavs_list[0]
        pcm_data = _numpy_to_pcm_s16le(samples)

        # Yield 20ms chunks for progressive playback
        bytes_per_sample = 2  # S16LE
        chunk_samples = sample_rate * 20 // 1000  # 20ms worth of samples
        chunk_size = chunk_samples * bytes_per_sample

        offset = 0
        while offset < len(pcm_data):
            end = min(offset + chunk_size, len(pcm_data))
            yield AudioChunk(
                data=pcm_data[offset:end],
                sample_rate=sample_rate,
                format="pcm_s16le",
                is_final=False,
            )
            offset = end

        # Final marker
        yield AudioChunk(
            data=b"",
            sample_rate=sample_rate,
            format="pcm_s16le",
            is_final=True,
        )

    async def close(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._cached_prompts.clear()
            logger.info("Qwen3-TTS model released")
