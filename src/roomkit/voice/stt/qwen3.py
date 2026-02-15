"""Qwen3-ASR speech-to-text provider."""

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
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


@dataclass
class Qwen3ASRConfig:
    """Configuration for the Qwen3-ASR STT provider.

    Attributes:
        model_id: HuggingFace model ID.
        backend: Inference backend — ``"transformers"`` (batch only) or
            ``"vllm"`` (batch + streaming).
        device_map: Torch device mapping (e.g. ``"auto"``, ``"cuda:0"``).
        dtype: Model dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        language: Language code for recognition (e.g. ``"en"``, ``"zh"``).
            ``None`` for automatic language detection.
        chunk_size_sec: Streaming chunk duration in seconds.
        unfixed_chunk_num: Number of unfixed chunks to re-decode in streaming.
        unfixed_token_num: Number of unfixed tokens in streaming output.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_new_tokens: Maximum output tokens.
    """

    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    backend: str = "transformers"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    language: str | None = None
    chunk_size_sec: float = 2.0
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    gpu_memory_utilization: float = 0.3
    max_new_tokens: int = 2048


def _pcm_s16le_to_float32_np(data: bytes) -> Any:
    """Convert PCM signed 16-bit little-endian bytes to numpy float32 array in [-1, 1]."""
    import numpy as np

    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
    return np.array(samples, dtype=np.float32) / 32768.0


class Qwen3ASRProvider(STTProvider):
    """Speech-to-text provider using Qwen3-ASR.

    Supports batch transcription with both ``transformers`` and ``vllm`` backends,
    and streaming transcription with the ``vllm`` backend only.
    """

    def __init__(self, config: Qwen3ASRConfig) -> None:
        try:
            import qwen_asr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "qwen-asr is required for Qwen3ASRProvider. "
                "Install it with: pip install roomkit[qwen-asr]"
            ) from exc
        self._config = config
        self._model: Any = None

    @property
    def name(self) -> str:
        return "Qwen3ASR"

    @property
    def supports_streaming(self) -> bool:
        return self._config.backend == "vllm"

    def _load_model(self) -> Any:
        """Load the Qwen3-ASR model (synchronous, meant to run in a thread)."""
        if self._model is not None:
            return self._model

        from qwen_asr import Qwen3ASRModel

        cfg = self._config
        logger.info(
            "Loading Qwen3-ASR model: model_id=%s, backend=%s, device_map=%s, dtype=%s",
            cfg.model_id,
            cfg.backend,
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

        if cfg.backend == "vllm":
            self._model = Qwen3ASRModel.LLM(
                cfg.model_id,
                dtype=resolved_dtype,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_new_tokens=cfg.max_new_tokens,
            )
        else:
            self._model = Qwen3ASRModel.from_pretrained(
                cfg.model_id,
                device_map=cfg.device_map,
                dtype=resolved_dtype,
            )

        logger.info("Qwen3-ASR model loaded successfully (backend=%s)", cfg.backend)
        return self._model

    async def warmup(self) -> None:
        """Pre-load the model (CUDA init can be slow)."""
        await asyncio.to_thread(self._load_model)
        logger.info("Qwen3-ASR model warmed up (backend=%s)", self._config.backend)

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        """Transcribe complete audio.

        Args:
            audio: Audio content or raw audio chunk (PCM S16LE expected).

        Returns:
            TranscriptionResult with text.
        """
        if hasattr(audio, "url"):
            raise ValueError(
                "Qwen3ASRProvider does not support URL-based AudioContent. "
                "Provide raw AudioChunk data instead."
            )

        samples = _pcm_s16le_to_float32_np(audio.data)
        sample_rate = getattr(audio, "sample_rate", 16000) or 16000
        model = self._load_model()
        cfg = self._config

        def _run() -> str:
            kwargs: dict[str, Any] = {"audio": samples, "sr": sample_rate}
            if cfg.language is not None:
                kwargs["language"] = cfg.language
            result = model.transcribe(**kwargs)
            if isinstance(result, list):
                return str(result[0]).strip() if result else ""
            return str(result).strip()

        text = await asyncio.to_thread(_run)
        return TranscriptionResult(text=text)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results.

        Only supported with ``vllm`` backend.  With ``transformers`` backend,
        falls back to the base class buffered batch transcription.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptionResult with partial and final transcripts.
        """
        if self._config.backend != "vllm":
            async for result in super().transcribe_stream(audio_stream):
                yield result
            return

        model = self._load_model()
        cfg = self._config

        state = await asyncio.to_thread(
            model.init_streaming_state,
            cfg.chunk_size_sec,
            cfg.unfixed_chunk_num,
            cfg.unfixed_token_num,
        )

        last_text = ""

        async for chunk in audio_stream:
            if chunk.data:
                samples = _pcm_s16le_to_float32_np(chunk.data)
                sample_rate = chunk.sample_rate or 16000

                def _feed(s: Any = state, a: Any = samples, sr: int = sample_rate) -> str:
                    kwargs: dict[str, Any] = {"state": s, "audio": a, "sr": sr}
                    if cfg.language is not None:
                        kwargs["language"] = cfg.language
                    result = model.streaming_transcribe(**kwargs)
                    return str(result).strip() if result else ""

                text = await asyncio.to_thread(_feed)
                if text and text != last_text:
                    yield TranscriptionResult(text=text, is_final=False)
                    last_text = text

            if chunk.is_final:
                break

        def _finish(s: Any = state) -> str:
            result = model.finish_streaming_transcribe(state=s)
            return str(result).strip() if result else ""

        final_text = await asyncio.to_thread(_finish)
        if final_text:
            yield TranscriptionResult(text=final_text, is_final=True)

    async def close(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Qwen3-ASR model released")
