"""NeuTTS text-to-speech provider with zero-shot voice cloning."""

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

_SAMPLE_RATE = 24000


@dataclass
class NeuTTSVoiceConfig:
    """Configuration for a NeuTTS voice clone reference.

    Attributes:
        ref_audio: Path to reference WAV (3-15s clean speech, 16kHz mono).
        ref_text: Transcript of the reference audio.
    """

    ref_audio: str
    ref_text: str


@dataclass
class NeuTTSConfig:
    """Configuration for the NeuTTS provider.

    Attributes:
        backbone_repo: HuggingFace repo ID or local path to the GGUF backbone.
        codec_repo: HuggingFace repo ID or local path to the NeuCodec model.
        device: Device for inference (``"cpu"`` or ``"cuda"``).
        voices: Named voice clones mapped to reference audio configs.
        streaming_pre_buffer: Number of chunks to accumulate before yielding
            during ``synthesize_stream``.  Each chunk is ~500ms.  Pre-buffering
            prevents playback underruns on CPU where codec decode may be slower
            than real-time.  Set to ``0`` to disable (e.g. on GPU).
    """

    backbone_repo: str = "neuphonic/neutts-nano-french-q8-gguf"
    codec_repo: str = "neuphonic/neucodec"
    device: str = "cpu"
    voices: dict[str, NeuTTSVoiceConfig] = field(default_factory=dict)
    streaming_pre_buffer: int = 2


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


class NeuTTSProvider(TTSProvider):
    """Text-to-speech provider using NeuTTS with zero-shot voice cloning.

    NeuTTS is an LLM-based TTS (Qwen2.5 backbone + NeuCodec) that generates
    24kHz audio with native streaming support via GGUF quantized models.
    """

    def __init__(self, config: NeuTTSConfig) -> None:
        self._patch_perth()
        try:
            import neutts  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "neutts is required for NeuTTSProvider. "
                "Install it with: pip install roomkit[neutts]"
            ) from exc
        self._config = config
        self._model: Any = None
        self._cached_refs: dict[str, Any] = {}

    @staticmethod
    def _patch_perth() -> None:
        """Work around a neutts bug with broken Perth watermarking.

        neutts catches ``ImportError``/``AttributeError`` from
        ``perth.PerthImplicitWatermarker()`` but not ``TypeError``.  When perth
        is installed with setuptools>=81 (which dropped ``pkg_resources``),
        ``PerthImplicitWatermarker`` can end up as ``None``.  Calling ``None()``
        raises ``TypeError`` and crashes ``NeuTTS.__init__``.

        We remove the broken attribute so neutts sees an ``AttributeError``
        (which it already handles) instead.

        Note: even if ``import perth`` fails, Python may cache a partially-loaded
        module in ``sys.modules`` — we must check that too.
        """
        import contextlib
        import sys

        with contextlib.suppress(Exception):
            import perth  # noqa: F401

        perth_mod = sys.modules.get("perth")
        if (
            perth_mod is not None
            and hasattr(perth_mod, "PerthImplicitWatermarker")
            and not callable(perth_mod.PerthImplicitWatermarker)
        ):
            delattr(perth_mod, "PerthImplicitWatermarker")

    @property
    def name(self) -> str:
        return "NeuTTS"

    @property
    def default_voice(self) -> str | None:
        voices = list(self._config.voices.keys())
        return voices[0] if voices else None

    def _resolve_voice(self, voice: str | None) -> NeuTTSVoiceConfig:
        """Resolve a voice name to its NeuTTSVoiceConfig."""
        voices = self._config.voices
        if not voices:
            raise ValueError(
                "No voices configured. Add at least one voice to NeuTTSConfig.voices."
            )
        if voice is None:
            return next(iter(voices.values()))
        if voice not in voices:
            raise ValueError(f"Voice '{voice}' not found. Available: {list(voices.keys())}")
        return voices[voice]

    def _load_model(self) -> Any:
        """Load the NeuTTS model (synchronous, meant to run in a thread)."""
        if self._model is not None:
            return self._model

        from neutts import NeuTTS

        cfg = self._config

        # NeuTTS uses different device strings per component:
        #   - GGUF backbone (llama.cpp): "gpu" for GPU, "cpu" for CPU
        #   - PyTorch backbone / codec: "cuda" for GPU, "cpu" for CPU
        # We accept "cuda" or "gpu" and translate appropriately.
        is_gguf = cfg.backbone_repo.endswith("gguf")
        if cfg.device in ("cuda", "gpu"):
            backbone_device = "gpu" if is_gguf else "cuda"
            codec_device = "cuda"
        else:
            backbone_device = "cpu"
            codec_device = "cpu"

        logger.info(
            "Loading NeuTTS model: backbone=%s (%s), codec=%s (%s)",
            cfg.backbone_repo,
            backbone_device,
            cfg.codec_repo,
            codec_device,
        )

        self._model = NeuTTS(
            backbone_repo=cfg.backbone_repo,
            backbone_device=backbone_device,
            codec_repo=cfg.codec_repo,
            codec_device=codec_device,
        )
        logger.info("NeuTTS model loaded successfully")
        return self._model

    def _encode_references(self) -> None:
        """Pre-encode reference audio for all configured voices."""
        model = self._model
        if model is None:
            return

        for voice_name, voice_cfg in self._config.voices.items():
            if voice_name not in self._cached_refs:
                logger.info("Encoding reference audio for '%s'", voice_name)
                ref_codes = model.encode_reference(voice_cfg.ref_audio)
                self._cached_refs[voice_name] = ref_codes
                logger.info("Reference codes cached for '%s'", voice_name)

    async def warmup(self) -> None:
        """Pre-load the model and encode reference audio."""
        await asyncio.to_thread(self._load_model)
        await asyncio.to_thread(self._encode_references)

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

        # Use cached ref codes or encode on the fly
        ref_codes = self._cached_refs.get(voice_name) if voice_name else None
        if ref_codes is None:
            ref_codes = model.encode_reference(voice_cfg.ref_audio)
            if voice_name is not None:
                self._cached_refs[voice_name] = ref_codes

        def _run() -> Any:
            return model.infer(
                text=text,
                ref_codes=ref_codes,
                ref_text=voice_cfg.ref_text,
            )

        samples = await asyncio.to_thread(_run)

        pcm_data = _numpy_to_pcm_s16le(samples)
        wav_data = _wrap_wav(pcm_data, _SAMPLE_RATE)
        data_url = f"data:audio/wav;base64,{base64.b64encode(wav_data).decode()}"

        duration = len(samples) / _SAMPLE_RATE

        return AudioContentModel(
            url=data_url,
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from NeuTTS inference.

        Uses ``infer_stream()`` which yields numpy chunks as they're decoded.
        Only available with GGUF quantized backbones.

        Pre-buffers ``streaming_pre_buffer`` chunks (~500ms each) before
        yielding the first one to prevent playback underruns when codec
        decode is slower than real-time (common on CPU).

        Args:
            text: Text to synthesize.
            voice: Voice name from configured voices (uses first voice if not specified).

        Yields:
            AudioChunk with PCM S16LE audio data at 24kHz.
        """
        voice_cfg = self._resolve_voice(voice)
        voice_name = voice or self.default_voice
        model = self._load_model()

        ref_codes = self._cached_refs.get(voice_name) if voice_name else None
        if ref_codes is None:
            ref_codes = model.encode_reference(voice_cfg.ref_audio)
            if voice_name is not None:
                self._cached_refs[voice_name] = ref_codes

        # Run the blocking generator in a thread, pushing chunks via a queue
        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _stream() -> None:
            # Disable Perth watermarking during streaming: it's applied per-chunk
            # before overlap-add, so adjacent chunks get incompatible watermarks
            # that interfere in the overlap region → crackling artifacts.
            # Non-streaming infer() watermarks the full audio at once (no issue).
            watermarker = getattr(model, "watermarker", None)
            if watermarker is not None:
                model.watermarker = None
            try:
                for chunk_samples in model.infer_stream(
                    text=text,
                    ref_codes=ref_codes,
                    ref_text=voice_cfg.ref_text,
                ):
                    pcm = _numpy_to_pcm_s16le(chunk_samples)
                    audio_chunk = AudioChunk(
                        data=pcm,
                        sample_rate=_SAMPLE_RATE,
                        format="pcm_s16le",
                        is_final=False,
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, audio_chunk)
            finally:
                if watermarker is not None:
                    model.watermarker = watermarker
            # Signal completion
            loop.call_soon_threadsafe(queue.put_nowait, None)

        thread_future = asyncio.ensure_future(asyncio.to_thread(_stream))

        # Pre-buffer: accumulate N chunks before yielding so the playback
        # buffer has a head start over the codec decode.  Without this,
        # the callback-based speaker output underruns between chunks,
        # inserting silence (= crackling).
        pre_buffer_count = self._config.streaming_pre_buffer
        pre_buffer: list[AudioChunk] = []

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if len(pre_buffer) < pre_buffer_count:
                    pre_buffer.append(chunk)
                    if len(pre_buffer) == pre_buffer_count:
                        logger.debug(
                            "Pre-buffer full (%d chunks), starting playback",
                            pre_buffer_count,
                        )
                        for buffered in pre_buffer:
                            yield buffered
                        pre_buffer.clear()
                else:
                    yield chunk

            # Flush any remaining pre-buffer (utterance shorter than
            # pre_buffer_count chunks)
            for buffered in pre_buffer:
                yield buffered
        finally:
            await thread_future

        # Final marker
        yield AudioChunk(
            data=b"",
            sample_rate=_SAMPLE_RATE,
            format="pcm_s16le",
            is_final=True,
        )

    async def close(self) -> None:
        """Release model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._cached_refs.clear()
            logger.info("NeuTTS model released")
