"""Gradium speech-to-text provider."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)

# Gradium STT expects 24kHz PCM input
_GRADIUM_SAMPLE_RATE = 24000


@dataclass
class GradiumSTTConfig:
    """Configuration for Gradium STT provider."""

    api_key: str
    region: str = "us"
    model_name: str = "default"
    input_format: str = "pcm"  # pcm | wav | opus
    # Language code: en | fr | de | es | pt.  Grounds the model to a
    # specific language and improves transcription quality.
    language: str | None = None
    json_config: dict[str, Any] | None = field(default=None, repr=False)
    # Pre-connect buffer: accumulate this much real audio (ms) before
    # opening the WebSocket.  The server gets a burst of context on
    # connect, avoiding lost first words from model warmup.  Set to 0
    # to connect immediately (first chunk is still sent on connect).
    connect_buffer_ms: int = 300
    # Model processing delay in frames (80ms each).  The server needs
    # this many frames before it starts producing text.  Silence is
    # prepended to cover the delay so real speech isn't lost.
    # Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
    delay_in_frames: int = 7
    # VAD inactivity_prob threshold (0.0-1.0).  The 2-second-horizon
    # prediction must exceed this for vad_steps consecutive steps
    # before the turn ends.
    vad_threshold: float = 0.9
    # Consecutive steps above vad_threshold required to confirm
    # end-of-turn.  Each step = 80ms, so 10 = 800ms.
    vad_steps: int = 10
    # Fallback: if the server stops sending steps, end the turn after
    # this many seconds of no messages at all.
    timeout_s: float = 3.0


def _resample(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit signed little-endian PCM using numpy."""
    if src_rate == dst_rate:
        return data

    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    ratio = dst_rate / src_rate
    out_len = int(len(samples) * ratio)
    if out_len == 0:
        return b""

    indices = np.arange(out_len) / ratio
    idx = indices.astype(np.intp)
    frac = indices - idx
    # Clamp to valid range for interpolation
    next_idx = np.minimum(idx + 1, len(samples) - 1)
    resampled = samples[idx] * (1.0 - frac) + samples[next_idx] * frac
    return bytes(np.clip(resampled, -32768, 32767).astype(np.int16).tobytes())


async def _close_stream(stream: Any, *, timeout: float = 2.0) -> bool:
    """Close a Gradium STT stream, freeing the WebSocket session slot.

    Returns True if the close completed cleanly, False on timeout/error.
    """
    try:
        raw = getattr(stream, "_stream", None)
        if raw is not None and hasattr(raw, "aclose"):
            await asyncio.wait_for(raw.aclose(), timeout=timeout)
        return True
    except Exception:
        logger.debug("Error closing Gradium STT stream", exc_info=True)
        return False


class GradiumSTTProvider(STTProvider):
    """Gradium speech-to-text provider with streaming support."""

    def __init__(self, config: GradiumSTTConfig) -> None:
        self._config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        return "GradiumSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    def _get_client(self) -> Any:
        if self._client is None:
            from gradium import GradiumClient

            self._client = GradiumClient(
                base_url=f"https://{self._config.region}.api.gradium.ai/api/",
                api_key=self._config.api_key,
            )
        return self._client

    def _build_setup(self) -> dict[str, Any]:
        """Build the STTSetup dict for the SDK."""
        setup: dict[str, Any] = {
            "model_name": self._config.model_name,
            "input_format": self._config.input_format,
        }
        jc: dict[str, Any] = {"delay_in_frames": self._config.delay_in_frames}
        if self._config.language is not None:
            jc["language"] = self._config.language
        if self._config.json_config is not None:
            jc.update(self._config.json_config)
        if jc:
            setup["json_config"] = jc
        return setup

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        """Transcribe complete audio to text using the Gradium SDK."""
        if hasattr(audio, "url"):
            raise ValueError(
                "GradiumSTTProvider does not support URL-based AudioContent. "
                "Provide raw AudioChunk data instead."
            )

        audio_data = audio.data
        src_rate = getattr(audio, "sample_rate", 16000)

        # Resample to 24kHz for Gradium
        resampled = _resample(audio_data, src_rate, _GRADIUM_SAMPLE_RATE)

        client = self._get_client()
        result = await client.stt(self._build_setup(), resampled)
        return TranscriptionResult(text=result.text)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with real-time results.

        Iterates ``stream._stream`` directly to access all message types:

        - ``text``: growing partial transcript for the current segment.
        - ``end_text``: marks a segment as complete.
        - ``step``: VAD heartbeat every ~80ms (used as a clock tick).

        Turn detection: when ``inactivity_prob`` (2s horizon) exceeds
        ``vad_threshold`` and there is accumulated text, the turn ends.

        Yields:
            - ``is_final=False`` for each ``text`` update (partial)
            - ``is_final=True`` when VAD confirms silence (then returns)
        """
        client = self._get_client()
        vad_threshold = self._config.vad_threshold
        vad_steps_required = self._config.vad_steps
        timeout_s = self._config.timeout_s

        # --- Pre-connect buffer: accumulate real audio before opening the
        # WebSocket so the server gets a context burst on connect. --------
        pre_buffer: list[bytes] = []
        pre_buffer_bytes = 0
        target_bytes = int(_GRADIUM_SAMPLE_RATE * 2 * self._config.connect_buffer_ms / 1000)
        stream_exhausted = False

        async for chunk in audio_stream:
            if chunk.data:
                src_rate = chunk.sample_rate or 16000
                data = _resample(chunk.data, src_rate, _GRADIUM_SAMPLE_RATE)
                pre_buffer.append(data)
                pre_buffer_bytes += len(data)
            if chunk.is_final:
                stream_exhausted = True
                break
            if pre_buffer_bytes >= target_bytes:
                break

        audio_done = asyncio.Event()

        async def audio_gen() -> AsyncIterator[bytes]:
            # Prepend silence matching delay_in_frames so the model's
            # processing delay falls on silence, not real speech.
            delay = self._config.delay_in_frames
            if delay > 0:
                silence_bytes = delay * 1920 * 2  # 16-bit, 80ms frames
                yield b"\x00" * silence_bytes
            if pre_buffer:
                yield b"".join(pre_buffer)
            if stream_exhausted:
                audio_done.set()
                return
            async for chunk in audio_stream:
                if chunk.data:
                    src_rate = chunk.sample_rate or 16000
                    yield _resample(chunk.data, src_rate, _GRADIUM_SAMPLE_RATE)
                if chunk.is_final:
                    break
            audio_done.set()

        pre_ms = int(pre_buffer_bytes / (_GRADIUM_SAMPLE_RATE * 2) * 1000)
        logger.info("Gradium stream connecting (pre-buffered %dms)", pre_ms)
        stream = await client.stt_stream(self._build_setup(), audio_gen())
        logger.info("Gradium stream connected")

        segments: list[str] = []
        current_partial = ""
        consecutive_inactive = 0
        first_text_at = 0.0
        last_text_at = 0.0
        msg_count = 0
        speech_start_signalled = False

        raw_iter = stream._stream.__aiter__()  # noqa: SLF001

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(raw_iter.__anext__(), timeout=timeout_s)
                except (TimeoutError, StopAsyncIteration):
                    break

                msg_count += 1
                type_ = msg.get("type")

                if type_ == "text":
                    text = msg.get("text", "")
                    if text:
                        now = time.monotonic()
                        speech_start = False
                        if not segments and not current_partial:
                            first_text_at = now
                            if not speech_start_signalled:
                                speech_start_signalled = True
                                speech_start = True
                        last_text_at = now
                        current_partial = text
                        consecutive_inactive = 0  # speech activity
                        full = " ".join([*segments, current_partial])
                        yield TranscriptionResult(
                            text=full,
                            is_final=False,
                            is_speech_start=speech_start,
                        )

                elif type_ == "end_text":
                    if current_partial:
                        segments.append(current_partial)
                        current_partial = ""
                        logger.info(
                            "STT segment complete, accumulated: %s",
                            " ".join(segments),
                        )

                elif type_ == "step":
                    vad = msg.get("vad", [])
                    if len(vad) >= 3:
                        inactivity = vad[2].get("inactivity_prob", 0.0)
                        if not (segments or current_partial):
                            continue
                        if inactivity > vad_threshold:
                            consecutive_inactive += 1
                        else:
                            consecutive_inactive = 0
                        if consecutive_inactive >= vad_steps_required:
                            logger.info(
                                "Turn end: inactivity=%.3f, steps=%d (%dms)",
                                inactivity,
                                consecutive_inactive,
                                consecutive_inactive * 80,
                            )
                            break

            # Yield accumulated text as final.
            if segments or current_partial:
                parts = list(segments)
                if current_partial:
                    parts.append(current_partial)
                text = " ".join(parts)
                now = time.monotonic()
                turn_ms = int((now - first_text_at) * 1000) if first_text_at else 0
                silence_ms = int((now - last_text_at) * 1000) if last_text_at else 0
                logger.info(
                    "Turn complete (turn=%dms, silence=%dms): %s",
                    turn_ms,
                    silence_ms,
                    text,
                )
                yield TranscriptionResult(text=text, is_final=True)
        finally:
            logger.info("Gradium stream closing (msgs=%d, segs=%d)", msg_count, len(segments))
            clean = await _close_stream(stream)
            if not clean:
                logger.info("Gradium stream close was not clean, resetting client")
                self._client = None

    async def close(self) -> None:
        """Release resources."""
        self._client = None
