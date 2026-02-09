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
    json_config: dict[str, Any] | None = field(default=None, repr=False)
    # VAD turn-detection: use 3rd prediction (2s horizon) inactivity_prob.
    # When this exceeds the threshold for enough consecutive steps AND
    # we have completed segments, the accumulated text is yielded as final.
    vad_turn_threshold: float = 0.5
    # Number of consecutive VAD steps above threshold required to confirm
    # end-of-turn.  Steps arrive every ~80ms, so 12 = ~960ms of sustained
    # inactivity.  Must be long enough to cover within-sentence pauses
    # (breaths, commas) which can be 300-600ms.
    vad_turn_steps: int = 6


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

    The SDK's ``stream._stream`` is an async generator wrapping an
    ``aiohttp.ClientSession`` + WebSocket via ``async with``.  Calling
    ``aclose()`` triggers the generator's ``finally`` block which awaits
    the send/receive tasks and closes the WebSocket + session.

    A timeout is applied because the SDK's sender task may be blocked
    reading from an audio generator that's still waiting on queue data.
    Without a timeout, ``aclose()`` hangs indefinitely and prevents
    the continuous STT loop from reconnecting.

    Returns:
        True if the close completed cleanly, False if it timed out or
        errored (caller should reset the client).
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
        if self._config.json_config is not None:
            setup["json_config"] = self._config.json_config
        return setup

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        """Transcribe complete audio to text using the Gradium SDK."""
        # Extract raw audio bytes and sample rate
        if hasattr(audio, "url"):
            import httpx

            async with httpx.AsyncClient() as fetch_client:
                resp = await fetch_client.get(audio.url)
                resp.raise_for_status()
                audio_data = resp.content
                src_rate = 16000
        else:
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

        Keeps a single long-lived WebSocket open and yields
        ``is_final=True`` at turn boundaries without closing the
        stream.  The SDK's ``iter_text()`` silently drops ``end_text``
        and ``step`` events, so we iterate ``stream._stream`` directly.

        Protocol (from Gradium API docs):

        - ``text``: growing partial transcript for the current segment.
          Each message replaces the previous.
        - ``end_text``: marks the current segment as complete (carries
          ``stop_s`` but no text).
        - ``step``: VAD info every ~80ms with ``vad[2]["inactivity_prob"]``
          indicating probability the speaker has finished their turn.

        Turn detection uses ``step`` messages: when ``inactivity_prob``
        exceeds the configured threshold for enough consecutive steps
        AND we have accumulated text, the result is yielded as
        ``is_final=True`` and the generator **returns**, closing the
        WebSocket.  The caller (``run_continuous`` in VoiceChannel)
        reconnects for the next turn, giving each turn a fresh Gradium
        session — this avoids server-side segment overlap across turns.

        Yields:
            - ``is_final=False`` for each ``text`` update (partial)
            - ``is_final=True`` when VAD detects turn completion (then returns)
        """
        client = self._get_client()
        threshold = self._config.vad_turn_threshold
        required_steps = self._config.vad_turn_steps

        async def audio_gen() -> AsyncIterator[bytes]:
            async for chunk in audio_stream:
                if chunk.data:
                    src_rate = chunk.sample_rate or 16000
                    yield _resample(chunk.data, src_rate, _GRADIUM_SAMPLE_RATE)
                if chunk.is_final:
                    break

        stream = await client.stt_stream(self._build_setup(), audio_gen())

        segments: list[str] = []  # completed segments from end_text
        current_partial = ""  # latest text for the in-progress segment
        consecutive_inactive = 0  # consecutive VAD steps above threshold

        # Timing for latency diagnostics
        first_text_at = 0.0  # first text message of this turn
        last_text_at = 0.0  # last text message (speech end proxy)

        try:
            async for msg in stream._stream:  # noqa: SLF001
                type_ = msg.get("type")
                if type_ == "text":
                    text = msg.get("text", "")
                    if text:
                        now = time.monotonic()
                        if not segments and not current_partial:
                            first_text_at = now
                        last_text_at = now
                        current_partial = text
                        consecutive_inactive = 0  # speech detected
                        # Yield partial: completed segments + current
                        full = " ".join([*segments, current_partial])
                        yield TranscriptionResult(text=full, is_final=False)

                elif type_ == "end_text":
                    if current_partial:
                        segments.append(current_partial)
                        current_partial = ""
                        logger.info(
                            "STT segment complete, accumulated: %s",
                            " ".join(segments),
                        )

                elif type_ == "step":
                    # VAD turn detection: 3rd prediction (2s horizon)
                    vad = msg.get("vad", [])
                    if len(vad) >= 3:
                        probs = [v.get("inactivity_prob", 0.0) for v in vad[:3]]
                        inactivity = probs[2]  # 2s horizon

                        if inactivity > threshold:
                            consecutive_inactive += 1
                        else:
                            if consecutive_inactive > 0 and segments:
                                logger.debug(
                                    "VAD reset at step %d (prob=[%.2f,%.2f,%.2f])",
                                    consecutive_inactive,
                                    *probs,
                                )
                            consecutive_inactive = 0

                        if segments or current_partial:
                            logger.debug(
                                "VAD step %d/%d prob=[%.2f,%.2f,%.2f] segs=%d partial=%r",
                                consecutive_inactive,
                                required_steps,
                                *probs,
                                len(segments),
                                current_partial or None,
                            )

                        if consecutive_inactive >= required_steps and (
                            segments or current_partial
                        ):
                            # Turn complete — yield and close stream
                            parts = list(segments)
                            if current_partial:
                                parts.append(current_partial)
                            text = " ".join(parts)

                            now = time.monotonic()
                            turn_dur_ms = int((now - first_text_at) * 1000) if first_text_at else 0
                            silence_ms = int((now - last_text_at) * 1000) if last_text_at else 0

                            logger.info(
                                "VAD turn complete (turn=%dms, silence=%dms, vad_steps=%d): %s",
                                turn_dur_ms,
                                silence_ms,
                                required_steps,
                                text,
                            )
                            yield TranscriptionResult(text=text, is_final=True)
                            # Return to close the WebSocket; run_continuous
                            # reconnects for the next turn.
                            return
        finally:
            clean = await _close_stream(stream)
            if not clean:
                # Aborted close may corrupt the SDK's aiohttp session.
                # Force a fresh client on the next reconnect.
                logger.debug("Stream close was not clean, resetting client")
                self._client = None

    async def close(self) -> None:
        """Release resources."""
        self._client = None
