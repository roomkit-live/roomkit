"""Gradium speech-to-text provider."""

from __future__ import annotations

import asyncio
import contextlib
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

        When the audio stream ends (caller sends ``None`` sentinel)
        without the server closing the WebSocket, a drain timeout
        gives the server a few seconds to send final VAD messages
        before yielding accumulated text as final.

        Yields:
            - ``is_final=False`` for each ``text`` update (partial)
            - ``is_final=True`` when VAD detects turn completion (then returns)
        """
        client = self._get_client()
        threshold = self._config.vad_turn_threshold
        required_steps = self._config.vad_turn_steps

        audio_done = asyncio.Event()

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

        async def audio_gen() -> AsyncIterator[bytes]:
            # Prepend silence matching delay_in_frames so the model's
            # processing delay falls on silence, not real speech.
            # Each frame = 80ms = 1920 samples at 24kHz = 3840 bytes.
            delay = self._config.delay_in_frames
            if delay > 0:
                silence_bytes = delay * 1920 * 2  # 16-bit samples
                yield b"\x00" * silence_bytes
            # Yield pre-buffered real audio as a single burst.
            if pre_buffer:
                yield b"".join(pre_buffer)
            if stream_exhausted:
                audio_done.set()
                return
            # Then stream directly — no extra buffering.
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

        segments: list[str] = []  # completed segments from end_text
        current_partial = ""  # latest text for the in-progress segment
        consecutive_inactive = 0  # consecutive VAD steps above threshold
        msg_count = 0  # total messages received

        # Timing for latency diagnostics
        first_text_at = 0.0  # first text message of this turn
        last_text_at = 0.0  # last text message (speech end proxy)

        # Pump server messages into a queue so we can apply a drain
        # timeout after audio ends.  Without this, the iteration on
        # stream._stream blocks indefinitely when the Gradium server
        # keeps the WebSocket open after audio stops.
        msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _pump() -> None:
            try:
                async for msg in stream._stream:  # noqa: SLF001
                    await msg_queue.put(msg)
            except Exception:
                logger.debug("Gradium message pump ended with error", exc_info=True)
            await msg_queue.put(None)  # sentinel

        pump_task = asyncio.get_running_loop().create_task(_pump(), name="gradium_msg_pump")

        # Seconds to wait for final server messages after audio ends.
        drain_timeout_s = 3.0

        try:
            drain_deadline: float | None = None
            while True:
                # Once audio has ended, start a drain countdown
                if drain_deadline is None and audio_done.is_set():
                    drain_deadline = time.monotonic() + drain_timeout_s
                    logger.debug(
                        "Audio ended, draining server messages (%.0fs)",
                        drain_timeout_s,
                    )

                if drain_deadline is not None:
                    remaining = drain_deadline - time.monotonic()
                    if remaining <= 0:
                        logger.info("Drain timeout %.0fs after audio ended", drain_timeout_s)
                        break
                    timeout: float | None = remaining
                else:
                    # Poll with short timeout so we notice audio_done
                    # promptly even if the server stops sending messages.
                    timeout = 0.5

                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=timeout)
                except TimeoutError:
                    if drain_deadline is not None:
                        logger.info("Drain timeout %.0fs after audio ended", drain_timeout_s)
                        break
                    # Normal poll timeout — re-check audio_done
                    continue

                if msg is None:
                    break

                msg_count += 1
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
                        logger.debug("Gradium partial: %r", full)
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
                            return

            # Stream ended or drain timed out without VAD turn completion.
            # Yield accumulated text as final so it isn't silently lost.
            if segments or current_partial:
                parts = list(segments)
                if current_partial:
                    parts.append(current_partial)
                text = " ".join(parts)
                logger.info(
                    "Stream ended with pending text, yielding as final: %s",
                    text,
                )
                yield TranscriptionResult(text=text, is_final=True)
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
            logger.info("Gradium stream closing (msgs=%d, segs=%d)", msg_count, len(segments))
            clean = await _close_stream(stream)
            if not clean:
                # Aborted close may corrupt the SDK's aiohttp session.
                # Force a fresh client on the next reconnect.
                logger.info("Gradium stream close was not clean, resetting client")
                self._client = None

    async def close(self) -> None:
        """Release resources."""
        self._client = None
