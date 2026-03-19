"""Grok (xAI) text-to-speech provider."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)

# Available voices on the xAI TTS API.
GROK_VOICES = ("eve", "ara", "rex", "sal", "leo")

# Supported output codecs and their MIME types / AudioChunk format strings.
_CODEC_META: dict[str, tuple[str, str]] = {
    "pcm": ("audio/pcm", "pcm_s16le"),
    "wav": ("audio/wav", "pcm_s16le"),
    "mp3": ("audio/mpeg", "mp3"),
    "mulaw": ("audio/basic", "mulaw"),
    "alaw": ("audio/alaw", "alaw"),
}


@dataclass
class GrokTTSConfig:
    """Configuration for xAI Grok TTS provider.

    Args:
        api_key: xAI API key (or set ``XAI_API_KEY`` env var).
        voice_id: One of ``eve``, ``ara``, ``rex``, ``sal``, ``leo``.
        language: BCP-47 language code or ``auto``.
        codec: Output codec — ``pcm``, ``wav``, ``mp3``, ``mulaw``, ``alaw``.
        sample_rate: Output sample rate in Hz.
        bit_rate: MP3 bit rate (only used when *codec* is ``mp3``).
        base_url: Override the REST API base URL.
        ws_url: Override the WebSocket streaming URL.
        timeout: HTTP request timeout in seconds.
    """

    api_key: str
    voice_id: str = "eve"
    language: str = "en"
    codec: str = "pcm"
    sample_rate: int = 24000
    bit_rate: int = 128000
    base_url: str = "https://api.x.ai/v1"
    ws_url: str = "wss://api.x.ai/v1/tts"
    timeout: float = 60.0


class GrokTTSProvider(TTSProvider):
    """xAI Grok text-to-speech provider with WebSocket streaming support.

    Supports:
    * ``synthesize()`` — REST endpoint, returns full audio.
    * ``synthesize_stream()`` — REST with chunked reading.
    * ``synthesize_stream_input()`` — bidirectional WebSocket for real-time
      text-to-speech (send text deltas, receive audio deltas).
    """

    def __init__(self, config: GrokTTSConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "GrokTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

    @property
    def supports_streaming_input(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # REST synthesis
    # ------------------------------------------------------------------

    def _build_request_body(self, text: str, voice: str | None) -> dict[str, object]:
        """Build the JSON payload for the /tts endpoint."""
        voice_id = voice or self._config.voice_id
        body: dict[str, object] = {
            "text": text,
            "voice_id": voice_id,
            "language": self._config.language,
        }
        output_format: dict[str, object] = {
            "codec": self._config.codec,
            "sample_rate": self._config.sample_rate,
        }
        if self._config.codec == "mp3":
            output_format["bit_rate"] = self._config.bit_rate
        body["output_format"] = output_format
        return body

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio via the REST endpoint.

        Args:
            text: Text to synthesize (max 15 000 characters).
            voice: Voice ID override.

        Returns:
            AudioContent with a data-URL of the generated audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        client = self._get_client()
        body = self._build_request_body(text, voice)

        t0 = time.monotonic()
        response = await client.post("/tts", json=body)
        response.raise_for_status()
        ttfb_ms = (time.monotonic() - t0) * 1000
        logger.debug("GrokTTS synthesize TTFB: %.1f ms", ttfb_ms)

        audio_bytes = response.content
        mime_type, _ = _CODEC_META.get(self._config.codec, ("audio/mpeg", "mp3"))
        data_url = f"data:{mime_type};base64,{base64.b64encode(audio_bytes).decode()}"

        # Estimate duration from PCM byte count when possible.
        duration: float | None = None
        if self._config.codec == "pcm":
            # Raw PCM: 16-bit mono → 2 bytes per sample
            samples = len(audio_bytes) // 2
            if self._config.sample_rate:
                duration = samples / self._config.sample_rate
        elif self._config.codec == "wav":
            # WAV has a 44-byte RIFF header before the PCM data
            pcm_bytes = max(0, len(audio_bytes) - 44)
            if self._config.sample_rate:
                duration = (pcm_bytes // 2) / self._config.sample_rate

        return AudioContentModel(
            url=data_url,
            mime_type=mime_type,
            transcript=text,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # HTTP streaming (chunked response)
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks from the REST endpoint.

        The xAI API returns the audio as a single response, so we chunk it
        ourselves for pipeline compatibility.
        """
        client = self._get_client()
        body = self._build_request_body(text, voice)
        _, fmt = _CODEC_META.get(self._config.codec, ("audio/mpeg", "mp3"))

        async with client.stream("POST", "/tts", json=body) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                logger.error(
                    "GrokTTS stream error %d: %s", response.status_code, error_body.decode()
                )
                response.raise_for_status()

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield AudioChunk(
                        data=chunk,
                        sample_rate=self._config.sample_rate,
                        format=fmt,
                        is_final=False,
                    )

        yield AudioChunk(
            data=b"",
            sample_rate=self._config.sample_rate,
            format=fmt,
            is_final=True,
        )

    # ------------------------------------------------------------------
    # WebSocket streaming (bidirectional)
    # ------------------------------------------------------------------

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio from streaming text input via WebSocket.

        Protocol (xAI):
        * Client → ``{"type": "text.delta", "delta": "..."}``
        * Client → ``{"type": "text.done"}``
        * Server → ``{"type": "audio.delta", "delta": "<base64>"}``
        * Server → ``{"type": "audio.done", "trace_id": "..."}``

        The connection stays open after ``audio.done`` allowing multiple
        turns, but we close after the first turn completes.

        Args:
            text_stream: Async iterator yielding text chunks.
            voice: Voice ID override.

        Yields:
            AudioChunk with decoded audio data.
        """
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for Grok TTS streaming: pip install websockets"
            ) from exc

        voice_id = voice or self._config.voice_id
        _, fmt = _CODEC_META.get(self._config.codec, ("audio/mpeg", "mp3"))

        params = (
            f"?language={self._config.language}"
            f"&voice={voice_id}"
            f"&codec={self._config.codec}"
            f"&sample_rate={self._config.sample_rate}"
        )
        if self._config.codec == "mp3":
            params += f"&bit_rate={self._config.bit_rate}"
        uri = f"{self._config.ws_url}{params}"

        async with websockets.connect(
            uri,
            additional_headers={"Authorization": f"Bearer {self._config.api_key}"},
            open_timeout=30,
        ) as ws:
            # Sender task: forward text deltas then signal done.
            async def send_text() -> None:
                try:
                    async for text_chunk in text_stream:
                        if text_chunk:
                            await ws.send(json.dumps({"type": "text.delta", "delta": text_chunk}))
                    await ws.send(json.dumps({"type": "text.done"}))
                except websockets.exceptions.ConnectionClosed as exc:
                    logger.error("GrokTTS WebSocket closed while sending: %s", exc)
                except Exception:
                    logger.exception("Error in upstream text stream (not GrokTTS)")
                    await ws.close()

            sender_task = asyncio.create_task(send_text())

            try:
                async for message in ws:
                    if isinstance(message, str):
                        event = json.loads(message)
                        event_type = event.get("type", "")

                        if event_type == "audio.delta":
                            audio_bytes = base64.b64decode(event["delta"])
                            yield AudioChunk(
                                data=audio_bytes,
                                sample_rate=self._config.sample_rate,
                                format=fmt,
                                is_final=False,
                            )
                        elif event_type == "audio.done":
                            yield AudioChunk(
                                data=b"",
                                sample_rate=self._config.sample_rate,
                                format=fmt,
                                is_final=True,
                            )
                            break
            finally:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
