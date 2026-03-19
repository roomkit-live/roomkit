"""ElevenLabs text-to-speech provider.

Supports expressive mode via the ``eleven_v3_conversational`` model.
When ``expressive=True``, synthesis uses v3 Conversational TTS which
understands expressive tags such as ``[laughs]``, ``[whispers]``,
``[sighs]``, ``[slow]``, and ``[excited]`` embedded in the text.

.. note::

    Do **not** combine expressive mode with :class:`StripBrackets` — that
    filter removes all ``[...]`` content, including the expressive tags.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)

# Model ID constants
MODEL_MULTILINGUAL_V2 = "eleven_multilingual_v2"
MODEL_TURBO_V2_5 = "eleven_turbo_v2_5"
MODEL_FLASH_V2_5 = "eleven_flash_v2_5"
MODEL_V3 = "eleven_v3_conversational"

# Expressive tags recognised by v3 Conversational TTS.
EXPRESSIVE_TAGS = frozenset({"[laughs]", "[whispers]", "[sighs]", "[slow]", "[excited]"})


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs TTS provider.

    Set ``expressive=True`` to enable expressive mode (v3 Conversational
    model).  This overrides ``model_id`` and disables voice settings that
    are not supported by v3 (``style``, ``use_speaker_boost``).
    """

    api_key: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    model_id: str = MODEL_MULTILINGUAL_V2
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    output_format: str = "mp3_44100_128"  # mp3, pcm_16000, pcm_22050, etc.
    # Streaming options
    optimize_streaming_latency: int = 3  # 0-4, higher = lower latency
    # Expressive mode — uses v3 Conversational TTS with emotion/tone tags
    expressive: bool = False


@dataclass
class ElevenLabsVoice:
    """Voice metadata from ElevenLabs."""

    voice_id: str
    name: str
    category: str = "premade"
    labels: dict[str, str] = field(default_factory=dict)


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs text-to-speech provider with streaming support.

    When *expressive mode* is enabled (``config.expressive=True``), the
    provider uses the ``eleven_v3_conversational`` model which supports
    inline expressive tags (``[laughs]``, ``[whispers]``, etc.) and adapts
    tone and timing based on conversational context.
    """

    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        if config.expressive:
            self._config.model_id = MODEL_V3
        self._client: httpx.AsyncClient | None = None
        self._voices_cache: dict[str, ElevenLabsVoice] | None = None

    @property
    def name(self) -> str:
        return "ElevenLabsTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

    @property
    def supports_streaming_input(self) -> bool:
        return True

    def _is_v3_model(self) -> bool:
        """Return True when the selected model is a v3 variant."""
        return "v3" in self._config.model_id

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.elevenlabs.io/v1",
                headers={
                    "xi-api-key": self._config.api_key,
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def _build_voice_settings(self) -> dict[str, float | bool]:
        """Build voice settings for synthesis.

        v3 Conversational only supports ``stability`` and
        ``similarity_boost``; ``style`` and ``use_speaker_boost`` are
        omitted when a v3 model is active.
        """
        settings: dict[str, float | bool] = {
            "stability": self._config.stability,
            "similarity_boost": self._config.similarity_boost,
        }
        if not self._is_v3_model():
            settings["style"] = self._config.style
            settings["use_speaker_boost"] = self._config.use_speaker_boost
        return settings

    async def list_voices(self) -> list[ElevenLabsVoice]:
        """List available voices from ElevenLabs."""
        if self._voices_cache is not None:
            return list(self._voices_cache.values())

        client = self._get_client()
        response = await client.get("/voices")
        response.raise_for_status()
        data = response.json()

        self._voices_cache = {}
        for voice in data.get("voices", []):
            v = ElevenLabsVoice(
                voice_id=voice["voice_id"],
                name=voice["name"],
                category=voice.get("category", "premade"),
                labels=voice.get("labels", {}),
            )
            self._voices_cache[v.voice_id] = v

        return list(self._voices_cache.values())

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).

        Returns:
            AudioContent with URL to generated audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        voice_id = voice or self._config.voice_id
        client = self._get_client()

        t0 = time.monotonic()
        response = await client.post(
            f"/text-to-speech/{voice_id}",
            json={
                "text": text,
                "model_id": self._config.model_id,
                "voice_settings": self._build_voice_settings(),
            },
            params={"output_format": self._config.output_format},
        )
        response.raise_for_status()

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.tts.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "elevenlabs", "model": self._config.model_id},
        )

        # ElevenLabs returns raw audio bytes
        # We need to save/upload this somewhere to get a URL
        # For now, return a data URL (base64 encoded)
        import base64

        audio_bytes = response.content
        mime_type = self._get_mime_type()
        data_url = f"data:{mime_type};base64,{base64.b64encode(audio_bytes).decode()}"

        # Estimate duration (rough: ~150 words/minute, ~5 chars/word)
        words = len(text.split())
        duration = words / 150 * 60  # seconds

        return AudioContentModel(
            url=data_url,
            mime_type=mime_type,
            transcript=text,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks as they're generated.

        Uses ElevenLabs streaming API for low-latency synthesis.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).

        Yields:
            AudioChunk with raw audio data.
        """
        voice_id = voice or self._config.voice_id
        client = self._get_client()

        # Use streaming endpoint
        params: dict[str, str | int] = {
            "output_format": self._config.output_format,
        }
        # optimize_streaming_latency is not supported by v3 models
        if not self._is_v3_model():
            params["optimize_streaming_latency"] = self._config.optimize_streaming_latency

        async with client.stream(
            "POST",
            f"/text-to-speech/{voice_id}/stream",
            json={
                "text": text,
                "model_id": self._config.model_id,
                "voice_settings": self._build_voice_settings(),
            },
            params=params,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                logger.error("ElevenLabs stream error %d: %s", response.status_code, body.decode())
                response.raise_for_status()

            chunk_index = 0
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield AudioChunk(
                        data=chunk,
                        sample_rate=self._get_sample_rate(),
                        format=self._get_audio_format(),
                        is_final=False,
                    )
                    chunk_index += 1

            # Send final chunk marker
            yield AudioChunk(
                data=b"",
                sample_rate=self._get_sample_rate(),
                format=self._get_audio_format(),
                is_final=True,
            )

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio from streaming text input.

        Uses ElevenLabs WebSocket API for real-time text-to-speech.

        Args:
            text_stream: Async iterator of text chunks.
            voice: Voice ID (uses default_voice if not specified).

        Yields:
            AudioChunk with raw audio data.
        """
        import asyncio
        import json

        import websockets

        voice_id = voice or self._config.voice_id
        model_id = self._config.model_id

        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            f"?model_id={model_id}"
            f"&output_format={self._config.output_format}"
        )
        if not self._is_v3_model():
            ws_url += f"&optimize_streaming_latency={self._config.optimize_streaming_latency}"

        async with websockets.connect(
            ws_url,
            additional_headers=[("xi-api-key", self._config.api_key)],
            open_timeout=30,
        ) as ws:
            # Send initial BOS (beginning of stream) message
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "voice_settings": self._build_voice_settings(),
                    }
                )
            )

            # Start text sender task
            async def send_text() -> None:
                try:
                    async for text_chunk in text_stream:
                        if text_chunk:
                            await ws.send(json.dumps({"text": text_chunk}))
                    # Send EOS (end of stream) message
                    await ws.send(json.dumps({"text": ""}))
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error("ElevenLabs WebSocket closed while sending: %s", e)
                except Exception as e:
                    logger.error(
                        "Error in upstream text stream (not ElevenLabs): %s: %s",
                        type(e).__name__,
                        e,
                    )
                    await ws.close()

            sender_task = asyncio.create_task(send_text())

            try:
                async for message in ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get("audio"):
                            import base64

                            audio_bytes = base64.b64decode(data["audio"])
                            yield AudioChunk(
                                data=audio_bytes,
                                sample_rate=self._get_sample_rate(),
                                format=self._get_audio_format(),
                                is_final=data.get("isFinal", False),
                            )
                        elif data.get("isFinal"):
                            yield AudioChunk(
                                data=b"",
                                sample_rate=self._get_sample_rate(),
                                format=self._get_audio_format(),
                                is_final=True,
                            )
                            break
            finally:
                sender_task.cancel()
                import contextlib

                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task

    def _get_mime_type(self) -> str:
        """Get MIME type from output format."""
        fmt = self._config.output_format
        if fmt.startswith("mp3"):
            return "audio/mpeg"
        elif fmt.startswith("pcm"):
            return "audio/pcm"
        elif fmt.startswith("ulaw"):
            return "audio/basic"
        return "audio/mpeg"

    def _get_sample_rate(self) -> int:
        """Get sample rate from output format."""
        fmt = self._config.output_format
        if "44100" in fmt:
            return 44100
        elif "24000" in fmt:
            return 24000
        elif "22050" in fmt:
            return 22050
        elif "16000" in fmt:
            return 16000
        return 44100

    def _get_audio_format(self) -> str:
        """Get audio format string."""
        fmt = self._config.output_format
        if fmt.startswith("mp3"):
            return "mp3"
        elif fmt.startswith("pcm"):
            return "pcm_s16le"
        elif fmt.startswith("ulaw"):
            return "ulaw"
        return "mp3"

    async def close(self) -> None:  # noqa: B027
        """Release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
