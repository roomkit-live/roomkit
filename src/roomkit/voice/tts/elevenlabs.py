"""ElevenLabs text-to-speech provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs TTS provider."""

    api_key: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    output_format: str = "mp3_44100_128"  # mp3, pcm_16000, pcm_22050, etc.
    # Streaming options
    optimize_streaming_latency: int = 3  # 0-4, higher = lower latency


@dataclass
class ElevenLabsVoice:
    """Voice metadata from ElevenLabs."""

    voice_id: str
    name: str
    category: str = "premade"
    labels: dict[str, str] = field(default_factory=dict)


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs text-to-speech provider with streaming support."""

    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._voices_cache: dict[str, ElevenLabsVoice] | None = None

    @property
    def name(self) -> str:
        return "ElevenLabsTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

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
        """Build voice settings for synthesis."""
        return {
            "stability": self._config.stability,
            "similarity_boost": self._config.similarity_boost,
            "style": self._config.style,
            "use_speaker_boost": self._config.use_speaker_boost,
        }

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

    async def synthesize(
        self, text: str, *, voice: str | None = None
    ) -> AudioContent:
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
        async with client.stream(
            "POST",
            f"/text-to-speech/{voice_id}/stream",
            json={
                "text": text,
                "model_id": self._config.model_id,
                "voice_settings": self._build_voice_settings(),
            },
            params={
                "output_format": self._config.output_format,
                "optimize_streaming_latency": self._config.optimize_streaming_latency,
            },
        ) as response:
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
            f"&optimize_streaming_latency={self._config.optimize_streaming_latency}"
        )

        async with websockets.connect(
            ws_url,
            additional_headers=[("xi-api-key", self._config.api_key)],
        ) as ws:
            # Send initial BOS (beginning of stream) message
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "voice_settings": self._build_voice_settings(),
                        "xi_api_key": self._config.api_key,
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
                except Exception as e:
                    logger.error("Error sending text to ElevenLabs: %s", e)

            sender_task = asyncio.create_task(send_text())

            try:
                async for message in ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        if "audio" in data:
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
