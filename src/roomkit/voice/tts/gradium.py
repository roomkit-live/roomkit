"""Gradium text-to-speech provider."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class GradiumTTSConfig:
    """Configuration for Gradium TTS provider."""

    api_key: str
    voice_id: str = "default"
    region: str = "us"
    model_name: str = "default"
    output_format: str = "pcm_16000"  # matches pipeline's 16kHz default
    json_config: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class GradiumVoice:
    """Voice metadata from Gradium."""

    uid: str
    name: str


class GradiumTTSProvider(TTSProvider):
    """Gradium text-to-speech provider with streaming support."""

    def __init__(self, config: GradiumTTSConfig) -> None:
        self._config = config
        self._client: Any = None
        self._voices_cache: list[GradiumVoice] | None = None

    @property
    def name(self) -> str:
        return "GradiumTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

    def _get_client(self) -> Any:
        if self._client is None:
            from gradium import GradiumClient

            self._client = GradiumClient(
                base_url=f"https://{self._config.region}.api.gradium.ai/api/",
                api_key=self._config.api_key,
            )
        return self._client

    def _get_sample_rate(self) -> int:
        """Get sample rate from output format."""
        fmt = self._config.output_format
        if "48000" in fmt:
            return 48000
        elif "24000" in fmt:
            return 24000
        elif "16000" in fmt:
            return 16000
        elif "8000" in fmt:
            return 8000
        # wav/pcm/opus without explicit rate default to 24kHz
        return 24000

    def _get_audio_format(self) -> str:
        """Get audio format string."""
        fmt = self._config.output_format
        if fmt.startswith("pcm") or fmt == "wav":
            return "pcm_s16le"
        elif fmt.startswith("opus"):
            return "opus"
        elif fmt.startswith("ulaw"):
            return "ulaw"
        elif fmt.startswith("alaw"):
            return "alaw"
        return "pcm_s16le"

    def _get_mime_type(self) -> str:
        """Get MIME type from output format."""
        fmt = self._config.output_format
        if fmt == "wav":
            return "audio/wav"
        elif fmt.startswith("pcm"):
            return "audio/pcm"
        elif fmt.startswith("opus"):
            return "audio/opus"
        elif fmt.startswith(("ulaw", "alaw")):
            return "audio/basic"
        return "audio/pcm"

    def _build_setup(self, voice: str | None = None) -> dict[str, Any]:
        """Build the TTSSetup dict for the SDK."""
        voice_value = voice or self._config.voice_id
        setup: dict[str, Any] = {
            "model_name": self._config.model_name,
            "output_format": self._config.output_format,
        }
        # SDK distinguishes voice (profile name) from voice_id (UID).
        # Use voice_id only when the value looks like a UID (not a name).
        if voice_value and voice_value != "default":
            setup["voice_id"] = voice_value
        else:
            setup["voice"] = voice_value
        if self._config.json_config is not None:
            setup["json_config"] = self._config.json_config
        return setup

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio using the Gradium SDK."""
        from roomkit.models.event import AudioContent as AudioContentModel

        client = self._get_client()
        result = await client.tts(self._build_setup(voice), text)

        mime_type = self._get_mime_type()
        data_url = f"data:{mime_type};base64,{base64.b64encode(result.raw_data).decode()}"

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
        """Stream audio chunks as they're generated."""
        client = self._get_client()
        stream = await client.tts_stream(self._build_setup(voice), text)
        sample_rate = stream.sample_rate or self._get_sample_rate()

        async for chunk in stream.iter_bytes():
            if chunk:
                yield AudioChunk(
                    data=chunk,
                    sample_rate=sample_rate,
                    format=self._get_audio_format(),
                    is_final=False,
                )

        yield AudioChunk(
            data=b"",
            sample_rate=sample_rate,
            format=self._get_audio_format(),
            is_final=True,
        )

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio from streaming text input.

        The Gradium SDK accepts an AsyncGenerator[str] for text input directly.
        """
        client = self._get_client()
        stream = await client.tts_stream(self._build_setup(voice), text_stream)
        sample_rate = stream.sample_rate or self._get_sample_rate()

        async for chunk in stream.iter_bytes():
            if chunk:
                yield AudioChunk(
                    data=chunk,
                    sample_rate=sample_rate,
                    format=self._get_audio_format(),
                    is_final=False,
                )

        yield AudioChunk(
            data=b"",
            sample_rate=sample_rate,
            format=self._get_audio_format(),
            is_final=True,
        )

    async def list_voices(self) -> list[GradiumVoice]:
        """List available voices from Gradium."""
        if self._voices_cache is not None:
            return list(self._voices_cache)

        client = self._get_client()
        voices = await client.voice_get(include_catalog=True)

        # voice_get returns a dict or list depending on the API response
        voice_list = voices if isinstance(voices, list) else voices.get("voices", [])
        self._voices_cache = [
            GradiumVoice(
                uid=v["uid"],
                name=v.get("name", v["uid"]),
            )
            for v in voice_list
        ]
        return list(self._voices_cache)

    async def close(self) -> None:
        """Release resources."""
        self._client = None
