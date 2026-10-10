"""Fluxions text-to-speech provider: Vui hosted by fluxions.ai.

`fluxions.ai <https://fluxions.ai>`_ serves the Vui model behind an API key, so
it needs no GPU. Its speech endpoint renders one text per request: it takes no
previous turns, no user audio and no session, so this provider declares no
conversation context (:attr:`TTSContextLevel.NONE`). For replies generated
inside the dialogue, run Vui locally with :class:`~roomkit.voice.tts.vui.VuiTTSProvider`.

Audio is 24 kHz mono 16-bit PCM, streamed as it renders. Fluxions keeps each
render in the account's history. Install with ``pip install roomkit[fluxions]``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from roomkit.providers.utils import http_timeout
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import VoiceInfo, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.fluxions")

SAMPLE_RATE = 24000


@dataclass
class FluxionsTTSConfig:
    """Configuration for :class:`FluxionsTTSProvider`.

    Attributes:
        api_key: Fluxions API key.
        voice: A voice's short id (``maeve``) or its full id (``maeve.h…``,
            ``u-…`` for a cloned voice). A short id follows the model Fluxions
            currently serves.
        temperature: Sampling temperature; ``None`` keeps the server's default.
        max_secs: Longest render, in seconds; ``None`` lets the server size it
            from the text.
        verify_chunks: Let the server check each rendered chunk, and render it
            again when it fails (slower first audio).
        base_url: API root.
        timeout: Read budget, in seconds: a cold start takes up to about 30 s.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    voice: str = "maeve"
    temperature: float | None = None
    max_secs: float | None = None
    verify_chunks: bool = False
    base_url: str = "https://api.fluxions.ai"
    timeout: float = 60.0
    connect_timeout: float = 5.0


class FluxionsTTSProvider(TTSProvider):
    """Vui hosted by fluxions.ai: each text rendered on its own, streamed."""

    def __init__(self, config: FluxionsTTSConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._catalog: list[dict[str, Any]] | None = None

    @property
    def name(self) -> str:
        return "FluxionsTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers={"Authorization": self._config.api_key},
                timeout=http_timeout(self._config),
            )
        return self._client

    async def warmup(self) -> None:
        """Fetch the voice lists and resolve the configured voice.

        It renders nothing: after an idle period, the first render can still
        wait for Fluxions to start a worker (up to about 30 s).
        """
        if await self._entry(self._config.voice) is None:
            logger.warning(
                "Fluxions lists no voice %r: renders pass it as given", self._config.voice
            )

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """The voices Fluxions hosts, then the account's cloned voices.

        Each ``id`` is one ``voice`` accepts: a hosted voice's short id, a
        cloned voice's id.
        """
        voices = [
            VoiceInfo(
                id=v.get("id") or v["voice_id"],
                name=v.get("name") or v.get("id") or v["voice_id"],
                gender=v.get("gender"),
                accent=v.get("accent"),
                description=v.get("description"),
                attributes={"style": v["style"]} if v.get("style") else {},
            )
            for v in await self._voices()
            if not v.get("hidden")
        ]
        return filter_voices(voices, language=language, gender=gender, query=query)

    async def _voices(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """The hosted voices, then the account's cloned ones not already among them."""
        if self._catalog is None or refresh:
            client = self._get_client()
            hosted_response = await client.get("/vui/voices")
            hosted_response.raise_for_status()
            mine_response = await client.get("/vui/v1/voices/mine")
            mine_response.raise_for_status()
            hosted = hosted_response.json()["voices"]
            listed = {v["voice_id"] for v in hosted}
            mine = [v for v in mine_response.json()["voices"] if v["voice_id"] not in listed]
            self._catalog = hosted + mine
        return self._catalog

    async def _entry(self, voice: str, *, refresh: bool = False) -> dict[str, Any] | None:
        for entry in await self._voices(refresh=refresh):
            if voice in (entry.get("id"), entry["voice_id"]):
                return entry
        return None

    async def _voice_id(self, voice: str, *, refresh: bool = False) -> str:
        """The id a render takes for *voice*: its short id resolves to the current model's."""
        entry = await self._entry(voice, refresh=refresh)
        return str(entry["voice_id"]) if entry else voice  # an unlisted id is passed as given

    def _body(self, text: str, voice_id: str) -> dict[str, object]:
        body: dict[str, object] = {"voice": voice_id, "input": text, "response_format": "pcm"}
        if self._config.temperature is not None:
            body["temperature"] = self._config.temperature
        if self._config.max_secs is not None:
            body["max_secs"] = self._config.max_secs
        if self._config.verify_chunks:
            body["verify_chunks"] = True
        return body

    @contextlib.asynccontextmanager
    async def _render(self, text: str, voice: str | None) -> AsyncGenerator[httpx.Response, None]:
        """Open the streamed render of *text*, listing the voices again once on a 404.

        A voice's full id carries the model's checkpoint, which changes with
        each model Fluxions ships: a 404 on a cached id lists the voices again.
        """
        name = voice or self._config.voice
        for refresh in (False, True):
            voice_id = await self._voice_id(name, refresh=refresh)
            body = self._body(text, voice_id)
            client = self._get_client()
            async with client.stream("POST", "/vui/v1/tts?stream=1", json=body) as response:
                if response.status_code == 404 and not refresh:
                    continue
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")
                    logger.error("Fluxions render failed (%d): %s", response.status_code, detail)
                    response.raise_for_status()
                yield response
                return

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as 24 kHz PCM; *context* is ignored (no context level)."""
        carry = b""
        async with self._render(text, voice) as response:
            async for chunk in response.aiter_bytes():
                data, carry = _whole_samples(carry + chunk)
                if data:
                    yield AudioChunk(data=data, sample_rate=SAMPLE_RATE)
        yield AudioChunk(data=b"", sample_rate=SAMPLE_RATE, is_final=True)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Render *text* whole, as a WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _whole_samples(data: bytes) -> tuple[bytes, bytes]:
    """Split *data* into whole 16-bit samples and the odd byte left over, if any."""
    cut = len(data) - len(data) % 2
    return data[:cut], data[cut:]
