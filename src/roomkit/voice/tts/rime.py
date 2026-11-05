"""Rime Coda and Mist v3 TTS over WebSocket v1 (``roomkit[rime]``).

Each synthesis owns its connection and context, so cancelling one reader
cancels only its synthesis. Streams carry mono PCM16; complete synthesis
returns a WAV data URL. No vendor SDK is required.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from roomkit.core.task_utils import cancel_and_wait
from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import collect_wav_content, whole_samples
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import VoiceInfo, filter_voices

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

_HOSTS = {"coda": "coda", "mistv3": "mist"}
_LANGUAGES = {
    "eng": "en",
    "fra": "fr",
    "ger": "de",
    "spa": "es",
    "por": "pt",
    "ita": "it",
    "jpn": "ja",
    "ara": "ar",
    "hin": "hi",
    "nld": "nl",
}
_VOICES_URL = "https://users.rime.ai/data/voices/voice_details.json"


class RimeTTSError(ProviderError):
    """A failed or incomplete Rime synthesis, with the service's request ID."""

    def __init__(
        self, message: str, *, kind: str = "protocol", request_id: str | None = None
    ) -> None:
        super().__init__(message, retryable=kind in {"unavailable", "timeout", "internal"})
        self.kind = kind
        self.request_id = request_id


@dataclass
class RimeTTSConfig:
    """TTS settings. A voice must support the selected model and language.

    ``time_scale_factor`` below one speeds speech up; above one slows it down.
    ``timeout_s`` bounds a whole synthesis, including streamed text input.
    ``websocket_url`` optionally points at a proxy or private Rime deployment.
    """

    api_key: str = field(repr=False)
    model: Literal["coda", "mistv3"] = "coda"
    voice_id: str = "astra"
    language: str = "en"
    sample_rate: int = 16000
    time_scale_factor: float = 1.0
    timeout_s: float = 60.0
    connect_timeout_s: float = 10.0
    websocket_url: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.api_key
            or not self.api_key.isascii()
            or any(c.isspace() for c in self.api_key)
        ):
            raise ValueError("RimeTTSConfig.api_key must be a nonempty ASCII token")
        if self.model not in _HOSTS:
            raise ValueError("Rime TTS model must be coda or mistv3")
        if not self.voice_id.strip() or not self.language.strip():
            raise ValueError("Rime TTS voice_id and language are required")
        if self.sample_rate not in (8000, 16000, 24000, 48000):
            raise ValueError("Rime TTS sample_rate must be 8000, 16000, 24000 or 48000")
        if not 0.4 <= self.time_scale_factor <= 2.5:
            raise ValueError("time_scale_factor must be between 0.4 and 2.5")
        if self.timeout_s <= 0 or self.connect_timeout_s <= 0:
            raise ValueError("Rime TTS timeouts must be positive")


class RimeTTSProvider(TTSProvider):
    """Streaming TTS on Coda or Mist v3, with independent synthesis contexts."""

    def __init__(self, config: RimeTTSConfig) -> None:
        self._config = config
        self._connections: set[Any] = set()

    @property
    def name(self) -> str:
        return "RimeTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice_id

    @property
    def supports_streaming_input(self) -> bool:
        return True

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """A starter voice available in both models; use list_voices for the catalog."""
        return [VoiceInfo(id="astra", name="Astra", language="en", gender="female")]

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """Query Rime's public catalog, restricted to the configured model."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("Install Rime dependencies with: pip install roomkit[rime]") from exc
        async with httpx.AsyncClient(timeout=self._config.connect_timeout_s) as client:
            response = await client.get(_VOICES_URL)
            response.raise_for_status()
            entries = response.json()
        voices = [
            VoiceInfo(
                id=entry["speaker"],
                name=entry["speaker"],
                language=_LANGUAGES.get(entry.get("lang"), entry.get("lang")),
                gender=entry.get("gender", "").lower() or None,
                accent=entry.get("dialect"),
                description=", ".join(entry.get("genre", [])) or None,
                attributes={"modelId": entry["modelId"]},
            )
            for entry in entries
            if entry.get("modelId") == self._config.model
        ]
        return filter_voices(voices, language=language, gender=gender, query=query)

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Collect a synthesis into a playable mono PCM16 WAV."""
        async with contextlib.aclosing(self.synthesize_stream(text, voice=voice)) as chunks:
            return await collect_wav_content(
                chunks, text=text, sample_rate=self._config.sample_rate
            )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream a complete text as PCM16. Rime does not consume dialogue context."""
        if not text.strip():
            return
        async with contextlib.aclosing(self._synthesize(text, None, voice)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Send text fragments unchanged while reading audio concurrently."""
        async with contextlib.aclosing(self._synthesize(None, text_stream, voice)) as chunks:
            async for chunk in chunks:
                yield chunk

    def _start(self, text: str | None, voice: str | None) -> dict[str, Any]:
        config = self._config
        start: dict[str, Any] = {
            "speaker": voice or config.voice_id,
            "language": config.language,
            "audioParameters": {
                "audioFormat": "audio/pcm",
                "samplingRate": config.sample_rate,
                "timeScaleFactor": config.time_scale_factor,
            },
        }
        if text is not None:
            start["text"] = text
        else:
            start["splitStrategy"] = "SPLIT_STRATEGY_ACCUMULATE_SENTENCE"
        return start

    async def _synthesize(
        self,
        text: str | None,
        source: AsyncIterator[str] | None,
        voice: str | None,
    ) -> AsyncGenerator[AudioChunk, None]:
        try:
            from websockets.asyncio.client import connect
            from websockets.typing import Subprotocol
        except ImportError as exc:
            raise ImportError("Install Rime dependencies with: pip install roomkit[rime]") from exc
        config = self._config
        url = config.websocket_url or f"wss://{_HOSTS[config.model]}.api.rime.ai/ws"
        async with (
            asyncio.timeout(config.timeout_s),
            connect(
                url,
                additional_headers={"Authorization": f"Bearer {config.api_key}"},
                subprotocols=[Subprotocol("rime.v1.json")],
                open_timeout=config.connect_timeout_s,
                close_timeout=2,
            ) as ws,
        ):
            self._connections.add(ws)
            try:
                ready = _event(await asyncio.wait_for(ws.recv(), config.connect_timeout_s))
                if ready.get("ready", {}).get("protocol") != 1:
                    raise RimeTTSError("Rime did not announce WebSocket protocol 1")
                context_id = uuid4().hex
                await ws.send(
                    json.dumps({"contextId": context_id, "start": self._start(text, voice)})
                )
                sender = asyncio.create_task(_send_text(ws, context_id, source))
                try:
                    async with contextlib.aclosing(self._audio(ws, context_id, sender)) as chunks:
                        async for chunk in chunks:
                            yield chunk
                finally:
                    await cancel_and_wait(sender)
            finally:
                self._connections.discard(ws)

    async def _audio(
        self,
        ws: Any,
        context_id: str,
        sender: asyncio.Task[None],
    ) -> AsyncGenerator[AudioChunk, None]:
        pending = b""
        request_id = None
        while True:
            event = _event(await _receive(ws, sender))
            if event.get("contextId", context_id) != context_id:
                continue
            if "started" in event:
                request_id = event["started"].get("requestId")
            if "audio" in event:
                data, pending = whole_samples(
                    pending + base64.b64decode(event["audio"], validate=True)
                )
                if data:
                    yield AudioChunk(data=data, sample_rate=self._config.sample_rate)
            if "done" in event:
                if pending:
                    raise RimeTTSError(
                        "Rime ended with an incomplete PCM sample", request_id=request_id
                    )
                yield AudioChunk(data=b"", sample_rate=self._config.sample_rate, is_final=True)
                return
            if "cancelled" in event:
                raise RimeTTSError(
                    "Rime synthesis was cancelled", kind="cancelled", request_id=request_id
                )

    async def close(self) -> None:
        """Cancel every active synthesis and release its WebSocket."""
        await asyncio.gather(*(ws.close() for ws in tuple(self._connections)))


def _event(raw: Any) -> dict[str, Any]:
    event = json.loads(raw)
    if "error" in event:
        error = event["error"]
        raise RimeTTSError(
            error.get("message", "Rime synthesis failed"),
            kind=error.get("kind", "internal"),
            request_id=error.get("requestId"),
        )
    return event


async def _send_text(ws: Any, context_id: str, source: AsyncIterator[str] | None) -> None:
    if source is None:
        return
    async for text in source:
        if text:
            await ws.send(json.dumps({"contextId": context_id, "text": text}))
    await ws.send(json.dumps({"contextId": context_id, "end": {}}))


async def _receive(ws: Any, sender: asyncio.Task[None]) -> Any:
    """Do not leave a reader waiting after the text source failed."""
    receiving = asyncio.create_task(ws.recv())
    try:
        if not sender.done():
            await asyncio.wait({receiving, sender}, return_when=asyncio.FIRST_COMPLETED)
        if sender.done():
            sender.result()
        return await receiving
    finally:
        await cancel_and_wait(receiving)
