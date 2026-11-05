"""Rime streaming STT: one explicitly ended utterance per WebSocket.

Use a pipeline VAD to delimit utterances in VoiceChannel. Rime does not
endpoint speech, identify speakers, or detect the language. Both complete
and streaming transcription use the same WebSocket protocol.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import wave
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from roomkit.core.task_utils import cancel_and_wait
from roomkit.models.event import AudioContent
from roomkit.providers.ai.base import ProviderError
from roomkit.providers.utils import parse_data_uri
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult, require_pcm16
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.stt.base import STTProvider

_RATE = 16000
_MAX_AUDIO_BYTES = 16 * 1024 * 1024


class RimeSTTError(ProviderError):
    """A refused or incomplete Rime transcription."""

    def __init__(
        self, message: str, *, kind: str = "protocol", request_id: str | None = None
    ) -> None:
        super().__init__(message, retryable=kind in {"unavailable", "timeout", "internal"})
        self.kind = kind
        self.request_id = request_id


@dataclass
class RimeSTTConfig:
    """Rime recognition settings, with a 120-second maximum stream lifetime.

    ``language`` is explicit, not auto-detected. ``context_terms`` hints at
    names and phrases; ``verbatim`` preserves spoken wording and fillers.
    """

    api_key: str = field(repr=False)
    language: str = "en"
    mode: Literal["written", "verbatim"] = "written"
    context_terms: list[str] = field(default_factory=list)
    websocket_url: str = "wss://stt.api.rime.ai/ws"
    timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    final_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if (
            not self.api_key
            or not self.api_key.isascii()
            or any(c.isspace() for c in self.api_key)
        ):
            raise ValueError("RimeSTTConfig.api_key must be a nonempty ASCII token")
        _language(self.language)
        if self.mode not in ("written", "verbatim"):
            raise ValueError("Rime STT mode must be written or verbatim")
        if any(not term.strip() for term in self.context_terms):
            raise ValueError("context_terms must contain nonempty phrases")
        if len(", ".join(self.context_terms).encode()) > 1024:
            raise ValueError("context_terms joined with ', ' must fit in 1024 UTF-8 bytes")
        if not 0 < self.timeout_s <= 120 or min(self.connect_timeout_s, self.final_timeout_s) <= 0:
            raise ValueError("Rime STT timeouts must be positive; timeout_s cannot exceed 120")


class RimeSTTProvider(STTProvider):
    """English/Spanish STT with revised partials and one final per utterance."""

    def __init__(self, config: RimeSTTConfig) -> None:
        self._config = config
        self._connections: set[Any] = set()

    @property
    def name(self) -> str:
        return "RimeSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe PCM or a WAV data URL. Remote URLs are not fetched."""
        frame = _frame(audio)

        async def chunks() -> AsyncIterator[AudioChunk]:
            pcm = await asyncio.to_thread(_pcm, frame)
            for offset in range(0, len(pcm), 3200):
                yield AudioChunk(data=pcm[offset : offset + 3200], sample_rate=_RATE)

        async with contextlib.aclosing(
            self.transcribe_stream(chunks(), language=language)
        ) as stream:
            async for result in stream:
                if result.is_final:
                    return result
        raise RimeSTTError("Rime ended without a final transcript")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Replace partials with each hypothesis; close audio on source exhaustion.

        PCM is resampled to 16 kHz mono. An ``is_final`` chunk also ends the
        input. A lost connection never promotes a partial to a final.
        """
        selected = _language(self._config.language if language is None else language)
        try:
            from websockets.asyncio.client import connect
            from websockets.typing import Subprotocol
        except ImportError as exc:
            raise ImportError("Install Rime dependencies with: pip install roomkit[rime]") from exc
        config = self._config
        async with (
            asyncio.timeout(config.timeout_s),
            connect(
                config.websocket_url,
                additional_headers={"Authorization": f"Bearer {config.api_key}"},
                subprotocols=[Subprotocol("rime.v1.json")],
                open_timeout=config.connect_timeout_s,
                close_timeout=2,
            ) as ws,
        ):
            self._connections.add(ws)
            try:
                await ws.send(
                    json.dumps(
                        {
                            "start": {
                                "outputContract": "STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES",
                                "language": selected,
                                "mode": f"TRANSCRIPTION_MODE_{config.mode.upper()}",
                                "contextTerms": config.context_terms,
                            }
                        }
                    )
                )
                event = _event(await asyncio.wait_for(ws.recv(), config.connect_timeout_s))
                if "started" not in event:
                    raise RimeSTTError("Rime did not acknowledge the transcription")
                sender = asyncio.create_task(_send_audio(ws, audio_stream))
                try:
                    async with contextlib.aclosing(self._results(ws, sender)) as results:
                        async for result in results:
                            yield result
                finally:
                    await cancel_and_wait(sender)
            finally:
                self._connections.discard(ws)

    async def _results(
        self,
        ws: Any,
        sender: asyncio.Task[None],
    ) -> AsyncGenerator[TranscriptionResult, None]:
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        while True:
            # The final wait is a total budget after end, not one per partial.
            raw, deadline = await _receive(ws, sender, deadline, self._config.final_timeout_s)
            event = _event(raw)
            if "hypothesis" in event:
                yield TranscriptionResult(text=event["hypothesis"].get("text", ""), is_final=False)
            if "done" in event:
                final = event["done"]
                yield TranscriptionResult(
                    text=final.get("text", ""),
                    is_final=True,
                    language=final.get("language", {}).get("tag"),
                )
                return
            if "cancelled" in event:
                raise RimeSTTError("Rime transcription was cancelled", kind="cancelled")
            if sender.done() and deadline is None:
                deadline = loop.time() + self._config.final_timeout_s

    async def close(self) -> None:
        """Cancel active utterances by closing their connections."""
        await asyncio.gather(*(ws.close() for ws in tuple(self._connections)))


def _language(value: str) -> str:
    if value not in {"en", "es"}:
        raise ValueError("Rime STT language must be en or es; automatic detection is unavailable")
    return value


def _frame(audio: AudioContent | AudioChunk | AudioFrame) -> AudioFrame:
    if isinstance(audio, AudioFrame):
        return audio
    if isinstance(audio, AudioChunk):
        require_pcm16(audio, "Rime STT")
        return AudioFrame(data=audio.data, sample_rate=audio.sample_rate, channels=audio.channels)
    if not audio.url.startswith("data:"):
        raise ValueError("Rime STT accepts PCM or WAV data URLs; fetch remote audio separately")
    mime, data = parse_data_uri(audio.url)
    if mime not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise ValueError("Rime STT AudioContent must be a WAV data URL")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return AudioFrame(
                data=wav.readframes(wav.getnframes()),
                sample_rate=wav.getframerate(),
                channels=wav.getnchannels(),
                sample_width=wav.getsampwidth(),
            )
    except (wave.Error, EOFError) as exc:
        raise ValueError("Rime STT could not decode the WAV") from exc


def _pcm(frame: AudioFrame) -> bytes:
    if frame.sample_width != 2:
        raise ValueError("Rime STT expects 16-bit PCM audio")
    return LinearResamplerProvider().resample(frame, _RATE, 1, 2, stream="rime").data


async def _send_audio(ws: Any, source: AsyncIterator[AudioChunk]) -> None:
    total = 0
    async for chunk in source:
        data = await asyncio.to_thread(_pcm, _frame(chunk))
        total += len(data)
        if total > _MAX_AUDIO_BYTES:
            raise ValueError("Rime STT utterance exceeds 16 MiB of PCM")
        for offset in range(0, len(data), 3200):
            await ws.send(
                json.dumps({"audio": base64.b64encode(data[offset : offset + 3200]).decode()})
            )
        if chunk.is_final:
            break
    await ws.send(json.dumps({"end": {}}))


def _event(raw: Any) -> dict[str, Any]:
    event = json.loads(raw)
    if "error" in event:
        error = event["error"]
        raise RimeSTTError(
            error.get("message", "Rime transcription failed"),
            kind=error.get("kind", "internal"),
            request_id=error.get("requestId"),
        )
    return event


async def _receive(
    ws: Any,
    sender: asyncio.Task[None],
    deadline: float | None,
    final_timeout: float,
) -> tuple[Any, float | None]:
    receiving = asyncio.create_task(ws.recv())
    try:
        if not sender.done():
            await asyncio.wait({receiving, sender}, return_when=asyncio.FIRST_COMPLETED)
        if sender.done():
            sender.result()
            if deadline is None:
                deadline = asyncio.get_running_loop().time() + final_timeout
        async with asyncio.timeout_at(deadline):
            return await receiving, deadline
    finally:
        await cancel_and_wait(receiving)
