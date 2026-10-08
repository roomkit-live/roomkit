"""Microsoft MAI-Transcribe speech-to-text provider (Microsoft Foundry).

``MAI-Transcribe-2-Streaming`` runs on a Microsoft Foundry resource, behind a
WebSocket modelled on the OpenAI Realtime transcription protocol
(``wss://<resource>.services.ai.azure.com/mai/v1/realtime``): PCM in, a
transcript that firms up as it goes out.

The service detects no turns. It transcribes what it receives and finalises
the audio sent so far when the client commits it, so this provider needs a
pipeline VAD in front of it. Behind one, a
:class:`~roomkit.channels.VoiceChannel` opens a stream per utterance and the
provider commits when the utterance ends; the final transcript follows.
Without one (continuous mode), the stream never ends and no final comes.

Each stream is one session on the service, which caps a session at one hour.
The service reports no language, confidence or speaker, so none is set.

The model is in public preview (October 2026). Install with
``pip install roomkit[azure-speech]``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import cancel_and_wait
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.stt.azure_mai_protocol import (
    SUPPORTED_LANGUAGES as SUPPORTED_LANGUAGES,
)
from roomkit.voice.stt.azure_mai_protocol import AzureMAISTTError as AzureMAISTTError
from roomkit.voice.stt.azure_mai_protocol import (
    Transcript,
    event_error,
    language_code,
    realtime_url,
    stream_error,
    to_result,
)
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger("roomkit.voice.stt.azure_mai")

DEFAULT_DEPLOYMENT = "MAI-Transcribe-2-Streaming"

# The two input rates the service takes; anything else is resampled to 16 kHz.
_RATES = frozenset({16000, 24000})
_DEFAULT_RATE = 16000

_APPEND_S = 0.1
"""Longest audio one append carries. The service takes audio in any cut, and
small appends keep a backlog (the pre-roll) from leaving as one large frame."""


def _import_websockets() -> tuple[Any, Any]:
    """``(connect, exceptions)`` from ``websockets``, which every stream needs."""
    try:
        from websockets import exceptions
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise ImportError(
            "websockets is required for AzureMAISTTProvider. "
            "Install with: pip install roomkit[azure-speech]"
        ) from exc
    return connect, exceptions


@dataclass
class AzureMAISTTConfig:
    """Configuration for :class:`AzureMAISTTProvider`.

    Attributes:
        endpoint: The Foundry resource URL, as the portal shows it
            (``https://<resource>.services.ai.azure.com``).
        api_key: The resource's key (sent as the ``api-key`` header).
        deployment: The name of the model's deployment on the resource,
            which Foundry sets to the model name unless renamed.
        language: Language to transcribe, as a BCP-47 tag or a code from
            :data:`SUPPORTED_LANGUAGES` (``"fr-CA"`` is sent as ``"fr"``).
            ``None`` lets the service detect it, utterance by utterance.
        handshake_timeout_s: How long a stream waits for the service to open
            the session and accept its configuration.
        final_timeout_s: How long a stream waits, once its audio is
            committed, for the final transcript.
    """

    endpoint: str
    api_key: str = field(repr=False)
    deployment: str = DEFAULT_DEPLOYMENT
    language: str | None = None
    handshake_timeout_s: float = 10.0
    final_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        realtime_url(self.endpoint)
        if not self.api_key:
            raise ValueError("AzureMAISTTConfig.api_key is required")
        if not self.api_key.isascii():
            # Refused here, not as a UnicodeEncodeError from the handshake.
            raise ValueError(
                "AzureMAISTTConfig.api_key holds a non-ASCII character (a placeholder "
                "such as '…' copied as is?): an HTTP header cannot carry it"
            )
        if not self.deployment.strip():
            raise ValueError("AzureMAISTTConfig.deployment must not be blank")
        language_code(self.language)
        if self.handshake_timeout_s <= 0 or self.final_timeout_s <= 0:
            raise ValueError("AzureMAISTTConfig timeouts must be positive")


@dataclass
class _Commit:
    """Whether the wait for the final ran out after the commit."""

    timed_out: bool = False


class AzureMAISTTProvider(STTProvider):
    """Speech-to-text on Microsoft's MAI-Transcribe, one session per stream."""

    def __init__(self, config: AzureMAISTTConfig) -> None:
        self._config = config
        self._url = realtime_url(config.endpoint)
        self._resampler = LinearResamplerProvider()

    @property
    def name(self) -> str:
        return "AzureMAISTT"

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
        """Transcribe one finished clip: a stream of the whole clip, committed once.

        The service has no batch endpoint for this model, so the clip goes
        through the same session a stream opens.

        Raises:
            ValueError: ``audio`` is an :class:`~roomkit.models.event.AudioContent`;
                pass the PCM as an ``AudioChunk`` or an ``AudioFrame``.
            AzureMAISTTError: The service refused or failed the stream.
        """
        if isinstance(audio, AudioFrame):
            data = self._supported_pcm(
                audio.data, audio.sample_rate, audio.channels, _DEFAULT_RATE, audio.sample_width
            )
            chunk = AudioChunk(data=data, sample_rate=_DEFAULT_RATE)
        elif isinstance(audio, AudioChunk):
            chunk = audio
        else:
            raise ValueError(
                "AzureMAISTTProvider does not take AudioContent; "
                "pass the PCM as an AudioChunk or an AudioFrame"
            )
        final = TranscriptionResult(text="")
        async for result in self.transcribe_stream(_single(chunk), language=language):
            if result.is_final:
                final = result
        return final

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream audio to the service, commit it when the stream ends, yield the transcript.

        Yields partials while the audio arrives, then one final once the
        stream ends, unless the service heard no words. The input rate is
        read off the first chunk: 16 and 24 kHz go through as they are, any
        other rate is resampled to 16 kHz.

        Args:
            audio_stream: The utterance; it ends when exhausted or on a chunk
                marked ``is_final``.
            language: Language for this stream, overriding the configured one.

        Raises:
            ValueError: ``language`` is not one the service supports.
            AzureMAISTTError: The service refused or failed the stream;
                ``retryable`` says whether a new stream may work.
        """
        code = language_code(language if language is not None else self._config.language)
        first = await _first_audio(audio_stream)
        if first is None:
            return
        rate = first.sample_rate if first.sample_rate in _RATES else _DEFAULT_RATE
        if rate != first.sample_rate:
            logger.debug("MAI-Transcribe: resampling %d Hz to %d Hz", first.sample_rate, rate)
        ws = await self._open_session(rate, code)
        commit = _Commit()
        sender = asyncio.create_task(self._send_audio(ws, first, audio_stream, rate, commit))
        transcript = Transcript()
        try:
            async with contextlib.aclosing(_events(ws)) as events:
                async for event in events:
                    result = to_result(event, transcript)
                    if result is None:
                        continue
                    if not result.is_final:
                        yield result
                        continue
                    if result.text:
                        yield result
                    return
            raise _ended_without_final(commit, self._config.final_timeout_s)
        finally:
            await cancel_and_wait(sender)
            await ws.close()

    async def _open_session(self, rate: int, language: str | None) -> Any:
        """Connect, wait for the session, and have it accept the stream's settings.

        Settings cannot change once audio is sent, so they go first.
        """
        connect, _ = _import_websockets()
        try:
            ws = await connect(
                self._url,
                additional_headers={"api-key": self._config.api_key},
                open_timeout=self._config.handshake_timeout_s,
            )
        except Exception as exc:
            raise stream_error(exc) from exc
        try:
            await self._await_event(ws, "session.created")
            await ws.send(json.dumps(self._session_update(rate, language)))
            await self._await_event(ws, "session.updated")
        except BaseException:
            await ws.close()
            raise
        logger.debug("MAI-Transcribe session open (%d Hz, language %s)", rate, language or "auto")
        return ws

    def _session_update(self, rate: int, language: str | None) -> dict[str, Any]:
        """Transcription only, PCM at ``rate``, and no server turn detection.

        Turn detection must be off for the commit to decide where an
        utterance ends; the service takes nothing else for it. Detection is
        asked by leaving ``language`` out: the endpoint refuses an explicit
        ``null`` (measured 2026-10-08), whatever its documentation shows.
        """
        transcription = {"model": self._config.deployment}
        if language is not None:
            transcription["language"] = language
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": transcription,
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }

    async def _await_event(self, ws: Any, kind: str) -> None:
        """Read until the service sends ``kind``; an error event raises."""
        try:
            async with asyncio.timeout(self._config.handshake_timeout_s):
                while True:
                    event = json.loads(await ws.recv())
                    if event.get("type") == "error":
                        raise event_error(event)
                    if event.get("type") == kind:
                        return
        except AzureMAISTTError:
            raise
        except Exception as exc:
            raise stream_error(exc) from exc

    async def _send_audio(
        self,
        ws: Any,
        first: AudioChunk,
        audio_stream: AsyncIterator[AudioChunk],
        rate: int,
        commit: _Commit,
    ) -> None:
        """Append the audio, commit it once the stream ends, then bound the wait for the final.

        A failing audio source still commits, so the service answers with the
        transcript of what it heard instead of waiting for more. If no final
        comes within ``final_timeout_s``, the socket is closed, which ends the
        reader.
        """
        _, exceptions = _import_websockets()
        try:
            await self._append(ws, first, rate)
            if not first.is_final:
                async for chunk in audio_stream:
                    if chunk.data:
                        await self._append(ws, chunk, rate)
                    if chunk.is_final:
                        break
        except exceptions.ConnectionClosed:
            # The reader sees the close and reports it; nothing to add here.
            logger.debug("MAI-Transcribe stream closed while sending audio")
            return
        except Exception:
            logger.exception("MAI-Transcribe audio source failed; committing what was sent")
        try:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except exceptions.ConnectionClosed:
            return
        await asyncio.sleep(self._config.final_timeout_s)
        commit.timed_out = True
        await ws.close()

    async def _append(self, ws: Any, chunk: AudioChunk, rate: int) -> None:
        """Send ``chunk`` as base64 PCM, in appends of at most :data:`_APPEND_S`."""
        pcm = self._supported_pcm(chunk.data, chunk.sample_rate, chunk.channels, rate)
        step = int(rate * _APPEND_S) * 2
        for start in range(0, len(pcm), step):
            piece = base64.b64encode(pcm[start : start + step]).decode("ascii")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": piece}))

    def _supported_pcm(
        self, data: bytes, sample_rate: int, channels: int, rate: int, sample_width: int = 2
    ) -> bytes:
        """``data`` as mono 16-bit PCM at ``rate``, converted when it is not."""
        if sample_rate == rate and channels == 1 and sample_width == 2:
            return data
        frame = AudioFrame(
            data=data, sample_rate=sample_rate, channels=channels, sample_width=sample_width
        )
        return self._resampler.resample(frame, rate, 1, 2, stream="azure-mai-stt").data


def _ended_without_final(commit: _Commit, timeout_s: float) -> AzureMAISTTError:
    """Why a stream ended with no final: the wait ran out, or the service hung up."""
    if commit.timed_out:
        return AzureMAISTTError(
            f"MAI-Transcribe sent no final transcript within {timeout_s:g} s of the commit",
            retryable=True,
        )
    return AzureMAISTTError(
        "MAI-Transcribe closed the session before the final transcript", retryable=True
    )


async def _single(chunk: AudioChunk) -> AsyncIterator[AudioChunk]:
    yield chunk


async def _first_audio(audio_stream: AsyncIterator[AudioChunk]) -> AudioChunk | None:
    """The first chunk carrying audio, or ``None`` if the stream ends without any.

    The session is opened only once there is audio to send, and its rate is
    set from this chunk's.
    """
    async for chunk in audio_stream:
        if chunk.data:
            return chunk
        if chunk.is_final:
            return None
    return None


async def _events(ws: Any) -> AsyncGenerator[dict[str, Any], None]:
    """The service's JSON events until it closes; an abnormal close raises."""
    _, exceptions = _import_websockets()
    try:
        async for message in ws:
            if isinstance(message, str):
                yield json.loads(message)
    except exceptions.ConnectionClosedError as exc:
        raise stream_error(exc) from exc
