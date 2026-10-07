"""Meta Muse Voice Transcribe speech-to-text provider (Meta Model API).

``muse-voice-transcribe-1.0`` is served two ways, and this provider speaks both:

* :meth:`MetaSTTProvider.transcribe_stream` — the realtime WebSocket
  (``wss://api.meta.ai/v1/asr/realtime``): PCM in, interim transcripts out,
  and in ``ENDPOINTING`` mode the model's own turn boundaries.
* :meth:`MetaSTTProvider.transcribe` — ``POST /v1/asr/transcribe`` for one
  finished clip (a WAV of at most 10 minutes and 32 MB).

How a stream ends depends on how the :class:`~roomkit.channels.VoiceChannel`
is set up, which the provider cannot see, so it is configuration:

* ``ENDPOINTING`` (default) — no pipeline VAD. The channel streams all audio
  (continuous mode) and the model decides where a turn ends, after about
  550 ms of silence (not configurable). Each ``speechComplete`` is a final.
* ``PUSH_TO_TALK`` — a pipeline VAD delimits utterances. The channel opens one
  stream per utterance, and the model answers one final once the stream ends.

* ``DIARIZATION`` — ``ENDPOINTING`` plus who spoke: each final carries its
  turn as one :class:`~roomkit.voice.base.SpeakerSegment` labelled ``"A"``,
  ``"B"``… (RFC §12.2.3). A change of voice also ends a turn, even without a
  pause. Labels hold within one stream only: a ``VoiceChannel`` carries them
  in continuous mode, where it keeps the stream across turns, and refuses the
  provider behind a VAD.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from roomkit.core.task_utils import cancel_and_wait
from roomkit.providers.utils import parse_data_uri
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.stt._pacing import AudioPace
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.meta_protocol import (
    WAV_MIME_TYPES,
    Turns,
    event_error,
    http_error,
    read_wav,
    rest_result,
    stream_error,
    to_result,
    wav_bytes,
)
from roomkit.voice.stt.meta_protocol import MetaSTTError as MetaSTTError

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger("roomkit.voice.stt.meta")

DEFAULT_MODEL = "muse-voice-transcribe-1.0"

_BURST_S = 3.0
"""Audio a stream sends at once at most. The realtime service refuses a stream
sent 7 s or more ahead at once ("Audio processing backlog too large"; 6 s
passes, measured 2026-10-07)."""

_CATCH_UP_SPEED = 2.0
"""How much faster than real time a backlog goes out past the burst: the
service keeps up at 4x (20 s accepted) and not at 8x (refused after 3 s),
measured 2026-10-07."""

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "Arabic",
        "Bengali",
        "Dutch",
        "English",
        "French",
        "German",
        "Hebrew",
        "Hindi",
        "Indonesian",
        "Italian",
        "Japanese",
        "Kannada",
        "Korean",
        "Malay",
        "Mandarin Chinese",
        "Marathi",
        "Polish",
        "Portuguese",
        "Spanish",
        "Tagalog",
        "Tamil",
        "Telugu",
        "Thai",
        "Turkish",
        "Vietnamese",
    }
)
"""The language names ``language_bias`` accepts, spelled as Meta documents them.

Checked here because the service does not: it answers a misspelt or unknown
name (``"french"``, ``"fr"``) exactly like a known one, and the bias then
silently does nothing.
"""

MetaSTTMode = Literal["ENDPOINTING", "PUSH_TO_TALK", "DIARIZATION"]

# The two input rates the service takes; anything else is resampled to the
# model's native 24 kHz.
_ENCODINGS: dict[int, str] = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}
_NATIVE_RATE = 24000


def _import_websockets() -> tuple[Any, Any]:
    """``(connect, exceptions)`` from ``websockets``, which streaming needs."""
    try:
        from websockets import exceptions
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise ImportError(
            "websockets is required for MetaSTTProvider streaming. "
            "Install with: pip install roomkit[meta-stt]"
        ) from exc
    return connect, exceptions


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for MetaSTTProvider.transcribe(). "
            "Install with: pip install roomkit[meta-stt]"
        ) from exc
    return httpx


@dataclass
class MetaSTTConfig:
    """Configuration for :class:`MetaSTTProvider`.

    Attributes:
        api_key: Meta Model API key (sent as ``Bearer``).
        model: Transcription model id.
        mode: How a stream ends — see the module docstring. ``ENDPOINTING``
            for a channel without a pipeline VAD, ``PUSH_TO_TALK`` behind one,
            ``DIARIZATION`` to have every final name its speaker.
        keywords: Terms to bias recognition toward (product names, people,
            places). A bias, not a guaranteed spelling.
        language_bias: Language *names* to bias toward, from
            :data:`SUPPORTED_LANGUAGES` (``["French"]``, not ``["fr"]``). A
            bias, not a pin: the model still transcribes what it hears, which
            is why the provider reports ``supports_language_override = False``.
        base_url: REST endpoint root. Override only to point at a proxy.
        realtime_url: Realtime WebSocket endpoint.
        timeout_s: Timeout of a REST transcription, in seconds.
        handshake_timeout_s: How long a stream waits for the service to
            acknowledge its configuration.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    mode: MetaSTTMode = "ENDPOINTING"
    keywords: list[str] = field(default_factory=list)
    language_bias: list[str] = field(default_factory=list)
    base_url: str = "https://api.meta.ai/v1"
    realtime_url: str = "wss://api.meta.ai/v1/asr/realtime"
    timeout_s: float = 60.0
    handshake_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("MetaSTTConfig.api_key is required")
        if not self.api_key.isascii():
            # Refused here, not as a UnicodeEncodeError from the HTTP client.
            raise ValueError(
                "MetaSTTConfig.api_key holds a non-ASCII character (a placeholder such as "
                "'…' copied as is?): an HTTP Authorization header cannot carry it"
            )
        if self.mode not in ("ENDPOINTING", "PUSH_TO_TALK", "DIARIZATION"):
            raise ValueError(
                f"mode must be 'ENDPOINTING', 'PUSH_TO_TALK' or 'DIARIZATION', got {self.mode!r}"
            )
        unknown = [name for name in self.language_bias if name not in SUPPORTED_LANGUAGES]
        if unknown:
            raise ValueError(
                f"language_bias takes Meta's language names, got {unknown!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )


class MetaSTTProvider(STTProvider):
    """Speech-to-text on Meta's Muse Voice Transcribe, streaming and batch."""

    def __init__(self, config: MetaSTTConfig) -> None:
        self._config = config
        self._resampler = LinearResamplerProvider()
        self._http: Any = None

    @property
    def name(self) -> str:
        return "MetaSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_diarization(self) -> bool:
        """True in ``DIARIZATION`` mode: every final then names its speaker."""
        return self._config.mode == "DIARIZATION"

    def _request_fields(self, mode: str) -> dict[str, Any]:
        """The settings every request carries, REST body or stream handshake."""
        fields: dict[str, Any] = {"model": self._config.model, "mode": mode}
        if self._config.keywords:
            fields["keywords"] = list(self._config.keywords)
        if self._config.language_bias:
            fields["languageBias"] = list(self._config.language_bias)
        return fields

    def _supported_pcm(
        self, data: bytes, sample_rate: int, channels: int, rate: int, sample_width: int = 2
    ) -> bytes:
        """``data`` as mono 16-bit PCM at ``rate``, converted when it is not."""
        if sample_rate == rate and channels == 1 and sample_width == 2:
            return data
        frame = AudioFrame(
            data=data, sample_rate=sample_rate, channels=channels, sample_width=sample_width
        )
        return self._resampler.resample(frame, rate, 1, 2, stream="meta-stt").data

    # ------------------------------------------------------------------
    # Batch: POST /v1/asr/transcribe
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one finished clip over the REST endpoint.

        The clip is sent as a WAV in ``PUSH_TO_TALK`` mode — one clip, one
        transcript — or in ``DIARIZATION`` mode when the provider is
        configured for it, and the service's turns then come back as speaker
        segments. Meta reports no confidence and no language, so neither is set.

        Raises:
            ValueError: ``audio`` is an http(s) URL, or a ``data:`` URI that
                does not carry a WAV. The endpoint takes an upload, and
                fetching a caller's URL from here would make RoomKit the
                client of whatever it names; a ``data:`` URI needs no fetch.
            MetaSTTError: The service refused or failed the request.
        """
        data, sample_rate, channels, width = _clip_pcm(audio)
        rate = sample_rate if sample_rate in _ENCODINGS else _NATIVE_RATE
        pcm = self._supported_pcm(data, sample_rate, channels, rate, width)
        mode = "DIARIZATION" if self.supports_diarization else "PUSH_TO_TALK"
        request = {**self._request_fields(mode), "audioEncoding": "WAV"}
        response = await self._http_client().post(
            f"{self._config.base_url}/asr/transcribe",
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": ("audio.wav", wav_bytes(pcm, rate), "audio/wav"),
            },
            headers={"Authorization": f"Bearer {self._config.api_key}"},
        )
        if response.status_code >= 400:
            raise http_error(response)
        return rest_result(response.json(), diarized=self.supports_diarization)

    def _http_client(self) -> Any:
        if self._http is None:
            httpx = _import_httpx()
            self._http = httpx.AsyncClient(timeout=self._config.timeout_s)
        return self._http

    # ------------------------------------------------------------------
    # Streaming: wss://api.meta.ai/v1/asr/realtime
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream audio to the realtime endpoint and yield its transcripts.

        Yields ``is_speech_start`` when the model hears speech begin
        (``ENDPOINTING`` and ``DIARIZATION``), cumulative partials while it
        listens, and a final per turn, with its speaker in ``DIARIZATION``.
        The input rate is read off the first chunk: 16 and 24 kHz go through
        as they are, any other rate is resampled to 24 kHz.

        Raises:
            MetaSTTError: The service refused the stream or closed it
                abnormally; ``retryable`` says whether a new stream may work.
        """
        first = await _first_audio(audio_stream)
        if first is None:
            return
        rate = first.sample_rate if first.sample_rate in _ENCODINGS else _NATIVE_RATE
        if rate != first.sample_rate:
            logger.debug("Meta STT: resampling %d Hz to %d Hz", first.sample_rate, rate)
        ws = await self._open_stream(rate)
        sender = asyncio.create_task(self._send_audio(ws, first, audio_stream, rate))
        turns = Turns() if self.supports_diarization else None
        try:
            async with contextlib.aclosing(_events(ws)) as events:
                async for event in events:
                    result = to_result(event, turns)
                    if result is not None:
                        yield result
        finally:
            await cancel_and_wait(sender)
            await ws.close()

    async def _open_stream(self, rate: int) -> Any:
        """Connect and have the service accept the stream's configuration."""
        connect, _ = _import_websockets()
        try:
            ws = await connect(self._config.realtime_url, max_size=None)
        except Exception as exc:
            raise stream_error(exc) from exc
        handshake = {
            "authorization": {"accessToken": f"Bearer {self._config.api_key}"},
            **self._request_fields(self._config.mode),
            "audioEncoding": _ENCODINGS[rate],
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        try:
            await ws.send(json.dumps(handshake))
            ack = json.loads(await asyncio.wait_for(ws.recv(), self._config.handshake_timeout_s))
        except Exception as exc:
            await ws.close()
            raise stream_error(exc) from exc
        if ack.get("type") == "error":
            await ws.close()
            raise event_error(ack)
        logger.debug("Meta STT stream open: session %s", ack.get("sessionId"))
        return ws

    async def _send_audio(
        self, ws: Any, first: AudioChunk, audio_stream: AsyncIterator[AudioChunk], rate: int
    ) -> None:
        """Send the audio as binary frames, then ``endStream`` once it ends.

        The audio is paced (RFC §12.2): a stream that opens on a backlog, after
        a reconnect, would otherwise send it at once, and the service refuses
        more unprocessed audio than it holds. The pace holds back reading from
        the source, so the audio it has not reached stays there, for the next
        stream if this one ends first.

        A failing audio source still ends the stream, so the service answers
        with the transcript of what it heard instead of waiting for more.
        """
        _, exceptions = _import_websockets()
        pace = AudioPace(_BURST_S, _CATCH_UP_SPEED)
        try:
            await _send_paced(ws, self._chunk_pcm(first, rate), rate, pace)
            if not first.is_final:
                async for chunk in audio_stream:
                    if chunk.data:
                        await _send_paced(ws, self._chunk_pcm(chunk, rate), rate, pace)
                    if chunk.is_final:
                        break
        except exceptions.ConnectionClosed:
            # The reader sees the close and reports it; nothing to add here.
            logger.debug("Meta STT stream closed while sending audio")
            return
        except Exception:
            logger.exception("Meta STT audio source failed; ending the stream")
        with contextlib.suppress(exceptions.ConnectionClosed):
            await ws.send(json.dumps({"type": "endStream"}))

    def _chunk_pcm(self, chunk: AudioChunk, rate: int) -> bytes:
        return self._supported_pcm(chunk.data, chunk.sample_rate, chunk.channels, rate)

    async def close(self) -> None:
        """Release the REST client."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def _clip_pcm(audio: AudioContent | AudioChunk | AudioFrame) -> tuple[bytes, int, int, int]:
    """``(pcm, sample_rate, channels, sample_width)`` of a clip to transcribe."""
    if isinstance(audio, AudioFrame):
        return audio.data, audio.sample_rate, audio.channels, audio.sample_width
    if isinstance(audio, AudioChunk):
        return audio.data, audio.sample_rate, audio.channels, 2
    if not audio.url.startswith("data:"):
        raise ValueError(
            "MetaSTTProvider does not fetch URLs; pass a data: URI, an AudioChunk or an AudioFrame"
        )
    mime, payload = parse_data_uri(audio.url, fallback_mime=audio.mime_type)
    if mime not in WAV_MIME_TYPES:
        raise ValueError(f"Meta transcribes WAV only, got {mime}")
    return read_wav(payload)


async def _send_paced(ws: Any, pcm: bytes, rate: int, pace: AudioPace) -> None:
    """Send ``pcm`` (mono 16-bit at ``rate``) once *pace* allows it."""
    delay = pace.delay(len(pcm) / (2 * rate))
    if delay:
        await asyncio.sleep(delay)
    await ws.send(pcm)


async def _first_audio(audio_stream: AsyncIterator[AudioChunk]) -> AudioChunk | None:
    """The first chunk carrying audio, or ``None`` if the stream ends without any.

    The connection is opened only once there is audio to send, and its
    encoding is set from this chunk's rate.
    """
    async for chunk in audio_stream:
        if chunk.data:
            return chunk
        if chunk.is_final:
            return None
    return None


async def _events(ws: Any) -> AsyncGenerator[dict[str, Any], None]:
    """The service's JSON frames until it closes; an abnormal close raises."""
    _, exceptions = _import_websockets()
    try:
        async for message in ws:
            if isinstance(message, str):
                yield json.loads(message)
    except exceptions.ConnectionClosedError as exc:
        raise stream_error(exc) from exc
