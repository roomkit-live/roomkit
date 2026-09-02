"""Deepgram speech-to-text provider using the official SDK."""

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


def _import_deepgram() -> Any:
    """Import the Deepgram SDK, raising a clear error if missing."""
    try:
        import deepgram

        return deepgram
    except ImportError as exc:
        raise ImportError(
            "deepgram-sdk is required for DeepgramSTTProvider. "
            "Install with: pip install roomkit[deepgram]"
        ) from exc


def _language_of(entry: Any) -> str | None:
    """A language code from a word, a ``languages`` entry, or a plain string."""
    if isinstance(entry, str):
        return entry or None
    code = entry.get("language") if isinstance(entry, dict) else getattr(entry, "language", None)
    return code if isinstance(code, str) and code else None


def _reported_language(alt: Any, channel: Any = None) -> str | None:
    """The language Deepgram reports for a result, ``None`` when it reports none.

    Nova-3 ``multi`` tags every word with its language and lists the
    languages heard in ``languages``. The word tags decide — the language
    most words carry is the one the speaker used, and a tie keeps the first
    heard — with ``languages`` as the fallback when words are missing.
    Prerecorded detection (``detect_language``) reports on the channel.
    A stream pinned to one language reports nothing, and this returns
    ``None`` rather than echoing the request. The SDK models accept these
    fields without typing them, and the v2 shape carries ``{language,
    score}`` objects where v1 carries strings, so every shape is read.
    """
    counts: dict[str, int] = {}
    for word in getattr(alt, "words", None) or []:
        code = _language_of(word)
        if code:
            counts[code] = counts.get(code, 0) + 1
    if counts:
        return max(counts, key=lambda code: counts[code])
    for entry in getattr(alt, "languages", None) or []:
        code = _language_of(entry)
        if code:
            return code
    detected = getattr(channel, "detected_language", None)
    return detected if isinstance(detected, str) and detected else None


@dataclass
class DeepgramConfig:
    """Configuration for Deepgram STT provider.

    See https://developers.deepgram.com/docs/getting-started
    for the full parameter reference.
    """

    api_key: str = field(repr=False)

    # Model & language
    model: str = "nova-2"
    language: str = "en"
    version: str | None = None

    # Formatting
    punctuate: bool = True
    smart_format: bool = True
    numerals: bool = False
    dictation: bool = False

    # Content filtering
    profanity_filter: bool = False
    redact: list[str] | None = None
    replace: list[str] | None = None
    detect_entities: bool = False

    # Speech features
    diarize: bool = False
    filler_words: bool = False
    multichannel: bool = False

    # Keyword boosting
    keywords: list[str] = field(default_factory=list)
    keyterm: list[str] = field(default_factory=list)
    search: list[str] = field(default_factory=list)

    # Real-time streaming options
    interim_results: bool = True
    endpointing: int | bool = 300
    utterance_end_ms: int | None = None
    vad_events: bool = True

    # Misc
    tag: str | None = None
    extra: list[str] = field(default_factory=list)
    mip_opt_out: bool = False


class DeepgramSTTProvider(STTProvider):
    """Deepgram speech-to-text provider using the official SDK.

    Uses ``deepgram-sdk`` for both batch and streaming transcription.
    """

    def __init__(self, config: DeepgramConfig) -> None:
        self._config = config
        self._dg = _import_deepgram()
        self._client = self._dg.AsyncDeepgramClient(api_key=config.api_key)

    @property
    def name(self) -> str:
        return "DeepgramSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    def _build_connect_options(
        self, sample_rate: int = 16000, language: str | None = None
    ) -> dict[str, Any]:
        """Build keyword arguments for the SDK connect() call.

        The SDK v6 connect() accepts all values as Optional[str].
        ``language`` replaces the configured one for this connection.
        """
        c = self._config

        def _b(v: bool) -> str:
            return str(v).lower()

        opts: dict[str, Any] = {
            "model": c.model,
            "language": language or c.language,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "punctuate": _b(c.punctuate),
            "smart_format": _b(c.smart_format),
            "diarize": _b(c.diarize),
            "numerals": _b(c.numerals),
            "profanity_filter": _b(c.profanity_filter),
            "detect_entities": _b(c.detect_entities),
            "multichannel": _b(c.multichannel),
            "dictation": _b(c.dictation),
            "interim_results": _b(c.interim_results),
            "vad_events": _b(c.vad_events),
        }
        if c.version is not None:
            opts["version"] = c.version
        if isinstance(c.endpointing, bool):
            opts["endpointing"] = _b(c.endpointing)
        else:
            opts["endpointing"] = str(c.endpointing)
        if c.utterance_end_ms is not None:
            opts["utterance_end_ms"] = str(c.utterance_end_ms)
        if c.tag is not None:
            opts["tag"] = c.tag
        if c.mip_opt_out:
            opts["mip_opt_out"] = "true"
        if c.keywords:
            opts["keywords"] = c.keywords
        if c.keyterm:
            opts["keyterm"] = c.keyterm
        if c.search:
            opts["search"] = c.search
        if c.redact:
            opts["redact"] = c.redact
        if c.replace:
            opts["replace"] = c.replace
        return opts

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe complete audio using the Deepgram REST API.

        URL-bearing ``AudioContent`` is dispatched via Deepgram's native
        ``transcribe_url`` so the fetch happens from Deepgram's network,
        not ours — this removes us from the SSRF surface entirely. Raw
        bytes (``AudioChunk`` / ``AudioFrame``) go through
        ``transcribe_file`` as before. ``language`` replaces the configured
        one for this call.
        """
        t0 = time.monotonic()
        effective_language = language or self._config.language

        if hasattr(audio, "url"):
            response = await self._client.listen.v1.media.transcribe_url(
                url=audio.url,
                model=self._config.model,
                language=effective_language,
                smart_format=self._config.smart_format,
                punctuate=self._config.punctuate,
            )
        else:
            audio_data = audio.data
            sample_rate = getattr(audio, "sample_rate", 16000)
            channels = getattr(audio, "channels", 1)

            response = await self._client.listen.v1.media.transcribe_file(
                request=audio_data,
                model=self._config.model,
                language=effective_language,
                smart_format=self._config.smart_format,
                punctuate=self._config.punctuate,
                encoding="linear16",
                multichannel=channels > 1 if channels else None,
                request_options={
                    "additional_query_parameters": {
                        "sample_rate": str(sample_rate),
                        "channels": str(channels),
                    },
                },
            )

        ttfb_ms = (time.monotonic() - t0) * 1000
        logger.debug("Deepgram batch transcription: %.0fms", ttfb_ms)

        try:
            channel = response.results.channels[0]
            alt = channel.alternatives[0]
            return TranscriptionResult(
                text=alt.transcript.strip(),
                confidence=alt.confidence,
                language=_reported_language(alt, channel),
            )
        except (AttributeError, IndexError):
            logger.warning("No transcript in Deepgram response")
            return TranscriptionResult(text="")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription using the Deepgram SDK WebSocket client.

        ``language`` replaces the configured one for this stream. Deepgram
        fixes the language in the connection URL, so it holds for the life
        of the stream; the caller opens a new stream to change it.
        """
        from deepgram.core.events import EventType  # noqa: N813

        # Read first chunk to detect sample rate
        first_chunk: AudioChunk | None = None
        async for chunk in audio_stream:
            first_chunk = chunk
            break

        if first_chunk is None:
            logger.debug("Deepgram stream: no audio chunks received")
            return

        sample_rate = first_chunk.sample_rate or 16000
        logger.debug(
            "Deepgram stream: first chunk %d bytes, sample_rate=%d",
            len(first_chunk.data),
            sample_rate,
        )

        opts = self._build_connect_options(sample_rate, language)
        logger.info(
            "Deepgram stream: connecting with model=%s, sample_rate=%s, opts=%s",
            opts.get("model"),
            opts.get("sample_rate"),
            {k: v for k, v in opts.items() if k not in ("model", "sample_rate", "encoding")},
        )

        # Fresh client per stream to avoid stale SDK state
        client = self._dg.AsyncDeepgramClient(api_key=self._config.api_key)

        # Results queue — SDK callbacks push, our async generator pulls
        result_queue: asyncio.Queue[TranscriptionResult | None] = asyncio.Queue()
        # Last stream error from the SDK's on_error callback. The stream then
        # closes via on_close; we raise it so the consumer marks the stream
        # failed (and reconnects) instead of seeing a clean, empty end.
        stream_error: list[Any] = []

        def on_message(message: Any) -> None:
            """Handle transcription results from the SDK."""
            try:
                if not hasattr(message, "channel"):
                    return
                alt = message.channel.alternatives[0]
                transcript = alt.transcript
                if not transcript:
                    return
                is_final = getattr(message, "is_final", False)
                confidence = getattr(alt, "confidence", None)
                words = getattr(alt, "words", [])
                result_queue.put_nowait(
                    TranscriptionResult(
                        text=transcript,
                        is_final=is_final,
                        confidence=confidence,
                        language=_reported_language(alt),
                        words=words,
                    )
                )
            except (AttributeError, IndexError):
                pass

        def on_error(error: Any) -> None:
            logger.error("Deepgram stream error: %s", error)
            stream_error.append(error)

        def on_close(_: Any) -> None:
            result_queue.put_nowait(None)  # sentinel

        async with client.listen.v1.connect(**opts) as connection:
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR, on_error)
            connection.on(EventType.CLOSE, on_close)

            # start_listening() runs the receive loop — must be a background
            # task so it doesn't block audio sending.
            listen_task = asyncio.create_task(connection.start_listening())
            logger.info("Deepgram stream: connected, sending audio...")

            # Sender task: feed audio chunks to Deepgram
            async def send_audio() -> None:
                chunks_sent = 0
                try:
                    if first_chunk.data:
                        await connection.send_media(first_chunk.data)
                        chunks_sent += 1
                        logger.debug(
                            "Deepgram: sent first chunk (%d bytes)", len(first_chunk.data)
                        )
                    async for chunk in audio_stream:
                        if chunk.data:
                            await connection.send_media(chunk.data)
                            chunks_sent += 1
                        if chunk.is_final:
                            break
                    logger.debug(
                        "Deepgram stream: sender done, %d chunks",
                        chunks_sent,
                    )
                except Exception as e:
                    logger.error("Error sending audio to Deepgram: %s", e)
                finally:
                    # Closing a stream whose socket is already gone is not a
                    # failure — the cancellation path arrives here with the
                    # connection torn down. Letting it raise would replace the
                    # CancelledError the caller awaits, so the cancellation
                    # surfaces as an unhandled error at interpreter shutdown
                    # instead of being suppressed.
                    try:
                        await connection.send_close_stream()
                    except Exception:
                        logger.debug(
                            "Deepgram stream: close frame not sent (connection already gone)",
                            exc_info=True,
                        )

            sender_task = asyncio.create_task(send_audio())

            try:
                while True:
                    result = await result_queue.get()
                    if result is None:
                        break
                    yield result
                if stream_error:
                    raise RuntimeError(f"Deepgram stream error: {stream_error[-1]}")
            finally:
                sender_task.cancel()
                listen_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task
                with contextlib.suppress(asyncio.CancelledError):
                    await listen_task

    async def close(self) -> None:
        """Release resources."""
        pass
