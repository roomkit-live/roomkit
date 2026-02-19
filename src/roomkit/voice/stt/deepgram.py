"""Deepgram speech-to-text provider."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


@dataclass
class DeepgramConfig:
    """Configuration for Deepgram STT provider.

    See https://developers.deepgram.com/reference/speech-to-text/listen-streaming
    for the full parameter reference.
    """

    api_key: str

    # Model & language
    model: str = "nova-3"
    language: str = "en"
    version: str | None = None

    # Formatting
    punctuate: bool = True
    smart_format: bool = True
    numerals: bool = False
    dictation: bool = False

    # Content filtering
    profanity_filter: bool = False
    redact: list[str] | None = None  # e.g. ["pci", "ssn"]
    replace: list[str] | None = None  # e.g. ["old:new"]
    detect_entities: bool = False

    # Speech features
    diarize: bool = False
    filler_words: bool = False
    multichannel: bool = False

    # Keyword boosting (keywords uses legacy format, keyterm is Nova-3 only)
    keywords: list[str] = field(default_factory=list)
    keyterm: list[str] = field(default_factory=list)
    search: list[str] = field(default_factory=list)

    # Real-time streaming options
    interim_results: bool = True
    endpointing: int | bool = 300  # ms of silence, or False to disable
    utterance_end_ms: int | None = None
    vad_events: bool = True

    # Misc
    tag: str | None = None
    extra: list[str] = field(default_factory=list)  # e.g. ["key:value"]
    mip_opt_out: bool = False


class DeepgramSTTProvider(STTProvider):
    """Deepgram speech-to-text provider with streaming support."""

    def __init__(self, config: DeepgramConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "DeepgramSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.deepgram.com/v1",
                headers={
                    "Authorization": f"Token {self._config.api_key}",
                    "Content-Type": "audio/wav",
                },
                timeout=60.0,
            )
        return self._client

    def _build_query_params(self) -> dict[str, Any]:
        """Build common query parameters for Deepgram API."""
        c = self._config
        params: dict[str, Any] = {
            "model": c.model,
            "language": c.language,
            "punctuate": c.punctuate,
            "diarize": c.diarize,
            "smart_format": c.smart_format,
            "filler_words": c.filler_words,
            "numerals": c.numerals,
            "profanity_filter": c.profanity_filter,
            "detect_entities": c.detect_entities,
            "multichannel": c.multichannel,
            "dictation": c.dictation,
        }
        # Optional scalar params
        if c.version is not None:
            params["version"] = c.version
        if c.tag is not None:
            params["tag"] = c.tag
        if c.mip_opt_out:
            params["mip_opt_out"] = True
        # Optional list params
        if c.keywords:
            params["keywords"] = c.keywords
        if c.keyterm:
            params["keyterm"] = c.keyterm
        if c.search:
            params["search"] = c.search
        if c.redact:
            params["redact"] = c.redact
        if c.replace:
            params["replace"] = c.replace
        if c.extra:
            params["extra"] = c.extra
        # Stringify bools
        return {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}

    @staticmethod
    def _to_query_string(params: dict[str, Any]) -> str:
        """Encode parameters as a query string, expanding list values."""
        pairs: list[str] = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    pairs.append(f"{k}={item}")
            else:
                pairs.append(f"{k}={v}")
        return "&".join(pairs)

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        """Transcribe complete audio to text.

        Args:
            audio: Audio content (URL), raw audio chunk, or audio frame.

        Returns:
            TranscriptionResult with text and metadata.
        """
        client = self._get_client()
        params = self._build_query_params()

        # Handle AudioContent (URL-based)
        if hasattr(audio, "url"):
            # Fetch audio from URL
            async with httpx.AsyncClient() as fetch_client:
                resp = await fetch_client.get(audio.url)
                resp.raise_for_status()
                audio_data = resp.content
                content_type = resp.headers.get("content-type", "audio/wav")
        else:
            # Handle AudioChunk or AudioFrame (raw bytes)
            audio_data = audio.data
            audio_format = getattr(audio, "format", "raw")

            # For raw PCM formats (including AudioFrame), set encoding params for Deepgram
            if audio_format in ("pcm_s16le", "linear16", "raw"):
                content_type = "audio/raw"
                params["encoding"] = "linear16"
                params["sample_rate"] = getattr(audio, "sample_rate", 16000)
                params["channels"] = getattr(audio, "channels", 1)
            else:
                content_type = f"audio/{audio_format}"

        # Call Deepgram API
        t0 = time.monotonic()
        response = await client.post(
            "/listen",
            params=params,
            content=audio_data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()
        result = response.json()

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.stt.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "deepgram", "model": self._config.model},
        )

        # Extract transcript
        try:
            alt = result["results"]["channels"][0]["alternatives"][0]
            transcript: str = alt.get("transcript", "")
            confidence = alt.get("confidence")
            language = result["results"]["channels"][0].get("detected_language")
            return TranscriptionResult(
                text=transcript.strip(),
                confidence=confidence,
                language=language,
            )
        except (KeyError, IndexError):
            logger.warning("No transcript in Deepgram response: %s", result)
            return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results using WebSocket.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptionResult with partial and final transcripts.
        """
        import json

        import websockets

        # Build WebSocket URL with query params
        params = self._build_query_params()
        params["interim_results"] = str(self._config.interim_results).lower()
        if isinstance(self._config.endpointing, bool):
            params["endpointing"] = str(self._config.endpointing).lower()
        else:
            params["endpointing"] = self._config.endpointing
        params["vad_events"] = str(self._config.vad_events).lower()
        if self._config.utterance_end_ms is not None:
            params["utterance_end_ms"] = self._config.utterance_end_ms
        params["encoding"] = "linear16"
        params["sample_rate"] = "16000"

        query_string = self._to_query_string(params)
        ws_url = f"wss://api.deepgram.com/v1/listen?{query_string}"

        headers = [("Authorization", f"Token {self._config.api_key}")]

        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            # Start sender task
            async def send_audio() -> None:
                try:
                    async for chunk in audio_stream:
                        if chunk.data:
                            await ws.send(chunk.data)
                        if chunk.is_final:
                            break
                    # Tell Deepgram to flush remaining audio and close
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception as e:
                    logger.error("Error sending audio to Deepgram: %s", e)

                # NOTE: Do NOT call ws.close() here — let Deepgram close
                # from its side after flushing final results.  Calling
                # ws.close() races with the receiver loop and can drop
                # final transcriptions.

            sender_task = asyncio.create_task(send_audio())

            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        continue

                    data = json.loads(message)

                    # Handle transcription results
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            alt = alternatives[0]
                            transcript = alt.get("transcript", "")
                            confidence = alt.get("confidence")
                            words = alt.get("words", [])
                            is_final = data.get("is_final", False)

                            if transcript:
                                yield TranscriptionResult(
                                    text=transcript,
                                    is_final=is_final,
                                    confidence=confidence,
                                    language=data.get("channel", {}).get("detected_language"),
                                    words=words,
                                )

                    # Handle speech events
                    elif data.get("type") == "SpeechStarted":
                        logger.debug("Speech started")
                    elif data.get("type") == "UtteranceEnd":
                        logger.debug("Utterance ended")

            finally:
                sender_task.cancel()
                import contextlib

                with contextlib.suppress(asyncio.CancelledError):
                    await sender_task

    async def close(self) -> None:  # noqa: B027
        """Release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
