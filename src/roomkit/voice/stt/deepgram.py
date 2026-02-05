"""Deepgram speech-to-text provider."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent

logger = logging.getLogger(__name__)


@dataclass
class DeepgramConfig:
    """Configuration for Deepgram STT provider."""

    api_key: str
    model: str = "nova-2"
    language: str = "en"
    punctuate: bool = True
    diarize: bool = False
    smart_format: bool = True
    filler_words: bool = False
    # Real-time streaming options
    interim_results: bool = True
    endpointing: int = 300  # ms of silence to end utterance
    vad_events: bool = True


class DeepgramSTTProvider(STTProvider):
    """Deepgram speech-to-text provider with streaming support."""

    def __init__(self, config: DeepgramConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "DeepgramSTT"

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
        """Build query parameters for Deepgram API."""
        params: dict[str, Any] = {
            "model": self._config.model,
            "language": self._config.language,
            "punctuate": self._config.punctuate,
            "diarize": self._config.diarize,
            "smart_format": self._config.smart_format,
            "filler_words": self._config.filler_words,
        }
        return {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}

    async def transcribe(self, audio: AudioContent | AudioChunk) -> str:
        """Transcribe complete audio to text.

        Args:
            audio: Audio content (URL) or raw audio chunk.

        Returns:
            Transcribed text.
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
            # Handle AudioChunk (raw bytes)
            audio_data = audio.data
            audio_format = getattr(audio, "format", "wav")

            # For raw PCM formats, set encoding params for Deepgram
            if audio_format in ("pcm_s16le", "linear16", "raw"):
                content_type = "audio/raw"
                params["encoding"] = "linear16"
                params["sample_rate"] = getattr(audio, "sample_rate", 16000)
                params["channels"] = getattr(audio, "channels", 1)
            else:
                content_type = f"audio/{audio_format}"

        # Call Deepgram API
        response = await client.post(
            "/listen",
            params=params,
            content=audio_data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()
        result = response.json()

        # Extract transcript
        try:
            transcript: str = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript.strip()
        except (KeyError, IndexError):
            logger.warning("No transcript in Deepgram response: %s", result)
            return ""

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results using WebSocket.

        Args:
            audio_stream: Async iterator of audio chunks.

        Yields:
            TranscriptionResult with partial and final transcripts.
        """
        import websockets

        # Build WebSocket URL with query params
        params = self._build_query_params()
        params["interim_results"] = str(self._config.interim_results).lower()
        params["endpointing"] = self._config.endpointing
        params["vad_events"] = str(self._config.vad_events).lower()
        params["encoding"] = "linear16"
        params["sample_rate"] = "16000"

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
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
                            # Send close frame to signal end of audio
                            await ws.send(b"")
                            break
                except Exception as e:
                    logger.error("Error sending audio to Deepgram: %s", e)

            sender_task = asyncio.create_task(send_audio())

            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        continue

                    import json

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
