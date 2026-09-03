"""Google Gemini speech-to-text provider — batch transcription of recordings.

Gemini has no speech-to-text endpoint. Transcription here is an *instruction* to
a multimodal model that takes audio as input, which is why this provider is
batch-only: the API accepts a complete recording, not a stream, and answers in
seconds. Google's own audio documentation points at Cloud Speech-to-Text for
dedicated real-time transcription, and that remains the right advice for live
turn-taking — reach for :mod:`~roomkit.voice.stt.deepgram`,
:mod:`~roomkit.voice.stt.gradium` or a local
:mod:`~roomkit.voice.stt.sherpa_onnx` model there.

What the batch shape buys is what a streaming recogniser structurally cannot
give: the model sees the whole recording before it answers, so one pass returns
the transcript, the speaker turns and the timestamps together — no diarization
stage, no merge. That makes this the provider for meeting recordings, voicemail
and imported audio files.

Two input paths, both verified against the live API on 2026-08-07:

* Inline — raw PCM or an encoded clip carried in the request. Fast, and bounded
  by the request size limit, so this provider only inlines small recordings.
* Uploaded — the Files API returns a URI the interaction refers to. This is the
  path a real meeting recording takes; uploads are deleted after use rather
  than left to expire.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import mimetypes
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from roomkit.providers.gemini.sdk import build_genai_client
from roomkit.voice.base import TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import AudioChunk

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",
        "audio/m4a",
        "audio/l16",
        "audio/s16le",
        "audio/opus",
        "audio/alaw",
        "audio/mulaw",
    }
)
"""Mime types the interactions endpoint accepts for audio input.

Taken from the service's own rejection message rather than the prose docs,
which list fewer: raw PCM (``audio/l16``, ``audio/s16le``) and the telephony
codecs are accepted but undocumented. Verified 2026-08-07.
"""

_PCM_MIME_TYPE = "audio/l16"
"""What roomkit's own frames and chunks are: 16-bit little-endian PCM."""

_FILES_API_HOST = "generativelanguage.googleapis.com"

_MAX_INLINE_BYTES = 15 * 1024 * 1024
"""Above this, a file is uploaded instead of inlined — the request has a size
limit and a base64 payload is a third larger than the file it carries."""

_TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "description": "BCP-47 code of the language spoken, e.g. 'fr-CA'.",
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "start": {"type": "string", "description": "MM:SS from the start"},
                    "end": {"type": "string", "description": "MM:SS from the start"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "start", "end", "text"],
            },
        },
    },
    "required": ["language", "segments"],
}


@dataclass(frozen=True)
class TranscriptSegment:
    """One speaker turn.

    Timestamps are ``MM:SS`` strings, as the model returns them. They are the
    model's reading of the recording, not a forced alignment: treat them as
    navigation, not as sync marks.
    """

    speaker: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class Transcript:
    """A whole recording, as speaker turns."""

    language: str
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        """The turns joined into a readable transcript, one line per speaker."""
        return "\n".join(f"{s.speaker}: {s.text}" for s in self.segments)

    @property
    def plain_text(self) -> str:
        """The spoken words alone, without speaker labels."""
        return " ".join(s.text for s in self.segments)


@dataclass
class GeminiSTTConfig:
    """Configuration for the Gemini batch STT provider.

    Args:
        api_key: Gemini API key (``GEMINI_API_KEY``).
        model: A text/multimodal Gemini model that accepts audio input — see
            :meth:`~roomkit.providers.gemini.ai.GeminiAIProvider.available_models`
            for the catalog roomkit keeps. The default is the current flash
            model: transcription is not a reasoning task, and flash is the
            cheapest way to buy the audio context window.
        language: Optional BCP-47 hint (e.g. ``"fr-CA"``). Left unset, the model
            identifies the language itself and reports it on the transcript.
        diarize: Ask for speaker labels. Worth turning off for a single-speaker
            recording, where labelling costs tokens and invents distinctions.
            A conference recorded per participant track needs no diarization at
            all — transcribe each track and merge on the timestamps.
        prompt: Extra instruction appended to the transcription request —
            vocabulary that matters ("the product is spelled RoomKit"),
            formatting rules, anything the model should know before it listens.
        timeout: Per-request timeout in seconds. Generous by design: a model
            answering on an hour of audio is not answering in milliseconds.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
        max_inline_bytes: Recordings larger than this are uploaded through the
            Files API instead of being inlined in the request.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.6-flash"
    language: str | None = None
    diarize: bool = True
    prompt: str | None = None
    timeout: float = 600.0
    connect_timeout: float = 5.0
    max_inline_bytes: int = _MAX_INLINE_BYTES

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be blank when provided")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be a positive finite number")
        if self.max_inline_bytes <= 0:
            raise ValueError("max_inline_bytes must be positive")


class GeminiSTTProvider(STTProvider):
    """Google Gemini speech-to-text provider.

    Batch only — :attr:`supports_streaming` is ``False``, so a
    :class:`~roomkit.channels.voice.VoiceChannel` transcribes on ``SPEECH_END``
    rather than streaming partials. Given seconds of model latency, the honest
    placement is ``batch_mode=True`` (dictation, voicemail) or transcription of
    a finished recording, not live turn-taking.

    Two entry points:

    * :meth:`transcribe` — the ABC contract, returning flat text.
    * :meth:`transcribe_recording` — the whole :class:`Transcript`, with speaker
      turns and timestamps, and the one that accepts a file path.
    """

    def __init__(self, config: GeminiSTTConfig) -> None:
        self._config = config
        self._client: Any = None
        self._http: Any = None

    @property
    def name(self) -> str:
        return "GeminiSTT"

    @property
    def supports_streaming(self) -> bool:
        """The API takes a complete recording; there is no stream to open."""
        return False

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            # The client carries the connect/read split; see ``build_genai_client``
            # for why it cannot go on the request.
            self._client, self._http = build_genai_client(
                self._config.api_key, self._config, provider="GeminiSTTProvider"
            )
        return self._client

    def _files_config(self) -> dict[str, Any]:
        """Per-call options for the Files API.

        Those calls go through the SDK's classic request path, which hands
        httpx ``timeout=None`` (no timeout at all) unless ``HttpOptions.timeout``
        is set, so a stalled upload of a long recording never returned. That
        option is one value in milliseconds and cannot split the connect from
        the read, so this is the flat ``timeout`` budget.
        """
        return {"http_options": {"timeout": int(self._config.timeout * 1000)}}

    def _build_prompt(self) -> str:
        lines = [
            "Transcribe the recording verbatim.",
            "Return only what is spoken: do not summarise, translate, or comment.",
            "Timestamps are MM:SS measured from the start of the recording.",
        ]
        if self._config.diarize:
            lines.append(
                'Label each speaker "Speaker 1", "Speaker 2", and so on, '
                "in order of first appearance, and keep a label attached to the "
                "same voice throughout."
            )
        else:
            lines.append('The recording has one speaker; label every segment "Speaker 1".')
        if self._config.language:
            lines.append(f"The recording is in {self._config.language}.")
        if self._config.prompt:
            lines.append(self._config.prompt)
        return "\n".join(lines)

    @staticmethod
    def _mime_for(path: Path) -> str:
        """Pick a mime type the endpoint accepts for *path*.

        ``mimetypes`` answers ``audio/x-wav`` for ``.wav`` on some platforms and
        the service rejects it, so the guess is normalised rather than trusted.
        """
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed in SUPPORTED_MIME_TYPES:
            return guessed
        by_suffix = {
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
            ".m4a": "audio/m4a",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".opus": "audio/opus",
            ".flac": "audio/flac",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
        }
        mime = by_suffix.get(path.suffix.lower())
        if mime is None:
            raise ValueError(
                f"Cannot infer a supported audio mime type for {path.name}. "
                f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
            )
        return mime

    async def _audio_part(self, source: Any) -> tuple[dict[str, Any], str | None]:
        """Turn *source* into an audio content block.

        Returns the block and, when the recording was uploaded, the file name to
        delete afterwards.
        """
        if isinstance(source, str | Path):
            text = str(source)
            if text.startswith(("data:", "http://", "https://", "file://")):
                return await self._part_from_url(text)
            return await self._part_from_path(Path(source))

        url = getattr(source, "url", None)
        if url is not None:
            return await self._part_from_url(str(url))

        data = getattr(source, "data", None)
        if not isinstance(data, bytes):
            raise TypeError(f"Cannot transcribe {type(source).__name__}: no audio bytes found")
        if not data:
            raise ValueError("Cannot transcribe empty audio")
        return (
            {
                "type": "audio",
                "data": base64.b64encode(data).decode(),
                "mime_type": _PCM_MIME_TYPE,
                "sample_rate": getattr(source, "sample_rate", 16000),
                "channels": getattr(source, "channels", 1),
            },
            None,
        )

    async def _part_from_path(self, path: Path) -> tuple[dict[str, Any], str | None]:
        mime = self._mime_for(path)
        try:
            info = await asyncio.to_thread(path.stat)
        except OSError as exc:
            raise FileNotFoundError(f"No such recording: {path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise FileNotFoundError(f"Not a recording file: {path}")

        size = info.st_size
        if size <= self._config.max_inline_bytes:
            data = await asyncio.to_thread(path.read_bytes)
            return (
                {
                    "type": "audio",
                    "data": base64.b64encode(data).decode(),
                    "mime_type": mime,
                },
                None,
            )

        logger.debug("Uploading %s (%d bytes) through the Files API", path.name, size)
        uploaded = await self._get_client().aio.files.upload(
            file=str(path), config=self._files_config()
        )
        # The upload guesses its own mime and can answer ``audio/x-wav``, which
        # the interactions endpoint rejects — send the normalised one.
        return ({"type": "audio", "uri": uploaded.uri, "mime_type": mime}, uploaded.name)

    async def _part_from_url(self, url: str) -> tuple[dict[str, Any], str | None]:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            mime = header[5:].split(";", 1)[0] or "audio/wav"
            if mime not in SUPPORTED_MIME_TYPES:
                raise ValueError(f"Unsupported audio mime type in data URL: {mime}")
            return ({"type": "audio", "data": payload, "mime_type": mime}, None)

        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            return await self._part_from_path(Path(parsed.path or url))
        if parsed.hostname == _FILES_API_HOST:
            return ({"type": "audio", "uri": url, "mime_type": "audio/wav"}, None)

        raise ValueError(
            f"GeminiSTT will not fetch {parsed.scheme}://{parsed.hostname} — the provider "
            "does not dereference arbitrary URLs. Pass a local path, raw audio, or a "
            "Files API URI."
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe a complete recording.

        Args:
            audio: Audio content (``data:`` URL, local path or Files API URI),
                raw audio chunk, or audio frame.

        Returns:
            TranscriptionResult whose ``text`` carries the spoken words and
            whose ``language`` carries what the model identified. Speaker turns
            and timestamps are dropped by this shape — call
            :meth:`transcribe_recording` for those.
        """
        transcript = await self.transcribe_recording(audio)
        return TranscriptionResult(
            text=transcript.plain_text,
            is_final=True,
            language=transcript.language or None,
        )

    async def transcribe_recording(self, source: Any) -> Transcript:
        """Transcribe a recording into speaker turns.

        Args:
            source: A path to a recording (``str`` or ``Path``), an
                ``AudioContent``, an ``AudioChunk`` or an ``AudioFrame``. Files
                larger than ``max_inline_bytes`` are uploaded through the Files
                API and deleted afterwards.

        Returns:
            The :class:`Transcript` — detected language, and one segment per
            speaker turn.

        Raises:
            RuntimeError: The model answered without a usable transcript.
        """
        part, uploaded_name = await self._audio_part(source)
        try:
            interaction = await self._get_client().aio.interactions.create(
                model=self._config.model,
                input=[part, {"type": "text", "text": self._build_prompt()}],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _TRANSCRIPT_SCHEMA,
                },
                # No per-request ``timeout``: the SDK would flatten it to one
                # float; the connect/read split is on the client (``_get_client``).
            )
        finally:
            if uploaded_name is not None:
                await self._delete_upload(uploaded_name)

        payload = getattr(interaction, "output_text", None)
        if not payload:
            raise RuntimeError(
                f"Gemini STT returned no transcript "
                f"(status={getattr(interaction, 'status', None)})"
            )
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini STT returned a transcript that is not JSON") from exc

        segments = [
            TranscriptSegment(
                speaker=str(item.get("speaker", "Speaker 1")),
                start=str(item.get("start", "")),
                end=str(item.get("end", "")),
                text=str(item.get("text", "")),
            )
            for item in parsed.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        return Transcript(language=str(parsed.get("language", "")), segments=segments)

    async def _delete_upload(self, name: str) -> None:
        """Remove an uploaded recording. Failing to is not worth an exception —
        the Files API expires uploads on its own."""
        try:
            await self._get_client().aio.files.delete(name=name, config=self._files_config())
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not delete uploaded recording %s", name, exc_info=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the genai client's connection pool and drop the reference."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        if client is None:
            return
        try:
            await client.aio.aclose()
            # The SDK leaves a client it was given open; it is ours to close.
            if http is not None:
                await http.aclose()
        except Exception:  # pragma: no cover - transport already gone
            logger.debug("GeminiSTT client close failed", exc_info=True)
