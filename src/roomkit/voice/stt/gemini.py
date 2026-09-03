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

Both paths live in :mod:`~roomkit.voice.stt.gemini_audio`; the transcript
models in :mod:`~roomkit.voice.stt.gemini_transcript`. This module keeps the
config, the client, the prompt and the transcription call, and remains the
public import path for all of them.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.voice.base import TranscriptionResult
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.gemini_audio import SUPPORTED_MIME_TYPES, audio_part, delete_upload
from roomkit.voice.stt.gemini_transcript import Transcript, TranscriptSegment

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import AudioChunk

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_MIME_TYPES",
    "GeminiSTTConfig",
    "GeminiSTTProvider",
    "Transcript",
    "TranscriptSegment",
]

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
        max_inline_bytes: Recordings larger than this are uploaded through the
            Files API instead of being inlined in the request.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.6-flash"
    language: str | None = None
    diarize: bool = True
    prompt: str | None = None
    timeout: float = 600.0
    max_inline_bytes: int = _MAX_INLINE_BYTES
    connect_timeout: float = 5.0

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
    # Client and prompt
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            # The client carries the connect/read split; see ``build_genai_client``
            # for why it cannot go on the request.
            built = build_genai_client(
                self._config, provider="GeminiSTTProvider", api_key=self._config.api_key
            )
            self._client, self._http = built.client, built.http
        return self._client

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
        part, uploaded_name = await audio_part(
            source, config=self._config, get_client=self._get_client
        )
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
                await delete_upload(uploaded_name, get_client=self._get_client)

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the genai client's connection pool and drop the reference."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)
