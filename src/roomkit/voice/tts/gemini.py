"""Google Gemini text-to-speech provider.

Gemini TTS is a *generative* speech model, not a conventional voice engine: it
performs the text rather than reading it, so a natural-language direction
("calm and reassuring") and inline audio tags steer the delivery. That
expressiveness costs latency — measured time-to-first-audio is seconds, not
milliseconds (see :class:`GeminiTTSConfig`) — which makes this the right
provider for prompts, announcements, and generated audio messages, and the
wrong one for live turn-taking. For conversational voice, use a low-latency
engine (:mod:`~roomkit.voice.tts.elevenlabs`, :mod:`~roomkit.voice.tts.gradium`)
or Gemini's speech-to-speech path
(:class:`~roomkit.providers.gemini.realtime.GeminiLiveProvider`), which sidesteps
the text round trip entirely.

Two request contracts live here, chosen by model family:

* The 3.1 and 2.5 models have no style field, so a direction can only travel
  inside the prompt. The text goes out as a ``<transcript>`` block under
  instructions that keep the model reciting the transcript rather than the
  direction; the text cannot close the block, so it can neither cut the
  transcript short nor open another (RFC §6.4).
* From 3.8 on, the model reads those instructions aloud (measured 2026-09-27:
  half the runs on ``gemini-3.8-flash-tts``, every run on the Lite model), so
  the text goes out alone and the direction rides as ``speech_metadata``.

Audio is 24 kHz, 16-bit, mono PCM on every model. The request accepts a
``sample_rate`` field, but the service ignores it and always answers at 24 kHz,
so this provider does not expose the knob — attach a resampler stage to the
outbound pipeline when the transport needs another rate. A streamed answer is
bare PCM on every model. A non-streamed one is bare PCM up to 3.1 and, from
3.8, a whole WAV file with a C2PA content-credentials chunk after its audio,
which :meth:`GeminiTTSProvider.synthesize` returns as it came.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit._text import fence
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.providers.gemini.voices import VOICES, voice_info_from_catalog
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import wav_duration_seconds, wrap_wav
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.voices import (
    DialogueTurn,
    VoiceInfo,
    check_dialogue,
    dialogue_transcript,
    filter_voices,
)

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger(__name__)

GEMINI_TTS_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)
"""TTS models the Gemini API serves, verified against ``models.list`` 2026-09-27.

The 3.8 and 3.1 models stream audio incrementally; the 2.5 models answer with
the whole clip in one delta. The ``native-audio`` models are absent on
purpose — they speak over the Live (bidi) API, which
:class:`~roomkit.providers.gemini.realtime.GeminiLiveProvider` covers.
"""

OUTPUT_SAMPLE_RATE = 24000
"""Sample rate of every Gemini TTS response — fixed by the service."""

_OUTPUT_CHANNELS = 1
_CATALOG_PAGE_SIZE = 1000
"""The largest page ``voices.list`` serves; the whole catalog is three pages."""

_DIALOGUE_SPEAKERS = 2
"""Speakers one Gemini TTS request can voice, per Google's speech-generation
guide; verified with two on ``gemini-3.8-flash-tts`` on 2026-09-27."""
_AUDIO_FORMAT = "pcm_s16le"

_PROMPTED_MODEL_PREFIXES = ("gemini-2.", "gemini-3.1-")
"""Model families that take a delivery direction only inside the prompt.

Every other id gets the 3.8 contract: the bare transcript, its direction as
``speech_metadata``. The list is closed on the old side on purpose — Google
ships no new model in these families, while an id newer than this module is
far likelier to follow 3.8 than 3.1, and the 3.1 prompt is what 3.8 reads aloud.
"""

_WAV_MIME_TYPES = frozenset({"audio/wav", "audio/wave", "audio/x-wav"})


def _uses_instruction_prompt(model: str) -> bool:
    return model.startswith(_PROMPTED_MODEL_PREFIXES)


def _is_wav(mime_type: str | None, audio: bytes) -> bool:
    """Whether the service answered a whole WAV file rather than bare PCM.

    The declared type decides, and the RIFF magic backs it up: a WAV file
    mistaken for PCM does not fail, it plays its header and trailing chunks as
    noise.
    """
    base_type = (mime_type or "").split(";", 1)[0].strip().lower()
    return base_type in _WAV_MIME_TYPES or (audio[:4] == b"RIFF" and audio[8:12] == b"WAVE")


@dataclass
class GeminiTTSConfig:
    """Configuration for the Gemini TTS provider.

    Args:
        api_key: Gemini API key (``GEMINI_API_KEY``).
        model: One of :data:`GEMINI_TTS_MODELS`. The default is Google's
            replacement for ``gemini-3.1-flash-tts-preview``;
            ``gemini-3.8-flash-lite-tts`` is the lower-latency,
            higher-throughput variant. Both stream incrementally, which is what
            lets playback start before the whole clip is generated.
        voice: Prebuilt voice name — see :meth:`GeminiTTSProvider.available_voices`.
        language: Optional BCP-47 hint (e.g. ``"fr-CA"``). Left unset, the model
            infers the language from the text.
        style_prompt: Natural-language direction for the whole utterance (e.g.
            ``"calm and reassuring"``). From 3.8 on it is sent as the ``style``
            of the text's ``speech_metadata``, a field the model takes as
            direction, never as words. The 3.1 and 2.5 models have no such
            field, so there it is written as a labelled ``Delivery direction:``
            line above the ``Transcript:`` label in the same ``input`` string,
            which is what keeps the model reciting the transcript instead of
            the direction; the 3.1 preview can occasionally read it aloud
            anyway. Every model performs a delivery cue written in the text
            itself (an audio tag, a sentence such as "whisper this"), whatever
            frame holds it (measured 2026-10-08, 3.8 included): remove such
            cues from text you do not trust in a ``BEFORE_TTS`` hook. For cues
            that steer a word or a phrase rather than the
            whole utterance, put audio tags inline in the text itself. Google
            documents ``<laugh>``, ``<sigh>`` and ``<short pause>`` for 3.8
            and ``[laughs]``, ``[whispers]`` for 3.1; the 3.8 models perform
            both spellings (measured 2026-09-27).
        timeout: Per-request timeout in seconds. Generous by design: measured
            time-to-first-audio for a one-sentence prompt is ~1.1 to ~1.6 s on
            the default model and ~0.7 s on the Lite one (2026-09-27), was up
            to ~8 s on 3.1, and long text is slower still.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.8-flash-tts"
    voice: str = "Kore"
    language: str | None = None
    style_prompt: str | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.voice.strip():
            raise ValueError("voice must not be empty")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be blank when provided")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be a positive finite number")


class GeminiTTSProvider(TTSProvider):
    """Google Gemini text-to-speech provider.

    Supports:

    * :meth:`synthesize` — one request, full clip as WAV.
    * :meth:`synthesize_stream` — audio deltas forwarded as they arrive.

    Streaming *text input* is not supported: the API takes a complete prompt,
    so there is no seam to feed token deltas into. A voice channel therefore
    delivers through :meth:`synthesize_stream` on the complete reply.
    """

    def __init__(self, config: GeminiTTSConfig) -> None:
        self._config = config
        self._client: Any = None
        self._http: Any = None

    @property
    def name(self) -> str:
        return "GeminiTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """The 30 prebuilt voices, shared with Gemini Live native audio."""
        return list(VOICES)

    async def list_voices(
        self,
        *,
        language: str | None = None,
        gender: str | None = None,
        query: str | None = None,
    ) -> list[VoiceInfo]:
        """Google's voice catalog, the account's custom voices first.

        The whole catalog is read — 2,089 voices in three pages, 0.4 s on
        2026-09-27 — and filtered here: the service's own filters mean
        something else (its ``language_code`` wants an exact tag, ``fr`` finds
        nothing, and its ``search`` for ``Kore`` answers Korean voices), and
        RFC §12.2 wants every provider to filter alike. Any ``id`` returned
        is accepted as ``voice``.
        """
        client = self._get_client()
        voices: list[VoiceInfo] = []
        page_token: str | None = None
        while True:
            page = await client.aio.voices.list(
                page_size=_CATALOG_PAGE_SIZE, page_token=page_token
            )
            voices.extend(voice_info_from_catalog(voice) for voice in page.voices or [])
            page_token = page.next_page_token
            if not page_token:
                break
        return filter_voices(voices, language=language, gender=gender, query=query)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            # The client carries the connect/read split; see ``build_genai_client``
            # for why it cannot go on the request.
            built = build_genai_client(
                self._config, provider="GeminiTTSProvider", api_key=self._config.api_key
            )
            self._client, self._http = built.client, built.http
        return self._client

    def _build_input(self, text: str) -> str | list[dict[str, Any]]:
        """Shape the ``input`` so the configured model speaks *text* and only that."""
        if _uses_instruction_prompt(self._config.model):
            return self._build_prompt(text)
        style = self._config.style_prompt
        if not style:
            return text
        return [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [{"type": "speech_metadata", "style": style}],
                    }
                ],
            }
        ]

    def _build_prompt(self, text: str) -> str:
        """Build an explicit direction/transcript prompt for reliable recitation:
        the text in a block it cannot close, so a text holding a
        ``Delivery direction:`` or ``Transcript:`` line of its own neither cuts
        the transcript short nor fails the request (measured 2026-10-08)."""
        lines = [
            "Synthesize speech from the transcript below.",
            "Speak only the text inside the transcript block, exactly as written; "
            "do not read these instructions, labels or tags aloud.",
        ]
        if self._config.style_prompt:
            lines.append(f"Delivery direction: {self._config.style_prompt}")
        lines.append(fence("transcript", text))
        return "\n".join(lines)

    def _generation_config(self, voice: str | None) -> dict[str, Any]:
        selected_voice = voice or self._config.voice
        if not selected_voice.strip():
            raise ValueError("voice must not be empty")
        speech: dict[str, Any] = {"voice": selected_voice}
        if self._config.language:
            speech["language"] = self._config.language
        return {"speech_config": [speech]}

    @staticmethod
    def _decode_base64(data: str) -> bytes:
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("Gemini TTS returned invalid base64 audio") from exc

    @classmethod
    def _decode_audio(
        cls, data: str, sample_rate: int | None, channels: int | None
    ) -> tuple[bytes, int, int]:
        """Decode and validate service PCM before exposing it downstream."""
        return cls._check_pcm(cls._decode_base64(data), sample_rate, channels)

    @staticmethod
    def _check_pcm(
        pcm: bytes, sample_rate: int | None, channels: int | None
    ) -> tuple[bytes, int, int]:
        effective_rate = sample_rate or OUTPUT_SAMPLE_RATE
        effective_channels = channels or _OUTPUT_CHANNELS
        if (
            not isinstance(effective_rate, int)
            or isinstance(effective_rate, bool)
            or effective_rate <= 0
        ):
            raise RuntimeError(f"Gemini TTS returned invalid sample rate: {effective_rate!r}")
        if (
            not isinstance(effective_channels, int)
            or isinstance(effective_channels, bool)
            or effective_channels <= 0
        ):
            raise RuntimeError(
                f"Gemini TTS returned invalid channel count: {effective_channels!r}"
            )
        if len(pcm) % (2 * effective_channels):
            raise RuntimeError("Gemini TTS returned a truncated PCM frame")
        return pcm, effective_rate, effective_channels

    @classmethod
    def _as_wav(cls, audio: Any) -> tuple[bytes, float]:
        """Return a non-streamed answer as a WAV file, with its duration in seconds.

        A WAV file the service answered (3.8 on) is kept as it came, provenance
        chunk included; bare PCM (up to 3.1) is wrapped in a header.
        """
        raw = cls._decode_base64(audio.data)
        if not _is_wav(getattr(audio, "mime_type", None), raw):
            pcm, sample_rate, channels = cls._check_pcm(raw, audio.sample_rate, audio.channels)
            return wrap_wav(pcm, sample_rate, channels), len(pcm) / 2 / channels / sample_rate
        try:
            return raw, wav_duration_seconds(raw)
        except ValueError as exc:
            raise RuntimeError(f"Gemini TTS returned an unreadable WAV file: {exc}") from exc

    async def _create(self, text: str, voice: str | None, *, stream: bool) -> Any:
        return await self._get_client().aio.interactions.create(
            model=self._config.model,
            input=self._build_input(text),
            stream=stream,
            # ``type`` only: the 3.1 models answer 400 to ``mime_type`` and
            # ``delivery``. Each model's default format (bare PCM, or a WAV
            # file from 3.8 on) is handled on the way back instead.
            response_format={"type": "audio"},
            generation_config=self._generation_config(voice),
            # No per-request ``timeout``: the SDK would flatten it to one float;
            # the connect/read split is on the client (``_get_client``).
        )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    @property
    def max_dialogue_speakers(self) -> int:
        """Two from 3.8 on; none on the 3.1 and 2.5 models, whose prompt-only
        contract this provider does not extend to several speakers."""
        return 0 if _uses_instruction_prompt(self._config.model) else _DIALOGUE_SPEAKERS

    async def synthesize_dialogue(
        self, turns: Sequence[DialogueTurn], voices: Mapping[str, str]
    ) -> AudioContent:
        """Voice a scripted exchange of up to two speakers in one clip.

        Each turn goes out as its own text item, its speaker (and its
        ``style``, when set) riding as ``speech_metadata``, and each speaker
        is bound to its voice in ``speech_config``. ``style_prompt`` does not
        apply: a dialogue directs each turn on its own.

        Raises:
            NotImplementedError: The configured model voices no dialogue.
            ValueError: More than two speakers, or a speaker ``voices`` does
                not map (both before any request).
            RuntimeError: The interaction completed without audio.
        """
        speakers = check_dialogue(
            turns, voices, max_speakers=self.max_dialogue_speakers, provider=self.name
        )
        speech_config: list[dict[str, Any]] = []
        for speaker in speakers:
            entry: dict[str, Any] = {"speaker": speaker, "voice": voices[speaker]}
            if self._config.language:
                entry["language"] = self._config.language
            speech_config.append(entry)
        interaction = await self._get_client().aio.interactions.create(
            model=self._config.model,
            input=[{"type": "user_input", "content": [_dialogue_item(t) for t in turns]}],
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
        )
        return self._audio_content(interaction, dialogue_transcript(turns))

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize the whole text in one request.

        Args:
            text: Text to speak. Must not be blank.
            voice: Prebuilt voice name overriding the configured one.

        Returns:
            AudioContent holding a WAV ``data:`` URL. A WAV file the service
            answered (3.8 on) is passed through unchanged, its C2PA
            content-credentials chunk included; bare PCM is wrapped in one.

        Raises:
            ValueError: *text* is empty or whitespace.
            RuntimeError: The interaction completed without audio, or with
                audio that does not decode.
        """
        if not text.strip():
            raise ValueError("GeminiTTS.synthesize() requires non-empty text")

        interaction = await self._create(text, voice, stream=False)
        return self._audio_content(interaction, text)

    @classmethod
    def _audio_content(cls, interaction: Any, transcript: str) -> AudioContent:
        """A non-streamed answer as a WAV ``data:`` URL with its duration."""
        from roomkit.models.event import AudioContent as AudioContentModel

        audio = getattr(interaction, "output_audio", None)
        if audio is None or not audio.data:
            raise RuntimeError(
                f"Gemini TTS returned no audio (status={getattr(interaction, 'status', None)})"
            )
        wav, duration = cls._as_wav(audio)
        return AudioContentModel(
            url=f"data:audio/wav;base64,{base64.b64encode(wav).decode()}",
            mime_type="audio/wav",
            transcript=transcript,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio deltas as the service emits them.

        Blank text yields nothing but the terminating chunk — TTS filters can
        strip a reply down to whitespace, and that is not worth a round trip
        the service would reject.
        """
        if not text.strip():
            yield AudioChunk(
                data=b"",
                sample_rate=OUTPUT_SAMPLE_RATE,
                channels=_OUTPUT_CHANNELS,
                format=_AUDIO_FORMAT,
                is_final=True,
            )
            return

        stream = await self._create(text, voice, stream=True)

        sample_rate = OUTPUT_SAMPLE_RATE
        channels = _OUTPUT_CHANNELS
        async for event in stream:
            delta = getattr(event, "delta", None)
            if delta is None or getattr(delta, "type", None) != "audio" or not delta.data:
                continue
            pcm, sample_rate, channels = self._decode_audio(
                delta.data, delta.sample_rate or sample_rate, delta.channels or channels
            )
            yield AudioChunk(
                data=pcm,
                sample_rate=sample_rate,
                channels=channels,
                format=_AUDIO_FORMAT,
                is_final=False,
            )

        yield AudioChunk(
            data=b"",
            sample_rate=sample_rate,
            channels=channels,
            format=_AUDIO_FORMAT,
            is_final=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the genai client's connection pool and drop the reference."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)


def _dialogue_item(turn: DialogueTurn) -> dict[str, Any]:
    """One turn as a text item, its speaker and style as ``speech_metadata``."""
    metadata: dict[str, Any] = {"type": "speech_metadata", "speaker": turn.speaker}
    if turn.style:
        metadata["style"] = turn.style
    return {"type": "text", "text": turn.text, "annotations": [metadata]}
