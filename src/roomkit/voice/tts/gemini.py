"""Google Gemini text-to-speech provider.

Gemini TTS is a *generative* speech model, not a conventional voice engine: the
prompt it receives is an instruction, so a natural-language direction ("Read
this cheerfully") can steer delivery. That expressiveness costs latency —
measured time-to-first-audio is seconds, not milliseconds (see
:class:`GeminiTTSConfig`) — which makes this the right provider for prompts,
announcements, and generated audio messages, and the wrong one for live
turn-taking. For conversational voice, use a low-latency engine
(:mod:`~roomkit.voice.tts.elevenlabs`, :mod:`~roomkit.voice.tts.gradium`) or
Gemini's speech-to-speech path
(:class:`~roomkit.providers.gemini.realtime.GeminiLiveProvider`), which sidesteps
the text round trip entirely.

Output is fixed by the API at 24 kHz, 16-bit, mono PCM. The request accepts a
``sample_rate`` field, but the service ignores it and always answers at 24 kHz,
so this provider does not expose the knob — attach a resampler stage to the
outbound pipeline when the transport needs another rate.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.providers.gemini.voices import VOICES
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.realtime.provider import VoiceInfo

logger = logging.getLogger(__name__)

GEMINI_TTS_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)
"""TTS models the Gemini API serves, verified against ``models.list`` 2026-08-06.

Only ``gemini-3.1-flash-tts-preview`` streams audio incrementally; the 2.5
models answer with the whole clip in one delta. The ``native-audio`` models are
absent on purpose — they speak over the Live (bidi) API, which
:class:`~roomkit.providers.gemini.realtime.GeminiLiveProvider` covers.
"""

OUTPUT_SAMPLE_RATE = 24000
"""Sample rate of every Gemini TTS response — fixed by the service."""

_OUTPUT_CHANNELS = 1
_AUDIO_FORMAT = "pcm_s16le"


@dataclass
class GeminiTTSConfig:
    """Configuration for the Gemini TTS provider.

    Args:
        api_key: Gemini API key (``GEMINI_API_KEY``).
        model: One of :data:`GEMINI_TTS_MODELS`. The default is the only model
            that streams incrementally, which is what lets playback start
            before the whole clip is generated.
        voice: Prebuilt voice name — see :meth:`GeminiTTSProvider.available_voices`.
        language: Optional BCP-47 hint (e.g. ``"fr-CA"``). Left unset, the model
            infers the language from the text.
        style_prompt: Natural-language delivery direction (e.g. ``"Read this in
            a calm, reassuring voice"``). The API has no style field — style is
            only expressible inside the prompt — so this is written as a
            labelled ``Delivery direction:`` line above the ``Transcript:``
            label in the same ``input`` string, which is what keeps the model
            reciting the transcript instead of the direction. Gemini 3.1 is a
            preview model and can occasionally read directions aloud anyway.
            For cues that steer a word or a phrase rather than the whole
            utterance, put audio tags such as ``[laughs]`` or ``[whispers]``
            inline in the text itself.
        timeout: Per-request timeout in seconds. Generous by design: measured
            time-to-first-audio for a one-sentence prompt ranges from ~1.2 s to
            ~8 s on the default model, and long text is slower still.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    model: str = "gemini-3.1-flash-tts-preview"
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

    def _build_prompt(self, text: str) -> str:
        """Build an explicit direction/transcript prompt for reliable recitation."""
        lines = [
            "Synthesize speech from the transcript below.",
            "Speak only the transcript; do not read these instructions or labels aloud.",
        ]
        if self._config.style_prompt:
            lines.append(f"Delivery direction: {self._config.style_prompt}")
        lines.extend(("Transcript:", text))
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
    def _decode_audio(
        data: str, sample_rate: int | None, channels: int | None
    ) -> tuple[bytes, int, int]:
        """Decode and validate service audio before exposing it downstream."""
        try:
            pcm = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("Gemini TTS returned invalid base64 audio") from exc

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

    async def _create(self, text: str, voice: str | None, *, stream: bool) -> Any:
        return await self._get_client().aio.interactions.create(
            model=self._config.model,
            input=self._build_prompt(text),
            stream=stream,
            # Audio ``mime_type`` and ``delivery`` are rejected by the service;
            # ``type`` is the only field it accepts here.
            response_format={"type": "audio"},
            generation_config=self._generation_config(voice),
            # No per-request ``timeout``: the SDK would flatten it to one float;
            # the connect/read split is on the client (``_get_client``).
        )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize the whole text in one request.

        Args:
            text: Text to speak. Must not be blank.
            voice: Prebuilt voice name overriding the configured one.

        Returns:
            AudioContent holding a WAV ``data:`` URL.

        Raises:
            ValueError: *text* is empty or whitespace.
            RuntimeError: The interaction completed without audio.
        """
        from roomkit.models.event import AudioContent as AudioContentModel

        if not text.strip():
            raise ValueError("GeminiTTS.synthesize() requires non-empty text")

        interaction = await self._create(text, voice, stream=False)
        audio = getattr(interaction, "output_audio", None)
        if audio is None or not audio.data:
            raise RuntimeError(
                f"Gemini TTS returned no audio (status={getattr(interaction, 'status', None)})"
            )

        pcm, sample_rate, channels = self._decode_audio(
            audio.data, audio.sample_rate, audio.channels
        )
        wav = wrap_wav(pcm, sample_rate, channels)

        return AudioContentModel(
            url=f"data:audio/wav;base64,{base64.b64encode(wav).decode()}",
            mime_type="audio/wav",
            transcript=text,
            duration_seconds=len(pcm) / 2 / channels / sample_rate,
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
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
