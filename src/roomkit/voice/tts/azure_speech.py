"""Azure Speech text-to-speech provider: MAI-Voice and Azure's neural voices.

Azure Speech renders SSML posted to
``https://<region>.tts.speech.microsoft.com/cognitiveservices/v1``. The voice
name picks the model: ``en-US-Harper:MAI-Voice-2.1-Flash`` is Microsoft's MAI
voice Harper on MAI-Voice-2.1-Flash (the low-latency model), and
``fr-CA-SylvieNeural`` one of Azure's neural voices, so one provider serves both.

Each text is rendered on its own (no conversation context). Audio is mono
16-bit PCM at the configured rate, read as the response arrives. The MAI-Voice
models are in public preview (October 2026). Install with
``pip install roomkit[azure-speech]``.
"""

from __future__ import annotations

import contextlib
import html
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from roomkit.providers.ai.base import ProviderError
from roomkit.providers.utils import http_timeout
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.audio_utils import collect_wav_content
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.tts.context import TTSContext

logger = logging.getLogger("roomkit.voice.tts.azure_speech")

DEFAULT_VOICE = "en-US-Harper:MAI-Voice-2.1-Flash"

OUTPUT_FORMATS: dict[int, str] = {
    8000: "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    22050: "raw-22050hz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    44100: "raw-44100hz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}
"""The headerless PCM formats Azure Speech renders, by sample rate."""

_LOCALE = re.compile(r"^([a-z]{2,3}-[A-Z]{2})-")
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
_REGION = re.compile(r"^[a-z0-9]+$")
# Characters XML 1.0 cannot carry at all, escaped or not: dropped, they are
# never speech.
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class AzureSpeechTTSError(ProviderError):
    """A render Azure Speech refused or failed."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(
            message,
            retryable=status_code == 429 or status_code >= 500,
            provider="AzureSpeechTTS",
            status_code=status_code,
        )


@dataclass
class AzureSpeechTTSConfig:
    """Configuration for :class:`AzureSpeechTTSProvider`.

    Attributes:
        api_key: The Speech (or Foundry) resource's key.
        region: The resource's region identifier (``swedencentral``,
            ``canadacentral``…). Exactly one of ``region`` and ``endpoint``.
        endpoint: The full synthesis URL, for a custom domain or a proxy,
            in place of ``region``.
        voice: Voice name as Azure spells it: ``<locale>-<name>:<model>`` for
            a MAI voice, ``<locale>-<name>Neural`` for a neural one.
        language: The SSML ``xml:lang``. ``None`` takes the locale the voice
            name starts with; a voice name without one needs it set.
        style: A speaking style the voice supports (``joyful``,
            ``customer_call_center``…), wrapped around every text with
            ``mstts:express-as``. ``None`` speaks in the voice's default style.
        sample_rate: Output rate, one of :data:`OUTPUT_FORMATS`.
        timeout: Read budget of a render, in seconds.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
    """

    api_key: str = field(repr=False)
    region: str | None = None
    endpoint: str | None = None
    voice: str = DEFAULT_VOICE
    language: str | None = None
    style: str | None = None
    sample_rate: int = 24000
    timeout: float = 30.0
    connect_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("AzureSpeechTTSConfig.api_key is required")
        if not self.api_key.isascii():
            raise ValueError(
                "AzureSpeechTTSConfig.api_key holds a non-ASCII character (a placeholder "
                "such as '…' copied as is?): an HTTP header cannot carry it"
            )
        synthesis_url(self.region, self.endpoint)
        if self.sample_rate not in OUTPUT_FORMATS:
            raise ValueError(
                f"sample_rate must be one of {sorted(OUTPUT_FORMATS)}, got {self.sample_rate}"
            )
        if self.language is not None and not _LANGUAGE_TAG.match(self.language):
            raise ValueError(f"language must be a BCP-47 tag such as fr-CA, got {self.language!r}")
        ssml_language(self.voice, self.language)


def synthesis_url(region: str | None, endpoint: str | None) -> str:
    """The synthesis URL, from a region or given whole.

    Raises:
        ValueError: Neither or both are set, the region is not an identifier,
            or the endpoint is not https (plain http on a loopback host only).
    """
    if (region is None) == (endpoint is None):
        raise ValueError("set exactly one of region and endpoint")
    if region is not None:
        if not _REGION.match(region):
            raise ValueError(f"region must be an identifier such as swedencentral, got {region!r}")
        return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    url = httpx.URL(str(endpoint))
    if url.scheme != "https" and not (url.scheme == "http" and url.host in _LOOPBACK_HOSTS):
        raise ValueError("endpoint must use https: a plain connection would send the key in clear")
    return str(url)


def ssml_language(voice: str, language: str | None) -> str:
    """The ``xml:lang`` of a render: *language*, else the voice name's locale.

    Raises:
        ValueError: No language is set and the voice name carries no locale.
    """
    if language is not None:
        return language
    match = _LOCALE.match(voice)
    if match is None:
        raise ValueError(f"voice {voice!r} names no locale: set AzureSpeechTTSConfig.language")
    return match.group(1)


def build_ssml(text: str, *, voice: str, language: str, style: str | None) -> str:
    """The SSML document of one render, with every value escaped.

    *text* is spoken as written: markup in it is escaped, never interpreted,
    and characters XML cannot carry are dropped.
    """
    body = html.escape(_NOT_XML.sub("", text), quote=False)
    if style is not None:
        body = f'<mstts:express-as style="{html.escape(style)}">{body}</mstts:express-as>'
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        f'xml:lang="{html.escape(language)}">'
        f'<voice name="{html.escape(voice)}">{body}</voice></speak>'
    )


class AzureSpeechTTSProvider(TTSProvider):
    """Azure Speech voices, MAI-Voice included: each text rendered on its own, streamed."""

    def __init__(self, config: AzureSpeechTTSConfig) -> None:
        self._config = config
        self._url = synthesis_url(config.region, config.endpoint)
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "AzureSpeechTTS"

    @property
    def default_voice(self) -> str:
        return self._config.voice

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Ocp-Apim-Subscription-Key": self._config.api_key,
                    "User-Agent": "roomkit",
                },
                timeout=http_timeout(self._config),
            )
        return self._client

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream *text* as PCM at the configured rate; *context* is ignored.

        A text with nothing to speak makes no request.

        Raises:
            ValueError: *voice* names no locale and no language is configured.
            AzureSpeechTTSError: Azure Speech refused or failed the render.
        """
        rate = self._config.sample_rate
        name = voice or self._config.voice
        ssml = build_ssml(
            text,
            voice=name,
            language=ssml_language(name, self._config.language),
            style=self._config.style,
        )
        if _NOT_XML.sub("", text).strip():
            carry = b""
            async with self._render(ssml) as response:
                async for data in response.aiter_bytes():
                    pcm, carry = _whole_samples(carry + data)
                    if pcm:
                        yield AudioChunk(data=pcm, sample_rate=rate)
        yield AudioChunk(data=b"", sample_rate=rate, is_final=True)

    @contextlib.asynccontextmanager
    async def _render(self, ssml: str) -> AsyncIterator[httpx.Response]:
        """Open the streamed render of *ssml*; a refused render raises."""
        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": OUTPUT_FORMATS[self._config.sample_rate],
        }
        client = self._get_client()
        async with client.stream(
            "POST", self._url, content=ssml.encode(), headers=headers
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:200]
                raise AzureSpeechTTSError(
                    f"Azure Speech render failed ({response.status_code}): "
                    f"{detail or response.reason_phrase}",
                    status_code=response.status_code,
                )
            yield response

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Render *text* whole, as a WAV data URL."""
        stream = self.synthesize_stream(text, voice=voice)
        return await collect_wav_content(stream, text=text, sample_rate=self._config.sample_rate)

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _whole_samples(data: bytes) -> tuple[bytes, bytes]:
    """Split *data* into whole 16-bit samples and the odd byte left over, if any."""
    cut = len(data) - len(data) % 2
    return data[:cut], data[cut:]
