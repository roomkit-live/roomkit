"""AzureSpeechTTSProvider against a fake Azure Speech endpoint (httpx.MockTransport).

The requests follow Microsoft's MAI-Voice page (Learn, 2026-09-30) and the
long-standing Azure Speech REST contract. No test here has met the live service.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from xml.dom import minidom

import httpx
import pytest

from roomkit.voice.tts.azure_speech import (
    OUTPUT_FORMATS,
    AzureSpeechTTSConfig,
    AzureSpeechTTSError,
    AzureSpeechTTSProvider,
    build_ssml,
    synthesis_url,
)
from roomkit.voice.tts.context import TTSContextLevel


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeAzureSpeech:
    """Renders every request as ``audio``, or answers ``status``."""

    def __init__(self, audio: list[bytes] | None = None, status: int = 200) -> None:
        self.audio = audio if audio is not None else [b"\x01\x00\x02\x00"]
        self.status = status
        self.requests: list[httpx.Request] = []
        self.streams: list[_Chunks] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="Quota exceeded")
        self.streams.append(_Chunks(self.audio))
        return httpx.Response(200, stream=self.streams[-1])


def _provider(fake: FakeAzureSpeech, **config: object) -> AzureSpeechTTSProvider:
    config.setdefault("region", "swedencentral")
    provider = AzureSpeechTTSProvider(AzureSpeechTTSConfig(api_key="az-key", **config))  # type: ignore[arg-type]
    real = provider._get_client()
    provider._client = httpx.AsyncClient(
        headers=real.headers, timeout=real.timeout, transport=httpx.MockTransport(fake.handle)
    )
    return provider


async def _pcm(provider: AzureSpeechTTSProvider, text: str = "Hi.", **kwargs: object) -> bytes:
    return b"".join([c.data async for c in provider.synthesize_stream(text, **kwargs)])  # type: ignore[arg-type]


def _ssml(request: httpx.Request) -> minidom.Document:
    """The request body, parsed: it must be well-formed XML."""
    return minidom.parseString(request.content)


class TestRender:
    async def test_a_render_posts_ssml_for_raw_pcm(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake))

        [request] = fake.requests
        assert str(request.url) == (
            "https://swedencentral.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        assert request.headers["Ocp-Apim-Subscription-Key"] == "az-key"
        assert request.headers["Content-Type"] == "application/ssml+xml"
        assert request.headers["X-Microsoft-OutputFormat"] == "raw-24khz-16bit-mono-pcm"
        speak = _ssml(request).documentElement
        assert speak.getAttribute("xml:lang") == "en-US"
        [voice] = speak.getElementsByTagName("voice")
        assert voice.getAttribute("name") == "en-US-Harper:MAI-Voice-2.1-Flash"
        assert voice.firstChild.data == "Hi."

    async def test_the_voice_names_the_language(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake), voice="fr-FR-Soleil:MAI-Voice-2.1-Flash")

        speak = _ssml(fake.requests[0]).documentElement
        assert speak.getAttribute("xml:lang") == "fr-FR"
        assert speak.getElementsByTagName("voice")[0].getAttribute("name") == (
            "fr-FR-Soleil:MAI-Voice-2.1-Flash"
        )

    async def test_a_configured_language_wins(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake, voice="fr-CA-SylvieNeural", language="fr-CA"))

        assert _ssml(fake.requests[0]).documentElement.getAttribute("xml:lang") == "fr-CA"

    async def test_a_style_wraps_the_text(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake, style="joyful"), "Great news!")

        [express] = _ssml(fake.requests[0]).getElementsByTagName("mstts:express-as")
        assert express.getAttribute("style") == "joyful"
        assert express.firstChild.data == "Great news!"

    async def test_a_rate_wraps_the_text_inside_the_style(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake, style="joyful", rate="+15%"), "Great news!")

        document = _ssml(fake.requests[0])
        [express] = document.getElementsByTagName("mstts:express-as")
        [prosody] = express.getElementsByTagName("prosody")
        assert prosody.getAttribute("rate") == "+15%"
        assert prosody.firstChild.data == "Great news!"

    async def test_no_rate_leaves_the_voice_its_own_pace(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake))

        assert _ssml(fake.requests[0]).getElementsByTagName("prosody") == []

    async def test_text_is_spoken_as_written_never_read_as_markup(self) -> None:
        fake = FakeAzureSpeech()
        text = 'Use <break time="5s"/> & "quotes" </voice><voice name="x">'

        await _pcm(_provider(fake), text)

        document = _ssml(fake.requests[0])
        assert document.getElementsByTagName("break") == []
        [voice] = document.getElementsByTagName("voice")
        assert voice.firstChild.data == text

    async def test_characters_xml_cannot_carry_are_dropped(self) -> None:
        fake = FakeAzureSpeech()

        await _pcm(_provider(fake), "Hi\x00 there\x1b.")

        assert _ssml(fake.requests[0]).getElementsByTagName("voice")[0].firstChild.data == (
            "Hi there."
        )

    @pytest.mark.parametrize("text", ["", "   ", "\x00\x07"])
    async def test_nothing_to_speak_makes_no_request(self, text: str) -> None:
        fake = FakeAzureSpeech()

        chunks = [c async for c in _provider(fake).synthesize_stream(text)]

        assert fake.requests == []
        assert [(c.data, c.is_final) for c in chunks] == [(b"", True)]

    @pytest.mark.parametrize("rate", sorted(OUTPUT_FORMATS))
    async def test_the_rate_picks_the_output_format(self, rate: int) -> None:
        fake = FakeAzureSpeech()

        chunks = [c async for c in _provider(fake, sample_rate=rate).synthesize_stream("Hi.")]

        assert fake.requests[0].headers["X-Microsoft-OutputFormat"] == OUTPUT_FORMATS[rate]
        assert all(c.sample_rate == rate for c in chunks)

    async def test_every_chunk_holds_whole_samples(self) -> None:
        fake = FakeAzureSpeech(audio=[b"\x01", b"\x02\x03", b"\x04\x05\x06"])

        chunks = [c async for c in _provider(fake).synthesize_stream("Hi.")]

        assert b"".join(c.data for c in chunks) == b"\x01\x02\x03\x04\x05\x06"
        assert all(len(c.data) % 2 == 0 for c in chunks)
        assert chunks[-1].is_final and chunks[-1].data == b""

    async def test_closing_the_stream_closes_the_render(self) -> None:
        """A barge-in closes the stream: the HTTP response goes with it."""
        fake = FakeAzureSpeech(audio=[b"\x00\x00"] * 50)
        stream = _provider(fake).synthesize_stream("Hi.")

        await anext(stream)
        await stream.aclose()

        assert fake.streams[0].closed

    async def test_synthesize_returns_a_wav(self) -> None:
        fake = FakeAzureSpeech(audio=[b"\x01\x00" * 2400])

        audio = await _provider(fake).synthesize("Hi.")

        wav = base64.b64decode(audio.url.split(",", 1)[1])
        assert wav[:4] == b"RIFF"
        assert audio.duration_seconds == pytest.approx(0.1)
        assert audio.transcript == "Hi."

    @pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
    async def test_a_refused_render_raises(self, status: int, retryable: bool) -> None:
        fake = FakeAzureSpeech(status=status)

        with pytest.raises(AzureSpeechTTSError, match="Quota exceeded") as info:
            await _pcm(_provider(fake))

        assert info.value.status_code == status
        assert info.value.retryable is retryable

    async def test_a_voice_without_locale_needs_a_language(self) -> None:
        fake = FakeAzureSpeech()

        with pytest.raises(ValueError, match="names no locale"):
            await _pcm(_provider(fake), voice="MAI-Voice-2.1-Flash")
        assert fake.requests == []


class TestConfig:
    def test_a_region_gives_the_regional_endpoint(self) -> None:
        assert synthesis_url("canadacentral", None) == (
            "https://canadacentral.tts.speech.microsoft.com/cognitiveservices/v1"
        )

    def test_an_endpoint_is_used_as_given(self) -> None:
        url = "https://my-resource.cognitiveservices.azure.com/tts/cognitiveservices/v1"
        assert synthesis_url(None, url) == url

    @pytest.mark.parametrize(
        ("region", "endpoint"),
        [(None, None), ("swedencentral", "https://x.example/v1"), ("Sweden Central", None)],
    )
    def test_exactly_one_valid_target(self, region: str | None, endpoint: str | None) -> None:
        with pytest.raises(ValueError):
            AzureSpeechTTSConfig(api_key="k", region=region, endpoint=endpoint)

    def test_a_plain_endpoint_is_refused_off_loopback(self) -> None:
        with pytest.raises(ValueError, match="https"):
            synthesis_url(None, "http://my-resource.example/cognitiveservices/v1")
        assert synthesis_url(None, "http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"

    def test_an_unsupported_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            AzureSpeechTTSConfig(api_key="k", region="swedencentral", sample_rate=32000)

    def test_a_voice_without_locale_and_no_language_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no locale"):
            AzureSpeechTTSConfig(api_key="k", region="swedencentral", voice="MAI-Voice-2.1")

    @pytest.mark.parametrize("rate", ["+20%", "-10%", "12.5%", "1.2", "fast", "x-slow", "default"])
    def test_rates_ssml_takes_are_accepted(self, rate: str) -> None:
        assert AzureSpeechTTSConfig(api_key="k", region="eastus", rate=rate).rate == rate

    @pytest.mark.parametrize("rate", ["", "20", "+20", "1000%", "faster", '+20%"><x'])
    def test_a_rate_ssml_does_not_take_is_refused(self, rate: str) -> None:
        with pytest.raises(ValueError, match="rate must be"):
            AzureSpeechTTSConfig(api_key="k", region="eastus", rate=rate)

    def test_a_language_that_is_no_tag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="BCP-47"):
            AzureSpeechTTSConfig(api_key="k", region="swedencentral", language='fr"><x')

    def test_the_key_is_required_and_stays_out_of_repr(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            AzureSpeechTTSConfig(api_key="", region="swedencentral")
        assert "s3cret" not in repr(AzureSpeechTTSConfig(api_key="s3cret", region="eastus"))

    def test_attribute_values_are_escaped(self) -> None:
        ssml = build_ssml("Hi.", voice='a"b', language="en-US", style="x'><y")
        document = minidom.parseString(ssml)

        assert document.getElementsByTagName("voice")[0].getAttribute("name") == 'a"b'
        assert document.getElementsByTagName("mstts:express-as")[0].getAttribute("style") == (
            "x'><y"
        )

    def test_no_conversation_context(self) -> None:
        provider = AzureSpeechTTSProvider(AzureSpeechTTSConfig(api_key="k", region="eastus"))
        assert provider.context_level is TTSContextLevel.NONE
        assert provider.default_voice == "en-US-Harper:MAI-Voice-2.1-Flash"
        assert provider.name == "AzureSpeechTTS"

    async def test_close_releases_the_client(self) -> None:
        provider = _provider(FakeAzureSpeech())

        await provider.close()

        assert provider._client is None

    def test_lazy_getters(self) -> None:
        from roomkit.voice import get_azure_speech_tts_config, get_azure_speech_tts_provider

        assert get_azure_speech_tts_provider() is AzureSpeechTTSProvider
        assert get_azure_speech_tts_config() is AzureSpeechTTSConfig
