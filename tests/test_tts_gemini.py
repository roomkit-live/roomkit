"""Tests for the Google Gemini TTS provider.

The fake client mirrors what the live API actually returns (verified
2026-08-06): base64 ``audio/l16`` at 24 kHz mono, streamed as ``step.delta``
events whose ``delta.type`` is ``"audio"``, interleaved with lifecycle events
that carry no delta at all.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.gemini import (
    GEMINI_TTS_MODELS,
    OUTPUT_SAMPLE_RATE,
    GeminiTTSConfig,
    GeminiTTSProvider,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeAudio:
    data: str | None
    mime_type: str = "audio/l16"
    sample_rate: int | None = OUTPUT_SAMPLE_RATE
    channels: int | None = 1


@dataclass
class _FakeInteraction:
    output_audio: _FakeAudio | None
    status: str = "completed"


@dataclass
class _FakeDelta:
    data: str | None
    type: str = "audio"
    sample_rate: int | None = OUTPUT_SAMPLE_RATE
    channels: int | None = 1


@dataclass
class _FakeEvent:
    """A ``step.delta`` event, or a lifecycle event when *delta* is None."""

    delta: _FakeDelta | None = None


class _FakeInteractions:
    def __init__(self, pcm: bytes, frame_bytes: int = 1920) -> None:
        self._pcm = pcm
        self._frame_bytes = frame_bytes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        return _FakeInteraction(output_audio=_FakeAudio(data=base64.b64encode(self._pcm).decode()))

    async def _stream(self) -> AsyncIterator[_FakeEvent]:
        # interaction.created / step.start carry no delta.
        yield _FakeEvent()
        for off in range(0, len(self._pcm), self._frame_bytes):
            frame = self._pcm[off : off + self._frame_bytes]
            yield _FakeEvent(delta=_FakeDelta(data=base64.b64encode(frame).decode()))
        # A text delta on an audio interaction must be ignored, not decoded.
        yield _FakeEvent(delta=_FakeDelta(data="aGk=", type="text"))
        yield _FakeEvent()


class _FakeAio:
    def __init__(self, interactions: _FakeInteractions) -> None:
        self.interactions = interactions
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, pcm: bytes, frame_bytes: int = 1920) -> None:
        self.interactions = _FakeInteractions(pcm, frame_bytes)
        self.aio = _FakeAio(self.interactions)


def _pcm(seconds: float = 0.2) -> bytes:
    return b"\x01\x02" * int(OUTPUT_SAMPLE_RATE * seconds)


def _provider(pcm: bytes | None = None, **overrides: Any) -> tuple[GeminiTTSProvider, _FakeClient]:
    provider = GeminiTTSProvider(GeminiTTSConfig(api_key="test-key", **overrides))
    client = _FakeClient(pcm if pcm is not None else _pcm())
    provider._client = client
    return provider, client


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_and_default_voice(self) -> None:
        provider, _ = _provider()
        assert provider.name == "GeminiTTS"
        assert provider.default_voice == "Kore"

    def test_custom_voice_is_the_default_voice(self) -> None:
        provider, _ = _provider(voice="Sulafat")
        assert provider.default_voice == "Sulafat"

    def test_streaming_text_input_is_not_supported(self) -> None:
        """The API takes a complete prompt; claiming otherwise would mislead
        the voice channel into the streaming-delivery path."""
        provider, _ = _provider()
        assert provider.supports_streaming_input is False

    def test_available_voices_matches_the_live_catalog(self) -> None:
        voices = GeminiTTSProvider.available_voices()
        assert len(voices) == 30
        assert "Kore" in {v.id for v in voices}

        from roomkit.providers.gemini.voices import VOICES

        assert [v.id for v in voices] == [v.id for v in VOICES]

    def test_default_model_is_the_one_that_streams(self) -> None:
        provider, _ = _provider()
        assert provider._config.model == "gemini-3.1-flash-tts-preview"
        assert provider._config.model in GEMINI_TTS_MODELS

    def test_api_key_is_not_in_the_repr(self) -> None:
        config = GeminiTTSConfig(api_key="super-secret")
        assert "super-secret" not in repr(config)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"api_key": "  "}, "api_key"),
            ({"model": "  "}, "model"),
            ({"voice": "  "}, "voice"),
            ({"language": "  "}, "language"),
            ({"timeout": 0}, "timeout"),
            ({"timeout": float("inf")}, "timeout"),
            ({"connect_timeout": 0}, "connect_timeout"),
        ],
    )
    def test_invalid_config_is_rejected(self, overrides: dict[str, Any], message: str) -> None:
        values: dict[str, Any] = {"api_key": "test-key", **overrides}
        with pytest.raises(ValueError, match=message):
            GeminiTTSConfig(**values)


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_returns_playable_wav_with_duration(self) -> None:
        pcm = _pcm(0.5)
        provider, _ = _provider(pcm)

        audio = await provider.synthesize("Hello")

        assert audio.mime_type == "audio/wav"
        assert audio.transcript == "Hello"
        assert audio.duration_seconds == pytest.approx(0.5)
        assert audio.url.startswith("data:audio/wav;base64,")

        raw = base64.b64decode(audio.url.split(",", 1)[1])
        assert raw[:4] == b"RIFF"
        assert raw[8:12] == b"WAVE"
        # WAV header advertises the rate the service actually used.
        assert struct.unpack("<I", raw[24:28])[0] == OUTPUT_SAMPLE_RATE
        assert raw[44:] == pcm

    async def test_request_shape_matches_what_the_api_accepts(self) -> None:
        provider, client = _provider()

        await provider.synthesize("Hello", voice="Puck")

        call = client.interactions.calls[0]
        assert call["model"] == "gemini-3.1-flash-tts-preview"
        assert call["input"] == (
            "Synthesize speech from the transcript below.\n"
            "Speak only the transcript; do not read these instructions or labels aloud.\n"
            "Transcript:\nHello"
        )
        assert call["stream"] is False
        # ``mime_type``/``delivery`` are 400s on the live API — only ``type``.
        assert call["response_format"] == {"type": "audio"}
        assert call["generation_config"] == {"speech_config": [{"voice": "Puck"}]}
        # The split lives on the SDK's httpx client: a per-request timeout
        # would be flattened by google-genai to one float (RMK-149).
        assert "timeout" not in call

    async def test_language_is_forwarded_when_configured(self) -> None:
        provider, client = _provider(language="fr-CA")

        await provider.synthesize("Hello")

        speech = client.interactions.calls[0]["generation_config"]["speech_config"][0]
        assert speech == {"voice": "Kore", "language": "fr-CA"}

    async def test_style_prompt_is_separated_from_the_transcript(self) -> None:
        provider, client = _provider(style_prompt="Read this cheerfully")

        await provider.synthesize("Hello")

        assert client.interactions.calls[0]["input"] == (
            "Synthesize speech from the transcript below.\n"
            "Speak only the transcript; do not read these instructions or labels aloud.\n"
            "Delivery direction: Read this cheerfully\n"
            "Transcript:\nHello"
        )

    async def test_blank_text_is_rejected(self) -> None:
        provider, client = _provider()

        with pytest.raises(ValueError, match="non-empty text"):
            await provider.synthesize("   ")

        assert client.interactions.calls == []

    async def test_missing_audio_raises_with_the_status(self) -> None:
        provider, client = _provider()

        async def _no_audio(**kwargs: Any) -> _FakeInteraction:
            return _FakeInteraction(output_audio=None, status="failed")

        client.interactions.create = _no_audio  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="no audio.*failed"):
            await provider.synthesize("Hello")

    @pytest.mark.parametrize(
        ("audio", "message"),
        [
            (_FakeAudio(data="not base64"), "invalid base64"),
            (_FakeAudio(data="AQI=", sample_rate=-1), "invalid sample rate"),
            (_FakeAudio(data="AQI=", channels=-1), "invalid channel count"),
            (_FakeAudio(data="AQ=="), "truncated PCM"),
        ],
    )
    async def test_malformed_service_audio_is_rejected(
        self, audio: _FakeAudio, message: str
    ) -> None:
        provider, client = _provider()

        async def _malformed(**kwargs: Any) -> _FakeInteraction:
            return _FakeInteraction(output_audio=audio)

        client.interactions.create = _malformed  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match=message):
            await provider.synthesize("Hello")

    async def test_service_reported_rate_wins_over_the_documented_one(self) -> None:
        """The documented rate is a default, not an assertion — trust the response."""
        provider, client = _provider()

        async def _at_16k(**kwargs: Any) -> _FakeInteraction:
            return _FakeInteraction(
                output_audio=_FakeAudio(
                    data=base64.b64encode(_pcm(0.5)).decode(), sample_rate=16000
                )
            )

        client.interactions.create = _at_16k  # type: ignore[method-assign]

        audio = await provider.synthesize("Hello")

        raw = base64.b64decode(audio.url.split(",", 1)[1])
        assert struct.unpack("<I", raw[24:28])[0] == 16000
        assert audio.duration_seconds == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# synthesize_stream()
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    async def _collect(self, provider: GeminiTTSProvider, text: str) -> list[AudioChunk]:
        return [chunk async for chunk in provider.synthesize_stream(text)]

    async def test_audio_deltas_become_chunks_terminated_by_a_final_marker(self) -> None:
        pcm = _pcm(0.2)  # 9600 bytes → 5 frames of 1920
        provider, _ = _provider(pcm)

        chunks = await self._collect(provider, "Hello")

        assert [c.is_final for c in chunks] == [False] * 5 + [True]
        assert b"".join(c.data for c in chunks) == pcm
        assert chunks[-1].data == b""
        assert all(c.sample_rate == OUTPUT_SAMPLE_RATE for c in chunks)
        assert all(c.channels == 1 for c in chunks)
        assert all(c.format == "pcm_s16le" for c in chunks)

    async def test_non_audio_deltas_are_ignored(self) -> None:
        provider, _ = _provider(_pcm(0.04))

        chunks = await self._collect(provider, "Hello")

        # One audio frame + terminator: the trailing text delta is dropped.
        assert len(chunks) == 2
        assert chunks[0].data == _pcm(0.04)

    async def test_stream_is_requested_with_stream_true(self) -> None:
        provider, client = _provider()

        await self._collect(provider, "Hello")

        assert client.interactions.calls[0]["stream"] is True

    async def test_blank_text_terminates_without_a_round_trip(self) -> None:
        """TTS filters can reduce a reply to whitespace; that must not 400."""
        provider, client = _provider()

        chunks = await self._collect(provider, "  \n ")

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].data == b""
        assert client.interactions.calls == []

    async def test_final_chunk_carries_the_rate_the_service_used(self) -> None:
        provider, client = _provider()

        async def _at_16k(**kwargs: Any) -> AsyncIterator[_FakeEvent]:
            async def _gen() -> AsyncIterator[_FakeEvent]:
                yield _FakeEvent(
                    delta=_FakeDelta(data=base64.b64encode(_pcm(0.04)).decode(), sample_rate=16000)
                )

            return _gen()

        client.interactions.create = _at_16k  # type: ignore[method-assign]

        chunks = await self._collect(provider, "Hello")

        assert [c.sample_rate for c in chunks] == [16000, 16000]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_close_shuts_the_pool_and_drops_the_client(self) -> None:
        provider, client = _provider()

        await provider.close()

        assert client.aio.closed is True
        assert provider._client is None

    async def test_close_is_idempotent(self) -> None:
        provider, _ = _provider()

        await provider.close()
        await provider.close()  # no client left to close

        assert provider._client is None

    async def test_close_survives_a_broken_transport(self) -> None:
        """A dead pool must not turn shutdown into a raised exception."""
        provider, client = _provider()

        async def _boom() -> None:
            raise RuntimeError("transport already gone")

        client.aio.aclose = _boom  # type: ignore[method-assign]

        await provider.close()

        assert provider._client is None

    async def test_missing_sdk_raises_an_actionable_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        provider = GeminiTTSProvider(GeminiTTSConfig(api_key="test-key"))
        real_import = builtins.__import__

        def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google" or name.startswith("google.genai"):
                raise ImportError("no google-genai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)

        with pytest.raises(ImportError, match=r"roomkit\[gemini\]"):
            provider._get_client()


class TestPublicLoaders:
    def test_get_gemini_tts_provider(self) -> None:
        from roomkit.voice import get_gemini_tts_provider

        assert get_gemini_tts_provider() is GeminiTTSProvider

    def test_get_gemini_tts_config(self) -> None:
        from roomkit.voice import get_gemini_tts_config

        assert get_gemini_tts_config() is GeminiTTSConfig
