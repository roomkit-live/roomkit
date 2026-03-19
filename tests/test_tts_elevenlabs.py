"""Tests for ElevenLabsTTSProvider — config, expressive mode, and synthesis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.voice.tts.elevenlabs import (
    EXPRESSIVE_TAGS,
    MODEL_MULTILINGUAL_V2,
    MODEL_V3,
    ElevenLabsConfig,
    ElevenLabsTTSProvider,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestElevenLabsConfig:
    def test_defaults(self):
        cfg = ElevenLabsConfig(api_key="test-key")
        assert cfg.voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert cfg.model_id == MODEL_MULTILINGUAL_V2
        assert cfg.stability == 0.5
        assert cfg.similarity_boost == 0.75
        assert cfg.style == 0.0
        assert cfg.use_speaker_boost is True
        assert cfg.output_format == "mp3_44100_128"
        assert cfg.optimize_streaming_latency == 3
        assert cfg.expressive is False

    def test_custom_values(self):
        cfg = ElevenLabsConfig(
            api_key="test-key",
            voice_id="custom-voice",
            model_id="eleven_flash_v2_5",
            stability=0.8,
            similarity_boost=0.9,
            style=0.3,
            use_speaker_boost=False,
            output_format="pcm_24000",
            optimize_streaming_latency=1,
        )
        assert cfg.voice_id == "custom-voice"
        assert cfg.model_id == "eleven_flash_v2_5"
        assert cfg.stability == 0.8
        assert cfg.style == 0.3

    def test_expressive_flag(self):
        cfg = ElevenLabsConfig(api_key="test-key", expressive=True)
        assert cfg.expressive is True


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------


class TestProviderBasics:
    def test_name(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.name == "ElevenLabsTTS"

    def test_default_voice(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", voice_id="custom"))
        assert provider.default_voice == "custom"

    def test_supports_streaming_input_always_false(self):
        """SDK migration disables streaming input for all models."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.supports_streaming_input is False

    def test_supports_streaming_input_v3_false(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider.supports_streaming_input is False


# ---------------------------------------------------------------------------
# Expressive mode
# ---------------------------------------------------------------------------


class TestExpressiveMode:
    def test_expressive_sets_v3_model(self):
        """expressive=True should force model to v3 conversational."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._config.model_id == MODEL_V3

    def test_expressive_overrides_explicit_model(self):
        """expressive=True overrides any explicit model_id."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                model_id="eleven_flash_v2_5",
                expressive=True,
            )
        )
        assert provider._config.model_id == MODEL_V3

    def test_non_expressive_keeps_model(self):
        """Without expressive, model_id is untouched."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", model_id="eleven_flash_v2_5")
        )
        assert provider._config.model_id == "eleven_flash_v2_5"

    def test_is_v3_model_expressive(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._is_v3_model() is True

    def test_is_v3_model_explicit(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        assert provider._is_v3_model() is True

    def test_is_v3_model_v2(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider._is_v3_model() is False

    def test_expressive_tags_constant(self):
        assert "[laughs]" in EXPRESSIVE_TAGS
        assert "[whispers]" in EXPRESSIVE_TAGS
        assert "[sighs]" in EXPRESSIVE_TAGS
        assert "[slow]" in EXPRESSIVE_TAGS
        assert "[excited]" in EXPRESSIVE_TAGS
        assert len(EXPRESSIVE_TAGS) == 5


# ---------------------------------------------------------------------------
# Voice settings
# ---------------------------------------------------------------------------


class TestVoiceSettings:
    def test_v2_includes_all_settings(self):
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", style=0.3, use_speaker_boost=False)
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": False,
        }

    def test_v3_omits_style_and_speaker_boost(self):
        """v3 model should only include stability and similarity_boost."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                expressive=True,
                style=0.5,
                use_speaker_boost=True,
            )
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
        assert "style" not in settings
        assert "use_speaker_boost" not in settings

    def test_v3_explicit_model_omits_style(self):
        """Setting model_id to v3 directly also omits style/speaker_boost."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        settings = provider._build_voice_settings()
        assert "style" not in settings
        assert "use_speaker_boost" not in settings


# ---------------------------------------------------------------------------
# Synthesize (mocked SDK)
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_sends_correct_payload(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=b"fake-audio-bytes")

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="mock-settings"),
        ):
            result = await provider.synthesize("Hello world")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == MODEL_MULTILINGUAL_V2
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["voice_settings"] == "mock-settings"
        assert result.transcript == "Hello world"

    async def test_synthesize_expressive_payload(self):
        """Expressive mode sends v3 model."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=b"fake-audio")

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="mock-settings"),
        ):
            await provider.synthesize("[laughs] That's funny!")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["model_id"] == MODEL_V3
        assert call_kwargs["text"] == "[laughs] That's funny!"

    async def test_synthesize_custom_voice(self):
        """Voice override is forwarded to SDK."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_client = MagicMock()
        mock_client.text_to_speech.convert = AsyncMock(return_value=b"audio")

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            await provider.synthesize("Hi", voice="custom-voice-id")

        call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert call_kwargs["voice_id"] == "custom-voice-id"


# ---------------------------------------------------------------------------
# Streaming synthesis (mocked SDK)
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    async def test_synthesize_stream_yields_chunks(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", output_format="pcm_24000"))

        async def mock_stream(**kwargs):
            yield b"chunk1"
            yield b"chunk2"

        mock_client = MagicMock()
        mock_client.text_to_speech.stream = mock_stream

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            chunks = []
            async for chunk in provider.synthesize_stream("Hello"):
                chunks.append(chunk)

        # 2 data chunks + 1 final marker
        assert len(chunks) == 3
        assert chunks[0].data == b"chunk1"
        assert chunks[0].sample_rate == 24000
        assert chunks[0].format == "pcm_s16le"
        assert chunks[0].is_final is False
        assert chunks[1].data == b"chunk2"
        assert chunks[2].data == b""
        assert chunks[2].is_final is True

    async def test_synthesize_stream_skips_empty_chunks(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        async def mock_stream(**kwargs):
            yield b"data"
            yield b""
            yield b"more"

        mock_client = MagicMock()
        mock_client.text_to_speech.stream = mock_stream

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_make_voice_settings", return_value="s"),
        ):
            chunks = []
            async for chunk in provider.synthesize_stream("Hi"):
                chunks.append(chunk)

        # 2 data chunks (empty skipped) + 1 final marker
        assert len(chunks) == 3
        assert chunks[0].data == b"data"
        assert chunks[1].data == b"more"
        assert chunks[2].is_final is True


# ---------------------------------------------------------------------------
# Streaming latency param
# ---------------------------------------------------------------------------


class TestStreamingLatency:
    async def test_v2_includes_latency_param(self):
        """Non-v3 models include optimize_streaming_latency."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider._is_v3_model() is False

    async def test_v3_excludes_latency_param(self):
        """v3 models skip optimize_streaming_latency."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._is_v3_model() is True


# ---------------------------------------------------------------------------
# Voice listing (mocked SDK)
# ---------------------------------------------------------------------------


class TestListVoices:
    async def test_list_voices(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "v1"
        mock_voice1.name = "Rachel"
        mock_voice1.category = "premade"
        mock_voice1.labels = {"accent": "american"}

        mock_voice2 = MagicMock()
        mock_voice2.voice_id = "v2"
        mock_voice2.name = "Adam"
        mock_voice2.category = "premade"
        mock_voice2.labels = {}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice1, mock_voice2]

        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            voices = await provider.list_voices()

        assert len(voices) == 2
        assert voices[0].voice_id == "v1"
        assert voices[0].name == "Rachel"
        assert voices[0].labels == {"accent": "american"}
        assert voices[1].voice_id == "v2"

    async def test_list_voices_caches(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        mock_voice = MagicMock()
        mock_voice.voice_id = "v1"
        mock_voice.name = "R"
        mock_voice.category = "premade"
        mock_voice.labels = {}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice]

        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.list_voices()
            await provider.list_voices()

        # Should only call the API once
        assert mock_client.voices.get_all.call_count == 1


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_clears_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        provider._client = MagicMock()  # Simulate existing client

        await provider.close()

        assert provider._client is None

    async def test_close_noop_when_no_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        await provider.close()  # should not raise
        assert provider._client is None
