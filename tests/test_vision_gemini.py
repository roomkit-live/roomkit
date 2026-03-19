"""Tests for GeminiVisionProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider


class TestGeminiVisionConfig:
    def test_defaults(self) -> None:
        config = GeminiVisionConfig()
        assert config.model == "gemini-3.1-flash-lite-preview"
        assert config.api_key == ""
        assert config.max_tokens == 1024

    def test_custom(self) -> None:
        config = GeminiVisionConfig(
            api_key="AIza-test",
            model="gemini-2.0-flash",
            max_tokens=500,
        )
        assert config.api_key == "AIza-test"
        assert config.model == "gemini-2.0-flash"


class TestGeminiVisionProvider:
    def test_name(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        assert provider.name == "gemini-vision:gemini-3.1-flash-lite-preview"

    def test_custom_model_name(self) -> None:
        config = GeminiVisionConfig(api_key="test", model="gemini-2.0-flash")
        provider = GeminiVisionProvider(config)
        assert provider.name == "gemini-vision:gemini-2.0-flash"

    def test_config_constructor(self) -> None:
        config = GeminiVisionConfig(api_key="k", model="custom")
        provider = GeminiVisionProvider(config=config)
        assert provider.name == "gemini-vision:custom"

    async def test_analyze_frame(self) -> None:
        """Test analyze_frame with a mocked Gemini client."""
        config = GeminiVisionConfig(api_key="test-key", model="gemini-3.1-flash-lite-preview")
        provider = GeminiVisionProvider(config)

        # Mock response
        mock_response = MagicMock()
        mock_response.text = "A person at a desk with a monitor"
        mock_response.usage_metadata.prompt_token_count = 80
        mock_response.usage_metadata.candidates_token_count = 15

        # Mock client
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        # Mock types for Part.from_bytes
        mock_types = MagicMock()
        provider._types = mock_types

        frame = VideoFrame(
            data=b"\x00" * (64 * 48 * 3),
            codec="raw_rgb24",
            width=64,
            height=48,
        )

        result = await provider.analyze_frame(frame)

        assert result.description == "A person at a desk with a monitor"
        assert result.metadata["model"] == "gemini-3.1-flash-lite-preview"
        assert result.metadata["usage"]["prompt_tokens"] == 80
        assert result.metadata["usage"]["completion_tokens"] == 15

        mock_client.aio.models.generate_content.assert_called_once()

    async def test_analyze_frame_empty_response(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        provider._types = MagicMock()

        frame = VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)
        result = await provider.analyze_frame(frame)
        assert result.description == ""

    async def test_close(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        provider._client = MagicMock()
        provider._types = MagicMock()

        await provider.close()
        assert provider._client is None
        assert provider._types is None

    async def test_close_no_client(self) -> None:
        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="test"))
        await provider.close()  # no-op


class TestExports:
    def test_importable_from_subpackage(self) -> None:
        from roomkit.video import GeminiVisionConfig, GeminiVisionProvider

        assert GeminiVisionProvider is not None
        assert GeminiVisionConfig is not None

    def test_importable_from_video(self) -> None:
        from roomkit.video import GeminiVisionConfig, GeminiVisionProvider

        assert GeminiVisionProvider is not None
        assert GeminiVisionConfig is not None
