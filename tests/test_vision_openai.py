"""Tests for OpenAIVisionProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.encode import frame_to_jpeg
from roomkit.video.vision.openai import (
    OpenAIVisionConfig,
    OpenAIVisionProvider,
)


class TestOpenAIVisionConfig:
    def test_defaults(self) -> None:
        config = OpenAIVisionConfig()
        assert config.base_url == "http://localhost:11434/v1"
        assert config.model == "qwen3.5"
        assert config.api_key == "ollama"
        assert config.detail == "low"

    def test_custom(self) -> None:
        config = OpenAIVisionConfig(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            max_tokens=500,
        )
        assert config.model == "gpt-4o"
        assert config.base_url == "https://api.openai.com/v1"


class TestOpenAIVisionProvider:
    def test_name(self) -> None:
        provider = OpenAIVisionProvider(OpenAIVisionConfig(model="gpt-4o"))
        assert provider.name == "openai-vision:gpt-4o"

    def test_default_name(self) -> None:
        provider = OpenAIVisionProvider()
        assert provider.name == "openai-vision:qwen3.5"

    async def test_analyze_frame(self) -> None:
        """Test analyze_frame with a mocked OpenAI client."""
        provider = OpenAIVisionProvider(OpenAIVisionConfig(model="test-model"))

        # Mock the openai module and client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A person sitting at a desk with a laptop"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        # Create a small raw frame
        frame = VideoFrame(
            data=b"\x00" * (64 * 48 * 3),
            codec="raw_rgb24",
            width=64,
            height=48,
        )

        result = await provider.analyze_frame(frame)

        assert result.description == "A person sitting at a desk with a laptop"
        assert result.metadata["model"] == "test-model"
        assert result.metadata["usage"]["prompt_tokens"] == 100

        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    async def test_analyze_frame_empty_response(self) -> None:
        provider = OpenAIVisionProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 0

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        frame = VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)
        result = await provider.analyze_frame(frame)
        assert result.description == ""

    async def test_close(self) -> None:
        provider = OpenAIVisionProvider()
        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()
        mock_client.close.assert_called_once()
        assert provider._client is None

    async def test_close_no_client(self) -> None:
        provider = OpenAIVisionProvider()
        await provider.close()  # no-op, should not raise


class TestFrameToJpegBase64:
    @pytest.fixture
    def rgb_frame(self) -> VideoFrame:
        return VideoFrame(
            data=b"\xff\x00\x00" * (32 * 24),  # red pixels
            codec="raw_rgb24",
            width=32,
            height=24,
        )

    def test_encoded_codec_passthrough(self) -> None:
        """Encoded frames are returned as-is."""
        raw_data = b"\x00\x00\x00\x01\x67"  # Fake h264 NAL
        frame = VideoFrame(data=raw_data, codec="h264", width=640, height=480)
        result = frame_to_jpeg(frame)
        assert result == raw_data

    def test_rgb_with_cv2(self, rgb_frame: VideoFrame) -> None:
        """RGB frame encoded to JPEG via OpenCV."""
        pytest.importorskip("cv2", reason="opencv not installed")
        jpeg_bytes = frame_to_jpeg(rgb_frame)
        # JPEG starts with FFD8
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_rgb_with_pillow_fallback(self, rgb_frame: VideoFrame) -> None:
        """RGB frame encoded via Pillow when cv2 not available."""
        pytest.importorskip("PIL", reason="Pillow not installed")
        with patch.dict("sys.modules", {"cv2": None, "numpy": None}):
            jpeg_bytes = frame_to_jpeg(rgb_frame)
            assert jpeg_bytes[:2] == b"\xff\xd8"


class TestExports:
    def test_importable_from_subpackage(self) -> None:
        from roomkit.video import OpenAIVisionConfig, OpenAIVisionProvider

        assert OpenAIVisionProvider is not None
        assert OpenAIVisionConfig is not None

    def test_importable_from_video(self) -> None:
        from roomkit.video import OpenAIVisionConfig, OpenAIVisionProvider

        assert OpenAIVisionProvider is not None
        assert OpenAIVisionConfig is not None

    def test_importable_from_vision(self) -> None:
        from roomkit.video.vision import OpenAIVisionConfig, OpenAIVisionProvider

        assert OpenAIVisionProvider is not None
        assert OpenAIVisionConfig is not None
