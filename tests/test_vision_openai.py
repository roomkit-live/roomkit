"""Tests for OpenAIVisionProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.encode import frame_to_jpeg
from roomkit.video.vision.openai import (
    OpenAIVisionConfig,
    OpenAIVisionProvider,
)


class TestOpenAIVisionProvider:
    def test_name(self) -> None:
        provider = OpenAIVisionProvider(OpenAIVisionConfig(model="gpt-4o"))
        assert provider.name == "openai-vision:gpt-4o"

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

    @pytest.mark.parametrize(
        ("content", "description"),
        [
            ("<think>a desk, maybe</think>A person at a desk", "A person at a desk"),
            ("<think>still weighing whether it is a desk", ""),
        ],
        ids=["closed", "cut-before-its-close"],
    )
    async def test_a_think_block_is_no_part_of_the_description(
        self, content: str, description: str
    ) -> None:
        """Reasoning that leaks into the answer is taken out, a block the output
        cap cut before its close included, as on the chat providers (RMK-484)."""
        provider = OpenAIVisionProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        frame = VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)

        result = await provider.analyze_frame(frame)

        assert result.description == description

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

    @pytest.mark.parametrize(
        ("model", "base_url", "cap", "temperature"),
        [
            ("gpt-6.2-sol", "https://api.openai.com/v1", "max_completion_tokens", None),
            ("o3", "https://api.openai.com/v1", "max_completion_tokens", None),
            ("gpt-4.1-mini", "https://api.openai.com/v1", "max_completion_tokens", 0.3),
            ("gpt-4o", "https://api.openai.com/v1", "max_tokens", 0.3),
            ("gpt-6.2-sol", "http://localhost:11434/v1", "max_tokens", 0.3),
        ],
        ids=["gpt-6", "o-series", "gpt-4.1", "gpt-4o", "another-server"],
    )
    async def test_the_cap_and_temperature_go_as_openai_takes_them(
        self, model: str, base_url: str, cap: str, temperature: float | None
    ) -> None:
        """OpenAI's endpoint refuses ``max_tokens`` from GPT-4.1 and the
        o-series, and any temperature but the default from GPT-5, as
        ``OpenAIConfig`` profiles them; another server gets both as set."""
        provider = OpenAIVisionProvider(OpenAIVisionConfig(model=model, base_url=base_url))
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "A desk"
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = client

        await provider.analyze_frame(_frame())

        kwargs = client.chat.completions.create.call_args.kwargs
        other = "max_tokens" if cap == "max_completion_tokens" else "max_completion_tokens"
        assert kwargs[cap] == 100
        assert other not in kwargs
        assert kwargs.get("temperature") == temperature

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


_BUTTON: dict[str, object] = {
    "type": "object",
    "properties": {"found": {"type": "boolean"}, "label": {"type": "string"}},
    "required": ["found", "label"],
    "additionalProperties": False,
}


def _openai_vision(
    content: str, *, finish: str = "stop", refusal: str | None = None, **config: object
) -> OpenAIVisionProvider:
    provider = OpenAIVisionProvider(OpenAIVisionConfig(model="gpt-4o", **config))  # type: ignore[arg-type]
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.refusal = refusal
    response.choices[0].finish_reason = finish
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    provider._client = client
    return provider


def _frame() -> VideoFrame:
    return VideoFrame(data=b"\x00" * (64 * 48 * 3), codec="raw_rgb24", width=64, height=48)


class TestResponseSchema:
    """RFC §6.7 for vision: the description answers the schema, or the call raises."""

    async def test_the_schema_rides_a_strict_json_schema_format(self) -> None:
        provider = _openai_vision('{"found": false, "label": ""}')

        result = await provider.analyze_frame(_frame(), response_schema=_BUTTON)

        assert result.description == '{"found": false, "label": ""}'
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"] == {
            "name": "response",
            "schema": _BUTTON,
            "strict": True,
        }

    async def test_a_length_stop_raises_truncated(self) -> None:
        provider = _openai_vision('{"found": fa', finish="length")

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.analyze_frame(_frame(), response_schema=_BUTTON)

        assert exc.value.reason == "truncated"

    async def test_a_refusal_raises_refusal(self) -> None:
        provider = _openai_vision("", refusal="I can't help with that.")

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.analyze_frame(_frame(), response_schema=_BUTTON)

        assert exc.value.reason == "refusal"

    async def test_a_content_filter_stop_raises_refusal(self) -> None:
        provider = _openai_vision("", finish="content_filter")

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.analyze_frame(_frame(), response_schema=_BUTTON)

        assert exc.value.reason == "refusal"

    async def test_a_server_declared_without_support_refuses_before_the_call(self) -> None:
        provider = _openai_vision("{}", supports_response_schema=False)

        assert provider.supports_response_schema is False
        with pytest.raises(ResponseSchemaError) as exc:
            await provider.analyze_frame(_frame(), response_schema=_BUTTON)

        assert exc.value.reason == "unsupported"
        provider._client.chat.completions.create.assert_not_called()
