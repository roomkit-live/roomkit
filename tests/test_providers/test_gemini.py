"""Tests for the Google Gemini AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
    StreamDone,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.gemini.config import GeminiConfig


class _FakeStreamIterator:
    """Simulates an async iterator returned by generate_content_stream."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> _FakeStreamIterator:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _mock_genai_module() -> MagicMock:
    """Return a MagicMock that behaves like the google.genai module."""
    mod = MagicMock()

    # Mock types
    types = MagicMock()
    types.Content = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    types.Part.from_text = MagicMock(side_effect=lambda text: SimpleNamespace(text=text))
    types.Part.from_uri = MagicMock(
        side_effect=lambda file_uri, mime_type: SimpleNamespace(uri=file_uri, mime_type=mime_type)
    )
    types.Part.from_bytes = MagicMock(
        side_effect=lambda data, mime_type: SimpleNamespace(data=data, mime_type=mime_type)
    )
    types.GenerateContentConfig = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    types.Tool = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
    mod.types = types

    # Mock Client with async generate_content and generate_content_stream
    client_instance = MagicMock()
    client_instance.aio.models.generate_content = AsyncMock()
    client_instance.aio.models.generate_content_stream = AsyncMock()
    mod.Client.return_value = client_instance

    # Attach types to the module so 'from google.genai import types' works
    mod.types = types

    return mod


def _config(**overrides: Any) -> GeminiConfig:
    defaults: dict[str, Any] = {"api_key": "test-api-key"}
    defaults.update(overrides)
    return GeminiConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake Gemini response."""
    parts = [SimpleNamespace(text=text, function_call=None)]

    # Add tool call parts if provided
    if tool_calls:
        for tc in tool_calls:
            parts.append(
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name=tc["name"],
                        args=tc.get("args", {}),
                    ),
                )
            )

    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


def _genai_modules(mock_genai: MagicMock) -> dict[str, Any]:
    """Build sys.modules patch dict for Gemini tests."""
    return {
        "google": MagicMock(genai=mock_genai),
        "google.genai": mock_genai,
    }


def _stream_chunks(
    text_parts: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
) -> _FakeStreamIterator:
    """Build a fake stream iterator from text parts and/or tool calls."""
    chunks: list[SimpleNamespace] = []

    for text in text_parts or []:
        chunks.append(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=text, function_call=None)]
                        )
                    )
                ],
                usage_metadata=None,
            )
        )

    for tc in tool_calls or []:
        chunks.append(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    text=None,
                                    function_call=SimpleNamespace(
                                        name=tc["name"],
                                        args=tc.get("args", {}),
                                    ),
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=None,
            )
        )

    # Final chunk with usage
    chunks.append(
        SimpleNamespace(
            candidates=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
            ),
        )
    )

    return _FakeStreamIterator(chunks)


class TestGeminiAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hi there!"]
            )
            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.metadata["model"] == "gemini-3.1-flash-lite"

    @pytest.mark.asyncio
    async def test_data_uri_image_decoded_to_inline_bytes(self) -> None:
        """A ``data:`` URI image must become inline bytes, not a file URI.

        Gemini's ``from_uri`` expects a fetchable URI; a data URI handed
        to it ships a broken reference and the model never sees the image.
        The provider routes data URIs through ``from_bytes`` with the
        decoded payload.
        """
        import base64

        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            raw = b"\x89PNG\r\n\x1a\n fake image bytes"
            b64 = base64.b64encode(raw).decode()
            parts = provider._format_content(
                [
                    AITextPart(text="what is this?"),
                    AIImagePart(url=f"data:image/png;base64,{b64}", mime_type="image/png"),
                ]
            )

            mock_genai.types.Part.from_uri.assert_not_called()
            mock_genai.types.Part.from_bytes.assert_called_once_with(
                data=raw, mime_type="image/png"
            )
            assert parts[-1].data == raw
            assert parts[-1].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_real_uri_image_passed_through_from_uri(self) -> None:
        """A genuine fetchable URI still goes through ``from_uri``."""
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            parts = provider._format_content(
                [AIImagePart(url="https://example.com/cat.png", mime_type="image/png")]
            )

            mock_genai.types.Part.from_bytes.assert_not_called()
            mock_genai.types.Part.from_uri.assert_called_once_with(
                file_uri="https://example.com/cat.png", mime_type="image/png"
            )
            assert parts[-1].uri == "https://example.com/cat.png"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hello"]
            )
            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            assert provider._client.aio.models.generate_content_stream.called

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["hi"], prompt_tokens=42, completion_tokens=7
            )
            result = await provider.generate(_context())

            assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                tool_calls=[{"name": "search", "args": {"query": "test"}}],
            )
            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search for info",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ]
            )
            result = await provider.generate(ctx)

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_generate_api_error(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "API error"
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert "API error" in str(exc_info.value)
            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retryable(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "Rate limit exceeded 429"
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "gemini-3.1-flash-lite"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 1.0

    def test_supports_vision(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            assert provider.supports_vision is True

    def test_supports_streaming(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            assert provider.supports_streaming is True
            assert provider.supports_structured_streaming is True

    def test_model_name(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config(model="gemini-1.5-pro"))
            assert provider.model_name == "gemini-1.5-pro"

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            import importlib

            import roomkit.providers.gemini.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="google-genai is required"):
                mod.GeminiAIProvider(_config())

    @pytest.mark.asyncio
    async def test_structured_stream_yields_events(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Hello", " world"],
                prompt_tokens=5,
                completion_tokens=10,
            )

            events = []
            async for event in provider.generate_structured_stream(_context()):
                events.append(event)

            assert len(events) == 3
            assert isinstance(events[0], StreamTextDelta)
            assert events[0].text == "Hello"
            assert isinstance(events[1], StreamTextDelta)
            assert events[1].text == " world"
            assert isinstance(events[2], StreamDone)
            assert events[2].usage == {"prompt_tokens": 5, "completion_tokens": 10}
            assert events[2].metadata["model"] == "gemini-3.1-flash-lite"

    @pytest.mark.asyncio
    async def test_structured_stream_surfaces_thought_parts(self) -> None:
        # Gemini flags reasoning summaries with thought=True on the part; the
        # provider must surface those as thinking, plain text as text.
        from roomkit.providers.ai.base import StreamThinkingDelta

        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            thought = SimpleNamespace(text="reasoning", thought=True, function_call=None)
            answer = SimpleNamespace(text="answer", thought=False, function_call=None)
            chunk = SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[thought, answer]))],
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
            )
            provider._client.aio.models.generate_content_stream.return_value = _FakeStreamIterator(
                [chunk]
            )

            events = [
                e async for e in provider.generate_structured_stream(_context(thinking_budget=512))
            ]
            thinking = [e.thinking for e in events if isinstance(e, StreamThinkingDelta)]
            text = [e.text for e in events if isinstance(e, StreamTextDelta)]
            assert thinking == ["reasoning"]
            assert text == ["answer"]

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["one", "two", "three"],
            )

            parts = []
            async for text in provider.generate_stream(_context()):
                parts.append(text)

            assert parts == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_structured_stream_with_tool_calls(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.return_value = _stream_chunks(
                text_parts=["Let me search"],
                tool_calls=[{"name": "search", "args": {"q": "test"}}],
            )

            events = []
            async for event in provider.generate_structured_stream(_context()):
                events.append(event)

            text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
            tool_calls = [e for e in events if isinstance(e, StreamToolCall)]
            done_events = [e for e in events if isinstance(e, StreamDone)]

            assert len(text_deltas) == 1
            assert text_deltas[0].text == "Let me search"
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "search"
            assert tool_calls[0].arguments == {"q": "test"}
            assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_structured_stream_api_error(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.gemini.ai import GeminiAIProvider

            provider = GeminiAIProvider(_config())
            provider._client.aio.models.generate_content_stream.side_effect = Exception(
                "Stream error"
            )

            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.generate_structured_stream(_context()):
                    pass

            assert "Stream error" in str(exc_info.value)
            assert exc_info.value.provider == "gemini"
