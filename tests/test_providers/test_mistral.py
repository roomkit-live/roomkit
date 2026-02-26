"""Tests for the Mistral AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.mistral.config import MistralConfig


class _FakeStream:
    """Simulates an async iterator returned by Mistral's chat.stream_async."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def _mock_mistral_module() -> MagicMock:
    """Return a MagicMock that behaves like the mistralai module."""
    mod = MagicMock()
    # The constructor returns a client mock
    client = MagicMock()
    client.chat.stream_async = AsyncMock()
    mod.Mistral.return_value = client
    return mod


def _config(**overrides: Any) -> MistralConfig:
    defaults: dict[str, Any] = {"api_key": "test-key"}
    defaults.update(overrides)
    return MistralConfig(**defaults)


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


def _stream_events(
    text_chunks: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
) -> _FakeStream:
    """Build a fake Mistral stream from text chunks and/or tool calls."""
    events: list[SimpleNamespace] = []

    for text in text_chunks or []:
        events.append(
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text, tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            )
        )

    for tc in tool_calls or []:
        events.append(
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=tc.get("index", 0),
                                        id=tc.get("id", "call_1"),
                                        function=SimpleNamespace(
                                            name=tc["name"],
                                            arguments=tc.get("arguments", "{}"),
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            )
        )

    # Final chunk with usage and finish_reason
    events.append(
        SimpleNamespace(
            data=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason=finish_reason,
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
            )
        )
    )

    return _FakeStream(events)


class TestMistralAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["Hi there!"]
            )
            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.metadata["model"] == "mistral-large-latest"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(text_chunks=["Hello"])
            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            assert provider._client.chat.stream_async.called

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["hi"], prompt_tokens=42, completion_tokens=7
            )
            result = await provider.generate(_context())

            assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                tool_calls=[{"name": "search", "arguments": '{"query": "test"}', "id": "call_1"}],
            )
            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search",
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
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.side_effect = Exception("API error")

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert "API error" in str(exc_info.value)
            assert exc_info.value.provider == "mistral"

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retryable(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.side_effect = Exception("Rate limit exceeded 429")

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "mistral-large-latest"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.server_url is None

    def test_supports_vision_pixtral(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config(model="pixtral-large-latest"))
            assert provider.supports_vision is True

    def test_no_vision_for_standard_models(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config(model="mistral-large-latest"))
            assert provider.supports_vision is False

    def test_supports_streaming(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            assert provider.supports_streaming is True
            assert provider.supports_structured_streaming is True

    def test_model_name(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config(model="mistral-small-latest"))
            assert provider.model_name == "mistral-small-latest"

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"mistralai": None}):
            import importlib

            import roomkit.providers.mistral.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="mistralai is required"):
                mod.MistralAIProvider(_config())

    @pytest.mark.asyncio
    async def test_structured_stream_yields_events(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["Hello", " world"],
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

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["one", "two", "three"],
            )

            parts = []
            async for text in provider.generate_stream(_context()):
                parts.append(text)

            assert parts == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_structured_stream_with_tool_calls(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["Let me search"],
                tool_calls=[{"name": "search", "arguments": '{"q": "test"}', "id": "call_1"}],
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
    async def test_structured_stream_with_think_tags(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(
                text_chunks=["<think>reasoning", "</think>", "answer"],
            )

            events = []
            async for event in provider.generate_structured_stream(_context()):
                events.append(event)

            thinking = [e for e in events if isinstance(e, StreamThinkingDelta)]
            text = [e for e in events if isinstance(e, StreamTextDelta)]

            assert len(thinking) == 1
            assert thinking[0].thinking == "reasoning"
            assert len(text) == 1
            assert text[0].text == "answer"

    @pytest.mark.asyncio
    async def test_thinking_part_round_trip(self) -> None:
        """AIThinkingPart in history is re-wrapped as <think> tags."""
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.return_value = _stream_events(text_chunks=["ok"])

            ctx = _context(
                messages=[
                    AIMessage(
                        role="assistant",
                        content=[
                            AIThinkingPart(thinking="I should search"),
                            AIToolCallPart(id="call_1", name="search", arguments={"q": "test"}),
                        ],
                    ),
                    AIMessage(role="user", content="thanks"),
                ]
            )
            result = await provider.generate(ctx)
            assert result.content == "ok"

            # Verify the messages sent include the thinking as <think> tags
            call_kwargs = provider._client.chat.stream_async.call_args
            messages = call_kwargs.kwargs["messages"]
            assistant_msg = messages[0]
            assert "<think>I should search</think>" in (assistant_msg["content"] or "")

    @pytest.mark.asyncio
    async def test_structured_stream_api_error(self) -> None:
        with patch.dict("sys.modules", {"mistralai": _mock_mistral_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(_config())
            provider._client.chat.stream_async.side_effect = Exception("Stream error")

            with pytest.raises(ProviderError) as exc_info:
                async for _ in provider.generate_structured_stream(_context()):
                    pass

            assert "Stream error" in str(exc_info.value)
            assert exc_info.value.provider == "mistral"
