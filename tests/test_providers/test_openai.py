"""Tests for the OpenAI AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.openai.config import OpenAIConfig


class _FakeAPIStatusError(Exception):
    """Stub for openai.APIStatusError used in tests."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _mock_openai_module() -> MagicMock:
    """Return a MagicMock that behaves like the openai module."""
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    return mod


def _config(**overrides: Any) -> OpenAIConfig:
    defaults: dict[str, Any] = {"api_key": "sk-test-key"}
    defaults.update(overrides)
    return OpenAIConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake OpenAI chat completion response."""
    mock_tool_calls = None
    if tool_calls:
        mock_tool_calls = [
            SimpleNamespace(
                id=tc.get("id", "call_123"),
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc.get("arguments", "{}"),
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=mock_tool_calls),
                finish_reason=finish_reason,
            ),
        ],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


class TestOpenAIAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"
            assert result.metadata["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0] == {"role": "system", "content": "You are helpful."}
            assert messages[1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(prompt_tokens=42, completion_tokens=7)
            )

            result = await provider.generate(_context())

            assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_api_error(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )

            with pytest.raises(Exception, match="API rate limit"):
                await provider.generate(_context())

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "gpt-4o"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.base_url is None

    def test_config_with_base_url(self) -> None:
        cfg = _config(base_url="http://localhost:11434/v1")
        assert cfg.base_url == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_in_provider_error(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_sdk_error_non_retryable(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("bad request", status_code=400)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_sdk_error_no_status_code(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code is None

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.openai.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.OpenAIAIProvider(_config())

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search for info",
                        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                    )
                ]
            )
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["type"] == "function"
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="",
                    tool_calls=[
                        {"id": "call_abc", "name": "search", "arguments": '{"query": "cats"}'}
                    ],
                )
            )

            result = await provider.generate(_context())

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_abc"
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "cats"}
