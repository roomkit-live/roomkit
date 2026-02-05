"""Tests for the Anthropic AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.anthropic.config import AnthropicConfig


class _FakeAPIStatusError(Exception):
    """Stub for anthropic.APIStatusError used in tests."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _mock_anthropic_module() -> MagicMock:
    """Return a MagicMock that behaves like the anthropic module."""
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    return mod


def _config(**overrides: Any) -> AnthropicConfig:
    defaults: dict[str, Any] = {"api_key": "sk-test-key"}
    defaults.update(overrides)
    return AnthropicConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 10,
    output_tokens: int = 25,
    tool_use: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake Anthropic message response."""
    content = [SimpleNamespace(type="text", text=text)]
    if tool_use:
        for tu in tool_use:
            content.append(
                SimpleNamespace(
                    type="tool_use",
                    id=tu.get("id", "tool_123"),
                    name=tu["name"],
                    input=tu.get("input", {}),
                )
            )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


class TestAnthropicAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "end_turn"
            assert result.metadata["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(return_value=_mock_response())

            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            call_kwargs = provider._client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(
                return_value=_mock_response(input_tokens=42, output_tokens=7)
            )

            result = await provider.generate(_context())

            assert result.usage == {"input_tokens": 42, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_api_error(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )

            with pytest.raises(Exception, match="API rate limit"):
                await provider.generate(_context())

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_in_provider_error(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.messages.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_sdk_error_non_retryable(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("bad request", status_code=400)
            provider._client.messages.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_sdk_error_no_status_code(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code is None

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force re-import so the lazy import runs against the patched modules
            import importlib

            import roomkit.providers.anthropic.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="anthropic is required"):
                mod.AnthropicAIProvider(_config())

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(return_value=_mock_response())

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

            call_kwargs = provider._client.messages.create.call_args[1]
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["name"] == "search"
            assert call_kwargs["tools"][0]["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        with patch.dict("sys.modules", {"anthropic": _mock_anthropic_module()}):
            from roomkit.providers.anthropic.ai import AnthropicAIProvider

            provider = AnthropicAIProvider(_config())
            provider._client = MagicMock()
            provider._client.messages.create = AsyncMock(
                return_value=_mock_response(
                    text="",
                    tool_use=[{"id": "call_123", "name": "search", "input": {"query": "cats"}}],
                )
            )

            result = await provider.generate(_context())

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_123"
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "cats"}
