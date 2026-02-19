"""Tests for the Azure AI Studio provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.azure.config import AzureAIConfig


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


def _config(**overrides: Any) -> AzureAIConfig:
    defaults: dict[str, Any] = {
        "api_key": "azure-test-key",
        "azure_endpoint": "https://my-project.services.ai.azure.com",
        "model": "DeepSeek-R1",
    }
    defaults.update(overrides)
    return AzureAIConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "DeepSeek-R1",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake chat completion response."""
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


class TestAzureAIConfig:
    def test_required_fields(self) -> None:
        cfg = _config()
        assert cfg.azure_endpoint == "https://my-project.services.ai.azure.com"
        assert cfg.model == "DeepSeek-R1"
        assert cfg.api_version == "2024-12-01-preview"

    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.timeout == 120.0

    def test_custom_api_version(self) -> None:
        cfg = _config(api_version="2025-01-01")
        assert cfg.api_version == "2025-01-01"

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            AzureAIConfig(
                api_key="key",
                azure_endpoint="https://example.com",
            )

    def test_missing_endpoint_raises(self) -> None:
        with pytest.raises(ValidationError):
            AzureAIConfig(
                api_key="key",
                model="gpt-4o",
            )


class TestAzureAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"
            assert result.metadata["model"] == "DeepSeek-R1"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
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
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(prompt_tokens=42, completion_tokens=7)
            )

            result = await provider.generate(_context())

            assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}

    @pytest.mark.asyncio
    async def test_provider_name_is_azure(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            assert provider._provider_name == "azure"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_with_azure_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "azure"
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_sdk_error_non_retryable(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("bad request", status_code=400)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.provider == "azure"
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search for info",
                        parameters={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    )
                ]
            )
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="",
                    tool_calls=[
                        {
                            "id": "call_abc",
                            "name": "search",
                            "arguments": '{"query": "cats"}',
                        }
                    ],
                )
            )

            result = await provider.generate(_context())

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_abc"
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "cats"}

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.azure.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.AzureAIProvider(_config())

    @pytest.mark.asyncio
    async def test_inherits_build_messages(self) -> None:
        """Verify Azure provider uses inherited OpenAI message building."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.azure.ai import AzureAIProvider

            provider = AzureAIProvider(_config())

            messages = provider._build_messages(
                [AIMessage(role="user", content="Hello")],
                system_prompt="Be brief.",
            )

            assert messages[0] == {"role": "system", "content": "Be brief."}
            assert messages[1] == {"role": "user", "content": "Hello"}

    def test_client_created_with_azure_params(self) -> None:
        """Verify AsyncAzureOpenAI is created with correct params."""
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.azure.ai import AzureAIProvider

            cfg = _config()
            AzureAIProvider(cfg)

            mock_mod.AsyncAzureOpenAI.assert_called_once_with(
                api_key="azure-test-key",
                azure_endpoint="https://my-project.services.ai.azure.com",
                api_version="2024-12-01-preview",
                timeout=120.0,
            )
