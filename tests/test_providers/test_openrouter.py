"""Tests for the OpenRouter provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ModelInfo, ProviderError
from roomkit.providers.openrouter.config import OpenRouterConfig


class _FakeAPIStatusError(Exception):
    """Stub for openai.APIStatusError used in tests."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    """Stub for openai.APIConnectionError used in tests."""


def _mock_openai_module() -> MagicMock:
    """Return a MagicMock that behaves like the openai module."""
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


def _config(**overrides: Any) -> OpenRouterConfig:
    defaults: dict[str, Any] = {
        "api_key": "or-test-key",
        "model": "anthropic/claude-sonnet-4.5",
    }
    defaults.update(overrides)
    return OpenRouterConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "anthropic/claude-sonnet-4.5",
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


class TestOpenRouterConfig:
    def test_defaults_inherited_from_openai(self) -> None:
        cfg = _config()
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.timeout == 30.0
        assert cfg.site_url is None
        assert cfg.app_name is None

    def test_model_is_required(self) -> None:
        # The whole point of OpenRouter is choosing a model — no default.
        with pytest.raises(ValidationError):
            OpenRouterConfig(api_key="key")  # type: ignore[call-arg]

    def test_custom_base_url_and_attribution(self) -> None:
        cfg = _config(
            base_url="https://proxy.internal/api/v1",
            site_url="https://myapp.example",
            app_name="My App",
        )
        assert cfg.base_url == "https://proxy.internal/api/v1"
        assert cfg.site_url == "https://myapp.example"
        assert cfg.app_name == "My App"

    def test_inherits_openai_request_fields(self) -> None:
        # Subclassing OpenAIConfig means every request field the provider reads
        # exists here — guards against the Azure-style config-drift bug.
        cfg = _config(reasoning_effort="high", use_max_completion_tokens=True)
        assert cfg.reasoning_effort == "high"
        assert cfg.use_max_completion_tokens is True
        assert cfg.supports_custom_temperature is True


class TestOpenRouterAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            provider = OpenRouterAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_provider_name_is_openrouter(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            provider = OpenRouterAIProvider(_config())
            assert provider._provider_name == "openrouter"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_with_openrouter_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            provider = OpenRouterAIProvider(_config())
            provider._client = MagicMock()
            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "openrouter"
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_inherited_path(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            provider = OpenRouterAIProvider(_config())
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
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    def test_client_created_with_attribution_headers(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            OpenRouterAIProvider(_config(site_url="https://myapp.example", app_name="My App"))

            mock_mod.AsyncOpenAI.assert_called_once_with(
                api_key="or-test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=30.0,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://myapp.example",
                    "X-Title": "My App",
                },
            )

    def test_client_created_without_headers_when_unset(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            OpenRouterAIProvider(_config())

            _, kwargs = mock_mod.AsyncOpenAI.call_args
            assert kwargs["default_headers"] is None

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.openrouter.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.OpenRouterAIProvider(_config())


class TestOpenRouterCatalog:
    def test_available_models_nonempty_and_unique(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        models = OpenRouterAIProvider.available_models()
        assert models
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = [m.id for m in models]
        assert len(ids) == len(set(ids))
        # OpenRouter slugs are namespaced as "<provider>/<model>".
        assert all("/" in m.id for m in models)

    def test_parse_model_maps_metadata(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        info = OpenRouterAIProvider._parse_model(
            {
                "id": "anthropic/claude-sonnet-4.5",
                "name": "Anthropic: Claude Sonnet 4.5",
                "context_length": 1_000_000,
                "architecture": {"input_modalities": ["text", "image"]},
            }
        )
        assert info.id == "anthropic/claude-sonnet-4.5"
        assert info.display_name == "Anthropic: Claude Sonnet 4.5"
        assert info.context_window == 1_000_000
        assert info.supports_vision is True

    def test_parse_model_vision_false_for_text_only(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        info = OpenRouterAIProvider._parse_model(
            {"id": "deepseek/deepseek-v4-pro", "architecture": {"input_modalities": ["text"]}}
        )
        assert info.supports_vision is False

    def test_parse_model_vision_unknown_when_no_modalities(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        info = OpenRouterAIProvider._parse_model({"id": "some/model"})
        assert info.supports_vision is None
        assert info.context_window is None

    @pytest.mark.asyncio
    async def test_list_models_maps_and_merges(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        provider = OpenRouterAIProvider.__new__(OpenRouterAIProvider)
        provider._fetch_models_json = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "id": "openai/gpt-5.5",
                    "name": "OpenAI: GPT-5.5",
                    "context_length": 1_050_000,
                    "architecture": {"input_modalities": ["text", "image"]},
                },
                {"id": "some/unknown-model"},
            ]
        )

        models = {m.id: m for m in await provider.list_models()}
        assert models["openai/gpt-5.5"].display_name == "OpenAI: GPT-5.5"
        assert models["openai/gpt-5.5"].supports_vision is True
        # Unknown live id passes through with id only.
        assert models["some/unknown-model"].display_name is None

    @pytest.mark.asyncio
    async def test_list_models_backfills_from_curated(self) -> None:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        provider = OpenRouterAIProvider.__new__(OpenRouterAIProvider)
        # A curated slug reported live with no metadata gets its context window
        # backfilled from the offline catalog.
        provider._fetch_models_json = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "anthropic/claude-opus-4.8"}]
        )

        models = {m.id: m for m in await provider.list_models()}
        assert models["anthropic/claude-opus-4.8"].context_window == 1_000_000
        assert models["anthropic/claude-opus-4.8"].display_name == "Claude Opus 4.8"


class TestOpenRouterReasoning:
    """Thinking is requested via OpenRouter's unified ``reasoning`` object."""

    def _provider(self, **cfg_overrides: Any) -> Any:
        from roomkit.providers.openrouter.ai import OpenRouterAIProvider

        provider = OpenRouterAIProvider.__new__(OpenRouterAIProvider)
        provider._config = _config(**cfg_overrides)
        return provider

    def test_effort_from_config_when_no_budget(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, _context())
        assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}

    def test_budget_maps_to_max_tokens(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context(thinking_budget=4096))
        assert kwargs["extra_body"]["reasoning"] == {"max_tokens": 4096}

    def test_zero_budget_disables_reasoning(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(
            kwargs, _context(thinking_budget=0)
        )
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_omitted_when_no_effort_and_no_budget(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context())
        assert "extra_body" not in kwargs

    def test_reasoning_skipped_on_tool_turns(self) -> None:
        kwargs: dict[str, Any] = {}
        ctx = _context(tools=[AITool(name="x", description="d", parameters={})])
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, ctx)
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    async def test_config_extra_body_merges_with_reasoning(self) -> None:
        # Regression: a configured extra_body must not clobber the reasoning
        # object OpenRouter sets in _apply_sampling_kwargs — both ride the
        # request together through the full generate() path.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openrouter.ai import OpenRouterAIProvider

            provider = OpenRouterAIProvider(
                _config(reasoning_effort="high", extra_body={"top_k": 20})
            )
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            await provider.generate(_context())
            extra_body = provider._client.chat.completions.create.call_args[1]["extra_body"]
            assert extra_body["reasoning"] == {"effort": "high"}
            assert extra_body["top_k"] == 20

    def test_temperature_still_applied(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context(temperature=0.3))
        assert kwargs["temperature"] == 0.3
