"""Tests for the xAI (Grok) chat provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ModelInfo, ProviderError
from roomkit.providers.xai.config import XAIConfig


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


def _config(**overrides: Any) -> XAIConfig:
    defaults: dict[str, Any] = {"api_key": "xai-test-key"}
    defaults.update(overrides)
    return XAIConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "grok-4.5",
) -> SimpleNamespace:
    """Build a fake chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            ),
        ],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=25),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {"messages": [AIMessage(role="user", content="Hi")]}
    defaults.update(overrides)
    return AIContext(**defaults)


class TestXAIConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.base_url == "https://api.x.ai/v1"
        assert cfg.model == "grok-4.5"
        # xAI deprecated max_tokens, and supports stream usage — both flipped
        # relative to the OpenAI parent's defaults.
        assert cfg.use_max_completion_tokens is True
        assert cfg.include_stream_usage is True

    def test_inherits_openai_request_fields(self) -> None:
        # Subclassing OpenAIConfig means every request field the inherited
        # provider reads exists here — guards against config drift.
        cfg = _config(reasoning_effort="low", extra_body={"top_k": 20})
        assert cfg.reasoning_effort == "low"
        assert cfg.extra_body == {"top_k": 20}
        assert cfg.supports_custom_temperature is True
        assert cfg.temperature == 0.7
        assert cfg.timeout == 30.0

    def test_custom_base_url(self) -> None:
        assert _config(base_url="https://proxy.internal/v1").base_url == (
            "https://proxy.internal/v1"
        )


class TestXAIAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.xai.ai import XAIAIProvider

            provider = XAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"

    def test_provider_name_is_xai(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.xai.ai import XAIAIProvider

            assert XAIAIProvider(_config())._provider_name == "xai"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_with_xai_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.xai.ai import XAIAIProvider

            provider = XAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=_FakeAPIStatusError("rate limited", status_code=429)
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.provider == "xai"
            assert exc_info.value.status_code == 429
            assert exc_info.value.retryable is True

    def test_client_created_against_xai_endpoint(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.xai.ai import XAIAIProvider

            XAIAIProvider(_config())

            mock_mod.AsyncOpenAI.assert_called_once_with(
                api_key="xai-test-key",
                base_url="https://api.x.ai/v1",
                timeout=30.0,
                max_retries=0,
                default_headers=None,
            )

    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_inherited_path(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.xai.ai import XAIAIProvider

            provider = XAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(
                _context(
                    tools=[
                        AITool(
                            name="search",
                            description="Search for info",
                            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                        )
                    ]
                )
            )

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_output_cap_sent_as_max_completion_tokens(self) -> None:
        # xAI deprecated max_tokens in favour of max_completion_tokens, so the
        # config flips the parent's default and the cap must ride the newer key.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.xai.ai import XAIAIProvider

            provider = XAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context(max_tokens=512))

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["max_completion_tokens"] == 512
            assert "max_tokens" not in call_kwargs

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.xai.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.XAIAIProvider(_config())


class TestXAIVision:
    """Grok's whole text line is multimodal — the OpenAI prefix list is not."""

    def _provider(self, model: str) -> Any:
        from roomkit.providers.xai.ai import XAIAIProvider

        provider = XAIAIProvider.__new__(XAIAIProvider)
        provider._config = _config(model=model)
        return provider

    def test_catalog_model_supports_vision(self) -> None:
        # Regression: the inherited OpenAI implementation prefix-matches gpt-4o /
        # o1 / o3 and would report False for every grok id, silently dropping
        # image attachments.
        assert self._provider("grok-4.5").supports_vision is True
        assert self._provider("grok-4.20-0309-non-reasoning").supports_vision is True

    def test_unknown_model_defaults_to_vision(self) -> None:
        # An alias (grok-latest) or a model newer than the snapshot must not
        # lose a capability the whole family has.
        assert self._provider("grok-latest").supports_vision is True


class TestXAICatalog:
    def test_available_models_nonempty_and_unique(self) -> None:
        from roomkit.providers.xai.ai import XAIAIProvider

        models = XAIAIProvider.available_models()
        assert models
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = [m.id for m in models]
        assert len(ids) == len(set(ids))
        assert all(m.id.startswith("grok-") for m in models)

    def test_every_model_declares_context_window_and_vision(self) -> None:
        from roomkit.providers.xai.ai import XAIAIProvider

        for model in XAIAIProvider.available_models():
            assert model.context_window, f"{model.id} has no context window"
            assert model.supports_vision is True, f"{model.id} should be multimodal"

    def test_flagship_is_first(self) -> None:
        # The catalog order drives pickers; grok-4.5 is the current flagship and
        # the XAIConfig default.
        from roomkit.providers.xai.ai import XAIAIProvider

        assert XAIAIProvider.available_models()[0].id == "grok-4.5"
        assert _config().model == XAIAIProvider.available_models()[0].id


class TestXAIReasoning:
    """Reasoning depth rides the top-level ``reasoning_effort`` string."""

    def _provider(self, **cfg_overrides: Any) -> Any:
        from roomkit.providers.xai.ai import XAIAIProvider

        provider = XAIAIProvider.__new__(XAIAIProvider)
        provider._config = _config(**cfg_overrides)
        return provider

    def test_effort_sent_when_configured(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, _context())
        assert kwargs["reasoning_effort"] == "high"

    def test_omitted_when_unset(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context())
        assert "reasoning_effort" not in kwargs

    def test_effort_kept_on_tool_turns(self) -> None:
        # Diverges from the OpenAI parent and OpenRouter on purpose: Grok reasons
        # unconditionally, so effort is the only lever over the cost of an
        # agentic turn — the turns that spend the most.
        kwargs: dict[str, Any] = {}
        ctx = _context(tools=[AITool(name="x", description="d", parameters={})])
        self._provider(reasoning_effort="low")._apply_sampling_kwargs(kwargs, ctx)
        assert kwargs["reasoning_effort"] == "low"

    def test_withheld_from_non_reasoning_model(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(
            model="grok-4.20-0309-non-reasoning", reasoning_effort="high"
        )._apply_sampling_kwargs(kwargs, _context())
        assert "reasoning_effort" not in kwargs

    def test_sent_for_unknown_model(self) -> None:
        # An id the catalog does not know (alias, newer model) counts as capable.
        kwargs: dict[str, Any] = {}
        self._provider(model="grok-latest", reasoning_effort="medium")._apply_sampling_kwargs(
            kwargs, _context()
        )
        assert kwargs["reasoning_effort"] == "medium"

    def test_temperature_still_applied(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context(temperature=0.3))
        assert kwargs["temperature"] == 0.3

    def test_temperature_dropped_when_unsupported(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(supports_custom_temperature=False)._apply_sampling_kwargs(
            kwargs, _context(temperature=0.3)
        )
        assert "temperature" not in kwargs
