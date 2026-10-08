"""A turn with tools carries the reasoning the config says the server takes (RFC §6.7).

Where a provider cannot know the model behind its endpoint (OpenAI behind a
``base_url``, an Azure deployment name, a LiteLLM alias), one rule decides:
``supports_reasoning_effort_with_tools`` on the config. ``True`` sends the
turn's reasoning as on any turn, ``False`` leaves it out, unset leaves it out
and warns once. Every case runs on each of the three paths.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

_TOOL = AITool(name="lookup", description="x", parameters={})
_FLAG = "supports_reasoning_effort_with_tools"


def _openai_behind_base_url(**cfg: Any) -> OpenAIAIProvider:
    provider = OpenAIAIProvider.__new__(OpenAIAIProvider)
    provider._config = OpenAIConfig(
        api_key="k", model="mercury-2.5", base_url="https://api.compatible.test/v1", **cfg
    )
    return provider


def _azure(**cfg: Any) -> AzureAIProvider:
    provider = AzureAIProvider.__new__(AzureAIProvider)
    provider._config = AzureAIConfig(
        api_key="k", azure_endpoint="https://res.openai.azure.com", model="o4-mini-prod", **cfg
    )
    return provider


def _litellm(**cfg: Any) -> LiteLLMAIProvider:
    provider = LiteLLMAIProvider.__new__(LiteLLMAIProvider)
    provider._config = LiteLLMConfig(api_key="k", model="team-reasoner", **cfg)
    return provider


PATHS: dict[str, Callable[..., OpenAIAIProvider]] = {
    "openai-base-url": _openai_behind_base_url,
    "azure": _azure,
    "litellm": _litellm,
}


@pytest.fixture(params=sorted(PATHS))
def build(request: pytest.FixtureRequest) -> Callable[..., OpenAIAIProvider]:
    """The provider of one path whose model this provider cannot know."""
    return PATHS[request.param]


def _context(**overrides: Any) -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="Hi")], **overrides)


def _sent(provider: OpenAIAIProvider, context: AIContext) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    provider._apply_sampling_kwargs(kwargs, context)
    return kwargs


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING and _FLAG in r.getMessage()]


class TestTheConfigDecides:
    def test_a_server_said_to_take_it_gets_the_effort_on_a_tool_turn(
        self, build: Callable[..., OpenAIAIProvider]
    ) -> None:
        provider = build(reasoning_effort="low", **{_FLAG: True})

        assert _sent(provider, _context(tools=[_TOOL]))["reasoning_effort"] == "low"

    def test_the_turn_effort_outranks_the_config_there(
        self, build: Callable[..., OpenAIAIProvider]
    ) -> None:
        provider = build(reasoning_effort="high", **{_FLAG: True})

        sent = _sent(provider, _context(tools=[_TOOL], reasoning_effort="minimal"))

        assert sent["reasoning_effort"] == "minimal"

    def test_a_server_said_not_to_take_it_gets_nothing_and_no_warning(
        self, build: Callable[..., OpenAIAIProvider], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = build(reasoning_effort="low", **{_FLAG: False})

        with caplog.at_level(logging.WARNING):
            sent = _sent(provider, _context(tools=[_TOOL], thinking_budget=2048))

        assert "reasoning_effort" not in sent
        assert "extra_body" not in sent
        assert _warnings(caplog) == []

    def test_a_turn_without_tools_is_left_as_it_was(
        self, build: Callable[..., OpenAIAIProvider]
    ) -> None:
        provider = build(reasoning_effort="low", **{_FLAG: False})

        assert _sent(provider, _context())["reasoning_effort"] == "low"


class TestUnsaid:
    def test_it_is_left_out_with_one_warning_naming_the_field(
        self, build: Callable[..., OpenAIAIProvider], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = build(reasoning_effort="low")

        with caplog.at_level(logging.WARNING):
            first = _sent(provider, _context(tools=[_TOOL]))
            second = _sent(provider, _context(tools=[_TOOL]))

        assert "reasoning_effort" not in first
        assert "reasoning_effort" not in second
        assert len(_warnings(caplog)) == 1

    def test_nothing_stated_warns_of_nothing(
        self, build: Callable[..., OpenAIAIProvider], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = build()

        with caplog.at_level(logging.WARNING):
            sent = _sent(provider, _context(tools=[_TOOL]))

        assert "reasoning_effort" not in sent
        assert _warnings(caplog) == []


class TestPathRules:
    """What each path adds to the shared rule."""

    @pytest.mark.parametrize("path", ["openai-base-url", "azure"])
    def test_a_turn_that_states_off_sends_none_where_the_server_takes_it(self, path: str) -> None:
        provider = PATHS[path](reasoning_effort="low", **{_FLAG: True})

        sent = _sent(provider, _context(tools=[_TOOL], reasoning_effort="none"))

        assert sent["reasoning_effort"] == "none"

    def test_litellm_sends_the_budget_where_the_proxy_takes_it(self) -> None:
        provider = _litellm(**{_FLAG: True})

        sent = _sent(provider, _context(tools=[_TOOL], thinking_budget=2048))

        assert sent["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_litellm_warns_of_a_budget_it_leaves_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            sent = _sent(_litellm(), _context(tools=[_TOOL], thinking_budget=2048))

        assert "extra_body" not in sent
        assert len(_warnings(caplog)) == 1


class TestOnOpenAIsOwnEndpoint:
    """Unset, the catalogue decides; stated, the config outranks it."""

    @staticmethod
    def _official(model: str, **cfg: Any) -> OpenAIAIProvider:
        provider = OpenAIAIProvider.__new__(OpenAIAIProvider)
        provider._config = OpenAIConfig(api_key="k", model=model, **cfg)
        return provider

    def test_unset_a_tagged_model_follows_the_catalogue_without_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tools = _context(tools=[_TOOL])

        with caplog.at_level(logging.WARNING):
            before_5_4 = _sent(self._official("gpt-5-mini", reasoning_effort="low"), tools)
            from_5_4 = _sent(self._official("gpt-5.5", reasoning_effort="low"), tools)

        assert before_5_4["reasoning_effort"] == "low"
        assert from_5_4["reasoning_effort"] == "none"
        assert _warnings(caplog) == []

    @pytest.mark.parametrize(("stated", "expected"), [(True, "low"), (False, None)])
    def test_a_stated_value_outranks_the_catalogue(
        self, stated: bool, expected: str | None
    ) -> None:
        provider = self._official("gpt-5.5", reasoning_effort="low", **{_FLAG: stated})

        assert _sent(provider, _context(tools=[_TOOL])).get("reasoning_effort") == expected

    def test_a_model_the_catalogue_does_not_tag_is_left_out_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = self._official("gpt-7-preview", reasoning_effort="low")

        with caplog.at_level(logging.WARNING):
            sent = _sent(provider, _context(tools=[_TOOL]))

        assert "reasoning_effort" not in sent
        assert len(_warnings(caplog)) == 1


def _openai_module() -> MagicMock:
    module = MagicMock()
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    return module


def _completion() -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="mercury-2.5",
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, prompt_tokens_details=None),
    )


async def _chunks() -> Any:
    delta = SimpleNamespace(content="ok", tool_calls=None)
    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")], usage=None)


class TestOnTheWire:
    """Both request doors, ``generate`` and the stream, send it."""

    @staticmethod
    def _provider() -> OpenAIAIProvider:
        config = OpenAIConfig(
            api_key="k",
            model="mercury-2.5",
            base_url="https://api.compatible.test/v1",
            reasoning_effort="low",
            **{_FLAG: True},
        )
        with patch.dict("sys.modules", {"openai": _openai_module()}):
            provider = OpenAIAIProvider(config)
        provider._client = MagicMock()
        return provider

    async def test_generate(self) -> None:
        provider = self._provider()
        provider._client.chat.completions.create = AsyncMock(return_value=_completion())

        await provider.generate(_context(tools=[_TOOL]))

        sent = provider._client.chat.completions.create.call_args.kwargs
        assert sent["reasoning_effort"] == "low"
        assert sent["tools"]

    async def test_stream(self) -> None:
        provider = self._provider()
        provider._client.chat.completions.create = AsyncMock(return_value=_chunks())

        async for _ in provider.generate_structured_stream(_context(tools=[_TOOL])):
            pass

        sent = provider._client.chat.completions.create.call_args.kwargs
        assert sent["reasoning_effort"] == "low"
        assert sent["tools"]
