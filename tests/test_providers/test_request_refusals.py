"""What a provider knows of its vendor before the request (RFC §6.7, RMK-309).

A model OpenAI's Chat Completions does not serve, or one that takes no
function tools there, fails before the request; Anthropic flags a tool result
whose call did not succeed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITool,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.anthropic.request import build_messages
from roomkit.providers.openai.config import OpenAIConfig

_TOOLS = [AITool(name="now", description="d")]


def _openai(model: str, **config: Any) -> Any:
    module = MagicMock()
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    with patch.dict(sys.modules, {"openai": module}):
        from roomkit.providers.openai.ai import OpenAIAIProvider

        provider = OpenAIAIProvider(OpenAIConfig(api_key="k", model=model, **config))
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("sent"))
    return provider


def _context(*, tools: bool) -> AIContext:
    return AIContext(
        messages=[AIMessage(role="user", content="hi")], tools=_TOOLS if tools else []
    )


class TestOpenAIModelsChatRefuses:
    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
    async def test_tools_for_a_model_that_takes_none_fail_before_the_request(
        self, model: str
    ) -> None:
        provider = _openai(model)

        with pytest.raises(ProviderError, match="takes no function tools") as raised:
            await provider.generate(_context(tools=True))

        assert raised.value.retryable is False
        assert raised.value.before_request is True
        provider._client.chat.completions.create.assert_not_awaited()

    async def test_the_same_model_without_tools_is_asked(self) -> None:
        provider = _openai("gpt-6-astra")

        with pytest.raises(ProviderError, match="sent"):
            await provider.generate(_context(tools=False))

        provider._client.chat.completions.create.assert_awaited_once()

    @pytest.mark.parametrize("model", ["gpt-5.5-pro", "gpt-5.4-pro", "o3-pro"])
    async def test_a_responses_only_model_fails_before_the_request(self, model: str) -> None:
        provider = _openai(model)

        with pytest.raises(ProviderError, match="Responses API only") as raised:
            [e async for e in provider.generate_structured_stream(_context(tools=False))]

        assert raised.value.before_request is True

        provider._client.chat.completions.create.assert_not_awaited()

    async def test_a_server_behind_a_base_url_decides(self) -> None:
        provider = _openai("gpt-6-astra", base_url="http://localhost:8000/v1")

        with pytest.raises(ProviderError, match="sent"):
            await provider.generate(_context(tools=True))

        provider._client.chat.completions.create.assert_awaited_once()


def _result(*, is_error: bool) -> AIMessage:
    part = AIToolResultPart(tool_call_id="t1", name="now", result="r", is_error=is_error)
    return AIMessage(role="tool", content=[part])


class TestAnthropicErrorFlag:
    def test_a_result_whose_call_did_not_succeed_is_flagged(self) -> None:
        [message] = build_messages([_result(is_error=True)])

        assert message["content"][0]["is_error"] is True

    def test_a_served_result_carries_no_flag(self) -> None:
        [message] = build_messages([_result(is_error=False)])

        assert "is_error" not in message["content"][0]
