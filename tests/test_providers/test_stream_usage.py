"""A streamed turn of the OpenAI family reports its usage and a cost (RMK-312).

Every tool round streams, and the server sends usage in a stream only when
``stream_options.include_usage`` asks for it: the family asks by default, and
a server that rejects the option sets ``include_stream_usage=False``.
Cerebras keeps False: it sends usage unasked. ``reasoning_tokens`` is the
thinking share of ``output_tokens``; xAI, which reports it beside the
completion count, has it folded in.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, StreamDone
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.vllm.config import VLLMConfig
from roomkit.providers.xai.ai import XAIAIProvider

_MODEL = "gpt-5.6-sol"
_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 30,
    "total_tokens": 130,
    "prompt_tokens_details": {"cached_tokens": 80},
    "completion_tokens_details": {"reasoning_tokens": 20},
}


_Config = OpenAIConfig | AzureAIConfig | VLLMConfig


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(OpenAIConfig(api_key="k", model=_MODEL), True, id="openai"),
        pytest.param(DeepSeekConfig(api_key="k", model="deepseek-chat"), True, id="deepseek"),
        pytest.param(
            AzureAIConfig(api_key="k", azure_endpoint="https://x", model="d"), True, id="azure"
        ),
        pytest.param(VLLMConfig(model="m"), True, id="vllm"),
        pytest.param(CerebrasConfig(api_key="k", model="gpt-oss-120b"), False, id="cerebras"),
    ],
)
def test_the_openai_family_asks_for_stream_usage_by_default(
    config: _Config, expected: bool
) -> None:
    assert config.include_stream_usage is expected


def _stream(
    requests: list[dict[str, Any]], usage: dict[str, Any] = _USAGE
) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        frames = [
            {"choices": [{"index": 0, "delta": {"content": "Sunny."}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        if body.get("stream_options", {}).get("include_usage"):
            frames.append({"choices": [], "usage": usage})
        chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": _MODEL}
        payload = "".join(f"data: {json.dumps({**chunk, **f})}\n\n" for f in frames)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload + "data: [DONE]\n\n",
        )

    return handle


@asynccontextmanager
async def _provider(
    handler: Callable[[httpx.Request], httpx.Response], **config: Any
) -> AsyncIterator[OpenAIAIProvider]:
    pytest.importorskip("openai")
    provider = OpenAIAIProvider(
        OpenAIConfig(api_key="k", model=_MODEL, **config), transport=httpx.MockTransport(handler)
    )
    try:
        yield provider
    finally:
        await provider.close()


async def _streamed_usage(provider: OpenAIAIProvider) -> dict[str, int]:
    context = AIContext(messages=[AIMessage(role="user", content="Weather?")])
    events = [event async for event in provider.generate_structured_stream(context)]
    assert isinstance(events[-1], StreamDone)
    return events[-1].usage


async def test_a_streamed_turn_reports_its_usage_and_a_cost() -> None:
    requests: list[dict[str, Any]] = []
    async with _provider(_stream(requests)) as provider:
        usage = await _streamed_usage(provider)
        entry = provider.catalog_entry()

    assert requests[0]["stream_options"] == {"include_usage": True}
    # reasoning_tokens is a detail of output_tokens, which already counts it.
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 30,
        "cache_read_input_tokens": 80,
        "reasoning_tokens": 20,
    }
    assert entry is not None and entry.pricing is not None
    assert entry.pricing.cost_for(usage) > 0
    assert entry.pricing.cost_for(usage) == entry.pricing.cost_for(
        {k: v for k, v in usage.items() if k != "reasoning_tokens"}
    )


async def test_a_server_that_rejects_the_option_can_turn_it_off() -> None:
    requests: list[dict[str, Any]] = []
    async with _provider(_stream(requests), include_stream_usage=False) as provider:
        usage = await _streamed_usage(provider)

    assert "stream_options" not in requests[0]
    assert usage == {}


@pytest.mark.parametrize(
    ("sent", "read"),
    [
        ({"completion_tokens": 7}, {"output_tokens": 7}),
        # A count the SDK does not declare reaches the provider as written:
        # one that is no whole number is none, where it raised TypeError.
        (
            {"prompt_tokens": 100, "completion_tokens": 7, "cache_write_tokens": "12"},
            {"input_tokens": 100, "output_tokens": 7},
        ),
        # One it declares it reads as pydantic's lax int does: "12" is 12.
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cache_write_tokens": "12"},
            },
            {"input_tokens": 88, "output_tokens": 7, "cache_creation_input_tokens": 12},
        ),
    ],
    ids=["no-prompt", "a-usage-cache-write-of-another-shape", "a-details-cache-write-as-text"],
)
async def test_a_count_reads_as_the_sdk_built_it(
    sent: dict[str, Any], read: dict[str, int]
) -> None:
    """What the server wrote, through the SDK's own objects (RFC §6.4, RMK-1441)."""
    async with _provider(_stream([], sent)) as provider:
        assert await _streamed_usage(provider) == read


def _xai_usage(completion: int) -> dict[str, int]:
    """xAI's usage for 279 prompt and 89 reasoning tokens, 374 in all."""
    return XAIAIProvider._usage_from(
        SimpleNamespace(
            prompt_tokens=279,
            completion_tokens=completion,
            total_tokens=374,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=89),
        )
    )


def test_xai_reasoning_reported_beside_the_completion_is_billed_as_output() -> None:
    apart = _xai_usage(completion=6)
    inside = _xai_usage(completion=95)

    assert apart == {"input_tokens": 279, "output_tokens": 95, "reasoning_tokens": 89}
    # Shaped like OpenAI's, the reasoning is already in the completion: never twice.
    assert inside == apart


def test_xai_reasoning_beside_an_omitted_completion_leaves_the_output_unknown() -> None:
    """A completion the server leaves out stays unknown (RFC §6.4, RMK-1441),
    even where the total says the reasoning was reported apart."""
    usage = XAIAIProvider._usage_from(
        SimpleNamespace(
            prompt_tokens=279,
            completion_tokens=None,
            total_tokens=368,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=89),
        )
    )

    assert usage == {"input_tokens": 279, "reasoning_tokens": 89}
