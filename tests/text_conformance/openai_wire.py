"""The OpenAI Chat Completions wire: OpenAI and the ten providers built on it.

Each provider keeps a real ``openai`` client whose HTTP transport answers the
script as the vendor's server writes it, server-sent events when streamed, so
the SDK itself builds every object the provider reads. A request is recorded as
the JSON body the SDK put on the wire.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import openai

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.cerebras.ai import CerebrasAIProvider
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.llamacpp.ai import LlamaCppAIProvider
from roomkit.providers.llamacpp.config import LlamaCppConfig
from roomkit.providers.meta.ai import MetaAIProvider
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.vllm import VLLMConfig, _VLLMProvider, create_vllm_provider
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig
from tests.text_conformance.chat_wire import (
    FINISH,
    ChatDriver,
    assistant_items,
    call_pieces,
    wire_arguments,
)
from tests.text_conformance.driver import (
    CACHE_WRITE_USAGE,
    MALFORMED_CALL,
    REDACTED_REASONING,
    SIGNED_REASONING,
    Driver,
    ReasoningConvention,
)
from tests.text_conformance.script import Call, Item, Script, Usage

_CANNOT = {
    SIGNED_REASONING: "Chat Completions carries no reasoning signature",
    REDACTED_REASONING: "Chat Completions has no redacted reasoning",
    CACHE_WRITE_USAGE: "this vendor's Chat Completions usage has no cache-write counter",
    MALFORMED_CALL: "Chat Completions has no stop reason for a call the server could not parse",
}
# OpenRouter and LiteLLM carry signed and redacted reasoning (reasoning_details,
# thinking_blocks): what they cannot do here is the provider's choice.
_OPENROUTER_CANNOT = {
    MALFORMED_CALL: _CANNOT[MALFORMED_CALL],
    SIGNED_REASONING: (
        "the provider replays a round's reasoning as <think> text, not as "
        "reasoning_details, which every upstream answers (measured 2026-09-30)"
    ),
    REDACTED_REASONING: "the provider reads no reasoning_details, encrypted ones included",
}
_LITELLM_CANNOT = {
    **_CANNOT,
    SIGNED_REASONING: "the provider reads reasoning_content, not LiteLLM's thinking_blocks",
    REDACTED_REASONING: "the provider reads reasoning_content, not LiteLLM's thinking_blocks",
}


def _openai_usage(usage: Usage) -> dict[str, Any]:
    """OpenAI's: the prompt counts its cached prefix, the completion its
    reasoning."""
    prompt = usage.input + usage.cache_read
    return {
        "prompt_tokens": prompt,
        "completion_tokens": usage.output,
        "total_tokens": prompt + usage.output,
        "prompt_tokens_details": {"cached_tokens": usage.cache_read},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning},
    }


def _openrouter_usage(usage: Usage) -> dict[str, Any]:
    """OpenRouter's: OpenAI's, the prompt counting its cache writes too."""
    shaped = _openai_usage(usage)
    prompt = usage.input + usage.cache_read + usage.cache_write
    shaped["prompt_tokens"] = prompt
    shaped["total_tokens"] = prompt + usage.output
    shaped["prompt_tokens_details"]["cache_write_tokens"] = usage.cache_write
    return shaped


def _deepseek_usage(usage: Usage) -> dict[str, Any]:
    """DeepSeek's: OpenAI's, with its own hit and miss counters beside."""
    return {
        **_openai_usage(usage),
        "prompt_cache_hit_tokens": usage.cache_read,
        "prompt_cache_miss_tokens": usage.input,
    }


def _xai_usage(usage: Usage) -> dict[str, Any]:
    """xAI's: reasoning counted beside the completion, the total over all three."""
    shaped = _openai_usage(usage)
    completion = usage.output - usage.reasoning
    shaped["completion_tokens"] = completion
    shaped["total_tokens"] = shaped["prompt_tokens"] + completion + usage.reasoning
    return shaped


@dataclass(frozen=True)
class _Vendor:
    """What one provider's server writes its own way on the shared wire
    (measured 2026-10-02 for DeepSeek, xAI and OpenRouter)."""

    usage: Callable[[Usage], dict[str, Any]] = _openai_usage
    reasoning_field: str = "reasoning_content"
    """The field a response's reasoning comes in, and the one earlier reasoning
    goes back in on a wire that replays it in a field."""


def _fragment(call: Call, first: bool, piece: Any) -> dict[str, Any]:
    if call.functionless:
        # A custom tool's entry: an id and a type, no function to run.
        return {"index": call.index or 0, "id": call.id, "type": "custom"}
    fragment: dict[str, Any] = {"function": {"arguments": piece}}
    if call.index is not None:
        fragment["index"] = call.index
    if first:
        fragment["type"] = "function"
        fragment["function"]["name"] = call.name
        if call.id is not None:
            fragment["id"] = call.id
    return fragment


def _chunk(delta: dict[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": choices,
    }


def _call_chunks(script: Script) -> list[dict[str, Any]]:
    cut = [call_pieces(call) for call in script.calls]
    chunks: list[dict[str, Any]] = []
    if script.calls_in_one_chunk:
        firsts = [_fragment(c, True, p[0]) for c, p in zip(script.calls, cut, strict=True)]
        chunks.append(_chunk({"tool_calls": firsts}))
        for call, rest in zip(script.calls, cut, strict=True):
            chunks.extend(_chunk({"tool_calls": [_fragment(call, False, p)]}) for p in rest[1:])
        return chunks
    for call, parts in zip(script.calls, cut, strict=True):
        for n, piece in enumerate(parts):
            chunks.append(_chunk({"tool_calls": [_fragment(call, n == 0, piece)]}))
    return chunks


def _stream(script: Script, vendor: _Vendor) -> bytes:
    """The response as the server-sent events the SDK reads, usage last."""
    chunks = [
        _chunk({vendor.reasoning_field: block.text}) for block in script.reasoning if block.text
    ]
    if script.text:
        chunks.append(_chunk({"content": script.text}))
    chunks.extend(_call_chunks(script))
    finish = FINISH[script.finish]
    if finish is not None:
        chunks.append(_chunk({}, finish))
    # The usage on a chunk of its own, with no choice, as the server sends it
    # under ``stream_options.include_usage`` (``usage_alone`` or not).
    chunks.append({**_chunk(), "usage": vendor.usage(script.usage)})
    chunks = [{**chunk, "model": script.answered_by} for chunk in chunks]
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*lines, "data: [DONE]\n\n"]).encode()


def _response_call(call: Call) -> dict[str, Any]:
    if call.functionless:
        return {"id": call.id, "type": "custom", "custom": {"name": call.name, "input": ""}}
    rendered: dict[str, Any] = {
        "type": "function",
        "function": {"name": call.name, "arguments": wire_arguments(call)},
    }
    if call.id is not None:
        rendered["id"] = call.id
    return rendered


def _response(script: Script, vendor: _Vendor) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": script.text or None}
    if script.calls:
        message["tool_calls"] = [_response_call(call) for call in script.calls]
    reasoning = "".join(block.text for block in script.reasoning)
    if reasoning:
        message[vendor.reasoning_field] = reasoning
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 0,
        "model": script.answered_by,
        "choices": [{"index": 0, "message": message, "finish_reason": FINISH[script.finish]}],
        "usage": vendor.usage(script.usage),
    }


def _client(script: Script, vendor: _Vendor, requests: list[Any]) -> openai.AsyncOpenAI:
    """The ``openai`` client whose HTTP transport answers *script*."""

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, headers=headers, content=_stream(script, vendor))
        return httpx.Response(200, json=_response(script, vendor))

    http = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    return openai.AsyncOpenAI(
        api_key="k", base_url="http://wire.test/v1", http_client=http, max_retries=0
    )


class _StartedServer:
    """A ``llama-server`` already running: nothing to download or start."""

    base_url = "http://127.0.0.1:1/v1"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _llamacpp() -> AIProvider:
    provider = LlamaCppAIProvider(LlamaCppConfig(model="unsloth/Qwen3-4B-GGUF:Q4_K_M"))
    provider._server = _StartedServer()  # type: ignore[assignment]
    return provider


class OpenAIWire(ChatDriver):
    """A provider built on OpenAI's Chat Completions client."""

    def __init__(
        self,
        provider_cls: type[AIProvider],
        build: Callable[[], AIProvider],
        *,
        label: str,
        vendor: _Vendor | None = None,
        reasoning: ReasoningConvention = "inline",
        cannot: Mapping[str, str] = _CANNOT,
        refused_names: tuple[str, ...] = (),
        accepted_names: tuple[str, ...] = ("lookup", "files.read"),
    ) -> None:
        super().__init__()
        self._build = build
        self._vendor = vendor or _Vendor()
        self.label = label
        self.covers = (provider_cls,)
        self.cannot = cannot
        self.reasoning = reasoning
        self.refused_names = refused_names
        self.accepted_names = accepted_names

    def provider(self, script: Script) -> AIProvider:
        provider = self._build()
        provider._client = _client(script, self._vendor, self.requests)  # type: ignore[attr-defined]
        return provider

    def assistant_items(self, message: dict[str, Any]) -> list[Item]:
        field = self._vendor.reasoning_field if self.reasoning == "field" else None
        return assistant_items(message, field)


def wires() -> list[Driver]:
    """One driver per provider class on the OpenAI wire."""
    return [
        OpenAIWire(
            OpenAIAIProvider,
            lambda: OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-5.4")),
            label="openai",
            refused_names=("files.read",),
            accepted_names=("lookup",),
        ),
        OpenAIWire(
            AzureAIProvider,
            lambda: AzureAIProvider(
                AzureAIConfig(api_key="k", azure_endpoint="https://x.azure.com", model="m")
            ),
            label="azure",
        ),
        OpenAIWire(
            CerebrasAIProvider,
            lambda: CerebrasAIProvider(CerebrasConfig(api_key="k", model="gpt-oss-120b")),
            label="cerebras",
            vendor=_Vendor(reasoning_field="reasoning"),
            reasoning="field",
        ),
        OpenAIWire(
            DeepSeekAIProvider,
            lambda: DeepSeekAIProvider(DeepSeekConfig(api_key="k", model="deepseek-v4-pro")),
            label="deepseek",
            vendor=_Vendor(usage=_deepseek_usage),
            reasoning="field",
            refused_names=("files.read",),
            accepted_names=("lookup",),
        ),
        OpenAIWire(
            LiteLLMAIProvider,
            lambda: LiteLLMAIProvider(LiteLLMConfig(api_key="k", model="gpt-4o")),
            label="litellm",
            cannot=_LITELLM_CANNOT,
        ),
        OpenAIWire(LlamaCppAIProvider, _llamacpp, label="llamacpp"),
        OpenAIWire(
            MetaAIProvider,
            lambda: MetaAIProvider(MetaConfig(api_key="k", model="muse-spark")),
            label="meta",
        ),
        OpenAIWire(
            OpenRouterAIProvider,
            lambda: OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="openai/gpt-4o")),
            label="openrouter",
            vendor=_Vendor(usage=_openrouter_usage, reasoning_field="reasoning"),
            cannot=_OPENROUTER_CANNOT,
        ),
        OpenAIWire(
            QwenAIProvider,
            lambda: QwenAIProvider(QwenConfig(api_key="k", model="qwen-plus")),
            label="qwen",
        ),
        OpenAIWire(
            _VLLMProvider,
            lambda: create_vllm_provider(VLLMConfig(model="local")),
            label="vllm",
        ),
        OpenAIWire(
            XAIAIProvider,
            lambda: XAIAIProvider(XAIConfig(api_key="k", model="grok-4")),
            label="xai",
            vendor=_Vendor(usage=_xai_usage),
        ),
    ]
