"""The Mistral wire: Mistral's own SDK client, its HTTP answered in place.

The provider talks to a real ``mistralai`` client whose transport answers the
script as the server-sent events Mistral streams, so the SDK itself serializes
each request and decodes each event. A request is recorded as the JSON body the
SDK put on the wire.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from mistralai.client import Mistral

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from tests.text_conformance.chat_wire import FINISH, ChatDriver, assistant_items, wire_arguments
from tests.text_conformance.driver import (
    CACHE_WRITE_USAGE,
    COMPOSITION,
    FILTER_STOP,
    FUNCTIONLESS_CALL,
    MALFORMED_CALL,
    REASONING_USAGE,
    REDACTED_REASONING,
    SIGNED_REASONING,
    STREAM_WITHOUT_FINISH,
    Driver,
)
from tests.text_conformance.script import Call, Item, Reasoning, Script, Usage

_MODEL = "mistral-large-latest"


# Mistral names the context window filling up mid-answer ``model_length``.
_FINISH = {**FINISH, "context": "model_length"}


def _chunk(delta: dict[str, Any], finish: str | None = None, usage: Any = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "cmpl-0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": _MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _usage(usage: Usage) -> dict[str, Any]:
    # As Mistral sends it (measured 2026-10-02); the SDK's chat UsageInfo does
    # not declare the detail field, so it hands it over as a dict.
    prompt = usage.input + usage.cache_read + usage.cache_write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": usage.output,
        "total_tokens": prompt + usage.output,
        "prompt_tokens_details": {"cached_tokens": usage.cache_read},
    }


def _thinking(block: Reasoning) -> dict[str, Any]:
    text = [{"type": "text", "text": block.text}]
    chunk: dict[str, Any] = {"type": "thinking", "thinking": text}
    if block.signature:
        chunk["signature"] = block.signature
    return chunk


def _call(call: Call) -> dict[str, Any]:
    # Mistral sends a call whole: the SDK's FunctionCall requires its name on
    # every call, so arguments never stream in fragments and ``Call.fragments``
    # has no Mistral form. An absent id or index is left out, as the server
    # leaves it out; the SDK then fills them ("null", 0).
    wire: dict[str, Any] = {"function": {"name": call.name, "arguments": wire_arguments(call)}}
    if call.id is not None:
        wire["id"] = call.id
    if call.index is not None:
        wire["index"] = call.index
    return wire


def _call_chunks(script: Script) -> list[dict[str, Any]]:
    calls = [_call(call) for call in script.calls]
    if script.calls_in_one_chunk:
        return [_chunk({"tool_calls": calls})] if calls else []
    return [_chunk({"tool_calls": [call]}) for call in calls]


def _events(script: Script) -> bytes:
    """The response as the server-sent events the SDK reads."""
    chunks = [_chunk({"role": "assistant", "content": ""})]
    chunks.extend(
        _chunk({"content": [_thinking(block)]})
        for block in script.reasoning
        if block.redacted is None
    )
    if script.text:
        chunks.append(_chunk({"content": script.text}))
    chunks.extend(_call_chunks(script))
    finish = _FINISH[script.finish]
    if finish is not None:
        # Mistral's server puts the usage on the chunk that stops (measured
        # 2026-10-02); an OpenAI-compatible server behind its SDK may send it
        # on a chunk of its own, with no choice.
        usage = _usage(script.usage)
        chunks.append(_chunk({"content": ""}, finish, None if script.usage_alone else usage))
        if script.usage_alone:
            chunks.append({**_chunk({}), "choices": [], "usage": usage})
    chunks = [{**chunk, "model": script.answered_by} for chunk in chunks]
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if finish is not None:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _round_items(message: dict[str, Any]) -> list[Item]:
    """An assistant round, its ThinkChunks read as the signed blocks they are."""
    content = message.get("content")
    if not isinstance(content, list):
        return assistant_items(message)
    thinking: list[Item] = [
        ("thinking", "".join(t.get("text", "") for t in c["thinking"]), c.get("signature"))
        for c in content
        if c.get("type") == "thinking"
    ]
    rest = [c for c in content if c.get("type") != "thinking"]
    return thinking + assistant_items({**message, "content": rest})


class MistralWire(ChatDriver):
    label = "mistral"
    covers = (MistralAIProvider,)
    cannot = {
        COMPOSITION: "Mistral streams each call whole, its arguments in one piece",
        CACHE_WRITE_USAGE: "Mistral's usage has no cache-write counter",
        MALFORMED_CALL: "Mistral has no stop reason for a call it could not parse",
        FILTER_STOP: "Mistral's finish reasons are stop, length, model_length, error, tool_calls",
        REDACTED_REASONING: "a Mistral ThinkChunk carries text and a signature, no redacted form",
        STREAM_WITHOUT_FINISH: "Mistral streams each call whole in one chunk: none stops mid-call",
        FUNCTIONLESS_CALL: "the SDK's ToolCall requires its function",
        # Measured 2026-10-02 on mistral-medium-latest with reasoning_effort high.
        REASONING_USAGE: "Mistral counts reasoning inside completion_tokens, with no breakdown",
        SIGNED_REASONING: (
            "no Mistral model measured signs a ThinkChunk; the provider keeps no "
            "signature (RMK-385)"
        ),
    }
    reasoning = "inline"
    refused_names = ("files:read",)
    accepted_names = ("lookup", "files.read")

    def provider(self, script: Script) -> AIProvider:
        provider = MistralAIProvider(MistralConfig(api_key="k", model=_MODEL))
        events = _events(script)

        async def answer(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=events
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(answer))
        provider._client = Mistral(api_key="k", async_client=http)
        return provider

    def assistant_items(self, message: dict[str, Any]) -> list[Item]:
        return _round_items(message)


def wires() -> list[Driver]:
    return [MistralWire()]
