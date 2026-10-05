"""The PolarGrid wire: Chat Completions through polargrid-sdk's own client.

The driver keeps the SDK's client and replaces only its HTTP layer, which
answers the script as PolarGrid's server writes it. The SDK itself then checks
each request and builds the objects the provider reads: its
``ChatCompletionResponse``, and a ``ChatCompletionChunk`` per streamed line; a
stream's usage, which the SDK cannot carry, comes through the provider's SDK
patch. ``requests`` holds the bodies posted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import polargrid

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from tests.text_conformance.chat_wire import FINISH, ChatDriver, call_pieces, wire_arguments
from tests.text_conformance.driver import (
    CACHE_USAGE,
    CACHE_WRITE_USAGE,
    CALL_INDEX,
    FUNCTIONLESS_CALL,
    MALFORMED_CALL,
    OBJECT_ARGUMENTS_RESPONSE,
    REASONING_USAGE,
    REDACTED_REASONING,
    RESPONSE_CALL_WITHOUT_ID,
    SIGNED_REASONING,
    Driver,
)
from tests.text_conformance.script import Call, Script

_MODEL = "qwen-3.8-27b"


def _present(**fields: Any) -> dict[str, Any]:
    """A JSON object as the server writes it: no key for what it does not send."""
    return {key: value for key, value in fields.items() if value is not None}


def _content(script: Script) -> str | None:
    # Qwen on PolarGrid reasons inline, ahead of its answer, in the content.
    reasoning = "".join(block.text for block in script.reasoning)
    content = f"<think>{reasoning}</think>{script.text}" if reasoning else script.text
    return content or None


def _usage(script: Script) -> dict[str, int]:
    usage = script.usage
    # The server counts every prompt token, cached or not, as a prompt token;
    # the SDK's TokenUsage has no field to set the cached ones apart.
    prompt = usage.input + usage.cache_read + usage.cache_write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": usage.output,
        "total_tokens": prompt + usage.output,
    }


def _chunk(
    delta: dict[str, Any] | None = None,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return _present(
        id="chatcmpl-0",
        object="chat.completion.chunk",
        created=0,
        model=_MODEL,
        choices=choices,
        usage=usage,
    )


def _fragment(call: Call, first: bool, piece: Any) -> dict[str, Any]:
    if not first:
        return _present(index=call.index, function={"arguments": piece})
    return _present(
        index=call.index,
        id=call.id,
        type="function",
        function={"name": call.name, "arguments": piece},
    )


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


def _stream(script: Script, body: dict[str, Any]) -> list[dict[str, Any]]:
    """The stream's lines, each a JSON chunk; the usage last, when the request
    asks for it, as the server sends it (measured 2026-10-02)."""
    content = _content(script)
    chunks = [_chunk({"role": "assistant", "content": content})] if content else []
    chunks.extend(_call_chunks(script))
    finish = FINISH[script.finish]
    if finish is not None:
        chunks.append(_chunk({}, finish))
    if (body.get("stream_options") or {}).get("include_usage"):
        # On a chunk of its own, with no choice (``usage_alone`` or not).
        chunks.append(_chunk(usage=_usage(script)))
    return [{**chunk, "model": script.answered_by} for chunk in chunks]


def _response(script: Script) -> dict[str, Any]:
    calls = [
        _present(
            id=call.id,
            type="function",
            function={"name": call.name, "arguments": wire_arguments(call)},
        )
        for call in script.calls
    ]
    message = _present(role="assistant", content=_content(script), tool_calls=calls or None)
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 0,
        "model": script.answered_by,
        "choices": [{"index": 0, "message": message, "finish_reason": FINISH[script.finish]}],
        "usage": _usage(script),
    }


def _client(script: Script, requests: list[Any]) -> polargrid.PolarGrid:
    """polargrid-sdk's client whose HTTP layer answers *script*."""
    # Nothing listens there: a request that misses the two seams fails at once.
    client = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")

    async def make_request(
        endpoint: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        requests.append(body)
        return _response(script)

    async def stream_post(endpoint: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        requests.append(body)
        for chunk in _stream(script, body):
            yield chunk

    client._make_request = make_request  # type: ignore[method-assign]
    client._stream_post = stream_post  # type: ignore[method-assign]
    return client


class PolarGridWire(ChatDriver):
    label = "polargrid"
    covers = (PolarGridAIProvider,)
    cannot = {
        CALL_INDEX: "polargrid-sdk's ToolCallDelta requires an index; it skips a chunk without",
        SIGNED_REASONING: "Qwen reasons as <think> text in the content, with no signature",
        REDACTED_REASONING: "Qwen reasons as <think> text in the content, never redacted",
        CACHE_USAGE: "polargrid-sdk's TokenUsage holds prompt, completion and total tokens",
        REASONING_USAGE: "polargrid-sdk's TokenUsage holds prompt, completion and total tokens",
        RESPONSE_CALL_WITHOUT_ID: "polargrid-sdk's ToolCall requires an id on a response",
        CACHE_WRITE_USAGE: "polargrid-sdk's TokenUsage holds prompt, completion and total tokens",
        MALFORMED_CALL: "PolarGrid has no stop reason for a call the server could not parse",
        FUNCTIONLESS_CALL: "polargrid-sdk's ToolCall requires its function on a response",
        OBJECT_ARGUMENTS_RESPONSE: (
            "polargrid-sdk's ToolCall types arguments as a string and refuses an object "
            "when it parses the response"
        ),
    }
    reasoning = "dropped"
    # PolarGrid states no tool name rule, and the provider checks none.
    accepted_names = ("lookup", "files.read")

    def provider(self, script: Script) -> AIProvider:
        provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model=_MODEL))
        provider._client = _client(script, self.requests)
        return provider


def wires() -> list[Driver]:
    return [PolarGridWire()]
