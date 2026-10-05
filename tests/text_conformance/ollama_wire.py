"""The Ollama native chat wire: whole calls with no ids, results paired by tool name.

The provider keeps the real ``ollama.AsyncClient``; only its HTTP transport is
fake. A request is recorded as the JSON body put on the wire, by the SDK or,
when it declares tools, by the provider's SDK patch, so what they drop (an
empty field) is dropped here too, and every answer is the SDK's own
``ChatResponse`` read back from its JSON. The server's own tool struct drops
a ``$ref`` and the constraints; that happens past the wire, out of reach.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import ollama

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from tests.text_conformance.chat_wire import ChatDriver
from tests.text_conformance.driver import (
    ARGUMENT_TEXT,
    CACHE_USAGE,
    CACHE_WRITE_USAGE,
    CALL_INDEX,
    COMPOSITION,
    FILTER_STOP,
    FUNCTIONLESS_CALL,
    MALFORMED_CALL,
    REASONING_USAGE,
    REDACTED_REASONING,
    REPEATED_ID,
    SERVER_ID,
    SIGNED_REASONING,
    STREAM_WITHOUT_FINISH,
    THINK_TAGS,
    USAGE_ALONE,
    WRITTEN_UNREADABLE,
    Driver,
)
from tests.text_conformance.script import Call, Item, Script

# Ollama has no tool stop reason: a turn that ends on calls is done with "stop".
# A response that filled the context window (num_ctx) ends "length" too.
_DONE = {"stop": "stop", "tool": "stop", "cut": "length", "context": "length", "none": None}
_MODEL = "qwen3:8b"
_AT = "2026-10-02T00:00:00Z"


def _calls(calls: tuple[Call, ...]) -> list[ollama.Message.ToolCall]:
    # The server sends each call parsed and whole; a script's id, index and
    # fragments have no field on this wire.
    return [
        ollama.Message.ToolCall(
            function=ollama.Message.ToolCall.Function(
                name=call.name, arguments=json.loads(call.arguments or "{}")
            )
        )
        for call in calls
    ]


def _message(**fields: Any) -> ollama.Message:
    return ollama.Message(role="assistant", **{"content": "", **fields})


def _chunk(message: ollama.Message, **fields: Any) -> ollama.ChatResponse:
    return ollama.ChatResponse(
        model=_MODEL, created_at=_AT, message=message, **{"done": False, **fields}
    )


def _answered(script: Script, chunks: list[ollama.ChatResponse]) -> list[ollama.ChatResponse]:
    """Each chunk naming the model that answered."""
    return [chunk.model_copy(update={"model": script.answered_by}) for chunk in chunks]


def _done(script: Script, message: ollama.Message) -> ollama.ChatResponse:
    return _chunk(
        message,
        done=True,
        done_reason=_DONE[script.finish],
        prompt_eval_count=script.usage.input,
        eval_count=script.usage.output,
    )


def _stream(script: Script) -> list[ollama.ChatResponse]:
    chunks = [_chunk(_message(thinking=block.text)) for block in script.reasoning if block.text]
    if script.text:
        chunks.append(_chunk(_message(content=script.text)))
    calls = _calls(script.calls)
    if script.calls_in_one_chunk and calls:
        chunks.append(_chunk(_message(tool_calls=calls)))
    else:
        chunks.extend(_chunk(_message(tool_calls=[call])) for call in calls)
    if script.finish != "none":
        chunks.append(_done(script, _message()))
    return chunks


def _response(script: Script) -> ollama.ChatResponse:
    message = _message(
        content=script.text,
        thinking="".join(block.text for block in script.reasoning) or None,
        tool_calls=_calls(script.calls) or None,
    )
    return _done(script, message)


def _transport(script: Script, requests: list[Any]) -> httpx.MockTransport:
    """Ollama's ``/api/chat``: one JSON object, or one per line when streamed."""

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        chunks = _answered(script, _stream(script) if body.get("stream") else [_response(script)])
        lines = [chunk.model_dump_json(exclude_none=True) for chunk in chunks]
        return httpx.Response(200, content="\n".join(lines).encode())

    return httpx.MockTransport(answer)


def _assistant_items(message: dict[str, Any]) -> list[Item]:
    items: list[Item] = []
    if message.get("thinking"):
        items.append(("field", message["thinking"]))
    if message.get("content"):
        items.append(("text", message["content"]))
    for call in message.get("tool_calls") or []:
        function = call["function"]
        items.append(("call", function["name"], function["arguments"]))
    return items


class OllamaWire(ChatDriver):
    label = "ollama"
    covers = (OllamaAIProvider,)
    cannot = {
        ARGUMENT_TEXT: "Ollama parses a call server-side and sends its arguments as an object",
        WRITTEN_UNREADABLE: "a call's arguments are a JSON object (SDK: Mapping[str, Any])",
        CALL_INDEX: "Ollama sends each call whole, with no stream index",
        COMPOSITION: "Ollama sends each call whole, never its arguments as composed",
        REPEATED_ID: "an Ollama call carries no id (SDK Message.ToolCall holds only function)",
        STREAM_WITHOUT_FINISH: "a call arrives whole in one chunk; no stream stops inside one",
        SIGNED_REASONING: "Ollama's thinking is a plain text field, with no signature",
        REDACTED_REASONING: "Ollama has no redacted reasoning",
        THINK_TAGS: "Ollama separates reasoning server-side, into its thinking field",
        CACHE_USAGE: "Ollama reports prompt_eval_count and eval_count only",
        CACHE_WRITE_USAGE: "Ollama reports prompt_eval_count and eval_count only",
        MALFORMED_CALL: "Ollama has no stop reason for a call it could not parse",
        FILTER_STOP: "Ollama's done reasons are stop, length, load and unload",
        REASONING_USAGE: "Ollama counts thinking inside eval_count",
        FUNCTIONLESS_CALL: "an Ollama call is its function (SDK Message.ToolCall)",
        USAGE_ALONE: "the counts ride the done chunk, never a chunk of their own",
        SERVER_ID: "an Ollama call carries no id (SDK Message.ToolCall holds only function)",
    }
    reasoning = "field"
    calls_by_name = True
    accepted_names = ("lookup", "files.read")

    def provider(self, script: Script) -> AIProvider:
        provider = OllamaAIProvider(OllamaConfig(model=_MODEL))
        provider._client = ollama.AsyncClient(
            host="http://ollama.test", transport=_transport(script, self.requests)
        )
        return provider

    def replayed(self, request: Any) -> list[Item]:
        items: list[Item] = []
        for message in request["messages"]:
            if message["role"] == "assistant":
                items.extend(_assistant_items(message))
            elif message["role"] == "tool":
                content = message.get("content", "")
                items.append(("result", message.get("tool_name"), content, None))
            elif message.get("images"):
                # Ollama reads images on a user message, never on a tool one.
                items.append(("image",))
        return items


def wires() -> list[Driver]:
    return [OllamaWire()]
