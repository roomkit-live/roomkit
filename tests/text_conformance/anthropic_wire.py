"""The Anthropic Messages wire: content blocks streamed one after another."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from tests.text_conformance.chat_wire import pieces
from tests.text_conformance.driver import (
    CALL_INDEX,
    CALLS_IN_ONE_CHUNK,
    FUNCTIONLESS_CALL,
    MALFORMED_CALL,
    THINK_TAGS,
    USAGE_ALONE,
    WRITTEN_UNREADABLE,
    Driver,
)
from tests.text_conformance.script import Item, Script

_STOP = {
    "stop": "end_turn",
    "tool": "tool_use",
    "cut": "max_tokens",
    "context": "model_context_window_exceeded",
    "filtered": "refusal",
    "none": None,
}


class _Stream:
    """``client.messages.stream(...)``: an async context manager of SDK events."""

    def __init__(self, events: list[Any], final: Any) -> None:
        self._events, self._final = events, final

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        async def events() -> Any:
            for event in self._events:
                yield event

        return events()

    async def get_final_message(self) -> Any:
        return self._final


def _event(kind: str, index: int, **fields: Any) -> Any:
    return SimpleNamespace(type=kind, index=index, **fields)


def _delta(index: int, **delta: Any) -> Any:
    return _event("content_block_delta", index, delta=SimpleNamespace(**delta))


def _start(index: int, **block: Any) -> Any:
    return _event("content_block_start", index, content_block=SimpleNamespace(**block))


def _reasoning_events(script: Script, index: int) -> tuple[list[Any], int]:
    events: list[Any] = []
    for block in script.reasoning:
        if block.redacted is not None:
            events.append(_start(index, type="redacted_thinking", data=block.redacted))
        else:
            events.append(_start(index, type="thinking"))
            events.append(_delta(index, type="thinking_delta", thinking=block.text))
            if block.signature:
                events.append(_delta(index, type="signature_delta", signature=block.signature))
        events.append(_event("content_block_stop", index))
        index += 1
    return events, index


def _events(script: Script) -> tuple[list[Any], list[Any]]:
    """The stream's events, and the tool blocks the stream left open."""
    events, index = _reasoning_events(script, 0)
    if script.text:
        events.append(_start(index, type="text"))
        events.append(_delta(index, type="text_delta", text=script.text))
        events.append(_event("content_block_stop", index))
        index += 1
    left_open: list[Any] = []
    for n, call in enumerate(script.calls):
        call_id = call.id or f"toolu_{n}"
        events.append(_start(index, type="tool_use", id=call_id, name=call.name))
        for piece in pieces(call.arguments, call.fragments):
            events.append(_delta(index, type="input_json_delta", partial_json=piece))
        last = n == len(script.calls) - 1
        if last and script.finish in ("cut", "context", "filtered", "none"):
            # Cut mid-block: the stream never closes it; the final message
            # holds what the SDK parsed of it.
            left_open.append(
                SimpleNamespace(type="tool_use", id=call_id, name=call.name, input={})
            )
        else:
            events.append(_event("content_block_stop", index))
        index += 1
    return events, left_open


def _final(script: Script, left_open: list[Any]) -> Any:
    usage = script.usage
    return SimpleNamespace(
        content=left_open,
        usage=SimpleNamespace(
            input_tokens=usage.input,
            output_tokens=usage.output,
            cache_creation_input_tokens=usage.cache_write,
            cache_read_input_tokens=usage.cache_read,
            # A detail of output_tokens, which counts it.
            output_tokens_details=SimpleNamespace(thinking_tokens=usage.reasoning),
        ),
        stop_reason=_STOP[script.finish],
        model=script.answered_by,
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def _tool_result_items(block: dict[str, Any]) -> list[Item]:
    content = block.get("content", "")
    error = bool(block.get("is_error"))
    items: list[Item] = [("result", block["tool_use_id"], _content_text(content), error)]
    if isinstance(content, list) and any(b.get("type") == "image" for b in content):
        items.append(("image",))
    return items


_BLOCK_ITEMS: dict[str, Callable[[dict[str, Any]], list[Item]]] = {
    "thinking": lambda block: [("thinking", block["thinking"], block.get("signature"))],
    "redacted_thinking": lambda block: [("redacted", block["data"])],
    "text": lambda block: [("text", block["text"])],
    "tool_use": lambda block: [("call", block["id"], block["input"])],
    "tool_result": _tool_result_items,
}


def _block_items(block: dict[str, Any]) -> list[Item]:
    read = _BLOCK_ITEMS.get(block.get("type", ""))
    return read(block) if read is not None else []


class AnthropicWire(Driver):
    label = "anthropic"
    covers = (AnthropicAIProvider,)
    cannot = {
        CALL_INDEX: "every block carries its own index",
        CALLS_IN_ONE_CHUNK: "each call is a content block of its own",
        WRITTEN_UNREADABLE: "a closed tool_use block always parses; one that does not was cut",
        MALFORMED_CALL: "Anthropic has no stop reason for a call it could not parse",
        THINK_TAGS: "reasoning comes in thinking blocks; text is the answer's",
        FUNCTIONLESS_CALL: "a tool_use block always names its tool",
        USAGE_ALONE: "usage rides message_start and message_delta, never a chunk of its own",
    }
    reasoning = "blocks"
    error_flag = True
    refused_names = ("files.read",)

    def provider(self, script: Script) -> AIProvider:
        provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
        events, left_open = _events(script)

        def stream(**kwargs: Any) -> _Stream:
            self.requests.append(kwargs)
            return _Stream(events, _final(script, left_open))

        provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        return provider

    def declared(self, request: Any) -> dict[str, dict[str, Any]]:
        return {tool["name"]: tool["input_schema"] for tool in request.get("tools") or []}

    def replayed(self, request: Any) -> list[Item]:
        items: list[Item] = []
        for message in request["messages"]:
            content = message["content"]
            if isinstance(content, str):
                if message["role"] == "assistant" and content:
                    items.append(("text", content))
                continue
            for block in content:
                if message["role"] == "assistant" or block.get("type") == "tool_result":
                    items.extend(_block_items(block))
        return items


def wires() -> list[Driver]:
    return [AnthropicWire()]
