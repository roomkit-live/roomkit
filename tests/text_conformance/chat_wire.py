"""What the wires shaped like Chat Completions share: how a script's text is
cut into stream pieces, and how a request's tools and turns are read back."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from tests.text_conformance.driver import Driver
from tests.text_conformance.script import Call, Item

# An OpenAI-compatible server ends a response that filled the context window
# ``length``, the output cap's word: it caps the output at what the window has
# left.
FINISH = {
    "stop": "stop",
    "tool": "tool_calls",
    "cut": "length",
    "context": "length",
    "filtered": "content_filter",
    "none": None,
}


def wire_arguments(call: Call) -> Any:
    """*call*'s arguments as the server sends them: text, or an object."""
    return json.loads(call.arguments) if call.as_object else call.arguments


def call_pieces(call: Call) -> list[Any]:
    """*call*'s arguments as a stream sends them: text in pieces, an object whole."""
    return [wire_arguments(call)] if call.as_object else pieces(call.arguments, call.fragments)


def pieces(text: str, count: int) -> list[str]:
    """*text* in *count* pieces, as a stream would cut it."""
    size = max(1, -(-len(text) // max(count, 1)))
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def iterate(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def assistant_items(message: dict[str, Any], field: str | None = None) -> list[Item]:
    """An assistant message: its reasoning in *field*, what it said, and its calls."""
    items: list[Item] = []
    if field is not None and message.get(field):
        items.append(("field", message[field]))
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    if content:
        items.append(("text", content))
    for call in message.get("tool_calls") or []:
        function = call["function"]
        items.append(("call", call["id"], json.loads(function["arguments"])))
    return items


def carries_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        part.get("type") == "image_url" for part in content if isinstance(part, dict)
    )


class ChatDriver(Driver):
    """A driver whose requests are Chat Completions bodies, as dicts."""

    def declared(self, request: Any) -> dict[str, dict[str, Any]]:
        return {
            tool["function"]["name"]: tool["function"]["parameters"]
            for tool in request.get("tools") or []
        }

    def replayed(self, request: Any) -> list[Item]:
        items: list[Item] = []
        for message in request["messages"]:
            if message["role"] == "assistant":
                items.extend(self.assistant_items(message))
            elif message["role"] == "tool":
                items.append(("result", message["tool_call_id"], message["content"], None))
            elif carries_image(message):
                items.append(("image",))
        return items

    def assistant_items(self, message: dict[str, Any]) -> list[Item]:
        """An assistant message as this wire writes it."""
        return assistant_items(message)
