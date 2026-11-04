"""A conversation rendered as Chat Completions messages, for every provider
that speaks the format (RFC §6.7).

OpenAI and its derivatives, Mistral and PolarGrid send the same messages but
for a few vendor choices, which each provider states as a :class:`ChatDialect`:
the field a vendor reads a model's earlier reasoning from, whether a tool message
names its tool, whether text is sent flat. Everything else is rendered once, here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from roomkit.providers.ai.base import (
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.image_parts import image_part_uri


@dataclass(frozen=True)
class ChatDialect:
    """What one provider renders differently from OpenAI's own endpoint."""

    thinking_field: str | None = None
    """The assistant message field a vendor reads a model's earlier reasoning
    from (Cerebras ``reasoning``; DeepSeek, MiniMax and Kimi
    ``reasoning_content``). ``None`` sends no earlier reasoning (RFC §6.4): any
    other server would read it as text the model said, and its prompt cache would
    not survive it."""
    names_tool_results: bool = False
    """Each tool message names its tool (Mistral, PolarGrid)."""
    flattens_text: bool = False
    """A message of text parts goes as one string, and an empty message is not
    sent; only one carrying an image goes as parts (PolarGrid)."""
    round_thinking_required: bool = False
    """Every round that called tools carries the thinking field, even a round
    that did not reason: DeepSeek in thinking mode refuses a round of the turn
    in progress without its ``reasoning_content``, unless it issued the round's
    call ids itself a moment ago, and Kimi any round without it."""
    round_thinking_placeholder: str = ""
    """What a required thinking field holds for a round that did not reason:
    DeepSeek takes it empty, and Kimi reads an empty one as missing."""

    def __post_init__(self) -> None:
        if self.round_thinking_required and self.thinking_field is None:
            raise ValueError("round_thinking_required needs a thinking_field to send")
        if self.round_thinking_placeholder and not self.round_thinking_required:
            raise ValueError("round_thinking_placeholder needs round_thinking_required")


OPENAI_CHAT = ChatDialect()
"""OpenAI's own rendering, which every compatible server reads."""


def chat_messages(
    messages: list[AIMessage],
    system_prompt: str | None,
    dialect: ChatDialect,
    *,
    provider: str,
) -> list[dict[str, Any]]:
    """The conversation as *dialect* sends it, the system prompt first.

    *provider* names the vendor in the error an image it cannot send raises.
    """
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for message in messages:
        result.extend(_rendered(message, dialect, provider))
    return result


def _rendered(message: AIMessage, dialect: ChatDialect, provider: str) -> list[dict[str, Any]]:
    """The messages one conversation message becomes: none, one, or a tool
    message per result and a user message carrying their images."""
    content = message.content
    if isinstance(content, str):
        if dialect.flattens_text and not content:
            return []
        return [{"role": message.role, "content": content}]
    calls = [p for p in content if isinstance(p, AIToolCallPart)]
    if calls:
        return [_call_round(message, calls, dialect, provider)]
    results = [p for p in content if isinstance(p, AIToolResultPart)]
    if results:
        return _tool_messages(results, dialect, provider)
    return _plain(message, dialect, provider)


def _call_round(
    message: AIMessage, calls: list[AIToolCallPart], dialect: ChatDialect, provider: str
) -> dict[str, Any]:
    """An assistant round that called tools, what it said beside its calls."""
    parts = list(message.content)
    content: str | list[dict[str, Any]]
    if dialect.flattens_text:
        content = _flat(parts, provider)
    else:
        content = "".join(p.text for p in parts if isinstance(p, AITextPart))
    rendered: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ],
    }
    return _with_thinking_field(
        rendered, message, dialect, required=dialect.round_thinking_required
    )


def _tool_messages(
    results: list[AIToolResultPart], dialect: ChatDialect, provider: str
) -> list[dict[str, Any]]:
    """One tool message per result, then the results' images on a user message.

    A tool message carries text only (an image part is user-only), so an
    image result keeps its text there and its image rides a user message
    after every tool message: the call/result pairing stays valid.
    """
    rendered: list[dict[str, Any]] = []
    images: list[AIImagePart] = []
    for result in results:
        text, result_images = result.split_for_message()
        message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": text,
        }
        if dialect.names_tool_results:
            message["name"] = result.name
        rendered.append(message)
        images.extend(result_images)
    if images:
        rendered.append({"role": "user", "content": [_image(i, provider) for i in images]})
    return rendered


def _plain(message: AIMessage, dialect: ChatDialect, provider: str) -> list[dict[str, Any]]:
    """A message without tool parts: its text and images, its reasoning in the
    dialect's field when it has one."""
    parts = list(message.content)
    if dialect.flattens_text:
        flat = _flat(parts, provider)
        return [{"role": message.role, "content": flat}] if flat else []
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, AITextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, AIImagePart):
            blocks.append(_image(part, provider))
    # A message whose only part is reasoning keeps an empty string for content.
    content = blocks if blocks or not parts else ""
    rendered = {"role": message.role, "content": content}
    return [_with_thinking_field(rendered, message, dialect)]


def _flat(parts: list[Any], provider: str) -> str | list[dict[str, Any]]:
    """Text parts as one string, or text and image blocks when an image is
    among them; reasoning is not sent."""
    if not any(isinstance(p, AIImagePart) for p in parts):
        return "".join(p.text for p in parts if isinstance(p, AITextPart))
    return [
        {"type": "text", "text": p.text} if isinstance(p, AITextPart) else _image(p, provider)
        for p in parts
        if isinstance(p, AITextPart | AIImagePart)
    ]


def _with_thinking_field(
    rendered: dict[str, Any], message: AIMessage, dialect: ChatDialect, *, required: bool = False
) -> dict[str, Any]:
    """*rendered*, an assistant's reasoning in the dialect's field when it has
    one; a *required* field goes without any, holding the dialect's placeholder."""
    field = dialect.thinking_field
    if field is not None and message.role == "assistant":
        thinking = _thinking(list(message.content))
        if thinking or required:
            rendered[field] = thinking or dialect.round_thinking_placeholder
    return rendered


def _thinking(parts: list[Any]) -> str:
    return "".join(p.thinking for p in parts if isinstance(p, AIThinkingPart))


def _image(part: AIImagePart, provider: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_part_uri(part, provider=provider)}}
