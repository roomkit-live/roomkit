"""Translate RoomKit messages and context into an Anthropic Messages request.

The provider in ``ai.py`` owns the clients, the call and the stream. This
module owns the other direction: how an ``AIMessage`` history (text, images,
tool calls and results, thinking blocks with their signatures) and the turn's
context become the kwargs ``messages.stream`` accepts, prompt-cache markers
included.
"""

from __future__ import annotations

import base64
from typing import Any, cast

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.image.base import image_part_payload

# Block types that accept a cache_control marker — notably NOT
# ``thinking`` blocks, which the API rejects as cache targets.
_CACHEABLE_BLOCK_TYPES = ("text", "tool_result", "tool_use", "image")


def format_content(
    content: (
        str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
    ),
) -> str | list[dict[str, Any]]:
    """Format message content for Anthropic API.

    Converts AITextPart/AIImagePart/AIToolCallPart/AIToolResultPart/AIThinkingPart
    to Anthropic's content block format.
    """
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, AITextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, AIImagePart):
            parts.append(_image_block(part))
        elif isinstance(part, AIToolCallPart):
            parts.append(
                {
                    "type": "tool_use",
                    "id": part.id,
                    "name": part.name,
                    "input": part.arguments,
                }
            )
        elif isinstance(part, AIToolResultPart):
            parts.append(
                {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id,
                    "content": _tool_result_content(part.result),
                }
            )
        elif isinstance(part, AIThinkingPart):
            # Anthropic requires thinking blocks preserved in conversation
            # history for round-trip fidelity across tool-loop turns.
            block: dict[str, Any] = {
                "type": "thinking",
                "thinking": part.thinking,
            }
            if part.signature:
                block["signature"] = part.signature
            parts.append(block)
    return parts


def _image_block(part: AIImagePart) -> dict[str, Any]:
    """Anthropic image content block from a data: URI or a plain URL.

    A data URI goes through the reader every provider shares: the media type
    comes from the header, then from the part, then the default, the payload
    is validated and sent canonical, and a malformed one is refused here —
    before the request leaves, naming the cause — rather than as the API's
    400 on an empty ``media_type`` that named nothing.
    """
    if not part.url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    media_type, data = image_part_payload(part, provider="anthropic")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def _tool_result_content(
    result: str | list[AITextPart | AIImagePart],
) -> str | list[dict[str, Any]]:
    """Render a tool result as Anthropic ``tool_result`` content.

    A string passes through unchanged; a list of parts becomes text and
    image content blocks — the Messages API accepts image blocks inside a
    ``tool_result``, which is how a screenshot tool reaches the model.
    """
    if isinstance(result, str):
        return result
    blocks: list[dict[str, Any]] = []
    for part in result:
        if isinstance(part, AITextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, AIImagePart):
            blocks.append(_image_block(part))
    return blocks


def build_messages(messages: list[AIMessage]) -> list[dict[str, Any]]:
    """Build Anthropic-formatted messages, mapping tool roles to user."""
    result: list[dict[str, Any]] = []
    for m in messages:
        role = "user" if m.role == "tool" else m.role
        result.append({"role": role, "content": format_content(m.content)})
    return result


def build_kwargs(config: AnthropicConfig, context: AIContext) -> dict[str, Any]:
    """Build kwargs dict shared by generate and streaming paths."""
    messages = build_messages(context.messages)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": context.max_tokens or config.max_tokens,
        "messages": messages,
    }
    if context.system_prompt:
        kwargs["system"] = context.system_prompt
    if context.thinking_budget is not None and context.thinking_budget > 0:
        # Extended thinking. Anthropic ignores temperature while thinking,
        # so it's dropped here regardless of model. Newer models (Opus
        # 4.7/4.8, Fable 5) reject the budget_tokens shape and want
        # adaptive thinking instead; ``display: "summarized"`` keeps the
        # reasoning trace visible (its default is "omitted" on those models).
        if config.use_adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        else:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": context.thinking_budget,
            }
        kwargs.pop("temperature", None)
    elif context.temperature is not None and config.supports_custom_temperature:
        kwargs["temperature"] = context.temperature
    if context.tools:
        kwargs["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in context.tools
        ]
    if config.enable_prompt_caching:
        _apply_cache_control(kwargs)
    return kwargs


def _apply_cache_control(kwargs: dict[str, Any]) -> None:
    """Mark the stable request prefix for Anthropic prompt caching.

    Layout (the API allows at most 4 markers): the tools array, the
    system prompt, and the last two eligible messages — the incremental-
    suffix pattern: on round N everything up to round N-1's marker is a
    prefix hit re-read at the cached rate instead of full price. Content
    below the provider's cacheable minimum is silently uncached by the
    API; markers there are harmless.
    """
    marker = {"type": "ephemeral"}
    tools = kwargs.get("tools")
    if isinstance(tools, list) and tools:
        tools[-1]["cache_control"] = marker
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        kwargs["system"] = [{"type": "text", "text": system, "cache_control": marker}]
    marked = 0
    for msg in reversed(kwargs.get("messages", [])):
        if marked >= 2:
            break
        content = msg.get("content")
        if isinstance(content, str):
            if not content:
                continue
            msg["content"] = [{"type": "text", "text": content, "cache_control": marker}]
            marked += 1
            continue
        if isinstance(content, list):
            for raw in reversed(content):
                if not isinstance(raw, dict):
                    continue
                block = cast(dict[str, Any], raw)
                if block.get("type") in _CACHEABLE_BLOCK_TYPES:
                    block["cache_control"] = marker
                    marked += 1
                    break
    return
