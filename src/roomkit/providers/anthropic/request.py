"""Translate RoomKit messages and context into an Anthropic Messages request.

The provider in ``ai.py`` owns the clients, the call and the stream. This
module owns the other direction: how an ``AIMessage`` history (text, images,
tool calls and results, thinking blocks with their signatures) and the turn's
context become the kwargs ``messages.stream`` accepts, prompt-cache markers
included.
"""

from __future__ import annotations

from typing import Any, cast

from roomkit._text import fence, identifier
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.image_parts import image_part_base64
from roomkit.providers.ai.reasoning import thinking_switch
from roomkit.providers.ai.tool_declaration import ToolNameRule, declared_parameters
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.vendor_endpoint import ANTHROPIC_BASE_URL, is_vendor_endpoint

ANTHROPIC_TOOL_NAMES = ToolNameRule("anthropic", r"[A-Za-z0-9_-]{1,128}")
"""The tool names Anthropic accepts (measured 2026-10-02)."""

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
    # What must follow every tool_result block of the message: the result of
    # a call that references tools (``_result_beside_references``).
    trailing: list[dict[str, Any]] = []
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
            parts.append(_tool_result_block(part))
            trailing.extend(_result_beside_references(part))
        elif isinstance(part, AIThinkingPart) and (part.signature or part.redacted is not None):
            parts.append(_thinking_block(part))
        # A reasoning block without a signature is another vendor's (Anthropic
        # signs each one it sends, RFC §6.4): Anthropic refuses it, thinking
        # on or off, and takes the round without it (measured 2026-10-03).
    return parts + trailing


def _thinking_block(part: AIThinkingPart) -> dict[str, Any]:
    """A reasoning block as Anthropic sent it: each block of a tool round goes
    back as received, a redacted one as its opaque data (RFC §6.4)."""
    if part.redacted is not None:
        return {"type": "redacted_thinking", "data": part.redacted}
    block: dict[str, Any] = {"type": "thinking", "thinking": part.thinking}
    if part.signature:
        block["signature"] = part.signature
    return block


def _tool_result_block(part: AIToolResultPart) -> dict[str, Any]:
    """The ``tool_result`` block of a call, flagged an error when the call was
    refused, failed, blocked, served by nothing or cancelled (RFC §6.7).

    One that makes tools callable carries their ``tool_reference`` blocks
    alone, where each deferred definition expands out of the cached prefix:
    the API refuses a reference mixed with any other content, so the call's
    result follows the message's tool results instead.
    """
    content: str | list[dict[str, Any]] = (
        [{"type": "tool_reference", "tool_name": name} for name in part.references]
        if part.references
        else _tool_result_content(part.result)
    )
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": part.tool_call_id,
        "content": content,
    }
    if part.is_error:
        block["is_error"] = True
    return block


def _result_beside_references(part: AIToolResultPart) -> list[dict[str, Any]]:
    """The result of a call whose ``tool_result`` carries references, as the
    blocks that follow the message's tool results, named after the call."""
    if not part.references:
        return []
    body = _tool_result_content(part.result)
    # What the tool returned is data, set apart in a block it cannot close.
    label = f"[Result of {identifier(part.name, 'tool')}]"
    if isinstance(body, str):
        return [{"type": "text", "text": f"{label}\n{fence('tool_result', body)}"}] if body else []
    fenced = [
        {**block, "text": fence("tool_result", block["text"])}
        if block.get("type") == "text"
        else block
        for block in body
    ]
    return [{"type": "text", "text": label}, *fenced] if body else []


def _image_block(part: AIImagePart) -> dict[str, Any]:
    """Anthropic image content block from a data: URI or a plain URL.

    A data URI goes through the shared reader: media type from the header,
    then the part, then the default; payload validated and sent canonical; a
    malformed one refused before the request leaves.
    """
    if not part.url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    media_type, payload = image_part_base64(part, provider="anthropic")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": payload},
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
    """Build Anthropic-formatted messages, mapping tool roles to user.

    A ``system`` message of the history (a memory provider's summary, an
    instruction) becomes a user turn, as on Gemini: the Messages API takes
    the roles ``user`` and ``assistant`` only, the system prompt being its
    top-level ``system``.
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        role = "user" if m.role in ("tool", "system") else m.role
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
    kwargs.update(_thinking_or_temperature(config, context))
    if context.tools:
        if is_vendor_endpoint(config.base_url, ANTHROPIC_BASE_URL):
            # Behind another base_url the server decides its names (RFC §6.7).
            ANTHROPIC_TOOL_NAMES.check(t.name for t in context.tools)
        kwargs["tools"] = _tool_definitions(context.tools)
    if context.response_schema is not None:
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": context.response_schema}
        }
    if config.enable_prompt_caching:
        _apply_cache_control(kwargs)
    return kwargs


def _tool_definitions(tools: list[AITool]) -> list[dict[str, Any]]:
    """The tools as Anthropic declares them, a deferred one held unseen.

    The API refuses a request whose every tool is deferred: then none is.
    """
    defers = not all(t.defer_loading for t in tools)
    definitions: list[dict[str, Any]] = []
    for t in tools:
        definition = {
            "name": t.name,
            "description": t.description,
            "input_schema": declared_parameters(t.parameters),
        }
        if defers and t.defer_loading:
            definition["defer_loading"] = True
        definitions.append(definition)
    return definitions


def _thinking_or_temperature(config: AnthropicConfig, context: AIContext) -> dict[str, Any]:
    """The turn's ``thinking`` block, or else its ``temperature``, or nothing.

    Anthropic ignores temperature while thinking, so a thinking turn never
    sends one. Newer models (Opus 4.7/4.8, Fable 5) reject the
    ``budget_tokens`` shape and want adaptive thinking instead;
    ``display: "summarized"`` keeps the reasoning trace visible (its default
    is "omitted" on those models).

    Thinking is on as the turn states it (RFC §6.7): a positive budget, or
    ``enable_thinking=True``. A model without adaptive thinking needs its
    budget, so ``enable_thinking=True`` alone leaves it off there.

    ``messages.stream()`` has no ``temperature`` parameter in anthropic 1.x,
    while the models profiled as taking one still do, so it rides
    ``extra_body``, which the SDK merges into the request JSON as it is.
    """
    budget = context.thinking_budget
    if thinking_switch(context):
        if config.use_adaptive_thinking:
            return {"thinking": {"type": "adaptive", "display": "summarized"}}
        if budget:
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if context.temperature is not None and config.supports_custom_temperature:
        return {"extra_body": {"temperature": context.temperature}}
    return {}


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
    # On the last tool the prefix holds: a deferred one is out of it, and the
    # API refuses a marker there.
    shown = [t for t in kwargs.get("tools") or [] if not t.get("defer_loading")]
    if shown:
        shown[-1]["cache_control"] = marker
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
