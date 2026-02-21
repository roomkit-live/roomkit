"""Fast approximate token estimation for context budget management."""

from __future__ import annotations

import json

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIToolCallPart,
    AIToolResultPart,
)


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ~ 4 characters for English text."""
    return len(text) // 4 + 1


def estimate_message_tokens(message: AIMessage) -> int:
    """Estimate tokens for a complete message including role overhead."""
    overhead = 4  # role, delimiters
    if isinstance(message.content, str):
        return overhead + estimate_tokens(message.content)
    total = overhead
    for part in message.content:
        if isinstance(part, AITextPart):
            total += estimate_tokens(part.text)
        elif isinstance(part, AIToolCallPart):
            args_str = (
                json.dumps(part.arguments)
                if isinstance(part.arguments, dict)
                else str(part.arguments)
            )
            total += estimate_tokens(part.name) + estimate_tokens(args_str)
        elif isinstance(part, AIToolResultPart):
            result_text = part.result if isinstance(part.result, str) else json.dumps(part.result)
            total += estimate_tokens(result_text)
        elif isinstance(part, AIImagePart):
            total += 1000  # rough estimate for vision tokens
    return total


def estimate_context_tokens(context: AIContext) -> int:
    """Estimate total tokens for an AIContext."""
    total = 0
    if context.system_prompt:
        total += estimate_tokens(context.system_prompt)
    for msg in context.messages:
        total += estimate_message_tokens(msg)
    if context.tools:
        for tool in context.tools:
            total += estimate_tokens(tool.name) + estimate_tokens(tool.description)
            if tool.parameters:
                total += estimate_tokens(json.dumps(tool.parameters))
    return total
