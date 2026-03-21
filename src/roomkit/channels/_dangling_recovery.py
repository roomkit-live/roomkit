"""Patch orphaned tool calls from barge-in interruptions."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AIMessage, AIToolCallPart, AIToolResultPart


def patch_dangling_tool_calls(messages: list[AIMessage]) -> list[AIMessage]:
    """Inject synthetic cancellation results for orphaned tool calls.

    When a user barge-in interrupts a tool loop, assistant messages with
    ``AIToolCallPart`` entries may exist without matching
    ``AIToolResultPart`` entries.  Provider APIs reject such orphans.

    This function scans the message list and patches any gaps with a
    cancellation notice.  When a tool-result message follows an assistant
    message with dangling calls, the cancellations are merged into that
    message to maintain contiguous result blocks.
    """
    seen_results = _collect_result_ids(messages)

    if not _has_dangling_calls(messages, seen_results):
        return messages

    return _build_patched_list(messages, seen_results)


def _collect_result_ids(messages: list[AIMessage]) -> set[str]:
    """Collect all tool_call_ids that have a result."""
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if isinstance(part, AIToolResultPart):
                seen.add(part.tool_call_id)
    return seen


def _has_dangling_calls(messages: list[AIMessage], seen: set[str]) -> bool:
    """Fast check: are there any orphaned tool calls?"""
    for msg in messages:
        if msg.role != "assistant" or not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if isinstance(part, AIToolCallPart) and part.id not in seen:
                return True
    return False


_CANCEL_MSG = "Tool call was cancelled \u2014 a new message arrived before it could complete."


def _build_patched_list(messages: list[AIMessage], seen: set[str]) -> list[AIMessage]:
    """Build a new message list with synthetic cancellation results."""
    patched: list[AIMessage] = []
    pending: list[Any] = []

    for msg in messages:
        # Merge pending cancellations into a following tool message
        if pending and msg.role == "tool" and isinstance(msg.content, list):
            patched.append(AIMessage(role="tool", content=list(pending) + list(msg.content)))
            pending = []
            continue

        # Flush pending before a non-tool message
        if pending:
            patched.append(AIMessage(role="tool", content=pending))
            pending = []

        patched.append(msg)

        if msg.role != "assistant" or not isinstance(msg.content, list):
            continue

        dangling = [p for p in msg.content if isinstance(p, AIToolCallPart) and p.id not in seen]
        if dangling:
            pending = [
                AIToolResultPart(tool_call_id=tc.id, name=tc.name, result=_CANCEL_MSG)
                for tc in dangling
            ]

    if pending:
        patched.append(AIMessage(role="tool", content=pending))

    return patched
