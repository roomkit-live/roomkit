"""What one event of an Anthropic messages stream hands the tool loop.

Each content block becomes the stream events RoomKit reads: reasoning deltas
naming their block (a redacted block as its opaque data), text deltas, a tool
call's composition and the call itself (RFC §6.4). The message's start
tells its input counters, which a generation cut before its end counts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roomkit.providers.ai.base import (
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamUsage,
    answered_by,
)
from roomkit.providers.anthropic.tool_blocks import ToolUseBlocks


def stream_events(event: Any, blocks: ToolUseBlocks) -> list[StreamEvent]:
    """The stream events one SDK event carries; none for an event that says
    nothing to the loop (message stop, pings)."""
    kind = getattr(event, "type", None)
    if kind == "message_start":
        told = input_usage(getattr(event.message, "usage", None))
        return [StreamUsage(usage=told)] if told else []
    if kind == "content_block_start":
        return _block_start(event, blocks)
    if kind == "content_block_delta" and hasattr(event.delta, "type"):
        read = _DELTAS.get(event.delta.type)
        out = read(event, blocks) if read is not None else None
        return [out] if out is not None else []
    if kind == "content_block_stop":
        call = blocks.close(event.index)
        return [call] if call is not None else []
    return []


def is_first_output(event: StreamEvent) -> bool:
    """Whether *event* is model output, which time-to-first-token measures."""
    return isinstance(event, StreamTextDelta) or (
        isinstance(event, StreamThinkingDelta) and bool(event.thinking)
    )


def input_usage(usage: Any) -> dict[str, int]:
    """The input counters a message's start tells (RFC §6.4): its input and
    its cache reads and writes, a cache counter at zero omitted. The output
    it tells then is no count yet."""
    told: dict[str, int] = {}
    if (tokens := getattr(usage, "input_tokens", None)) is not None:
        told["input_tokens"] = tokens
    for counter in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        if value := getattr(usage, counter, None):
            told[counter] = value
    return told


def done_event(final: Any, asked: str) -> StreamDone:
    """The done event of a finished message: its stop reason, usage and model,
    the one that answered as :func:`answered_by` names it (*asked* when the
    message names none)."""
    usage: dict[str, int] = {
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }
    # A cache counter at zero is omitted, as every other provider omits it.
    if cache_write := getattr(final.usage, "cache_creation_input_tokens", None):
        usage["cache_creation_input_tokens"] = cache_write
    if cache_read := getattr(final.usage, "cache_read_input_tokens", None):
        usage["cache_read_input_tokens"] = cache_read
    # A detail of output_tokens, which already counts it.
    details = getattr(final.usage, "output_tokens_details", None)
    thinking = (getattr(details, "thinking_tokens", 0) if details else 0) or 0
    if thinking:
        usage["reasoning_tokens"] = thinking
    metadata = answered_by(getattr(final, "model", None), asked, _refusal_category(final))
    return StreamDone(finish_reason=final.stop_reason, usage=usage, metadata=metadata)


def _refusal_category(final: Any) -> str | None:
    """Why the safeguards declined a refused message (``stop_details``, set
    only on a refusal): its category, such as ``reasoning_extraction``."""
    return getattr(getattr(final, "stop_details", None), "category", None)


def _block_start(event: Any, blocks: ToolUseBlocks) -> list[StreamEvent]:
    block = event.content_block
    kind = getattr(block, "type", None)
    if kind == "tool_use":
        return [blocks.open(event.index, block)]
    if kind == "redacted_thinking":
        # Replayed as received, its opaque data and all.
        return [StreamThinkingDelta(thinking="", redacted=block.data, block=event.index)]
    return []


def _thinking(event: Any, blocks: ToolUseBlocks) -> StreamEvent:
    return StreamThinkingDelta(thinking=event.delta.thinking, block=event.index)


def _signature(event: Any, blocks: ToolUseBlocks) -> StreamEvent:
    # A block's signature comes as its own delta after its text; it rides with
    # the block it signs, so each block goes back with its own (RFC §6.4).
    return StreamThinkingDelta(thinking="", signature=event.delta.signature, block=event.index)


def _text(event: Any, blocks: ToolUseBlocks) -> StreamEvent:
    return StreamTextDelta(text=event.delta.text)


def _arguments(event: Any, blocks: ToolUseBlocks) -> StreamEvent | None:
    return blocks.add(event.index, event.delta.partial_json)


_DELTAS: dict[str, Callable[[Any, ToolUseBlocks], StreamEvent | None]] = {
    "thinking_delta": _thinking,
    "signature_delta": _signature,
    "text_delta": _text,
    "input_json_delta": _arguments,
}
