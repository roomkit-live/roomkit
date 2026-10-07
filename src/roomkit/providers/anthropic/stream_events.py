"""What one event of an Anthropic messages stream hands the tool loop.

Each content block becomes the stream events RoomKit reads: reasoning deltas
naming their block (a redacted block as its opaque data), text deltas, a tool
call's composition and the call itself (RFC §6.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roomkit.providers.ai.base import (
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    answered_by,
)
from roomkit.providers.anthropic.tool_blocks import ToolUseBlocks


def stream_events(event: Any, blocks: ToolUseBlocks) -> list[StreamEvent]:
    """The stream events one SDK event carries; none for an event that says
    nothing to the loop (message start and stop, pings)."""
    kind = getattr(event, "type", None)
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
    metadata = answered_by(getattr(final, "model", None), asked)
    return StreamDone(finish_reason=final.stop_reason, usage=usage, metadata=metadata)


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
