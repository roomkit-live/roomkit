"""Anthropic's stop reasons in the provider contract's vocabulary (RFC §6.7)."""

from __future__ import annotations

from roomkit.providers.ai.finish_reason import FinishReason

_STOP_REASONS = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "refusal": FinishReason.REFUSAL,
}


def finish_reason_of(stop_reason: str | None) -> FinishReason | None:
    """A message's stop reason in the contract's vocabulary: ``None`` for
    none, and for one with no word there (``pause_turn``, a turn the server
    paused and that has not ended)."""
    return None if stop_reason is None else _STOP_REASONS.get(stop_reason)
