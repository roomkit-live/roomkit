"""Shared error translation for the Gemini providers."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import ProviderError, provider_error
from roomkit.providers.ai.finish_reason import FinishReason

#: Finish reasons that mean the model withheld the answer rather than wrote it:
#: a constrained answer ending on one of these is a refusal, not bad JSON.
REFUSAL_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
)


_FINISH_REASONS = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "MALFORMED_FUNCTION_CALL": FinishReason.MALFORMED_CALL,
    "UNEXPECTED_TOOL_CALL": FinishReason.UNEXPECTED_CALL,
    **dict.fromkeys(REFUSAL_FINISH_REASONS, FinishReason.CONTENT_FILTER),
}
"""Gemini's finish reasons in the contract's vocabulary (RFC §6.7). A round of
calls ends ``STOP``: Gemini has no word of its own for it."""


def finish_reason_of(name: str | None) -> FinishReason | None:
    """A Gemini finish reason, by its wire spelling (:func:`reason_name`), in
    the contract's vocabulary: ``None`` for none, and for one with no word
    there (``LANGUAGE``, ``OTHER``)."""
    return None if name is None else _FINISH_REASONS.get(name)


def reason_name(raw: Any) -> str | None:
    """The wire spelling of a finish or block reason (``"MAX_TOKENS"``).

    The SDK hands back an enum whose ``.name`` is that spelling; a plain
    string passes through.
    """
    if raw is None:
        return None
    return getattr(raw, "name", None) or str(raw)


def prompt_block_reason(response: Any) -> str | None:
    """Why Gemini refused the prompt itself, when it did.

    A blocked prompt produces no candidate at all, so no finish reason says so;
    only ``prompt_feedback.block_reason`` does.
    """
    feedback = getattr(response, "prompt_feedback", None)
    return reason_name(getattr(feedback, "block_reason", None) if feedback is not None else None)


def wrap_gemini_error(exc: Exception) -> ProviderError:
    """Wrap a ``google-genai`` exception into a :class:`ProviderError`.

    The SDK spells its status on ``code`` or ``status_code`` depending on the
    error class, and reads an error written into a 200 stream as its status;
    :func:`provider_error` reads it all, a body the SDK could not parse
    (``UnknownApiResponseError``, a ``ValueError``) as final. Shared by every
    Gemini provider so "is this retryable" has one answer rather than one per
    surface.
    """
    return provider_error(exc, provider="gemini")
