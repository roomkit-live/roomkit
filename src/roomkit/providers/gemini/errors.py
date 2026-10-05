"""Shared error translation for the Gemini providers."""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import ProviderError, failure_retryable

#: Finish reasons that mean the model withheld the answer rather than wrote it:
#: a constrained answer ending on one of these is a refusal, not bad JSON.
REFUSAL_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
)


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
    error class, and some failures carry neither: a transport failure (the
    httpx error the SDK lets through, RoomKit's client setting no retries of
    its own) or a message naming a transient status is then retryable. Shared
    by every Gemini provider so "is this retryable" has one answer rather than
    one per surface.
    """
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return ProviderError(
        str(exc),
        retryable=failure_retryable(status_code, exc),
        provider="gemini",
        status_code=status_code,
    )
