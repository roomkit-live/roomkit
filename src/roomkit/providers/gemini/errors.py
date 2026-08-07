"""Shared error translation for the Gemini providers."""

from __future__ import annotations

from roomkit.providers.ai.base import RETRYABLE_STATUS_CODES, ProviderError

_RETRYABLE_TERMS = ("rate", "limit", "429", "500", "502", "503")


def wrap_gemini_error(exc: Exception) -> ProviderError:
    """Wrap a ``google-genai`` exception into a :class:`ProviderError`.

    The SDK spells its status on ``code`` or ``status_code`` depending on the
    error class, and some transport failures carry neither — hence the fallback
    to matching the message. Shared by every Gemini provider so "is this
    retryable" has one answer rather than one per surface.
    """
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    retryable = (
        status_code in RETRYABLE_STATUS_CODES
        if status_code
        else any(term in str(exc).lower() for term in _RETRYABLE_TERMS)
    )
    return ProviderError(
        str(exc),
        retryable=retryable,
        provider="gemini",
        status_code=status_code,
    )
