"""Central redaction policy for log content that may contain PII.

By default RoomKit does not write raw message content to logs — STT
transcripts, TTS/AI responses, screen-agent input, and similar. Each such log
site passes its value through :func:`redact`, which emits a length-only
placeholder unless content logging is explicitly enabled. This keeps personal
data out of production logs by default (e.g. for Québec Law 25 / privacy
requirements).

Enable full content logging (for local debugging) with the
``ROOMKIT_LOG_CONTENT`` environment variable (``1``/``true``/``yes``/``on``) or
programmatically via :func:`set_content_logging`. Even when enabled, gated
sites log at DEBUG level, so the log level is a second gate.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}

_content_logging = os.environ.get("ROOMKIT_LOG_CONTENT", "").strip().lower() in _TRUTHY


def set_content_logging(enabled: bool) -> None:
    """Enable or disable logging of raw message content process-wide."""
    global _content_logging
    _content_logging = enabled


def content_logging_enabled() -> bool:
    """Whether raw content may be written to logs (default ``False``)."""
    return _content_logging


def redact(value: object) -> str:
    """Return *value* for logging, redacted unless content logging is enabled.

    When disabled (the default) returns a length-only placeholder that leaks no
    content; when enabled returns ``str(value)`` verbatim.
    """
    if value is None:
        return "<none>"
    text = value if isinstance(value, str) else str(value)
    if _content_logging:
        return text
    return f"<redacted:{len(text)} chars>"
