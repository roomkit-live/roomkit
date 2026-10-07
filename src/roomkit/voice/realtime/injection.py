"""Provider-reported submission boundary for realtime text injection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from roomkit._text import quoted

LINE_LIMIT = 2000
"""The characters of an ``assistant`` line an instruction quotes."""


class VoiceInjectionResult(BaseModel):
    """Report what the provider knows after ``inject_text``.

    ``sent`` means its send operation completed, not that audio was heard.
    ``not_sent`` guarantees no submission or pending submission exists.
    ``unknown`` covers acceptance that cannot be established, including input
    queued locally for later sending. Only ``not_sent`` may allow retry.
    Exceptions and ``None`` returns are conservatively unknown.
    """

    status: Literal["sent", "not_sent", "unknown"]
    reason: str | None = None
    retryable: bool = False


def say_line_instruction(line: str) -> str:
    """Phrase an ``assistant`` line as an instruction to say it (RFC §12.4).

    For a provider with no primitive that makes the agent speak a given text:
    the line reaches the model as a direction, never as something the user
    said, which the model would answer. The line is quoted (RFC §6.4): it
    cannot end the quote and add to the direction.
    """
    return f"Say this to the user now, as your next words, then listen: {quoted(line, LINE_LIMIT)}"
