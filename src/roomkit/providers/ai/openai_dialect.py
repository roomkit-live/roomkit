"""The OpenAI Chat Completions dialect, read on the way back.

Several providers speak it: ``OpenAIAIProvider`` and its derivatives, Mistral
and PolarGrid in their own packages. What they share is how a response is
read, not how a request is built: the two reasoning conventions (inline
``<think>`` tags and a dedicated field), the way a tool call is fragmented
across stream chunks, and the structured context-overflow fact off a status
error. It lives with the AI provider ABC rather than under one vendor because
three vendor packages consume it, and nothing here shapes a request.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from roomkit.providers.ai.base import StreamToolCallDelta

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def field_reasoning(carrier: Any) -> str | None:
    """Read a reasoning trace carried in its own field rather than inline.

    ``<think>`` tags are one of two conventions OpenAI-compatible servers use.
    The other is a dedicated field beside ``content`` — ``reasoning_content``
    for DeepSeek, Qwen and vLLM's reasoning parsers, ``reasoning`` for
    OpenRouter. Both a streaming delta and a complete message carry it under
    the same names, so both paths read it here.
    """
    value = getattr(carrier, "reasoning_content", None) or getattr(carrier, "reasoning", None)
    return value if isinstance(value, str) and value else None


def merge_thinking(inline: str | None, field: str | None) -> str | None:
    """Combine the two reasoning conventions into one trace.

    A server uses one or the other, so in practice exactly one side is set;
    concatenating rather than picking a winner means neither is dropped if one
    ever emits both.
    """
    if not field:
        return inline
    return f"{inline}{field}" if inline else field


class ThinkTagParser:
    """Stateful parser for ``<think>...</think>`` tags in a text stream.

    vLLM / Ollama models (DeepSeek-R1, QwQ, etc.) emit reasoning inside
    ``<think>`` tags before the answer text.  This parser classifies
    incoming chunks into ``"thinking"`` and ``"text"`` segments, handling
    tags that are split across chunk boundaries.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_thinking = False
        self._buf = ""

    def feed(self, chunk: str) -> list[tuple[Literal["thinking", "text"], str]]:
        """Process *chunk* and return classified ``(kind, content)`` pairs."""
        self._buf += chunk
        results: list[tuple[Literal["thinking", "text"], str]] = []

        while self._buf:
            tag = self._CLOSE if self._in_thinking else self._OPEN
            idx = self._buf.find(tag)

            if idx >= 0:
                before = self._buf[:idx]
                if before:
                    kind: Literal["thinking", "text"] = "thinking" if self._in_thinking else "text"
                    results.append((kind, before))
                self._buf = self._buf[idx + len(tag) :]
                self._in_thinking = not self._in_thinking
            else:
                # Hold back a suffix that could be a partial tag start.
                hold = self._partial_tag_len(self._buf, tag)
                if hold:
                    emit = self._buf[:-hold]
                    if emit:
                        kind = "thinking" if self._in_thinking else "text"
                        results.append((kind, emit))
                    self._buf = self._buf[-hold:]
                    break
                # No partial match — emit everything.
                kind = "thinking" if self._in_thinking else "text"
                results.append((kind, self._buf))
                self._buf = ""

        return results

    def flush(self) -> list[tuple[Literal["thinking", "text"], str]]:
        """Emit any remaining buffered content."""
        if not self._buf:
            return []
        kind: Literal["thinking", "text"] = "thinking" if self._in_thinking else "text"
        result = [(kind, self._buf)]
        self._buf = ""
        return result

    @staticmethod
    def _partial_tag_len(text: str, tag: str) -> int:
        """Length of the longest suffix of *text* that is a prefix of *tag*."""
        max_check = min(len(text), len(tag) - 1)
        for length in range(max_check, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0


def overflow_fact(exc: object) -> bool | None:
    """The structured overflow fact off an OpenAI-style status error, if any.

    OpenAI and Azure carry ``error.code == "context_length_exceeded"`` in the
    error body — a first-hand fact, unlike the message wording. The other
    compatible vendors put integers or generic strings in ``code``, so a miss
    is ``None`` (nobody classified), never ``False``: their overflows are
    still caught by the shared phrase fallback.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == "context_length_exceeded":
            return True
    return None


def fold_tool_call_fragment(
    slot: dict[str, str], index: int, name: str | None, fragment: str
) -> StreamToolCallDelta | None:
    """Fold one tool-call fragment into its slot; return the event it warrants.

    Every provider speaking OpenAI's dialect fragments a tool call the same
    way: the name arrives on one fragment, the arguments accumulate across the
    rest, and a composition event is due whenever either actually changed. What
    differs between them is only how the fragment is read — attributes here and
    in Mistral, a plain dict in PolarGrid — so the accessors stay at the call
    site and the folding lives once, here, beside the other helpers those
    providers already share.

    ``slot`` is mutated in place: it is the caller's accumulator for this
    call's index, and the complete :class:`StreamToolCall` is still built from
    it once the stream ends.
    """
    first_name = bool(name) and not slot["name"]
    if name:
        slot["name"] = name
    if fragment:
        slot["arguments"] += fragment
    if not slot["name"] or not (first_name or fragment):
        return None
    return StreamToolCallDelta(
        id=slot["id"], name=slot["name"], index=index, arguments_delta=fragment
    )


def extract_think_tags(text: str) -> tuple[str | None, str]:
    """Extract ``<think>...</think>`` content from *text*.

    Returns:
        ``(thinking, clean_text)`` — *thinking* is ``None`` when no tags
        are present.
    """
    matches = _THINK_RE.findall(text)
    if not matches:
        return None, text
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    clean = _THINK_RE.sub("", text).strip()
    return thinking or None, clean
