"""The OpenAI Chat Completions dialect, read on the way back.

Several providers speak it: ``OpenAIAIProvider`` and its derivatives, Mistral
and PolarGrid in their own packages. What they share is how a response is
read, not how a request is built: the two reasoning conventions (inline
``<think>`` tags and a dedicated field), the way a tool call is fragmented
across stream chunks, the structured context-overflow fact off a status
error, and why a constrained answer was withheld. It lives with the AI provider
ABC rather than under one vendor because three vendor packages consume it. The
one request piece here is the strict ``json_schema`` response format (RFC
§6.7), which every speaker of the dialect asks for in the same words; the
messages a request carries are rendered by ``chat_request``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from roomkit.providers.ai.base import (
    AIToolCall,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.tool_calls import (
    CallIds,
    arguments_cut,
    call_garbled,
    call_partial,
    minted_call_id,
    tool_arguments,
)


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
    error body — a first-hand fact, unlike the message wording. The SDK hands
    over that ``error`` object itself as the exception's ``body``. The other
    compatible vendors put integers or generic strings in ``code``, so a miss
    is ``None`` (nobody classified), never ``False``: their overflows are
    still caught by the shared phrase fallback.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("code") == "context_length_exceeded":
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


class ToolCallSlots:
    """Streamed tool-call fragments folded into calls, one slot per call.

    A fragment belongs to the call its stream index names, unless it starts
    another call: a server that sends each call whole may tag every one with
    index 0 (Mistral's SDK defaults it), and folding them together would run
    one call made of two. Each slot holds its call's id from the start, the
    server's or a minted one, and keeps it once a composition event named the
    call by it, so the events carry the id the call ends with (RFC §6.4).
    """

    def __init__(self) -> None:
        self._slots: list[dict[str, str]] = []
        self._by_index: dict[int, int] = {}
        self._taken: set[str] = set()

    def fold(
        self,
        index: int | None,
        call_id: str | None,
        name: str | None,
        fragment: str | Mapping[str, Any] | None,
    ) -> StreamToolCallDelta | None:
        """Fold one fragment in; return the composition event it warrants.

        A fragment a server sends as an object (Mistral's SDK types it
        ``Dict | str``; some compatible servers do it on OpenAI's wire) reads
        as the JSON text it stands for.
        """
        fragment = _argument_text(fragment)
        key = index if index is not None else 0
        position = self._by_index.get(key)
        if position is not None and self._starts_another_call(position, call_id, name, fragment):
            held = self._slots[position]
            if not name and call_id in (None, "", held["id"]):
                # Opened by a new JSON object under no new id or name: a
                # second call to the held call's tool.
                name = held["name"]
            position = None
        if position is None:
            self._slots.append(self._new_slot(name))
            position = self._by_index[key] = len(self._slots) - 1
        slot = self._slots[position]
        self._adopt_server_id(slot, call_id)
        delta = fold_tool_call_fragment(slot, position, name, fragment)
        if delta is not None:
            slot["announced"] = "1"
        return delta

    def _new_slot(self, name: str | None) -> dict[str, str]:
        call_id = minted_call_id(name or "tool")
        self._taken.add(call_id)
        return {"id": call_id, "minted": "1", "announced": "", "name": "", "arguments": ""}

    def _adopt_server_id(self, slot: dict[str, str], call_id: str | None) -> None:
        """Give a call the server's id in place of its minted one, unless an
        event already named the call by the minted one or another call of the
        response holds the server's."""
        if not call_id or not slot["minted"] or slot["announced"] or call_id in self._taken:
            return
        self._taken.discard(slot["id"])
        self._taken.add(call_id)
        slot["id"], slot["minted"] = call_id, ""

    def _starts_another_call(
        self, position: int, call_id: str | None, name: str | None, fragment: str
    ) -> bool:
        """Whether a fragment on an occupied index is the start of another call.

        Another server id says so. Without one: another name once the held
        call is whole (no arguments, or complete JSON); the same name bringing
        arguments of its own after whole ones, a second call to the same tool;
        or, after whole arguments, a fragment that opens another JSON object,
        which no continuation of them can. A server may repeat a call's name
        on every fragment, so the same name with nothing or blanks starts
        nothing.
        """
        slot = self._slots[position]
        if call_id and not slot["minted"] and call_id != slot["id"]:
            return True
        held = slot["arguments"]
        whole = not arguments_cut(held)
        if not slot["name"]:
            return False
        if not name:
            return bool(held.strip()) and whole and fragment.lstrip().startswith("{")
        if name != slot["name"]:
            return whole
        return bool(fragment.strip()) and bool(held.strip()) and whole

    def calls(self, finish_reason: str | None) -> list[StreamToolCall]:
        """The complete calls, each with its own id and its arguments as a
        mapping; one whose arguments do not read is partial, and cut when the
        response was cut short over them, which only the last call can be.

        A slot that never got a name or an argument is no call: an entry with
        no function (a custom tool's call), as a response's reader skips it
        (:func:`message_tool_calls`). Opening the slot still keeps the
        server's id for a call whose function arrives on a later fragment.
        """
        slots = [slot for slot in self._slots if slot["name"] or slot["arguments"]]
        final = len(slots) - 1
        return [
            StreamToolCall(
                id=slot["id"],
                name=slot["name"],
                arguments=tool_arguments(slot["arguments"]),
                partial=call_partial(slot["arguments"], finish_reason, last=n == final),
                garbled=call_garbled(slot["arguments"], finish_reason, last=n == final),
            )
            for n, slot in enumerate(slots)
        ]


def _argument_text(fragment: str | Mapping[str, Any] | None) -> str:
    """A streamed call's arguments fragment as text, an object as its JSON."""
    if isinstance(fragment, Mapping):
        return json.dumps(dict(fragment))
    return fragment or ""


def message_tool_calls(message: Any, finish_reason: str | None) -> list[AIToolCall]:
    """The calls of a chat completion's message, as the loop reads them
    (RFC §6.4): each with an id of its own, its arguments read as text or as
    an object, cut or not; a call with no name keeps an empty one, for the
    loop to refuse, never an error raised while reading the response."""
    raw_calls = [
        call
        for call in getattr(message, "tool_calls", None) or []
        if getattr(call, "function", None) is not None
    ]
    ids = CallIds()
    final = len(raw_calls) - 1
    calls: list[AIToolCall] = []
    for n, call in enumerate(raw_calls):
        name = str(getattr(call.function, "name", "") or "")
        raw = getattr(call.function, "arguments", "")
        calls.append(
            AIToolCall(
                id=ids(getattr(call, "id", None), name),
                name=name,
                arguments=tool_arguments(raw),
                partial=call_partial(raw, finish_reason, last=n == final),
                garbled=call_garbled(raw, finish_reason, last=n == final),
            )
        )
    return calls


def extract_think_tags(text: str) -> tuple[str | None, str]:
    """Split a whole response's *text* into its ``<think>`` reasoning and its
    answer, as :class:`ThinkTagParser` splits a stream: a block the response
    stopped before closing (the output cap) is reasoning too, never answer.

    Returns:
        ``(thinking, clean_text)`` — *thinking* is ``None`` when no tags
        are present.
    """
    if ThinkTagParser._OPEN not in text:
        return None, text
    parser = ThinkTagParser()
    segments = parser.feed(text) + parser.flush()
    thinking = "\n".join(c.strip() for kind, c in segments if kind == "thinking" and c.strip())
    clean = "".join(c for kind, c in segments if kind == "text").strip()
    return thinking or None, clean


def json_schema_format(schema: dict[str, Any]) -> dict[str, Any]:
    """The ``response_format`` that constrains an answer to *schema*, strictly."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }


def choice_refusal(choice: Any) -> str | None:
    """Why a choice withheld its constrained answer, if it did.

    ``message.refusal`` is where a structured-output refusal lands on this
    dialect; a content filter (Azure's, and some compatible servers') says the
    same through the finish reason instead.
    """
    refusal = getattr(getattr(choice, "message", None), "refusal", None)
    if refusal:
        return refusal
    return "content_filter" if getattr(choice, "finish_reason", None) == "content_filter" else None
