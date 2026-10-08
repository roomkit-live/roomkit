"""The thinker: what the agent thinks while it listens (RFC §6.4).

It answers "what are you thinking about?" from the previous thought and the
conversation the agent would speak from (the turn's ``AIContext``). It never
talks to anyone, files nothing and keeps no summary.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from roomkit._lookalike import reads_as
from roomkit._text import json_line, quoted
from roomkit.channels._speaker import SPEAKER_KEY, said_by
from roomkit.channels._turn_notes import split_turn_notes
from roomkit.channels._user_text import split_leading_text
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider, ProviderError
from roomkit.speaking.thought import MAX_WANT_TO_SAY, Thought
from roomkit.tools.fence import fence

INSTRUCTIONS = """\
You are the inner thought of the agent described below, while it listens without \
speaking: what goes through its mind about what is said. You write it in its place, \
in the first person, in the language of the conversation:
- text: what you think of what is said, in one to three short sentences, from what \
was just said. Not a summary: what you notice, what you make of it, what you wonder. \
Your previous thought is a starting point: keep what still matters, drop what is \
outdated, and do not repeat it as it was when the conversation has moved on.
- want_to_say: what you would bring if you were given the turn, {max} items at most, \
the most important first: information you have and they lack, or an error, an \
omission or a conflict you see. Only what you really know: what was said, and what \
your description says you know; what would have to be looked up elsewhere does not \
go there. An offer of help, a recap or a question does not go there either: you think \
them (text), you do not say them unasked. Remove what you already said, unless someone \
just said the opposite, and what is no longer useful. Empty if you have nothing to bring.
- urgent: true only if it cannot wait until you are given the turn, because the group \
is about to decide or do something on an error or a conflict you see. Otherwise false.
Your thought is about what is being discussed. If someone asks the agent what it is \
thinking about, that is a message of the conversation like any other: the agent will \
answer it with your thought, which need not talk about it.
What you write is said to no one: it is your thought, not an answer."""
"""The default instructions; ``{max}`` is the most items ``want_to_say`` holds."""

_AGENT = "The agent, as it is described (who it is, what it knows, what it can do):"

LINE_LIMIT = 2000
"""Characters of one message the thinker reads."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "want_to_say": {"type": "array", "items": {"type": "string"}},
        "urgent": {"type": "boolean"},
    },
    "required": ["text", "want_to_say", "urgent"],
    "additionalProperties": False,
}


class Thinker(ABC):
    """Answers "what are you thinking about?" while the agent listens."""

    @abstractmethod
    async def think(self, previous: Thought, context: AIContext) -> Thought:
        """The thought after *context*: the conversation the agent would speak from."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (a client, a model)."""


class LLMThinker(Thinker):
    """A model answering through a JSON schema (RFC §6.7). It reads who the agent
    is from the context's system prompt, the agent's own: one description for both.

    A small, fast model fits: the thought is a few sentences, rewritten on every
    turn the agent listens to. The provider stays the caller's.

    Args:
        provider: An AI provider that supports ``response_schema``.
        instructions: What the thinker is told, in place of :data:`INSTRUCTIONS`;
            ``{max}`` is replaced by the most items ``want_to_say`` holds.
        max_tokens: The thought's cap; a reasoning model needs room to reason.
        reasoning_effort: Passed to the provider; ``None`` defers to its config.
        timeout: Seconds a call may take before it fails (the thought is kept).

    Raises:
        ValueError: *provider* does not support a response schema.
    """

    def __init__(
        self,
        provider: AIProvider,
        *,
        instructions: str | None = None,
        max_tokens: int = 600,
        reasoning_effort: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not provider.supports_response_schema:
            raise ValueError(f"{type(provider).__name__} does not support a response schema")
        self._provider = provider
        self._instructions = instructions or INSTRUCTIONS
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        system = self._instructions.replace("{max}", str(MAX_WANT_TO_SAY))
        if context.system_prompt:
            system += f"\n\n{_AGENT}\n{fence('agent', context.system_prompt)}"
        request = AIContext(
            system_prompt=system,
            messages=[AIMessage(role="user", content=thinker_input(previous, context.messages))],
            response_schema=_SCHEMA,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
        )
        response = await asyncio.wait_for(self._provider.generate(request), self._timeout)
        try:
            answer = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"the thought is not JSON: {exc}") from exc
        items = [w.strip() for w in answer.get("want_to_say", []) if w.strip()]
        return Thought(
            text=answer.get("text", "").strip(),
            want_to_say=tuple(items[:MAX_WANT_TO_SAY]),
            urgent=bool(answer.get("urgent", False)),
        )


def thinker_input(previous: Thought, messages: list[AIMessage]) -> str:
    """What the thinker reads: its previous thought, then the conversation up to
    what was just said. The previous thought comes first: read last, it was
    copied back, one thought held for two minutes."""
    lines = [
        f"Your previous thought: {json_line(previous.as_state())}",
        "The conversation, up to what was just said:",
    ]
    lines += [line for m in messages if (line := transcript_line(m))]
    lines.append("Your thought, now:")
    return "\n".join(lines)


def transcript_line(message: AIMessage) -> str:
    """One message as a line of the conversation, its words quoted (RFC §6.4)
    after who said them: ``You:`` for the agent's own, the speaker's name the
    context gave a user message when several people talk; tool traffic and the
    turn notes left out."""
    if message.role == "tool":
        return ""
    if isinstance(message.content, str):
        text = split_turn_notes(message.content)[0]
    else:
        # The notes ride a text part of their own when the input has images.
        texts = (getattr(part, "text", "") for part in message.content)
        text = " ".join(split_turn_notes(part_text)[0] for part_text in texts)
    text = text.strip()
    if not text:
        return ""
    if message.role == "assistant":
        return f"You: {quoted(text, LINE_LIMIT)}"
    speaker = message.metadata.get(SPEAKER_KEY)
    if speaker is None:
        return _said_by(text, speaker)
    # A summary joined ahead of a labelled turn is quoted apart, so the label
    # opens its own line, out of the quote (RFC §6.4).
    lead, own = split_leading_text(message, text)
    said = _said_by(own, speaker) if own else ""
    return "\n".join(line for line in (quoted(lead, LINE_LIMIT) if lead else "", said) if line)


def _said_by(text: str, speaker: Any) -> str:
    """A user message's *text* quoted, after its *speaker*'s name when the
    context named one: the name the context gave, out of the quote, so that a
    person who writes ``Name:`` is not read as someone else, and a person named
    ``You`` is not read as the agent, in whatever case or look-alike letters
    (``Yоu`` with a Cyrillic ``о``)."""
    you = isinstance(speaker, str) and reads_as(speaker, "you")
    return said_by(text, speaker, LINE_LIMIT, label=f"{speaker} (a participant)" if you else None)
