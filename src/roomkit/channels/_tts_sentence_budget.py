"""A voice channel's sentence budget: at most N sentences spoken per reply
(RFC §12.2 step 12s.e)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from roomkit.voice.tts.sentence_splitter import split_sentences

logger = logging.getLogger("roomkit.channels.voice")


class SentenceBudget:
    """Lets a streamed reply's first *limit* sentences through, and ends the
    reply at the next one.

    It reads the sentences as the TTS would (after the text filter and
    BEFORE_TTS), so a sentence a hook dropped does not count. Once the budget
    is said (:meth:`spent`), the channel stops reading the reply at its first
    text past it, before the turn asks for anything more (a tool call, a
    round): the channel returns before the stream's end, which the framework
    takes as an early stop (RFC §12.2 step 13s). What the splitter held of the
    sentence over the budget then reaches :meth:`run`, which does not pass it
    on. A reply of exactly *limit* sentences runs to its end.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._spoken: list[str] = []
        self.cut = False

    async def run(self, sentences: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield the first sentences up to the budget; stop at the one over it."""
        async for sentence in sentences:
            if self.spent():
                self.cut = True
                logger.info("Sentence over the budget (%d) not spoken", self._limit)
                return
            self._spoken.append(sentence)
            yield sentence

    def spent(self) -> bool:
        """Whether the budget's sentences are all said."""
        return len(self._spoken) == self._limit

    def text(self) -> str:
        """The sentences spoken."""
        return " ".join(s.strip() for s in self._spoken if s.strip())


def checked_budget(limit: int | None) -> int | None:
    """*limit* as a voice channel takes its sentence budget: none, or at least one."""
    if limit is not None and limit < 1:
        raise ValueError("max_sentences must be at least 1")
    return limit


async def first_sentences(text: str, limit: int) -> str:
    """The first *limit* sentences of a whole *text*, cut by the rule a streamed
    reply is cut by."""
    budget = SentenceBudget(limit)
    async for _ in budget.run(split_sentences(_whole(text))):
        pass
    return budget.text() if budget.cut else text


async def _whole(text: str) -> AsyncIterator[str]:
    yield text
