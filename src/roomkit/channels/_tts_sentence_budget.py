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
    BEFORE_TTS), so a sentence a hook dropped does not count. The sentence
    over the budget is not passed on and nothing more is read: the channel
    returns before the stream's end, which the framework takes as an early
    stop (RFC §12.2 step 13s). A reply of exactly *limit* sentences runs to
    its end.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._spoken: list[str] = []
        self.cut = False

    async def run(self, sentences: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield the first sentences up to the budget; stop at the one over it."""
        async for sentence in sentences:
            if len(self._spoken) == self._limit:
                self.cut = True
                logger.info("Reply cut at its sentence budget (%d)", self._limit)
                return
            self._spoken.append(sentence)
            yield sentence

    def text(self) -> str:
        """The sentences spoken."""
        return " ".join(s.strip() for s in self._spoken if s.strip())


async def first_sentences(text: str, limit: int) -> str:
    """The first *limit* sentences of a whole *text*, cut by the rule a streamed
    reply is cut by."""
    budget = SentenceBudget(limit)
    async for _ in budget.run(split_sentences(_whole(text))):
        pass
    return budget.text() if budget.cut else text


async def _whole(text: str) -> AsyncIterator[str]:
    yield text
