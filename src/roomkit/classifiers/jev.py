"""Jev, TypeSafe's System One model, as a classifier (RFC §6.8).

Jev is trained to answer typed questions with calibrated probabilities: a yes/no
question's probability of yes can be thresholded, a choice's spread read as
doubt. All the questions of one call are answered together, ~150 ms for a dozen.

Requires the ``typesafe`` extra::

    pip install roomkit[typesafe]
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from roomkit.classifiers.base import (
    Answer,
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    ClassifierError,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
    YesNoAnswer,
    check_answers,
)


def _import_sdk() -> Any:
    """The TypeSafe SDK, or a clear error naming the extra."""
    try:
        import typesafe_sdk

        return typesafe_sdk
    except ImportError as exc:
        raise ImportError(
            "typesafe-sdk is required for JevClassifier. "
            "Install it with: pip install roomkit[typesafe]"
        ) from exc


class JevClassifier(Classifier):
    """Answers on Jev, with calibrated probabilities.

    Args:
        api_key: The TypeSafe key; the SDK reads ``TYPESAFE_API_KEY`` when None.
        model: A Jev model id; the SDK's default when None.
        timeout: Seconds the whole call may take, retries included, before
            ``ClassifierError``.
        client: An ``AsyncTypeSafeClient`` to use instead of one made here (it
            is then the caller's to close).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        timeout: float = 3.0,
        client: Any | None = None,
    ) -> None:
        self._sdk = _import_sdk()
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or self._sdk.AsyncTypeSafeClient(
            api_key=api_key, model=model, timeout=timeout
        )

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        asked = {name: self._question(question) for name, question in questions.items()}
        try:
            response = await asyncio.wait_for(self._client.system_one(state, asked), self._timeout)
        except TimeoutError as exc:
            raise ClassifierError(f"no answer within {self._timeout} s") from exc
        except self._sdk.TypeSafeError as exc:
            raise ClassifierError(str(exc)) from exc
        answers = Answers()
        for name, question in questions.items():
            answer = response.answers.get(name)
            if answer is not None:
                answers[name] = _answer(question, answer)
        return check_answers(questions, answers)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _question(self, question: Question) -> Any:
        sdk = self._sdk
        if isinstance(question, ChoiceQuestion):
            return sdk.Choice(instructions=question.instructions, criteria=dict(question.options))
        if isinstance(question, ScoreQuestion):
            return sdk.Score(instructions=question.instructions, criteria=list(question.levels))
        described = {"true": question.yes, "false": question.no}
        criteria = {k: v for k, v in described.items() if v is not None} or None
        return sdk.Noul(instructions=question.instructions, criteria=criteria)


def _answer(question: Question, answer: Any) -> Answer:
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(answer.choice, dict(answer.probabilities))
    if isinstance(question, ScoreQuestion):
        levels = range(len(question.levels))
        return ScoreAnswer(answer.score, tuple(answer.probabilities.get(i, 0.0) for i in levels))
    return YesNoAnswer(answer.noul)
