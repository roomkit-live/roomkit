"""OpenAI's Decisions API as a classifier (RFC §6.8).

``POST /v1/decisions`` (public beta, ``gpt-6-luna``) answers typed questions
about a text with probabilities: a predicate (a yes/no question) with the
probability that it holds, a choice with every option's, a score with each
level's and the expected level. All the questions of one call are answered
together, ~0.4 s for four. A structured state is sent as its JSON. OpenAI
advises calibrating thresholds on labelled data.

The OpenAI SDK 2.x has no ``decisions`` resource: the call goes through its
generic ``post``, with the client's key, retries and errors.

Requires the ``openai`` extra::

    pip install roomkit[openai]
"""

from __future__ import annotations

import asyncio
import json
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
    YesNoQuestion,
    check_answers,
)

DEFAULT_MODEL = "gpt-6-luna"
"""The one model the Decisions API takes in its beta."""


def _import_sdk() -> Any:
    """The OpenAI SDK, or a clear error naming the extra."""
    try:
        import openai

        return openai
    except ImportError as exc:
        raise ImportError(
            "openai is required for OpenAIClassifier. Install it with: pip install roomkit[openai]"
        ) from exc


class OpenAIClassifier(Classifier):
    """Answers on OpenAI's Decisions API, with probabilities.

    Args:
        api_key: The OpenAI key; the SDK reads ``OPENAI_API_KEY`` when None.
        model: A model the Decisions API takes.
        timeout: Seconds the whole call may take, retries included, before
            ``ClassifierError``.
        client: An ``AsyncOpenAI`` to use instead of one made here (it is then
            the caller's to close).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 3.0,
        client: Any | None = None,
    ) -> None:
        self._sdk = _import_sdk()
        self._model = model
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or self._sdk.AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        body = {
            "model": self._model,
            "input": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            "questions": [_question(name, question) for name, question in questions.items()],
        }
        try:
            payload = await asyncio.wait_for(
                self._client.post("/decisions", body=body, cast_to=object), self._timeout
            )
        except TimeoutError as exc:
            raise ClassifierError(f"no answer within {self._timeout} s") from exc
        except self._sdk.OpenAIError as exc:
            raise ClassifierError(str(exc)) from exc
        return check_answers(questions, _answers(questions, payload))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _question(name: str, question: Question) -> dict[str, Any]:
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "name": name,
            "instructions": question.instructions,
            "choices": [
                {"value": option, "description": described}
                for option, described in question.options.items()
            ],
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "name": name,
            "instructions": question.instructions,
            "levels": [{"label": level} for level in question.levels],
        }
    return {"type": "predicate", "name": name, "instructions": _predicate(question)}


def _predicate(question: YesNoQuestion) -> str:
    """A yes/no question as a predicate: the API takes instructions alone, so
    what a yes and a no cover joins them."""
    covered = [
        f"{side}: {text}" for side, text in (("Yes", question.yes), ("No", question.no)) if text
    ]
    return " ".join([question.instructions, *covered])


def _answers(questions: Mapping[str, Question], payload: Any) -> Answers:
    """The answers *payload* holds for *questions*, by name; a question the
    API declined fails the call."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        raise ClassifierError("the Decisions API answered without answers")
    answers = Answers()
    for answer in payload["answers"]:
        if not isinstance(answer, dict):
            raise ClassifierError("the Decisions API answered something it cannot read")
        name = answer.get("name")
        if answer.get("type") == "refusal":
            raise ClassifierError(f"the Decisions API declined to answer {name!r}")
        question = questions.get(name) if isinstance(name, str) else None
        if question is not None:
            answers[name] = _answer(question, answer)
    return answers


def _answer(question: Question, answer: dict[str, Any]) -> Answer:
    try:
        if isinstance(question, ChoiceQuestion):
            spread = {str(p["value"]): float(p["probability"]) for p in answer["probabilities"]}
            return ChoiceAnswer(str(answer["choice"]), spread)
        if isinstance(question, ScoreQuestion):
            by_level = {int(p["value"]): float(p["probability"]) for p in answer["probabilities"]}
            levels = range(len(question.levels))
            return ScoreAnswer(float(answer["score"]), tuple(by_level.get(i, 0.0) for i in levels))
        return YesNoAnswer(float(answer["probability"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ClassifierError(f"an answer the Decisions API gave cannot be read: {exc}") from exc
