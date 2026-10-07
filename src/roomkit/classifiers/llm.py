"""A classifier on any AI provider that answers under a JSON schema (RFC §6.8).

The model is asked for one answer per question: a yes or a no, an option, a
level. Its probabilities are therefore degenerate, not calibrated: the answer it
chose gets probability 1. Use it where a generative model is what you have; a
trained classifier (Jev) gives probabilities to threshold on.
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
    check_answers,
)
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider, ProviderError

_SYSTEM = (
    "You answer questions about the state below, each on its own, from the state "
    "alone. Answer every question with the JSON the schema asks: a yes/no question "
    "with true or false, a choice with one of its options, a score with the number "
    "of the level that fits (0 for the first)."
)


class LLMClassifier(Classifier):
    """Answers on *provider* under a JSON schema; probabilities degenerate (0 or 1).

    Args:
        provider: An AI provider that supports ``response_schema`` (RFC §6.7).
            It stays the caller's: ``close()`` leaves it open.
        timeout: Seconds it waits for the answer before raising ``ClassifierError``.
        max_tokens: The answer's cap; a reasoning model needs room for its reasoning.

    Raises:
        ValueError: *provider* does not support a response schema.
    """

    def __init__(self, provider: AIProvider, *, timeout: float = 10.0, max_tokens: int = 1000):
        if not provider.supports_response_schema:
            raise ValueError(f"{type(provider).__name__} does not support a response schema")
        self._provider = provider
        self._timeout = timeout
        self._max_tokens = max_tokens

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        context = AIContext(
            system_prompt=_SYSTEM,
            messages=[AIMessage(role="user", content=_prompt(state, questions))],
            response_schema=_schema(questions),
            max_tokens=self._max_tokens,
        )
        try:
            response = await asyncio.wait_for(self._provider.generate(context), self._timeout)
            reply = json.loads(response.content)
        except TimeoutError as exc:
            raise ClassifierError(f"no answer within {self._timeout} s") from exc
        except (ProviderError, json.JSONDecodeError) as exc:
            raise ClassifierError(str(exc)) from exc
        if not isinstance(reply, dict):
            raise ClassifierError("the answer is not a JSON object")
        answers = Answers()
        for name, question in questions.items():
            if name in reply:
                answers[name] = _answer(question, reply[name])
        return check_answers(questions, answers)


def _prompt(state: State, questions: Mapping[str, Question]) -> str:
    lines = [
        "State:",
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
        "",
        "Questions:",
    ]
    for name, question in questions.items():
        lines.append(f"- {name}: {question.instructions}")
        if isinstance(question, ChoiceQuestion):
            lines += [f"    {option}: {meaning}" for option, meaning in question.options.items()]
        elif isinstance(question, ScoreQuestion):
            lines += [f"    {i}: {level}" for i, level in enumerate(question.levels)]
        else:
            if question.yes:
                lines.append(f"    true: {question.yes}")
            if question.no:
                lines.append(f"    false: {question.no}")
    return "\n".join(lines)


def _schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, question in questions.items():
        if isinstance(question, ChoiceQuestion):
            properties[name] = {"type": "string", "enum": list(question.options)}
        elif isinstance(question, ScoreQuestion):
            # Only a string may carry an enum in the portable subset (RFC §6.7).
            properties[name] = {"type": "string", "enum": _level_names(question)}
        else:
            properties[name] = {"type": "boolean"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(questions),
        "additionalProperties": False,
    }


def _level_names(question: ScoreQuestion) -> list[str]:
    return [str(i) for i in range(len(question.levels))]


def _answer(question: Question, value: Any) -> Answer:
    if isinstance(question, ChoiceQuestion):
        if value not in question.options:
            raise ClassifierError(f"{value!r} is not an option")
        return ChoiceAnswer(value, {o: float(o == value) for o in question.options})
    if isinstance(question, ScoreQuestion):
        names = _level_names(question)
        if value not in names:
            raise ClassifierError(f"{value!r} is not a level")
        return ScoreAnswer(float(names.index(value)), tuple(float(n == value) for n in names))
    if not isinstance(value, bool):
        raise ClassifierError(f"{value!r} is not a yes or a no")
    return YesNoAnswer(float(value))
