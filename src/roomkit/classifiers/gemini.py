"""Gemini's Decisions API as a classifier (RFC §6.8).

``POST /v1beta/decisions`` (``gemini-decision-1``) scores typed questions
about an input in one forward pass rather than generating an answer: a
``noul`` (a yes/no question) with the probability of yes, a choice with every
option's, a score with each level's and the expected level. All the questions
of one call are answered together, 250-450 ms measured with the network. Google
documents the probabilities as calibrated; a yes/no one stays within
[0.01, 0.99]. A structured state is sent as its JSON.

The call goes through google-genai's ``decisions`` resource, on the client
every Gemini provider here shares (``providers/gemini/sdk.py``), and always
names v1beta: the API has no other version. The client it builds does not
retry: a 429 or a 5xx fails at once with the status and message the API gave,
rather than spending the bounded wait on retries and ending as a timeout. A
client passed in keeps its own retry settings.

Requires the ``gemini`` extra::

    pip install roomkit[gemini]
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
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
from roomkit.providers.ai.base import server_status
from roomkit.providers.gemini.sdk import (
    DEVELOPER_API_VERSION,
    build_genai_client,
    close_genai_client,
)

DEFAULT_MODEL = "gemini-decision-1"
"""The one model the Decisions API takes."""


def _require_decisions() -> None:
    """google-genai with its Decisions API (3.2 or later), or a clear error
    naming the extra."""
    try:
        import google.genai.decisions  # noqa: F401 - the 3.2 resource's types
    except ImportError as exc:
        raise ImportError(
            "google-genai 3.2 or later is required for GeminiClassifier. "
            "Install it with: pip install roomkit[gemini]"
        ) from exc


@dataclass
class _Budget:
    """The HTTP client's timeouts: the whole call's, the connect included."""

    timeout: float
    connect_timeout: float


class GeminiClassifier(Classifier):
    """Answers on Gemini's Decisions API, with calibrated probabilities.

    Example::

        from roomkit import GeminiClassifier, YesNoQuestion

        classifier = GeminiClassifier(api_key)
        answers = await classifier.classify(
            "Nova, can you book a table?",
            {"addressed": YesNoQuestion("Is the message addressed to Nova?")},
        )
        answers.yes("addressed")

    Args:
        api_key: The Gemini API key. Never read from the environment;
            required unless ``client`` is given.
        model: A model the Decisions API takes.
        timeout: Seconds the whole call may take before ``ClassifierError``.
        client: A ``google.genai.Client`` for the Gemini Developer API to use
            instead of one made here (it is then the caller's to close). Its
            calls name v1beta whatever version it was built for, and keep its
            retry settings; a Vertex AI client has no Decisions API. The
            client made here does not retry.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 3.0,
        client: Any | None = None,
    ) -> None:
        _require_decisions()
        self._model = model
        self._timeout = timeout
        self._http: Any = None
        if client is not None:
            self._client = client
        elif api_key:
            built = build_genai_client(
                _Budget(timeout, timeout),
                provider="GeminiClassifier",
                api_key=api_key,
                disable_retries=True,
            )
            self._client, self._http = built.client, built.http
        else:
            raise ValueError("GeminiClassifier needs an api_key, or a client")
        self._owns_client = client is None

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        call = self._client.aio.decisions.create(
            model=self._model,
            input=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            questions={name: _question(question) for name, question in questions.items()},
            api_version=DEVELOPER_API_VERSION,
        )
        try:
            response = await asyncio.wait_for(call, self._timeout)
        except TimeoutError as exc:
            raise ClassifierError(f"no answer within {self._timeout} s") from exc
        except Exception as exc:
            # The SDK's error classes for this resource are private: whatever
            # the call raised is reported with the status the server gave.
            raise ClassifierError(_failure(exc)) from exc
        return check_answers(questions, _answers(questions, response))

    async def close(self) -> None:
        if self._owns_client:
            await close_genai_client(self._client, self._http)


def _failure(exc: Exception) -> str:
    """What a failed call says: the HTTP status the server gave and its
    message, or the failure's type when no server answered."""
    status = server_status(exc)
    if status is not None:
        return f"the Decisions API answered HTTP {status}: {_server_message(exc)}"
    return f"the Decisions API call failed: {type(exc).__name__}: {exc}"


def _server_message(exc: Exception) -> str:
    """The message in the API's error body (``{"error": {"message": ...}}``),
    else what the exception says."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) and message else str(exc)


def _question(question: Question) -> dict[str, Any]:
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.options),
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": list(question.levels),
        }
    return _noul(question)


def _noul(question: YesNoQuestion) -> dict[str, Any]:
    """A yes/no question, with what a yes and a no cover when the caller said."""
    asked: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
    described = {"true_description": question.yes, "false_description": question.no}
    criteria = {side: text for side, text in described.items() if text is not None}
    if criteria:
        asked["criteria"] = criteria
    return asked


def _answers(questions: Mapping[str, Question], response: Any) -> Answers:
    """The answers *response* holds for *questions*, by name; an input the
    API's filters blocked fails the call, and ``check_answers`` refuses a
    question left without an answer of its type."""
    blocked = _block_reason(response)
    if blocked is not None:
        raise ClassifierError(f"the Decisions API blocked the input: {blocked}")
    held = getattr(response, "answers", None) or {}
    answers = Answers()
    for name, question in questions.items():
        answer = held.get(name)
        if _fits(question, answer):
            answers[name] = _answer(question, answer)
    return answers


def _block_reason(response: Any) -> str | None:
    """Why the API's input filters blocked the call, when they did."""
    feedback = getattr(response, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    if reason is None:
        return None
    message = getattr(feedback, "block_reason_message", None)
    return f"{reason} ({message})" if message else str(reason)


def _fits(question: Question, answer: Any) -> bool:
    """Whether *answer* is of its question's type, and a choice one of the
    options asked."""
    kind = getattr(answer, "type", None)
    if isinstance(question, ChoiceQuestion):
        return kind == "choice" and getattr(answer, "choice", None) in question.options
    if isinstance(question, ScoreQuestion):
        return kind == "score"
    return kind == "noul"


def _answer(question: Question, answer: Any) -> Answer:
    try:
        # float() refuses a None: the SDK hands back unvalidated a body that
        # fails its own validation.
        if isinstance(question, ChoiceQuestion):
            spread = {str(option): float(p) for option, p in answer.probabilities.items()}
            return ChoiceAnswer(str(answer.choice), spread)
        if isinstance(question, ScoreQuestion):
            by_level = {int(level): float(p) for level, p in answer.probabilities.items()}
            levels = tuple(by_level.get(i, 0.0) for i in range(len(question.levels)))
            return ScoreAnswer(float(answer.score), levels)
        return YesNoAnswer(float(answer.noul))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ClassifierError(f"an answer the Decisions API gave cannot be read: {exc}") from exc
