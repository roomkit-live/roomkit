"""OpenAIClassifier (RFC §6.8): RoomKit's questions sent as the Decisions API's,
its answers read back, its failures as ClassifierError. A fake client stands in
for the SDK's ``AsyncOpenAI``; the answers are shaped as the API returned them
on 2026-10-10 (``POST /v1/decisions``, ``gpt-6-luna``)."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from roomkit import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    OpenAIClassifier,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)

openai = pytest.importorskip("openai")

QUESTIONS = {
    "addressed": YesNoQuestion("Is the last message addressed to Nova?"),
    "done": YesNoQuestion("Did they finish?", yes="a complete thought"),
    "language": ChoiceQuestion("Which language?", {"fr": "French", "en": "English"}),
    "urgency": ScoreQuestion("How urgent?", ("can wait", "soon", "now")),
}

ANSWERS: list[dict[str, Any]] = [
    {"type": "predicate", "name": "addressed", "probability": 0.83},
    {"type": "predicate", "name": "done", "probability": 0.2},
    {
        "type": "choice",
        "name": "language",
        "choice": "fr",
        "probabilities": [
            {"value": "fr", "probability": 0.94},
            {"value": "en", "probability": 0.06},
        ],
        "confidence": 0.9,
    },
    {
        "type": "score",
        "name": "urgency",
        "score": 1.7,
        "probabilities": [
            {"value": 1, "label": "soon", "probability": 0.3},
            {"value": 2, "label": "now", "probability": 0.7},
        ],
        "confidence": 0.8,
    },
]


class FakeClient:
    def __init__(
        self,
        answers: list[dict[str, Any]] | None = None,
        *,
        payload: Any = None,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.payload = (
            payload
            if payload is not None
            else {
                "answers": ANSWERS if answers is None else answers,
                "model": "gpt-6-luna",
                "usage": {"input_tokens": 574, "output_tokens": 0, "total_tokens": 574},
            }
        )
        self.delay = delay
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], Any]] = []
        self.closed = False

    async def post(self, path: str, *, body: dict[str, Any], cast_to: Any) -> Any:
        self.calls.append((path, body, cast_to))
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.payload

    async def close(self) -> None:
        self.closed = True


async def test_questions_go_out_as_the_decisions_api_takes_them() -> None:
    client = FakeClient()
    await OpenAIClassifier(client=client).classify({"last": "Nova?"}, QUESTIONS)

    ((path, body, _cast),) = client.calls
    assert path == "/decisions"
    assert body["model"] == "gpt-6-luna"
    # A structured state goes as its JSON: the API reads a text.
    assert body["input"] == '{"last": "Nova?"}'
    addressed, done, language, urgency = body["questions"]
    assert addressed == {
        "type": "predicate",
        "name": "addressed",
        "instructions": "Is the last message addressed to Nova?",
    }
    # A predicate takes instructions alone: what a yes covers joins them.
    assert done["instructions"] == "Did they finish? Yes: a complete thought"
    assert language == {
        "type": "choice",
        "name": "language",
        "instructions": "Which language?",
        "choices": [
            {"value": "fr", "description": "French"},
            {"value": "en", "description": "English"},
        ],
    }
    assert urgency["levels"] == [{"label": "can wait"}, {"label": "soon"}, {"label": "now"}]


async def test_a_text_state_goes_as_it_is() -> None:
    client = FakeClient()
    await OpenAIClassifier(client=client).classify("Nova, are you there?", QUESTIONS)
    assert client.calls[0][1]["input"] == "Nova, are you there?"


async def test_answers_come_back_with_their_probabilities() -> None:
    answers = await OpenAIClassifier(client=FakeClient()).classify("x", QUESTIONS)
    assert answers["addressed"] == YesNoAnswer(0.83)
    assert answers["language"] == ChoiceAnswer("fr", {"fr": 0.94, "en": 0.06})
    # A level the API left out has probability 0.
    assert answers["urgency"] == ScoreAnswer(1.7, (0.0, 0.3, 0.7))


async def test_a_refused_question_fails_the_call() -> None:
    refused = [*ANSWERS[:3], {"type": "refusal", "name": "urgency"}]
    with pytest.raises(ClassifierError, match="declined to answer 'urgency'"):
        await OpenAIClassifier(client=FakeClient(refused)).classify("x", QUESTIONS)


async def test_a_missing_or_unreadable_answer_is_a_classifier_error() -> None:
    with pytest.raises(ClassifierError, match="no ScoreAnswer for 'urgency'"):
        await OpenAIClassifier(client=FakeClient(ANSWERS[:3])).classify("x", QUESTIONS)
    broken = [{"type": "predicate", "name": "addressed"}, *ANSWERS[1:]]
    with pytest.raises(ClassifierError, match="cannot be read"):
        await OpenAIClassifier(client=FakeClient(broken)).classify("x", QUESTIONS)
    with pytest.raises(ClassifierError, match="without answers"):
        await OpenAIClassifier(client=FakeClient(payload={"error": "?"})).classify("x", QUESTIONS)


async def test_an_sdk_error_or_the_wait_is_a_classifier_error() -> None:
    failing = FakeClient(error=openai.OpenAIError("Decision API is not enabled for this user"))
    with pytest.raises(ClassifierError, match="not enabled"):
        await OpenAIClassifier(client=failing).classify("x", QUESTIONS)
    slow = FakeClient(delay=1)
    with pytest.raises(ClassifierError, match="within 0.05 s"):
        await OpenAIClassifier(client=slow, timeout=0.05).classify("x", QUESTIONS)


async def test_close_closes_only_the_client_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
    lent = FakeClient()
    await OpenAIClassifier(client=lent).close()
    assert not lent.closed

    made: list[tuple[FakeClient, dict[str, Any]]] = []

    def factory(**kwargs: Any) -> FakeClient:
        made.append((FakeClient(), kwargs))
        return made[-1][0]

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    classifier = OpenAIClassifier("key", model="gpt-6-luna", timeout=1.5)
    assert made[0][1] == {"api_key": "key", "timeout": 1.5}
    await classifier.close()
    assert made[0][0].closed


def test_without_the_sdk_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r"roomkit\[openai\]"):
        OpenAIClassifier(client=FakeClient())
