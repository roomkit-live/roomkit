"""JevClassifier (RFC §6.8): RoomKit's questions sent as the TypeSafe SDK's, its
answers read back, its failures as ClassifierError. A fake client stands in for
the service; the questions and answers are the SDK's own types."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    JevClassifier,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)

sdk = pytest.importorskip("typesafe_sdk")

QUESTIONS = {
    "addressed": YesNoQuestion("Is the last message addressed to Nova?"),
    "done": YesNoQuestion("Did they finish?", yes="a complete thought"),
    "language": ChoiceQuestion("Which language?", {"fr": "French", "en": "English"}),
    "urgency": ScoreQuestion("How urgent?", ("can wait", "soon", "now")),
}

SDK_ANSWERS = {
    "addressed": sdk.NoulAnswer(noul=0.83),
    "done": sdk.NoulAnswer(noul=0.2),
    "language": sdk.ChoiceAnswer(
        choice="fr", confidence=0.9, probabilities={"fr": 0.94, "en": 0.06}
    ),
    "urgency": sdk.ScoreAnswer(
        score=1.7,
        confidence=0.8,
        legend={0: "can wait", 1: "soon", 2: "now"},
        probabilities={1: 0.3, 2: 0.7},
    ),
}


class FakeClient:
    def __init__(
        self,
        answers: Mapping[str, Any] = SDK_ANSWERS,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.answers = dict(answers)
        self.delay = delay
        self.error = error
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.closed = False

    async def system_one(self, state: Any, questions: Mapping[str, Any]) -> Any:
        self.calls.append((state, dict(questions)))
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(answers=self.answers)

    async def aclose(self) -> None:
        self.closed = True


async def test_questions_go_out_as_the_sdk_types() -> None:
    client = FakeClient()
    await JevClassifier(client=client).classify({"last": "Nova?"}, QUESTIONS)

    state, asked = client.calls[0]
    assert state == {"last": "Nova?"}
    assert asked["addressed"] == sdk.Noul(instructions="Is the last message addressed to Nova?")
    # An outcome left undescribed is not sent.
    assert asked["done"] == sdk.Noul(
        instructions="Did they finish?", criteria={"true": "a complete thought"}
    )
    assert asked["language"] == sdk.Choice(
        instructions="Which language?", criteria={"fr": "French", "en": "English"}
    )
    assert asked["urgency"] == sdk.Score(
        instructions="How urgent?", criteria=["can wait", "soon", "now"]
    )


async def test_answers_come_back_with_their_probabilities() -> None:
    answers = await JevClassifier(client=FakeClient()).classify("x", QUESTIONS)
    assert answers["addressed"] == YesNoAnswer(0.83)
    assert answers["language"] == ChoiceAnswer("fr", {"fr": 0.94, "en": 0.06})
    # A level the service left out has probability 0.
    assert answers["urgency"] == ScoreAnswer(1.7, (0.0, 0.3, 0.7))


async def test_a_missing_answer_an_sdk_error_or_the_wait_is_a_classifier_error() -> None:
    partial = {k: v for k, v in SDK_ANSWERS.items() if k != "urgency"}
    with pytest.raises(ClassifierError, match="no ScoreAnswer for 'urgency'"):
        await JevClassifier(client=FakeClient(partial)).classify("x", QUESTIONS)
    failing = FakeClient(error=sdk.TypeSafeError("quota exceeded"))
    with pytest.raises(ClassifierError, match="quota exceeded"):
        await JevClassifier(client=failing).classify("x", QUESTIONS)
    with pytest.raises(ClassifierError, match="within 0.05 s"):
        await JevClassifier(client=FakeClient(delay=1), timeout=0.05).classify("x", QUESTIONS)


async def test_close_closes_only_the_client_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
    lent = FakeClient()
    await JevClassifier(client=lent).close()
    assert not lent.closed

    made: list[tuple[FakeClient, dict[str, Any]]] = []

    def factory(**kwargs: Any) -> FakeClient:
        made.append((FakeClient(), kwargs))
        return made[-1][0]

    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", factory)
    classifier = JevClassifier("key", model="jev-1", timeout=1.5)
    assert made[0][1] == {"api_key": "key", "model": "jev-1", "timeout": 1.5}
    await classifier.close()
    assert made[0][0].closed


def test_without_the_sdk_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ImportError, match=r"roomkit\[typesafe\]"):
        JevClassifier(client=FakeClient())
