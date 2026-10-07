"""Classifiers: narrow, typed questions answered with probabilities (RFC §6.8).

A component that needs judgment where code needs understanding (was the agent
addressed, did the person finish) asks a classifier its questions together, in
one call, and composes the answers in code, where each one stays visible and
measurable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

State = str | Mapping[str, Any] | list[Any]
"""What the questions are about: a text, or JSON-compatible structured data."""


class ClassifierError(Exception):
    """A classifier could not answer: a refused request, an answer it cannot read,
    or the end of its bounded wait."""


@dataclass(frozen=True)
class YesNoQuestion:
    """Whether something holds: answered with the probability of yes."""

    instructions: str
    """The question, and what it means."""

    yes: str | None = None
    """What a yes covers, when that needs saying."""

    no: str | None = None
    """What a no covers."""


@dataclass(frozen=True)
class ChoiceQuestion:
    """One option among several: answered with every option's probability."""

    instructions: str
    options: Mapping[str, str]
    """Each option's name and what it covers; two at least."""

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a ChoiceQuestion needs two options at least")


@dataclass(frozen=True)
class ScoreQuestion:
    """A degree on an ordered scale: answered with the expected level."""

    instructions: str
    levels: tuple[str, ...]
    """The levels, lowest first, each described; two at least."""

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError("a ScoreQuestion needs two levels at least")


Question = YesNoQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class YesNoAnswer:
    probability: float
    """Of yes, in [0, 1]."""


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    """The most probable option."""

    probabilities: Mapping[str, float] = field(default_factory=dict)
    """Every option's probability."""


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    """The expected level: 0 for the first, a fraction between two levels."""

    probabilities: tuple[float, ...] = ()
    """Each level's probability, in the levels' order."""


Answer = YesNoAnswer | ChoiceAnswer | ScoreAnswer


class Answers(dict[str, Answer]):
    """The answers by question name, with typed reads.

    ``answers.yes("addressed")`` is the probability of yes, ``answers.choice("lang")``
    the chosen option, ``answers.score("directness")`` the expected level.
    """

    def yes(self, name: str) -> float:
        return self._typed(name, YesNoAnswer).probability

    def choice(self, name: str) -> str:
        return self._typed(name, ChoiceAnswer).choice

    def score(self, name: str) -> float:
        return self._typed(name, ScoreAnswer).score

    def _typed[A](self, name: str, kind: type[A]) -> A:
        answer = self[name]
        if not isinstance(answer, kind):
            got = type(answer).__name__
            raise TypeError(f"{name!r} was answered as {got}, not {kind.__name__}")
        return answer


class Classifier(ABC):
    """Answers typed questions about a state, every one or none (``ClassifierError``).

    It sends the questions' wording as the caller wrote it, and bounds its wait.
    An implementation says how far its probabilities can be trusted.
    """

    @abstractmethod
    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        """Every question in *questions* answered, under its name."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (a client, a model)."""


def answer_kind(question: Question) -> type[Answer]:
    """The answer type *question* takes."""
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer
    return YesNoAnswer


def check_answers(questions: Mapping[str, Question], answers: Answers) -> Answers:
    """*answers*, once every question has its answer of the right type."""
    for name, question in questions.items():
        kind = answer_kind(question)
        if not isinstance(answers.get(name), kind):
            raise ClassifierError(f"no {kind.__name__} for {name!r}")
    return answers
