"""A scripted classifier for tests."""

from __future__ import annotations

from collections.abc import Mapping

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
)


class MockClassifier(Classifier):
    """Answers from a script, by question name: a float for a yes/no question
    (the probability of yes), an option for a choice, a level for a score, or an
    :data:`~roomkit.classifiers.base.Answer`. A question the script does not name
    gets no, the first option, or the lowest level. Records every call.
    """

    def __init__(
        self,
        answers: Mapping[str, Answer | float | str] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._script = dict(answers or {})
        self._error = error
        self.calls: list[tuple[State, dict[str, Question]]] = []

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        self.calls.append((state, dict(questions)))
        if self._error is not None:
            raise self._error
        return Answers({name: self._answer(name, q) for name, q in questions.items()})

    def _answer(self, name: str, question: Question) -> Answer:
        scripted = self._script.get(name)
        if isinstance(scripted, YesNoAnswer | ChoiceAnswer | ScoreAnswer):
            return scripted
        if isinstance(question, ChoiceQuestion):
            choice = str(scripted) if scripted is not None else next(iter(question.options))
            if choice not in question.options:
                raise ClassifierError(f"{choice!r} is not an option of {name!r}")
            return ChoiceAnswer(choice, {o: float(o == choice) for o in question.options})
        if isinstance(question, ScoreQuestion):
            level = float(scripted) if scripted is not None else 0.0
            return ScoreAnswer(level)
        return YesNoAnswer(float(scripted) if scripted is not None else 0.0)
