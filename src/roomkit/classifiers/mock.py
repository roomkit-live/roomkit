"""A scripted classifier for tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

Script = Mapping[str, Answer | float | str]


class MockClassifier(Classifier):
    """Answers from a script, by question name: a float for a yes/no question
    (the probability of yes), an option for a choice, a level for a score, or an
    :data:`~roomkit.classifiers.base.Answer`. A question the script does not name
    gets no, the first option, or the lowest level. Given a list of scripts, the
    calls take them in order and the last one repeats. Records every call.
    """

    def __init__(
        self,
        answers: Script | Sequence[Script] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        scripts = [answers or {}] if answers is None or isinstance(answers, Mapping) else answers
        self._scripts: list[Script] = list(scripts) or [{}]
        self._error = error
        self.calls: list[tuple[State, dict[str, Question]]] = []

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        self.calls.append((state, dict(questions)))
        if self._error is not None:
            raise self._error
        script = self._scripts[min(len(self.calls), len(self._scripts)) - 1]
        return Answers({name: _answer(script, name, q) for name, q in questions.items()})


def _answer(script: Script, name: str, question: Question) -> Answer:
    scripted = script.get(name)
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
