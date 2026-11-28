"""A scripted classifier over the public mock, and its calls as a scenario reads them.

README, "Scripted agents" (`speak_policy.classifier`) and "Setup" (a discussion's
classifier `dispatch`): each call takes the next of the script's `calls`, the last
repeating, records the state and the questions it was handed, holds at its gate,
then fails with a classifier error or answers as the implementation's
``MockClassifier`` answers that call's script.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.classifiers import (
    Answers,
    ChoiceQuestion,
    Classifier,
    ClassifierError,
    MockClassifier,
    Question,
    ScoreQuestion,
    State,
    YesNoQuestion,
)

if TYPE_CHECKING:
    from tests.conformance._script import Gates

_CALL_MEMBERS = frozenset({"answers", "fail", "gate"})
_QUESTION_MEMBERS = frozenset({"instructions", "yes", "no", "options", "levels"})


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _call_problems(entry: Any, path: str) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"{path}: expected an object"]
    found = [f"{path}: unknown member {key}" for key in entry if key not in _CALL_MEMBERS]
    if ("answers" in entry) == ("fail" in entry):
        found.append(f"{path}: exactly one of answers and fail is required")
    answers = entry.get("answers", {})
    if not isinstance(answers, Mapping) or any(
        not (_number(value) or isinstance(value, str)) for value in answers.values()
    ):
        found.append(f"{path}.answers must map names to numbers or option names")
    for key in ("fail", "gate"):
        if key in entry and not isinstance(entry[key], str):
            found.append(f"{path}.{key} must be a string")
    return found


def classifier_problems(value: Any) -> list[str]:
    """What is wrong with a scripted classifier's controls."""
    if not isinstance(value, Mapping):
        return ["classifier must be an object"]
    found = [f"classifier: unknown member {key}" for key in value if key != "calls"]
    calls = value.get("calls")
    if not isinstance(calls, list) or not calls:
        return [*found, "classifier.calls must be a nonempty array"]
    for index, entry in enumerate(calls):
        found.extend(_call_problems(entry, f"classifier.calls[{index}]"))
    return found


def question_problems(value: Any, path: str) -> list[str]:
    """What is wrong with a question a scenario writes."""
    if not isinstance(value, Mapping):
        return [f"{path}: expected an object"]
    found = [f"{path}: unknown member {key}" for key in value if key not in _QUESTION_MEMBERS]
    if not isinstance(value.get("instructions"), str):
        found.append(f"{path}.instructions must be a string")
    kinds = [key for key in ("options", "levels") if key in value]
    if len(kinds) > 1 or (kinds and ("yes" in value or "no" in value)):
        found.append(f"{path}: a yes/no question, a choice or a score, not several")
    options = value.get("options", {"a": "", "b": ""})
    if (
        not isinstance(options, Mapping)
        or len(options) < 2
        or any(not isinstance(v, str) for v in options.values())
    ):
        found.append(f"{path}.options must map two names at least to strings")
    levels = value.get("levels", ["", ""])
    if (
        not isinstance(levels, list)
        or len(levels) < 2
        or any(not isinstance(v, str) for v in levels)
    ):
        found.append(f"{path}.levels must be an array of two strings at least")
    for key in ("yes", "no"):
        if value.get(key) is not None and not isinstance(value[key], str):
            found.append(f"{path}.{key} must be a string or null")
    return found


def question(spec: Mapping[str, Any]) -> Question:
    """The implementation's question *spec* writes."""
    if "options" in spec:
        return ChoiceQuestion(spec["instructions"], dict(spec["options"]))
    if "levels" in spec:
        return ScoreQuestion(spec["instructions"], tuple(spec["levels"]))
    return YesNoQuestion(spec["instructions"], spec.get("yes"), spec.get("no"))


def question_record(asked: Question) -> dict[str, Any]:
    """A question as ``expect.classifier_calls`` reads it."""
    record = dataclasses.asdict(asked)
    if isinstance(asked, ChoiceQuestion):
        record["options"] = dict(asked.options)
    if isinstance(asked, ScoreQuestion):
        record["levels"] = list(asked.levels)
    return record


@dataclass
class ClassifierRecords:
    """Where scripted classifiers record their calls, by the name
    ``expect.classifier_calls`` reads them under, and their scripts' problems."""

    calls: dict[str, list[Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def answer_problems(script: Mapping[str, Any], questions: Mapping[str, Question]) -> list[str]:
    """What in a call's *script* does not fit the *questions* it answers: a
    question not asked, a choice that is no option of it, a number for a
    choice or a name for a yes/no question or a score. The mock would fail
    the call, and the policy's fallback would hide the scenario's mistake."""
    found = []
    for name, value in script.items():
        asked = questions.get(name)
        if asked is None:
            found.append(f"answers.{name}: no question of that name was asked")
        elif isinstance(asked, ChoiceQuestion):
            if not isinstance(value, str) or value not in asked.options:
                found.append(f"answers.{name}: not an option of the choice asked")
        elif not _number(value):
            found.append(f"answers.{name}: the question asked takes a number")
    return found


class ScriptedClassifier(Classifier):
    """Answers each call as the public mock answers that call's script."""

    def __init__(
        self, spec: Mapping[str, Any], gates: Gates, calls: list[Any], problems: list[str]
    ) -> None:
        if problems_found := classifier_problems(spec):
            raise ValueError("; ".join(problems_found))
        self._entries = spec["calls"]
        self._gates, self._calls, self._problems = gates, calls, problems

    async def classify(self, state: State, questions: Mapping[str, Question]) -> Answers:
        self._calls.append(
            {
                "state": state,
                "questions": {name: question_record(q) for name, q in questions.items()},
            }
        )
        entry = self._entries[min(len(self._calls), len(self._entries)) - 1]
        if "gate" in entry:
            await self._gates.hold(entry["gate"])
        if "fail" in entry:
            raise ClassifierError(entry["fail"])
        if found := answer_problems(entry["answers"], questions):
            at = f"classifier call {len(self._calls)}"
            self._problems.extend(f"{at}: {problem}" for problem in found)
            raise ClassifierError("the scripted answers do not fit the questions asked")
        return await MockClassifier(dict(entry["answers"])).classify(state, questions)
