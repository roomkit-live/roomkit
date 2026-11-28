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
    if not isinstance(options, Mapping) or any(not isinstance(v, str) for v in options.values()):
        found.append(f"{path}.options must map names to strings")
    levels = value.get("levels", [])
    if not isinstance(levels, list) or any(not isinstance(v, str) for v in levels):
        found.append(f"{path}.levels must be an array of strings")
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


class ScriptedClassifier(Classifier):
    """Answers each call as the public mock answers that call's script."""

    def __init__(self, spec: Mapping[str, Any], gates: Gates, calls: list[Any]) -> None:
        if problems := classifier_problems(spec):
            raise ValueError("; ".join(problems))
        self._entries = spec["calls"]
        self._gates, self._calls = gates, calls

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
        return await MockClassifier(dict(entry["answers"])).classify(state, questions)
