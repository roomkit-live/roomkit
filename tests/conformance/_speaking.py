"""A scripted public SpeakPolicy, the implementation's classifier and answer-only
policies, and observations taken when they are called."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.classifiers import Answers, Classifier
from roomkit.speaking import (
    AnswerOnly,
    ClassifierSpeakPolicy,
    CutReply,
    SpeakDecision,
    SpeakDecisionEvent,
    SpeakPolicy,
    SpeakTurn,
)
from tests.conformance._classifier import (
    ScriptedClassifier,
    classifier_problems,
    question,
    question_problems,
)
from tests.conformance._thinking import thought_value

if TYPE_CHECKING:
    from tests.conformance._script import Gates


class SpeakPolicyFailedError(Exception):
    """A configured policy error, distinct from malformed scenario controls."""


def decision_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["decision must be an object"]
    found = [
        f"unknown decision member {key}"
        for key in value
        if key
        not in {
            "mode",
            "reason",
            "judgments",
            "notes",
            "final",
        }
    ]
    if value.get("mode") not in ("speak", "offer", "silent"):
        found.append("decision.mode must be speak, offer or silent")
    if "reason" in value and not isinstance(value["reason"], str):
        found.append("decision.reason must be a string")
    if "final" in value and not isinstance(value["final"], bool):
        found.append("decision.final must be a boolean")
    if value.get("final") is True and value.get("mode") != "silent":
        found.append("only a silent decision can be final")
    notes = value.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        found.append("decision.notes must be an array of strings")
    judgments = value.get("judgments", {})
    if not isinstance(judgments, Mapping) or any(
        not isinstance(number, (int, float)) or isinstance(number, bool)
        for number in judgments.values()
    ):
        found.append("decision.judgments must be a map of numbers")
    return found


_CLASSIFIER_POLICY = frozenset(
    {
        "classifier",
        "agent_name",
        "agent_role",
        "questions",
        "languages",
        "history",
        "proactivity",
        "composition",
    }
)
"""The members of the implementation's classifier policy (README, ``speak_policy``)."""


def _answer_only_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != {"people"}:
        return ["answer_only must be an object of people alone"]
    people = value["people"]
    if not isinstance(people, list) or not people or any(not isinstance(p, str) for p in people):
        return ["answer_only.people must be a nonempty array of strings"]
    return []


def _classifier_policy_problems(value: Mapping[str, Any]) -> list[str]:
    found = classifier_problems(value["classifier"])
    if "decisions" in value:
        found.append("decisions and classifier are two policies: give one")
    for key in ("agent_name", "agent_role"):
        if key in value and not isinstance(value[key], str):
            found.append(f"{key} must be a string")
    if "agent_name" not in value:
        found.append("a classifier policy needs agent_name")
    questions = value.get("questions", {})
    if not isinstance(questions, Mapping):
        found.append("questions must map names to questions")
    else:
        for name, asked in questions.items():
            found.extend(question_problems(asked, f"questions.{name}"))
    languages = value.get("languages", {})
    if not isinstance(languages, Mapping) or any(
        not isinstance(v, str) for v in languages.values()
    ):
        found.append("languages must map names to strings")
    if "history" in value and (
        not isinstance(value["history"], int) or isinstance(value["history"], bool)
    ):
        found.append("history must be an integer")
    proactivity = value.get("proactivity", 0.5)
    if not isinstance(proactivity, (int, float)) or isinstance(proactivity, bool):
        found.append("proactivity must be a number")
    if "composition" in value:
        composition = value["composition"]
        if not isinstance(composition, Mapping) or set(composition) != {"decision"}:
            found.append("composition must be an object of decision alone")
        else:
            found.extend(decision_problems(composition["decision"]))
    return found


def policy_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["expected an object"]
    allowed = {"decisions", "answer_only", *_CLASSIFIER_POLICY}
    found = [f"unknown member {key}" for key in value if key not in allowed]
    if "answer_only" in value:
        found.extend(_answer_only_problems(value["answer_only"]))
    if "classifier" in value:
        return [*found, *_classifier_policy_problems(value)]
    found.extend(f"{key} needs classifier" for key in value if key in _CLASSIFIER_POLICY)
    entries = value.get("decisions")
    if not isinstance(entries, list) or not entries:
        return [*found, "decisions must be a nonempty array"]
    for index, entry in enumerate(entries):
        path = f"decisions[{index}]"
        if not isinstance(entry, Mapping):
            found.append(f"{path}: expected an object")
            continue
        found.extend(
            f"{path}: unknown member {key}"
            for key in entry
            if key
            not in {
                "decision",
                "fail",
                "gate",
            }
        )
        if ("decision" in entry) == ("fail" in entry):
            found.append(f"{path}: exactly one of decision and fail is required")
        if "decision" in entry:
            found.extend(f"{path}: {problem}" for problem in decision_problems(entry["decision"]))
        if "fail" in entry and not isinstance(entry["fail"], str):
            found.append(f"{path}.fail must be a string")
        if "gate" in entry and not isinstance(entry["gate"], str):
            found.append(f"{path}.gate must be a string")
    return found


def channel_problems(channel: Mapping[str, Any], path: str) -> list[str]:
    found = []
    if "speak_policy" in channel:
        found.extend(f"{path}.speak_policy: {p}" for p in policy_problems(channel["speak_policy"]))
    if "speak_timeout_ms" in channel:
        value = channel["speak_timeout_ms"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            found.append(f"{path}.speak_timeout_ms must be a number")
    return found


def hook_problems(hook: Mapping[str, Any], path: str) -> list[str]:
    if hook.get("trigger") != "on_speak_decision":
        return []
    found = []
    for index, reaction in enumerate(hook.get("reactions", [])):
        result = reaction.get("result", {})
        if result.get("action") != "modify":
            continue
        patch = result.get("event")
        at = f"{path}.reactions[{index}].result.event"
        if not isinstance(patch, Mapping) or set(patch) != {"decision"}:
            found.append(f"{at}: a speak decision patch must contain only decision")
        else:
            found.extend(f"{at}: {problem}" for problem in decision_problems(patch["decision"]))
    return found


def decision(value: Mapping[str, Any]) -> SpeakDecision:
    if problems := decision_problems(value):
        raise ValueError("speak_policy: " + "; ".join(problems))
    return SpeakDecision(**{**value, "notes": tuple(value.get("notes", ()))})


class ScriptedSpeakPolicy(SpeakPolicy):
    def __init__(self, spec: Mapping[str, Any], gates: Gates, calls: list[Any]) -> None:
        if problems := policy_problems(spec):
            raise ValueError("speak_policy: " + "; ".join(problems))
        self._entries = spec["decisions"]
        self._gates, self._calls = gates, calls

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        self._calls.append(turn_record(turn))
        entry = self._entries[min(len(self._calls), len(self._entries)) - 1]
        if "gate" in entry:
            await self._gates.hold(entry["gate"])
        if "fail" in entry:
            raise SpeakPolicyFailedError(entry["fail"])
        return decision(entry["decision"])


class _HostComposition(ClassifierSpeakPolicy):
    """The classifier policy with its host's composition (README,
    ``speak_policy.composition``): every turn it judged is decided so."""

    def __init__(self, classifier: Classifier, decided: SpeakDecision, **options: Any) -> None:
        super().__init__(classifier, **options)
        self._decided = decided

    def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
        return self._decided


def classifier_policy(spec: Mapping[str, Any], classifier: Classifier) -> ClassifierSpeakPolicy:
    """The implementation's classifier policy *spec* describes, over *classifier*."""
    options: dict[str, Any] = {
        key: spec[key]
        for key in ("agent_name", "agent_role", "history", "proactivity")
        if key in spec
    }
    if "questions" in spec:
        options["questions"] = {name: question(q) for name, q in spec["questions"].items()}
    if "languages" in spec:
        options["languages"] = dict(spec["languages"])
    if "composition" in spec:
        return _HostComposition(classifier, decision(spec["composition"]["decision"]), **options)
    return ClassifierSpeakPolicy(classifier, **options)


def _policy(
    spec: Mapping[str, Any], gates: Gates, calls: list[Any], classified: list[Any]
) -> SpeakPolicy:
    """The policy *spec* describes, answer-only around it when it says so."""
    if "classifier" in spec:
        policy: SpeakPolicy = classifier_policy(
            spec, ScriptedClassifier(spec["classifier"], gates, classified)
        )
    else:
        policy = ScriptedSpeakPolicy(spec, gates, calls)
    if "answer_only" in spec:
        return AnswerOnly(policy, spec["answer_only"]["people"])
    return policy


def speaking_options(
    channel: Mapping[str, Any],
    gates: Gates,
    observations: dict[str, list[Any]] | None,
    classifier_calls: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    if problems := channel_problems(channel, "channel"):
        raise ValueError("; ".join(problems))
    options: dict[str, Any] = {}
    calls = observations.setdefault(channel["id"], []) if observations is not None else None
    if "speak_policy" in channel:
        if calls is None or classifier_calls is None:
            raise AssertionError("speak_policy: no observation sink")
        classified = classifier_calls.setdefault(channel["id"], [])
        options["speak_policy"] = _policy(channel["speak_policy"], gates, calls, classified)
    if "speak_timeout_ms" in channel:
        options["speak_timeout"] = channel["speak_timeout_ms"] / 1000
    return options


def turn_record(turn: SpeakTurn) -> dict[str, Any]:
    return {
        "event": turn.event.model_dump(mode="json"),
        "recent": [event.model_dump(mode="json") for event in turn.recent],
        "people": list(turn.people),
        "channel_id": turn.channel_id,
        "speakers": dict(turn.speakers),
        "thought": thought_value(turn.thought) if turn.thought is not None else None,
        "cut": cut_record(turn.cut) if turn.cut is not None else None,
    }


def cut_record(cut: CutReply) -> dict[str, Any]:
    """The turn's cut answer as ``expect.speak_turns`` reads it (README)."""
    return {"text": cut.text, "played_ms": cut.played_ms, "at": cut.at.isoformat()}


def decision_record(report: SpeakDecisionEvent) -> dict[str, Any]:
    return {
        "room_id": report.room_id,
        "channel_id": report.channel_id,
        "event": report.event.model_dump(mode="json"),
        "decision": {**dataclasses.asdict(report.decision), "notes": list(report.decision.notes)},
        "duration_ms": report.duration_ms,
        "asked_again": report.asked_again,
        "late": report.late,
        "thought": thought_value(report.thought) if report.thought is not None else None,
    }


def patched_decision(report: SpeakDecisionEvent, patch: Mapping[str, Any]) -> SpeakDecisionEvent:
    if not isinstance(patch, Mapping) or set(patch) != {"decision"}:
        raise ValueError("speak decision patch must contain only decision")
    return dataclasses.replace(report, decision=decision(patch["decision"]))
