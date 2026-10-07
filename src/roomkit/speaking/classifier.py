"""A speak policy on a classifier: narrow judgments composed in code (RFC §6.4).

The policy asks a :class:`~roomkit.classifiers.base.Classifier` its questions
about the turn in one call (was the agent addressed, did the speaker finish, are
they asking it to keep quiet...) and composes the answers in
:func:`compose`, readable code where every judgment stays visible: each one is
reported with the decision. The questions were measured on French
conversations with Jev (TypeSafe's System One model); their wording is English,
as the classifier reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from roomkit.classifiers.base import (
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.speaking.base import SpeakDecision, SpeakMode, SpeakPolicy, SpeakTurn

DIRECTNESS = ScoreQuestion(
    "How directly does `last_turn` bring in the assistant named in `agent.name`? When "
    "`people` lists a single person, that person is talking with the assistant: a greeting, "
    "a 'tu' or 'you', or a question that names no one else is direct. When `people` lists "
    "several people, a 'tu' or 'you' said to one of them does not bring in the assistant.",
    (
        "Not mentioned: `last_turn` does not name or refer to the assistant.",
        "Tentatively: the speaker wonders whether the assistant could help or might know.",
        "Indirectly: the speaker proposes that someone asks the assistant.",
        "Directly: the assistant is asked by name or as 'you', even with polite conditionals.",
    ),
)
DEFERRED = YesNoQuestion(
    "In `last_turn`, does the speaker postpone or decline asking the assistant or the "
    "question being discussed (for example 'later', 'after the meeting', 'we will check "
    "ourselves')?"
)
UNFINISHED = YesNoQuestion(
    "Does `last_turn` stop before the speaker has said what they want: a sentence cut off, "
    "or the speaker says they are still thinking or looking for something?"
)
HUSH = YesNoQuestion(
    "In `last_turn`, does the speaker ask the assistant not to answer, to stay quiet, or "
    "only to listen?"
)
QUIET_RULE = YesNoQuestion(
    "In `recent_turns`, did someone ask the assistant itself to stay quiet, not to answer, "
    "or only to listen, without lifting it since? Words said to someone else, or a "
    "description of what people are doing, do not count."
)
REQUEST = YesNoQuestion(
    "Does `last_turn` ask the assistant to answer a question or to do something now? A "
    "remark or a reproach about the assistant is not a request."
)
ANSWERED = YesNoQuestion(
    "Did the assistant ask a question in its last turn of `recent_turns`, and does "
    "`last_turn` answer it, even with 'no', 'never mind' or 'it was just a test'?"
)
"""An answer to the assistant's question neither names it nor asks it anything:
judged on ``last_turn`` alone, "Non, c'était un test" read directness 0.4 and
request 0.07."""

ANSWERS = YesNoQuestion(
    "Does one of `assistant_thought.want_to_say` answer a question asked, or a need "
    "stated, in `last_turn` itself?"
)
CORRECTS = YesNoQuestion(
    "Does one of `assistant_thought.want_to_say` correct, or warn about a problem with, "
    "something said or proposed in `last_turn` itself?"
)

QUESTIONS: Mapping[str, Question] = MappingProxyType(
    {
        "directness": DIRECTNESS,
        "deferred": DEFERRED,
        "unfinished": UNFINISHED,
        "hush": HUSH,
        "quiet_rule": QUIET_RULE,
        "request": REQUEST,
        "answered": ANSWERED,
    }
)
"""The questions :class:`ClassifierSpeakPolicy` asks, by the name :func:`compose`
reads."""

THOUGHT_QUESTIONS: Mapping[str, Question] = MappingProxyType(
    {"answers": ANSWERS, "corrects": CORRECTS}
)
"""Asked too when the agent has something to say (``SpeakTurn.thought``)."""

LANGUAGE_INSTRUCTIONS = (
    "In which language should the assistant answer the person who spoke in `last_turn`? "
    "It is the language that person speaks, judged on `last_turn` together with their "
    "turns in `recent_turns`. A short turn in another language, a word or an "
    "exclamation, maybe misheard by the speech recognizer, does not change it; a whole "
    "sentence in another language, or a request to switch, does."
)
"""Judged over the recent turns: alone, a « Quoi ? » transcribed « What? » switched
a French conversation to English."""

_OTHER_LANGUAGE = "other"


def compose(
    judgments: Mapping[str, float],
    *,
    alone: bool = False,
    proactivity: float = 0.5,
    urgent: bool = False,
) -> tuple[SpeakMode, str]:
    """The mode, and why, from the judgments by question name (missing ones read 0).

    ``directness`` is the expected level on 0 (not mentioned), 1 (tentatively),
    2 (indirectly), 3 (directly); the others are probabilities of yes. *alone*:
    one person talks with the agent, so a request is for it whatever the
    directness says. In order: the speaker not done, postponing or asking for
    quiet, or a standing request for quiet the turn does not lift, keeps the agent
    silent; an answer to its question, or being addressed, makes it speak; being
    only wondered about makes it offer, or speak when what it wants to say answers
    or corrects the turn (``answers``, ``corrects``). Not addressed, it offers when
    that reaches *proactivity* (lower is more eager), half of it when what it
    wants to say is *urgent*: urgency alone made an agent ask the same question
    four times over unrelated turns.
    """
    j = {name: judgments.get(name, 0.0) for name in (*QUESTIONS, *THOUGHT_QUESTIONS)}
    if j["unfinished"] >= 0.5:
        return "silent", "not finished"
    if j["deferred"] >= 0.5:
        return "silent", "postponed"
    if j["hush"] >= 0.5:
        return "silent", "asked to keep quiet"
    if j["quiet_rule"] >= 0.5 and j["request"] < 0.5:
        return "silent", "keeping quiet"
    if j["answered"] >= 0.5:
        return "speak", "answers its question"
    if addressed(j["directness"], j["request"], alone=alone):
        return "speak", "addressed"
    knows = max(j["answers"], j["corrects"])
    if j["directness"] >= 0.75:
        return ("speak", "knows the answer") if knows >= 0.5 else ("offer", "wondered about")
    if knows >= (proactivity / 2 if urgent else proactivity):
        return "offer", "urgent" if urgent else "has something to add"
    return "silent", "not addressed"


def addressed(directness: float, request: float, *, alone: bool = False) -> bool:
    """Whether the agent was addressed: directly or indirectly, asked for something
    tentatively, or asked anything when it is the person's only listener.

    0.75 rather than 0.5 opens the tentative band: an expected level near 0.5 is
    a split between "not mentioned" and "tentatively".
    """
    if request >= 0.5 and (alone or directness >= 0.75):
        return True
    return directness >= 1.5


def judgments_of(answers: Answers) -> dict[str, float]:
    """The answers as judgments: a yes/no's probability, a score's level, and a
    choice as ``name=choice`` with its probability."""
    judgments: dict[str, float] = {}
    for name, answer in answers.items():
        if isinstance(answer, YesNoAnswer):
            judgments[name] = answer.probability
        elif isinstance(answer, ScoreAnswer):
            judgments[name] = answer.score
        elif isinstance(answer, ChoiceAnswer):
            judgments[f"{name}={answer.choice}"] = answer.probabilities.get(answer.choice, 1.0)
    return judgments


class ClassifierSpeakPolicy(SpeakPolicy):
    """Narrow judgments about the turn in one classifier call, composed in code.

    The classifier stays the caller's: it may serve other components, and is
    closed by whoever made it.

    Args:
        classifier: Answers the questions; Jev's calibrated probabilities are
            what the thresholds were measured on.
        agent_name: The agent's name, as people call it.
        agent_role: What the agent is there for, in a few words, when it helps
            to judge whether a turn is for it.
        questions: Questions replacing :data:`QUESTIONS` by name, or added to
            them for a :meth:`compose` of your own.
        languages: The languages the agent answers in, each with the line its
            turn's notes carry when the speaker speaks it, best written in that
            language (``{"French": "Réponds en français uniquement."}``). Empty:
            the language is not judged.
        history: How many turns before the event the classifier reads.
        proactivity: How sure the policy must be that what the agent wants to
            say answers or corrects the turn before it offers unasked, in
            (0, 1]: lower is more eager. Only a thinker gives it something to say.
    """

    def __init__(
        self,
        classifier: Classifier,
        *,
        agent_name: str,
        agent_role: str = "",
        questions: Mapping[str, Question] | None = None,
        languages: Mapping[str, str] | None = None,
        history: int = 6,
        proactivity: float = 0.5,
    ) -> None:
        if not 0 < proactivity <= 1:
            raise ValueError("proactivity must be in (0, 1]")
        if languages and _OTHER_LANGUAGE in languages:
            raise ValueError(f"{_OTHER_LANGUAGE!r} is kept for a language not listed")
        self._classifier = classifier
        self._agent_name = agent_name
        self._agent_role = agent_role
        self._questions = {**QUESTIONS, **(questions or {})}
        self._languages = dict(languages or {})
        self._history = history
        self._proactivity = proactivity

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        if not _text(turn.event):
            return SpeakDecision("speak", reason="nothing to judge")
        answers = await self._classifier.classify(self.state(turn), self.questions(turn))
        return self.decision(turn, answers)

    def questions(self, turn: SpeakTurn) -> dict[str, Question]:
        """The questions asked on *turn*: :data:`QUESTIONS` as replaced, those of
        :data:`THOUGHT_QUESTIONS` when the agent has something to say, and the
        language when the policy has languages."""
        asked = dict(self._questions)
        if turn.thought is not None and turn.thought.want_to_say:
            asked |= THOUGHT_QUESTIONS
        if self._languages:
            options = {name: name for name in self._languages}
            options[_OTHER_LANGUAGE] = "Another language, or nothing to tell it from."
            asked["language"] = ChoiceQuestion(LANGUAGE_INSTRUCTIONS, options)
        return asked

    def state(self, turn: SpeakTurn) -> dict[str, Any]:
        """What the classifier reads: the agent, the people, the recent turns and
        the last one, each with its speaker, and the agent's thought when it has
        one."""
        agent: dict[str, str] = {"name": self._agent_name}
        if self._agent_role:
            agent["role"] = self._agent_role
        said = [e for e in turn.recent if _text(e)][-self._history :] if self._history else []
        state: dict[str, Any] = {
            "agent": agent,
            "people": list(turn.people),
            "recent_turns": [self._said(turn, e) for e in said],
            "last_turn": self._said(turn, turn.event),
        }
        if turn.thought is not None:
            state["assistant_thought"] = turn.thought.as_state()
        return state

    def decision(self, turn: SpeakTurn, answers: Answers) -> SpeakDecision:
        """The decision from the answers: :func:`compose`, and the language's note."""
        judgments = judgments_of(answers)
        thought = turn.thought
        mode, reason = compose(
            judgments,
            alone=len(turn.people) == 1,
            proactivity=self._proactivity,
            urgent=thought is not None and thought.urgent and bool(thought.want_to_say),
        )
        return SpeakDecision(mode, reason, judgments, self._language_notes(answers))

    def _language_notes(self, answers: Answers) -> tuple[str, ...]:
        answer = answers.get("language")
        if not isinstance(answer, ChoiceAnswer) or answer.choice not in self._languages:
            return ()
        return (self._languages[answer.choice],)

    def _said(self, turn: SpeakTurn, event: RoomEvent) -> dict[str, str]:
        if turn.by_agent(event):
            speaker = self._agent_name
        else:
            speaker = turn.speakers.get(event.id, "someone")
        return {"speaker": speaker, "text": _text(event)}


def _text(event: RoomEvent) -> str:
    return event.content.body if isinstance(event.content, TextContent) else ""
