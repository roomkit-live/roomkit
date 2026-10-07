"""Classifiers (RFC §6.8): the typed questions and answers, the mock, and the
classifier on any AI provider."""

from __future__ import annotations

import asyncio
import json

import pytest

from roomkit import (
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    LLMClassifier,
    MockClassifier,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
from roomkit.classifiers.base import Answers, check_answers
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider

QUESTIONS = {
    "addressed": YesNoQuestion("Is the last message addressed to Nova?"),
    "language": ChoiceQuestion(
        "Which language does the person speak?", {"fr": "French", "en": "English"}
    ),
    "urgency": ScoreQuestion("How urgent is it?", ("can wait", "soon", "now")),
}


class SlowProvider(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(1)
        return await super().generate(context)


# -- questions and answers ------------------------------------------------------


def test_a_choice_needs_two_options_and_a_score_two_levels() -> None:
    with pytest.raises(ValueError, match="two options"):
        ChoiceQuestion("Which?", {"only": "the one"})
    with pytest.raises(ValueError, match="two levels"):
        ScoreQuestion("How much?", ("all",))


def test_answers_read_typed() -> None:
    answers = Answers(
        addressed=YesNoAnswer(0.8),
        language=ChoiceAnswer("fr", {"fr": 0.9, "en": 0.1}),
        urgency=ScoreAnswer(1.4, (0.1, 0.4, 0.5)),
    )
    assert answers.yes("addressed") == 0.8
    assert answers.choice("language") == "fr"
    assert answers.score("urgency") == 1.4
    with pytest.raises(TypeError, match="'language' was answered as ChoiceAnswer"):
        answers.yes("language")
    with pytest.raises(KeyError):
        answers.yes("unknown")


def test_check_answers_wants_every_question_with_its_kind() -> None:
    complete = Answers(
        addressed=YesNoAnswer(1.0), language=ChoiceAnswer("en"), urgency=ScoreAnswer(0.0)
    )
    assert check_answers(QUESTIONS, complete) is complete
    with pytest.raises(ClassifierError, match="no ScoreAnswer for 'urgency'"):
        check_answers(QUESTIONS, Answers(addressed=YesNoAnswer(1.0), language=ChoiceAnswer("en")))
    with pytest.raises(ClassifierError, match="no YesNoAnswer for 'addressed'"):
        check_answers(QUESTIONS, Answers({**complete, "addressed": ScoreAnswer(1.0)}))


# -- MockClassifier -------------------------------------------------------------


async def test_mock_answers_from_its_script_and_records_calls() -> None:
    classifier = MockClassifier({"addressed": 0.9, "language": "en", "urgency": 2})
    answers = await classifier.classify("Nova, it's urgent", QUESTIONS)
    assert answers.yes("addressed") == 0.9
    assert answers.choice("language") == "en"
    assert answers["language"] == ChoiceAnswer("en", {"fr": 0.0, "en": 1.0})
    assert answers.score("urgency") == 2.0
    assert classifier.calls == [("Nova, it's urgent", QUESTIONS)]


async def test_mock_defaults_to_no_first_option_lowest_level() -> None:
    answers = await MockClassifier().classify({"text": "hello"}, QUESTIONS)
    assert (answers.yes("addressed"), answers.choice("language"), answers.score("urgency")) == (
        0.0,
        "fr",
        0.0,
    )


async def test_mock_takes_a_scripted_answer_and_raises_its_error() -> None:
    scripted = YesNoAnswer(0.42)
    answers = await MockClassifier({"addressed": scripted}).classify("x", QUESTIONS)
    assert answers["addressed"] is scripted
    with pytest.raises(ClassifierError, match="'de' is not an option"):
        await MockClassifier({"language": "de"}).classify("x", QUESTIONS)
    with pytest.raises(ClassifierError, match="down"):
        await MockClassifier(error=ClassifierError("down")).classify("x", QUESTIONS)


# -- LLMClassifier --------------------------------------------------------------


def _provider(*replies: str) -> MockAIProvider:
    return MockAIProvider(list(replies), response_schema=True)


async def test_llm_asks_under_a_schema_and_reads_the_answers() -> None:
    provider = _provider(json.dumps({"addressed": True, "language": "en", "urgency": "2"}))
    answers = await LLMClassifier(provider).classify({"last": "Nova, now!"}, QUESTIONS)

    assert answers["addressed"] == YesNoAnswer(1.0)
    assert answers["language"] == ChoiceAnswer("en", {"fr": 0.0, "en": 1.0})
    assert answers["urgency"] == ScoreAnswer(2.0, (0.0, 0.0, 1.0))
    context = provider.calls[0]
    assert context.response_schema == {
        "type": "object",
        "properties": {
            "addressed": {"type": "boolean"},
            "language": {"type": "string", "enum": ["fr", "en"]},
            "urgency": {"type": "string", "enum": ["0", "1", "2"]},
        },
        "required": ["addressed", "language", "urgency"],
        "additionalProperties": False,
    }
    prompt = context.messages[0].content
    assert isinstance(prompt, str)
    assert '{"last": "Nova, now!"}' in prompt
    assert "- language: Which language does the person speak?" in prompt
    assert "    en: English" in prompt
    assert "    2: now" in prompt


async def test_llm_describes_a_yes_and_a_no_when_given() -> None:
    provider = _provider(json.dumps({"done": False}))
    question = YesNoQuestion("Did they finish?", yes="a complete thought", no="cut mid-sentence")
    answers = await LLMClassifier(provider).classify("so I was", {"done": question})
    assert answers.yes("done") == 0.0
    prompt = provider.calls[0].messages[0].content
    assert isinstance(prompt, str)
    assert "    true: a complete thought\n    false: cut mid-sentence" in prompt


async def test_llm_turns_a_failed_call_into_a_classifier_error() -> None:
    refused = MockAIProvider(
        ai_responses=[AIResponse(content="", finish_reason="refusal")], response_schema=True
    )
    with pytest.raises(ClassifierError):
        await LLMClassifier(refused).classify("x", QUESTIONS)
    with pytest.raises(ClassifierError, match="within 0.05 s"):
        await LLMClassifier(SlowProvider(response_schema=True), timeout=0.05).classify(
            "x", QUESTIONS
        )


def test_llm_refuses_a_provider_without_response_schema() -> None:
    with pytest.raises(ValueError, match="does not support a response schema"):
        LLMClassifier(MockAIProvider())


async def test_llm_close_leaves_the_provider_to_its_owner() -> None:
    provider = _provider("{}")
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    provider.close = close  # type: ignore[method-assign]
    await LLMClassifier(provider).close()
    assert closed == []


async def test_mock_takes_one_script_per_call_the_last_repeating() -> None:
    classifier = MockClassifier([{"addressed": 0.9}, {"addressed": 0.1}])
    probabilities = [
        (await classifier.classify("x", QUESTIONS)).yes("addressed") for _ in range(3)
    ]
    assert probabilities == [0.9, 0.1, 0.1]
