"""GeminiClassifier (RFC §6.8): RoomKit's questions sent as the Decisions API's,
its answers read back, its failures as ClassifierError.

The wire is faked under google-genai, so the SDK's own request building and
response parsing run. Bodies are shaped as Google's Decisions API reference
shows them (aistudio.google.com/docs/decisions-api, read 2026-11-12):
``POST /v1beta/decisions``, ``gemini-decision-1``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any

import httpx
import pytest

genai = pytest.importorskip("google.genai")

from roomkit import (  # noqa: E402
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierError,
    GeminiClassifier,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)

QUESTIONS = {
    "addressed": YesNoQuestion("Is the last message addressed to Nova?"),
    "done": YesNoQuestion("Did they finish?", yes="a complete thought", no="cut off"),
    "language": ChoiceQuestion("Which language?", {"fr": "French", "en": "English"}),
    "urgency": ScoreQuestion("How urgent?", ("can wait", "soon", "now")),
}

ANSWERS: dict[str, Any] = {
    "addressed": {"type": "noul", "noul": 0.94, "confidence": 0.94},
    "done": {"type": "noul", "noul": 0.03, "confidence": 0.97},
    "language": {
        "type": "choice",
        "choice": "fr",
        "probabilities": {"fr": 0.96, "en": 0.04},
        "confidence": 0.96,
    },
    "urgency": {
        "type": "score",
        "score": 1.7,
        "legend": {"0": "can wait", "1": "soon", "2": "now"},
        "probabilities": {"1": 0.3, "2": 0.7},
        "confidence": 0.7,
    },
}

USAGE = {
    "total_input_tokens": 324,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 324}],
    "total_output_tokens": 0,
    "total_tokens": 324,
}

Handler = Callable[[httpx.Request], Any]


def answering(
    answers: dict[str, Any] | None = None,
    *,
    status: int = 200,
    body: Any = None,
    seen: list[httpx.Request] | None = None,
) -> Handler:
    """A Decisions API that records each request and answers *body*, or *answers*."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        payload = body if body is not None else {"answers": answers or ANSWERS, "usage": USAGE}
        return httpx.Response(status, json=payload)

    return handler


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    """Route the httpx client RoomKit builds for the SDK to a handler."""

    def install(handler: Handler) -> None:
        real = httpx.AsyncClient

        def through_fake(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", through_fake)

    return install


async def classify(state: Any, questions: Any = QUESTIONS, **kwargs: Any) -> Any:
    classifier = GeminiClassifier("k", **kwargs)
    try:
        return await classifier.classify(state, questions)
    finally:
        await classifier.close()


async def test_questions_go_out_as_the_decisions_api_takes_them(wire: Any) -> None:
    seen: list[httpx.Request] = []
    wire(answering(seen=seen))
    await classify({"last": "Nova?"})

    (request,) = seen
    assert request.url.path == "/v1beta/decisions"
    assert request.headers["x-goog-api-key"] == "k"
    body = json.loads(request.content)
    assert body["model"] == "gemini-decision-1"
    # A structured state goes as its JSON: the API reads it as a text.
    assert body["input"] == '{"last": "Nova?"}'
    assert body["questions"] == {
        "addressed": {"type": "noul", "instructions": "Is the last message addressed to Nova?"},
        "done": {
            "type": "noul",
            "instructions": "Did they finish?",
            "criteria": {"true_description": "a complete thought", "false_description": "cut off"},
        },
        "language": {
            "type": "choice",
            "instructions": "Which language?",
            "criteria": {"fr": "French", "en": "English"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["can wait", "soon", "now"],
        },
    }


async def test_a_text_state_goes_as_it_is(wire: Any) -> None:
    seen: list[httpx.Request] = []
    wire(answering(seen=seen))
    await classify("Nova, are you there?")
    assert json.loads(seen[0].content)["input"] == "Nova, are you there?"


async def test_answers_come_back_with_their_probabilities(wire: Any) -> None:
    wire(answering())
    answers = await classify("x")
    assert answers["addressed"] == YesNoAnswer(0.94)
    assert answers.yes("done") == 0.03
    assert answers["language"] == ChoiceAnswer("fr", {"fr": 0.96, "en": 0.04})
    # A level the API left out has probability 0.
    assert answers["urgency"] == ScoreAnswer(1.7, (0.0, 0.3, 0.7))


async def test_a_lent_client_is_asked_on_v1beta_and_left_open(wire: Any) -> None:
    seen: list[httpx.Request] = []
    wire(answering(seen=seen))
    http = httpx.AsyncClient()
    # Built for the SDK's default version, v1, where the API answers 404.
    client = genai.Client(api_key="k", http_options={"httpx_async_client": http})
    classifier = GeminiClassifier(client=client)
    answers = await classifier.classify("x", QUESTIONS)
    await classifier.close()

    assert answers.choice("language") == "fr"
    assert seen[0].url.path == "/v1beta/decisions"
    assert not http.is_closed
    await http.aclose()


@pytest.mark.parametrize(
    ("answers", "error"),
    [
        ({k: v for k, v in ANSWERS.items() if k != "urgency"}, "no ScoreAnswer for 'urgency'"),
        ({**ANSWERS, "urgency": ANSWERS["addressed"]}, "no ScoreAnswer for 'urgency'"),
        (
            {**ANSWERS, "language": {**ANSWERS["language"], "choice": "de"}},
            "no ChoiceAnswer for 'language'",
        ),
        # The SDK hands back a body that fails its validation unvalidated.
        ({**ANSWERS, "addressed": {"type": "noul", "noul": None}}, "cannot be read"),
        ({**ANSWERS, "urgency": {**ANSWERS["urgency"], "score": None}}, "cannot be read"),
    ],
    ids=["missing", "wrong-type", "unasked-option", "null-noul", "null-score"],
)
async def test_an_answer_short_of_the_question_fails_the_call(
    wire: Any, answers: dict[str, Any], error: str
) -> None:
    wire(answering(answers))
    with pytest.raises(ClassifierError, match=error):
        await classify("x")


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (
            200,
            {"prompt_feedback": {"block_reason": "safety", "block_reason_message": "harm"}},
            "blocked the input: safety",
        ),
        (
            400,
            {"error": {"code": "safety", "message": "Request blocked due to safety violations"}},
            "safety violations",
        ),
        (
            400,
            {"error": {"code": 400, "message": "question key 'a b' must match [A-Za-z0-9_-]+"}},
            "must match",
        ),
    ],
    ids=["blocked-input", "safety-400", "limit-400"],
)
async def test_a_refused_call_is_a_classifier_error(
    wire: Any, status: int, body: dict[str, Any], error: str
) -> None:
    wire(answering(status=status, body=body))
    with pytest.raises(ClassifierError, match=error):
        await classify("x")


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, "HTTP 429: The model is overloaded$"), (503, "HTTP 503: The model is overloaded$")],
    ids=["rate-limited", "overloaded"],
)
async def test_a_server_error_fails_at_once_with_its_status(
    wire: Any, status: int, error: str
) -> None:
    # Retried in the wait, it read "no answer within 1.0 s" and hid the status.
    seen: list[httpx.Request] = []
    body = {"error": {"code": status, "message": "The model is overloaded"}}
    wire(answering(status=status, body=body, seen=seen))
    with pytest.raises(ClassifierError, match=error):
        await classify("x", timeout=1.0)
    assert len(seen) == 1


async def test_a_connection_failure_is_a_classifier_error(wire: Any) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    wire(refuse)
    with pytest.raises(ClassifierError, match="connection"):
        await classify("x", timeout=1.0)


async def test_the_wait_is_bounded(wire: Any) -> None:
    async def never(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    wire(never)
    with pytest.raises(ClassifierError, match="within 0.05 s"):
        await classify("x", timeout=0.05)


def test_it_needs_a_key_or_a_client() -> None:
    with pytest.raises(ValueError, match="api_key, or a client"):
        GeminiClassifier()


def test_without_the_sdk_it_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai.decisions", None)
    with pytest.raises(ImportError, match=r"roomkit\[gemini\]"):
        GeminiClassifier("k")
