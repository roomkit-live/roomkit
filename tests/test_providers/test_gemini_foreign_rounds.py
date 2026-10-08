"""A round another vendor made, replayed to Gemini (RMK-398, RFC §6.4).

Gemini 3 refuses, in the current turn, a function call without its thought
signature (measured 2026-10-03 on gemini-3.8-flash): a round of that turn
none of whose calls carries one goes back as text, its results with it and
their images kept. An earlier turn's round, which Gemini does not check, and
a model that does not sign its calls keep the structured form.
"""

from __future__ import annotations

import base64

import pytest
from google.genai import types

from roomkit.providers.ai.base import (
    AIImagePart,
    AIMessage,
    AITextPart,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.request import format_messages
from tests.text_conformance.scenario import PNG


def _round(call_id: str, *, signed: bool = False) -> list[AIMessage]:
    metadata = {"thought_signature": base64.b64encode(b"S0").decode()} if signed else {}
    return [
        AIMessage(
            role="assistant",
            content=[
                AIToolCallPart(id=call_id, name="lookup", arguments={"q": "a"}, metadata=metadata)
            ],
        ),
        AIMessage(
            role="tool",
            content=[AIToolResultPart(tool_call_id=call_id, name="lookup", result="found")],
        ),
    ]


def _calls(contents: list[types.Content]) -> list[str]:
    return [p.function_call.name for c in contents for p in c.parts or [] if p.function_call]


def _texts(contents: list[types.Content]) -> list[str]:
    return [p.text for c in contents for p in c.parts or [] if p.text]


def test_a_foreign_round_of_the_current_turn_goes_back_as_text() -> None:
    messages = [AIMessage(role="user", content="go"), *_round("c1")]

    contents = format_messages(types, messages, signed_calls=True)

    assert _calls(contents) == []
    assert 'I called lookup({"q": "a"}).' in _texts(contents)
    assert "lookup returned:\n<tool_result>\nfound\n</tool_result>" in _texts(contents)


def test_an_earlier_turns_unsigned_round_keeps_its_form() -> None:
    """Gemini checks signatures in the current turn only (measured)."""
    messages = [
        AIMessage(role="user", content="first"),
        *_round("c1"),
        AIMessage(role="assistant", content="done"),
        AIMessage(role="user", content="second"),
        *_round("c2", signed=True),
    ]

    contents = format_messages(types, messages, signed_calls=True)

    assert _calls(contents) == ["lookup", "lookup"]


def test_a_foreign_rounds_image_result_reaches_the_model() -> None:
    result = AIToolResultPart(
        tool_call_id="c1",
        name="screenshot",
        result=[AITextPart(text="done"), AIImagePart(url=PNG)],
    )
    messages = [
        AIMessage(role="user", content="go"),
        AIMessage(
            role="assistant",
            content=[AIToolCallPart(id="c1", name="screenshot", arguments={})],
        ),
        AIMessage(role="tool", content=[result]),
    ]

    contents = format_messages(types, messages, signed_calls=True)

    [image] = [p for c in contents for p in c.parts or [] if p.inline_data]
    assert image.inline_data.mime_type == "image/png"
    assert "screenshot returned:\n<tool_result>\ndone\n</tool_result>" in _texts(contents)


@pytest.mark.parametrize(
    ("model", "signed"),
    [("gemini-3.8-flash", True), ("gemini-2.5-flash", False), ("gemini-9-unknown", True)],
)
def test_the_models_that_sign_their_calls(model: str, signed: bool) -> None:
    """Gemini 3 takes thinking levels and signs its calls; a model the
    catalogue does not know is taken as recent."""
    provider = GeminiAIProvider(GeminiConfig(api_key="k", model=model))

    assert provider._calls_need_signatures() is signed
