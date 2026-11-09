"""One rendering of a conversation as Chat Completions messages (RFC §6.7,
RMK-309).

OpenAI, its derivatives, Mistral and PolarGrid render through
``chat_messages``; what they do differently is their ``ChatDialect``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.chat_request import OPENAI_CHAT, ChatDialect, chat_messages
from roomkit.providers.cerebras.ai import CerebrasAIProvider
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.ai import DEEPSEEK_CHAT
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.kimi.ai import KIMI_CHAT
from roomkit.providers.kimi.config import KimiConfig
from roomkit.providers.minimax.ai import MINIMAX_CHAT
from roomkit.providers.minimax.config import MiniMaxConfig
from roomkit.providers.mistral.ai import MISTRAL_CHAT, MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.polargrid.ai import POLARGRID_CHAT

_IMAGE = "https://example.com/a.png"
_THOUGHT = AIThinkingPart(thinking="why")


def _round(*parts: object) -> AIMessage:
    return AIMessage(
        role="assistant",
        content=[*parts, AIToolCallPart(id="c1", name="lookup", arguments={"q": "x"})],
    )


def _results(*results: AIToolResultPart) -> AIMessage:
    return AIMessage(role="tool", content=list(results))


def _render(dialect: ChatDialect, *messages: AIMessage, system: str | None = None) -> list:
    return chat_messages(list(messages), system, dialect, provider="acme")


def test_the_system_prompt_comes_first() -> None:
    rendered = _render(OPENAI_CHAT, AIMessage(role="user", content="hi"), system="Be brief.")
    assert rendered == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]


def _builder(cls: type, config: Any) -> Any:
    provider = cls.__new__(cls)
    provider._config = config
    return provider


@pytest.mark.parametrize(
    "provider",
    [
        _builder(OpenAIAIProvider, OpenAIConfig(api_key="k", model="gpt-4.1")),
        _builder(MistralAIProvider, MistralConfig(api_key="k")),
    ],
    ids=["openai", "mistral"],
)
def test_a_provider_without_a_reasoning_field_sends_none(provider: Any) -> None:
    round_ = _round(AIThinkingPart(thinking="A-1 first."), AITextPart(text="Let me look."))

    (sent,) = provider._build_messages([round_])

    assert sent["content"] == "Let me look."
    assert [c["id"] for c in sent["tool_calls"]] == ["c1"]


def test_a_tool_round_sends_its_text_and_its_calls_and_no_reasoning() -> None:
    [rendered] = _render(OPENAI_CHAT, _round(AITextPart(text="Looking."), _THOUGHT))
    assert rendered == {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            }
        ],
    }


def test_a_replayed_call_s_arguments_on_the_chat_wire() -> None:
    """A replayed call's arguments as the chat wire writes them: json.dumps'
    default form, a space after each separator and every character past
    ASCII escaped, the keys in their order. roomkit-rs writes the same,
    under a test of the same name (HITL 9f5e61ba, RMK-1508)."""
    call = AIToolCallPart(id="c1", name="lookup", arguments={"q": "café", "n": 42})
    [rendered] = _render(OPENAI_CHAT, AIMessage(role="assistant", content=[call]))
    assert rendered["tool_calls"][0]["function"]["arguments"] == '{"q": "caf\\u00e9", "n": 42}'


def test_a_round_of_reasoning_and_calls_alone_has_no_content() -> None:
    [rendered] = _render(OPENAI_CHAT, _round(_THOUGHT))
    assert rendered["content"] is None


def test_a_plain_answer_sends_no_reasoning() -> None:
    message = AIMessage(role="assistant", content=[_THOUGHT, AITextPart(text="Done.")])
    [rendered] = _render(OPENAI_CHAT, message)
    assert rendered["content"] == [{"type": "text", "text": "Done."}]


def test_a_message_s_text_and_images_keep_their_order() -> None:
    message = AIMessage(
        role="user",
        content=[AITextPart(text="mine"), AIImagePart(url=_IMAGE), AITextPart(text="yours")],
    )
    [rendered] = _render(OPENAI_CHAT, message)
    assert rendered["content"] == [
        {"type": "text", "text": "mine"},
        {"type": "image_url", "image_url": {"url": _IMAGE}},
        {"type": "text", "text": "yours"},
    ]


def test_an_answer_of_reasoning_alone_is_an_empty_text() -> None:
    [rendered] = _render(OPENAI_CHAT, AIMessage(role="assistant", content=[_THOUGHT]))
    assert rendered == {"role": "assistant", "content": ""}


@pytest.mark.parametrize(
    ("dialect", "field"),
    [
        (CerebrasAIProvider._chat_dialect, "reasoning"),
        (DEEPSEEK_CHAT, "reasoning_content"),
        (MINIMAX_CHAT, "reasoning_content"),
        (KIMI_CHAT, "reasoning_content"),
    ],
    ids=["cerebras", "deepseek", "minimax", "kimi"],
)
def test_a_dialect_with_a_reasoning_field_carries_it_there(
    dialect: ChatDialect, field: str
) -> None:
    [round_] = _render(dialect, _round(_THOUGHT, AITextPart(text="Looking.")))
    [answer] = _render(dialect, AIMessage(role="assistant", content=[_THOUGHT]))
    assert (round_["content"], round_[field]) == ("Looking.", "why")
    assert answer == {"role": "assistant", "content": "", field: "why"}


def test_a_placeholder_needs_a_required_field() -> None:
    with pytest.raises(ValueError, match="round_thinking_required"):
        ChatDialect(thinking_field="reasoning_content", round_thinking_placeholder=" ")


def test_an_image_result_rides_a_user_message_after_the_tool_messages() -> None:
    rendered = _render(
        MISTRAL_CHAT,
        _results(
            AIToolResultPart(
                tool_call_id="c1",
                name="shot",
                result=[AITextPart(text="here"), AIImagePart(url=_IMAGE)],
            ),
            AIToolResultPart(tool_call_id="c2", name="lookup", result="found"),
        ),
    )
    assert rendered == [
        {"role": "tool", "tool_call_id": "c1", "name": "shot", "content": "here"},
        {"role": "tool", "tool_call_id": "c2", "name": "lookup", "content": "found"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _IMAGE}}]},
    ]


def test_openai_tool_messages_do_not_name_their_tool() -> None:
    [rendered] = _render(
        OPENAI_CHAT, _results(AIToolResultPart(tool_call_id="c1", name="x", result="r"))
    )
    assert "name" not in rendered


def test_polargrid_sends_text_flat_drops_reasoning_and_skips_empty_messages() -> None:
    rendered = _render(
        POLARGRID_CHAT,
        AIMessage(role="user", content=""),
        AIMessage(role="assistant", content=[_THOUGHT]),
        AIMessage(role="assistant", content=[_THOUGHT, AITextPart(text="Hi.")]),
        _round(_THOUGHT, AITextPart(text="Looking.")),
        AIMessage(role="user", content=[AITextPart(text="see"), AIImagePart(url=_IMAGE)]),
    )
    assert [(m["role"], m["content"]) for m in rendered] == [
        ("assistant", "Hi."),
        ("assistant", "Looking."),
        (
            "user",
            [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": _IMAGE}},
            ],
        ),
    ]


def test_a_reasoning_field_is_the_assistants_own() -> None:
    message = AIMessage(role="user", content=[_THOUGHT, AITextPart(text="u")])
    [rendered] = _render(CerebrasAIProvider._chat_dialect, message)
    assert "reasoning" not in rendered
    assert rendered["content"] == [{"type": "text", "text": "u"}]


def test_an_empty_part_list_stays_a_list() -> None:
    [rendered] = _render(OPENAI_CHAT, AIMessage(role="user", content=[]))
    assert rendered["content"] == []


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_a_message_holding_a_call_and_its_result_sends_the_call_alone(role: str) -> None:
    """A message whose parts hold both a call and a result renders as the
    assistant's round that called, its result left out (RMK-1408)."""
    held = _round(AIToolResultPart(tool_call_id="c1", name="lookup", result="found"))
    rendered = _render(OPENAI_CHAT, held.model_copy(update={"role": role}))
    assert rendered == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    ("role", "field"), [("assistant", {"reasoning_content": ""}), ("tool", {})]
)
def test_a_call_round_s_required_reasoning_field_is_the_assistants_alone(
    role: str, field: dict[str, str]
) -> None:
    """A dialect that requires a round's reasoning field writes it on the
    assistant's own round only: a message of another role holding a call is
    rendered as the assistant's round, without it (RMK-1408)."""
    held = _round(AIToolResultPart(tool_call_id="c1", name="lookup", result="found"))
    [rendered] = _render(DEEPSEEK_CHAT, held.model_copy(update={"role": role}))
    assert {k: v for k, v in rendered.items() if k == "reasoning_content"} == field


def _answer(captured: list[dict[str, Any]]) -> httpx.MockTransport:
    """A server keeping each request's body, answering every one in a word."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


async def _sent_round(**config: Any) -> dict[str, Any]:
    """The round as a server receives it, through the provider's public API."""
    captured: list[dict[str, Any]] = []
    settings = OpenAIConfig(
        api_key="k", base_url="https://models.example/v1", model="glm-5.3-flash", **config
    )
    provider = OpenAIAIProvider(settings, transport=_answer(captured))
    await provider.generate(AIContext(messages=[_round(_THOUGHT, AITextPart(text="Look."))]))
    return captured[0]["messages"][0]


async def test_a_configured_thinking_field_carries_a_round_s_reasoning_to_the_server() -> None:
    sent = await _sent_round(thinking_field="reasoning_content")

    assert sent["reasoning_content"] == "why"
    assert sent["content"] == "Look."


async def test_without_a_thinking_field_the_server_gets_no_reasoning() -> None:
    assert "reasoning_content" not in await _sent_round()


async def test_a_configured_thinking_field_leaves_a_round_that_did_not_reason_without_it() -> None:
    captured: list[dict[str, Any]] = []
    settings = OpenAIConfig(api_key="k", model="glm", thinking_field="reasoning_content")
    provider = OpenAIAIProvider(settings, transport=_answer(captured))

    await provider.generate(AIContext(messages=[_round(AITextPart(text="Look."))]))

    assert "reasoning_content" not in captured[0]["messages"][0]


@pytest.mark.parametrize("config", [DeepSeekConfig, MiniMaxConfig, KimiConfig, CerebrasConfig])
def test_a_config_whose_vendor_names_the_field_refuses_another(config: type[OpenAIConfig]) -> None:
    with pytest.raises(ValidationError, match="thinking_field"):
        config(api_key="k", model="m", thinking_field="reasoning_content")
    assert config(api_key="k", model="m").thinking_field is None
