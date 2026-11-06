"""A chat completion's response reads the same through ``generate()`` as on
the stream (RMK-484, RFC §6.4).

A call whose server lost its name reaches the loop with an empty one, which
the loop refuses, rather than an error raised while reading the response; a
response with no choice still reports what its request cost. One reader serves
the message's calls of OpenAI's wire and of PolarGrid.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITool,
    ProviderError,
    StreamDone,
    StreamToolCall,
)
from roomkit.providers.ai.openai_dialect import ToolCallSlots, message_tool_calls
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig

LOOKUP = AITool(name="lookup", description="d", parameters={"type": "object", "properties": {}})
USAGE = {"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11}
NAMELESS = {"id": "c1", "type": "function", "function": {"name": None, "arguments": "{}"}}


def _completion(choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "r",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": choices,
        "usage": USAGE,
    }


def _chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": choices,
        **extra,
    }


ANSWERS = {
    "nameless-call": (
        _completion(
            [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [NAMELESS]},
                }
            ]
        ),
        [
            _chunk(
                [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, **NAMELESS}]},
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
            _chunk([], usage=USAGE),
        ],
    ),
    "no-choice": (_completion([]), [_chunk([], usage=USAGE)]),
    # A custom tool's call carries no function: no call, on both modes (RMK-500).
    "function-less": (
        _completion(
            [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "type": "custom", "custom": {"name": "x"}}],
                    },
                }
            ]
        ),
        [
            _chunk(
                [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "custom"}]},
                        "finish_reason": None,
                    }
                ]
            ),
            _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        ],
    ),
    # A filtered answer whose message the server left null (RMK-500).
    "null-message": (
        _completion([{"index": 0, "finish_reason": "content_filter", "message": None}]),
        [_chunk([{"index": 0, "delta": {}, "finish_reason": "content_filter"}])],
    ),
}


def _provider(response: dict[str, Any], chunks: list[dict[str, Any]]) -> OpenAIAIProvider:
    def answer(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            lines = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(lines + "data: [DONE]\n\n").encode(),
            )
        return httpx.Response(200, json=response)

    config = OpenAIConfig(api_key="k", model="gpt-4.1-mini", include_stream_usage=True)
    provider = OpenAIAIProvider(config)
    provider._client = openai.AsyncOpenAI(
        api_key="k",
        base_url="http://wire.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
        max_retries=0,
    )
    return provider


async def _read(answer: str, mode: str) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """The calls and the usage a generation hands the loop."""
    provider = _provider(*ANSWERS[answer])
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[LOOKUP])
    if mode == "generate":
        response = await provider.generate(context)
        return [(c.name, c.arguments) for c in response.tool_calls], dict(response.usage)
    events = [event async for event in provider.generate_structured_stream(context)]
    calls = [(e.name, e.arguments) for e in events if isinstance(e, StreamToolCall)]
    done = next(e for e in events if isinstance(e, StreamDone))
    return calls, dict(done.usage)


@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_a_call_whose_name_was_lost_reaches_the_loop_nameless(mode: str) -> None:
    calls, _ = await _read("nameless-call", mode)

    assert calls == [("", {})]


@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_a_response_with_no_choice_reports_its_usage(mode: str) -> None:
    calls, usage = await _read("no-choice", mode)

    assert calls == []
    assert usage == {"input_tokens": 11, "output_tokens": 0}


def test_the_shared_reader_reads_any_sdk_s_message() -> None:
    """PolarGrid's SDK objects read as OpenAI's: by attribute, a nameless call
    kept, a call without a function skipped."""
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="c1", function=SimpleNamespace(name=None, arguments='{"q": 1}')),
            SimpleNamespace(id="c2", function=None),
        ]
    )

    [call] = message_tool_calls(message, "tool_calls")

    assert (call.id, call.name, call.arguments, call.partial) == ("c1", "", {"q": 1}, False)


def test_polargrid_reports_the_usage_of_a_response_with_no_choice() -> None:
    provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model="m"))
    response = SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=0), model="m"
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")])

    answer = provider._response_of(response, context)

    assert answer.tool_calls == []
    assert dict(answer.usage) == {"input_tokens": 11, "output_tokens": 0}


def test_ollama_keeps_a_call_whose_name_was_lost_nameless() -> None:
    """Ollama reads its own wire; a lost name is still an empty one, not "None"."""
    provider = OllamaAIProvider(OllamaConfig(model="m"))
    message = SimpleNamespace(
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=None, arguments={}))]
    )

    [call] = provider._extract_tool_calls(message)

    assert (call.name, call.arguments) == ("", {})


@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_an_entry_without_a_function_is_no_call(mode: str) -> None:
    calls, _ = await _read("function-less", mode)

    assert calls == []


@pytest.mark.parametrize("index", [0, None], ids=["indexed", "index-less"])
def test_a_call_whose_id_comes_before_its_function_keeps_the_server_id(
    index: int | None,
) -> None:
    """One fold serves OpenAI's wire, Mistral and PolarGrid (RFC §6.4)."""
    slots = ToolCallSlots()
    slots.fold(index, "call_srv", None, None)
    slots.fold(index, None, "lookup", '{"q": 1}')

    [call] = slots.calls("tool_calls")

    assert (call.id, call.name, call.arguments) == ("call_srv", "lookup", {"q": 1})


def test_polargrid_streams_no_call_for_an_entry_without_a_function() -> None:
    provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model="m"))
    slots = ToolCallSlots()
    deltas = [
        SimpleNamespace(index=0, id="ct_1", function=None),
        SimpleNamespace(index=1, id="c2", function={"name": "lookup", "arguments": "{}"}),
    ]

    provider._accumulate_tool_deltas(slots, deltas)

    assert [(c.id, c.name) for c in slots.calls("tool_calls")] == [("c2", "lookup")]


async def test_a_null_message_reads_as_an_empty_one() -> None:
    provider = _provider(*ANSWERS["null-message"])
    context = AIContext(messages=[AIMessage(role="user", content="go")])

    response = await provider.generate(context)

    assert (response.content, response.finish_reason, response.tool_calls) == (
        "",
        "content_filter",
        [],
    )


@pytest.mark.parametrize(
    ("base_url", "model", "vision"),
    [
        (None, "gpt-3.5-turbo-0125", False),
        ("http://local.test/v1", "qwen2.5-vl-7b-instruct", True),
    ],
    ids=["openai-endpoint", "behind-base-url"],
)
def test_a_server_behind_a_base_url_decides_what_its_model_reads(
    base_url: str | None, model: str, vision: bool
) -> None:
    """OpenAI's model names say nothing of a local model: images are passed
    through and the server answers (RMK-500)."""
    provider = OpenAIAIProvider(OpenAIConfig(api_key="k", model=model, base_url=base_url))

    assert provider.supports_vision is vision


@pytest.mark.parametrize(
    "provider",
    [
        OpenRouterAIProvider(OpenRouterConfig(api_key="k", model="anthropic/claude-opus-4")),
        AzureAIProvider(
            AzureAIConfig(
                api_key="k",
                azure_endpoint="https://r.openai.azure.com",
                api_version="2024-10-21",
                model="vision-prod",
            )
        ),
    ],
    ids=["openrouter", "azure"],
)
def test_a_router_or_a_deployment_decides_what_its_model_reads(provider: Any) -> None:
    """A ``vendor/model`` id or a deployment name is no OpenAI model name:
    images are passed through and the server answers (RMK-500)."""
    assert provider.supports_vision is True


def _streamed(*fragments: dict[str, Any]) -> list[dict[str, Any]]:
    """A stream whose chunks carry *fragments*, one call fragment each."""
    chunks = [
        _chunk([{"index": 0, "delta": {"tool_calls": [fragment]}, "finish_reason": None}])
        for fragment in fragments
    ]
    return [*chunks, _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])]


async def _streamed_calls(chunks: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """The calls a stream of *chunks* hands the loop."""
    provider = _provider(_completion([]), chunks)
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[LOOKUP])
    events = [event async for event in provider.generate_structured_stream(context)]
    return [(e.name, e.arguments) for e in events if isinstance(e, StreamToolCall)]


WHOLE = {
    "index": 0,
    "id": "c1",
    "type": "function",
    "function": {"name": "lookup", "arguments": "{}"},
}


@pytest.mark.parametrize("call_id", [0, [], {}, False], ids=["zero", "list", "object", "false"])
async def test_a_nameless_call_under_an_id_that_is_no_text_takes_no_name(call_id: Any) -> None:
    """A fragment opening another call under an id that is no text, read as
    false, is no second call to the held tool: its call keeps no name, for the
    loop to refuse (RMK-1372)."""
    second = {"index": 0, "id": call_id, "function": {"arguments": '{"q": 1}'}}

    calls = await _streamed_calls(_streamed(WHOLE, second))

    assert calls == [("lookup", {}), ("", {"q": 1})]


@pytest.mark.parametrize(
    "call_id", [7, True, [1], {"a": 1}], ids=["number", "true", "list", "object"]
)
async def test_an_id_that_is_no_text_on_a_call_already_named_is_ignored(call_id: Any) -> None:
    """A call already composed under the id the stream minted keeps it: an id
    that is no text, coming after, changes nothing and the read goes on
    (RMK-1372)."""
    first = {"index": 0, "type": "function", "function": {"name": "lookup", "arguments": '{"q"'}}
    second = {"index": 0, "id": call_id, "function": {"arguments": ": 1}"}}

    calls = await _streamed_calls(_streamed(first, second))

    assert calls == [("lookup", {"q": 1})]


@pytest.mark.parametrize("call_id", [7, True], ids=["number", "true"])
async def test_an_id_that_is_no_text_on_an_entry_without_a_function_is_no_call(
    call_id: Any,
) -> None:
    """An entry with no function is no call, whatever id it carries: the read
    goes on (RMK-1372)."""
    entry = {"index": 0, "id": call_id, "type": "custom"}
    named = {
        "index": 1,
        "id": "c2",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }

    calls = await _streamed_calls(_streamed(entry, named))

    assert calls == [("lookup", {})]


@pytest.mark.parametrize(
    "entry",
    [
        {
            "index": 0,
            "id": 7,
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        },
        {
            "index": 0,
            "id": True,
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        },
        {
            "index": 0,
            "id": [1],
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        },
        {"index": 0, "id": {"a": 1}, "type": "custom"},
    ],
    ids=["number", "true", "list", "object-without-function"],
)
async def test_an_id_that_is_no_text_on_a_new_call_fails_the_read(entry: dict[str, Any]) -> None:
    """A call that would run under an id that is no text, or an id no set
    can hold, fails the read rather than run (RMK-1372)."""
    with pytest.raises(ProviderError):
        await _streamed_calls(_streamed(entry))


@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize(
    ("answered", "named"), [("m", "m"), ("", "gpt-4.1-mini")], ids=["answered", "unnamed"]
)
async def test_a_response_names_the_model_that_answered_else_the_one_asked(
    mode: str, answered: str, named: str
) -> None:
    """Both modes name the model the server says answered, else the one the
    config asked for (RMK-1434)."""
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "Hi"},
    }
    completion = {**_completion([choice]), "model": answered}
    chunks = [
        {
            **_chunk([{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]),
            "model": answered,
        }
    ]
    provider = _provider(completion, chunks)
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    if mode == "generate":
        metadata = (await provider.generate(context)).metadata
    else:
        events = [event async for event in provider.generate_structured_stream(context)]
        metadata = next(e for e in events if isinstance(e, StreamDone)).metadata

    assert metadata == {"model": named}


@pytest.mark.parametrize("body", [[], [[]]], ids=["empty-list", "nested-list"])
async def test_a_body_that_is_no_object_is_no_completion(body: list[Any]) -> None:
    """A JSON body that is no object holds no completion, even an empty one
    (RMK-1434)."""
    provider = _provider(body, [])  # ty: ignore[invalid-argument-type]
    context = AIContext(messages=[AIMessage(role="user", content="go")])
    with pytest.raises(ProviderError):
        await provider.generate(context)


NAMED = {"index": 1, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


@pytest.mark.parametrize(
    ("held", "named"),
    [(1, True), (True, 1), (7, 7.0), (7, 7)],
    ids=["1-true", "true-1", "7-7.0", "7-7"],
)
async def test_an_id_that_is_no_text_another_entry_holds_is_not_taken_again(
    held: Any, named: Any
) -> None:
    """An id that is no text one entry holds, as Python compares it, is not
    taken by another call, which runs under its own (RMK-1372)."""
    entry = {"index": 0, "id": held, "type": "custom"}

    calls = await _streamed_calls(_streamed(entry, {**NAMED, "id": named}))

    assert calls == [("lookup", {})]


@pytest.mark.parametrize("again", [None, 7], ids=["no-id", "same-id"])
async def test_a_call_named_under_an_id_that_is_no_text_fails_where_it_is_named(
    again: int | None,
) -> None:
    """The call fails as its name comes, under the id an earlier fragment of
    its index gave: nothing after it reaches the reader (RMK-1372)."""
    provider = _provider(
        _completion([]),
        [
            _chunk(
                [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": 7, "type": "custom"}]}}]
            ),
            _chunk([{"index": 0, "delta": {"tool_calls": [{**NAMED, "index": 0, "id": again}]}}]),
            _chunk([{"index": 0, "delta": {"content": "after"}, "finish_reason": None}]),
            _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        ],
    )
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[LOOKUP])
    seen: list[Any] = []
    with pytest.raises(ProviderError):
        async for event in provider.generate_structured_stream(context):
            seen.append(event)

    assert seen == []
