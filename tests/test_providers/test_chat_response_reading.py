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

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, StreamDone, StreamToolCall
from roomkit.providers.ai.openai_dialect import message_tool_calls
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
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
