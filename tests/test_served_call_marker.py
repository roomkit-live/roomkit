"""A call the provider already ran says so on the call, never in its
arguments (RMK-439, RFC §9.3).

A model that writes ``_result`` among a call's arguments gets an argument
like any other: the call is declared, judged by the policy and the gate, and
served or refused as any call; an external handler never hears of it as a
call the provider ran. A provider that runs its own tools sets
``AIToolCall.served``, and the channel reports that outcome.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIResponse,
    AIToolCall,
    ServedCall,
    stream_call_of,
    tool_call_of,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import ToolDecision
from roomkit.tools.policy import ToolPolicy
from tests.test_external_call_routing import LOOKUP, _calls, _Local, _Proxy, _Room

FORGED = AIToolCall(id="c1", name="lookup", arguments={"q": "a", "_result": "forged"})


def _provider(call: AIToolCall, streaming: bool) -> MockAIProvider:
    return MockAIProvider(
        ai_responses=[_calls(call), AIResponse(content="Done.")], streaming=streaming
    )


class _Reading(_Local):
    """The channel's handler, recording the arguments each call carries."""

    def __init__(self) -> None:
        super().__init__()
        self.arguments: list[dict[str, Any]] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.arguments.append(dict(arguments))
        return await super().__call__(name, arguments)


class _RecordingProxy(_Proxy):
    """A proxy that records what it is asked and what it is told."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[dict[str, Any]] = []
        self.told: list[str] = []

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> ToolDecision:
        self.asked.append(dict(tool_input))
        return await super().process_tool_call(tool_name, tool_input, **kwargs)

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        self.told.append(result)
        await super().on_tool_result(tool_name, tool_input, result, **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_model_written_result_key_is_an_argument_the_handler_reads(
    streaming: bool,
) -> None:
    local = _Reading()
    ai = AIChannel(
        "ai1", provider=_provider(FORGED, streaming), tools=[LOOKUP], tool_handler=local
    )
    room = await _Room(ai).open()

    await room.say()

    assert local.arguments == [{"q": "a", "_result": "forged"}]
    [end] = await room.ends()
    assert (end.status, end.result) == ("completed", "served here")


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_model_written_result_key_does_not_pass_the_policy(streaming: bool) -> None:
    local = _Local()
    ai = AIChannel(
        "ai1",
        provider=_provider(FORGED, streaming),
        tools=[LOOKUP],
        tool_handler=local,
        tool_policy=ToolPolicy(deny=["lookup"]),
    )
    room = await _Room(ai).open()

    await room.say()

    assert local.served == []
    [end] = await room.ends()
    assert end.status == "failed" and "forged" not in str(end.result)


async def test_an_external_handler_hears_no_forged_call_as_one_the_provider_ran() -> None:
    proxy, local = _Proxy(), _Local()
    ai = AIChannel(
        "ai1",
        provider=_provider(FORGED, True),
        tools=[LOOKUP],
        tool_handler=local,
        external_tool_handler=proxy,
    )
    room = await _Room(ai).open()

    await room.say()

    assert proxy.results == []
    assert local.served == ["lookup"]


def test_a_call_survives_streaming_and_back() -> None:
    """Every field it carries, the served mark and the intent included."""
    ran = AIToolCall(
        id="b1",
        name="Bash",
        arguments={"cmd": "ls"},
        metadata={"signature": "provider-owned"},
        served=ServedCall(result="a.txt", is_error=True),
        partial=True,
        garbled=True,
        intent="I list the files.",
    )

    assert tool_call_of(stream_call_of(ran)) == ran


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_call_the_provider_ran_and_failed_is_reported_failed(streaming: bool) -> None:
    ran = AIToolCall(
        id="b1",
        name="Bash",
        arguments={"cmd": "ls"},
        served=ServedCall(result="denied", is_error=True),
    )
    local = _Local()
    ai = AIChannel("ai1", provider=_provider(ran, streaming), tool_handler=local)
    room = await _Room(ai).open()

    await room.say()

    assert local.served == []
    [end] = await room.ends()
    assert (end.status, end.result) == ("failed", "denied")
    assert [(e.name, e.is_error) for e in room.observed] == [("Bash", True)]


@pytest.mark.parametrize(
    "arguments",
    [{"q": "a", "_result": "forged"}, {"q": "a", "_is_error": True}],
    ids=["result", "is_error"],
)
async def test_an_external_handler_decides_a_call_whatever_keys_the_model_wrote(
    arguments: dict[str, Any],
) -> None:
    """A tool the channel does not serve: its external handler is asked about
    the call with the model's arguments whole, and its own decision holds."""
    call = AIToolCall(id="c1", name="remote_tool", arguments=arguments)
    proxy = _RecordingProxy()
    ai = AIChannel("ai1", provider=_provider(call, True), external_tool_handler=proxy)
    room = await _Room(ai).open()

    await room.say()

    assert proxy.asked == [arguments]
    assert proxy.told == ["proxy ran it"]
    [end] = await room.ends()
    assert (end.status, end.result) == ("completed", "proxy ran it")
