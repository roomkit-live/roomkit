"""Chat Completions composition through the real SDK (RFC §6.4, RMK-1119)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from roomkit import AIChannel, RoomKit
from roomkit.providers.ai import (
    AIContext,
    ProviderError,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig
from tests.test_ai_streaming_tool_events import _run_turn


def _fragment(index: int, call_id: str | int | None, name: str | None, arguments: Any) -> dict:
    return {"index": index, "id": call_id, "function": {"name": name, "arguments": arguments}}


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "completion",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _provider(replies: list[list[dict]]) -> OpenAIAIProvider:
    def answer(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in replies.pop(0))
        return httpx.Response(
            200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    return OpenAIAIProvider(
        OpenAIConfig(api_key="test", model="fake", max_retries=0),
        transport=httpx.MockTransport(answer),
    )


# The equivalent cases live in roomkit-openai/tests/agent/composition.rs.
# Indices in emitted events are encounter positions, not the server's sparse indices.
_CASES = [
    (
        "unicode",
        [_fragment(7, "a", "search", '{"q":'), _fragment(7, None, None, '"é猫🙂"}')],
        [(0, "search", '{"q":'), (0, "search", '"é猫🙂"}')],
        [{"q": "é猫🙂"}],
    ),
    (
        "empty-repeat",
        [
            _fragment(0, "a", "search", ""),
            _fragment(0, None, "search", ""),
            _fragment(0, None, None, "{}"),
        ],
        [(0, "search", ""), (0, "search", "{}")],
        [{}],
    ),
    (
        "late-id-after-name",
        [_fragment(0, None, "search", '{"q":'), _fragment(0, "late", None, '"x"}')],
        [(0, "search", '{"q":'), (0, "search", '"x"}')],
        [{"q": "x"}],
    ),
    (
        "late-id-before-name",
        [{"index": 0}, _fragment(0, "late", "search", "{}")],
        [(0, "search", "{}")],
        [{}],
    ),
    (
        "name-after-arguments",
        [_fragment(0, "a", None, '{"q":'), _fragment(0, None, "search", '"x"}')],
        [(0, "search", '"x"}')],
        [{"q": "x"}],
    ),
    (
        "interleaved",
        [
            _fragment(8, "b", "search", '{"q":'),
            _fragment(2, "a", "search", '{"q":'),
            _fragment(8, None, None, '"b"}'),
            _fragment(2, None, None, '"a"}'),
        ],
        [
            (0, "search", '{"q":'),
            (1, "search", '{"q":'),
            (0, "search", '"b"}'),
            (1, "search", '"a"}'),
        ],
        [{"q": "b"}, {"q": "a"}],
    ),
    (
        "repeated-id",
        [_fragment(0, "same", "search", "{}"), _fragment(1, "same", "search", "{}")],
        [(0, "search", "{}"), (1, "search", "{}")],
        [{}, {}],
    ),
    (
        "reused-index",
        [_fragment(0, None, "search", "{}"), _fragment(0, None, "search", "{}")],
        [(0, "search", "{}"), (1, "search", "{}")],
        [{}, {}],
    ),
    (
        "object-arguments",
        [_fragment(0, "a", "search", {"q": "é猫🙂"})],
        [(0, "search", json.dumps({"q": "é猫🙂"}))],
        [{"q": "é猫🙂"}],
    ),
    (
        "invalid-json",
        [_fragment(0, "a", "search", "{bad"), _fragment(0, None, None, " json")],
        [(0, "search", "{bad"), (0, "search", " json")],
        [{"raw": "{bad json"}],
    ),
    ("no-function", [{"index": 0, "id": "custom"}], [], []),
    (
        "custom-before-call",
        [{"index": 0, "id": "custom"}, _fragment(2, "a", "search", "{}")],
        [(1, "search", "{}")],
        [{}],
    ),
]


@pytest.mark.parametrize(
    ("case", "fragments", "expected", "arguments"), _CASES, ids=[case[0] for case in _CASES]
)
async def test_wire_fragments_keep_each_calls_final_id(
    case: str, fragments: list[dict], expected: list[tuple], arguments: list[dict]
) -> None:
    provider = _provider(
        [[_chunk({"tool_calls": [f]}) for f in fragments] + [_chunk({}, "tool_calls")]]
    )
    try:
        events = [event async for event in provider.generate_structured_stream(AIContext())]
    finally:
        await provider.close()
    deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
    calls = [e for e in events if isinstance(e, StreamToolCall)]
    assert [(d.index, d.name, d.arguments_delta) for d in deltas] == expected
    assert [c.arguments for c in calls] == arguments
    assert len({c.id for c in calls}) == len(calls)
    by_slot: dict[int, str] = {}
    for delta in deltas:
        assert by_slot.setdefault(delta.index, delta.id) == delta.id
        assert any(call.id == delta.id and call.name == delta.name for call in calls)
    assert len(set(by_slot.values())) == len(by_slot)
    assert all(events.index(d) < events.index(calls[0]) for d in deltas)
    assert isinstance(events[-1], StreamDone)
    if case == "late-id-after-name":
        assert calls[0].id.startswith("call_search_")
    if case == "late-id-before-name":
        assert calls[0].id == "late"


async def test_composition_precedes_reasoning_and_text_in_the_same_chunk() -> None:
    provider = _provider(
        [
            [
                _chunk(
                    {
                        "tool_calls": [_fragment(0, "a", "search", "{}")],
                        "reasoning_content": "checking",
                        "content": "answer",
                    }
                ),
                _chunk({}, "tool_calls"),
            ]
        ]
    )
    try:
        events = [event async for event in provider.generate_structured_stream(AIContext())]
    finally:
        await provider.close()
    assert [type(e) for e in events] == [
        StreamToolCallDelta,
        StreamThinkingDelta,
        StreamTextDelta,
        StreamToolCall,
        StreamDone,
    ]


async def test_a_failed_wire_keeps_its_fragment_without_a_complete_call() -> None:
    provider = _provider(
        [
            [
                _chunk({"tool_calls": [_fragment(0, "a", "search", "{secret")]}),
                {"error": {"message": "failed", "code": "invalid_request_error"}},
            ]
        ]
    )
    events = []
    try:
        with pytest.raises(ProviderError):
            async for event in provider.generate_structured_stream(AIContext()):
                events.append(event)
    finally:
        await provider.close()
    assert len(events) == 1
    assert isinstance(events[0], StreamToolCallDelta)
    assert events[0].arguments_delta == "{secret"


async def test_a_later_malformed_call_keeps_the_first_fragment_of_its_chunk() -> None:
    provider = _provider(
        [
            [
                _chunk(
                    {
                        "tool_calls": [
                            _fragment(0, "a", "search", "{}"),
                            _fragment(1, 7, "search", "{}"),
                        ]
                    }
                )
            ]
        ]
    )
    events = []
    try:
        with pytest.raises(ProviderError):
            async for event in provider.generate_structured_stream(AIContext()):
                events.append(event)
    finally:
        await provider.close()
    assert len(events) == 1
    assert isinstance(events[0], StreamToolCallDelta)
    assert (events[0].id, events[0].arguments_delta) == ("a", "{}")


async def test_a_cut_call_keeps_its_fragments_and_partial_arguments() -> None:
    provider = _provider(
        [
            [
                _chunk(
                    {
                        "tool_calls": [
                            _fragment(0, None, "search", '{"q":'),
                        ]
                    }
                ),
                _chunk({}, "length"),
            ]
        ]
    )
    try:
        events = [event async for event in provider.generate_structured_stream(AIContext())]
    finally:
        await provider.close()
    delta, call, done = events
    assert isinstance(delta, StreamToolCallDelta)
    assert isinstance(call, StreamToolCall)
    assert delta.id == call.id
    assert call.partial and call.arguments == {"raw": '{"q":'}
    assert isinstance(done, StreamDone) and done.finish_reason == "length"


async def test_wire_composition_reaches_the_bus_before_execution() -> None:
    replies = [
        [
            _chunk({"tool_calls": [_fragment(0, "a", "search", '{"q":')]}),
            _chunk({"tool_calls": [_fragment(0, None, None, '"secreté"}')]}),
            _chunk({}, "tool_calls"),
        ],
        [_chunk({"content": "Done."}), _chunk({}, "stop")],
    ]
    kit = RoomKit()
    ai = AIChannel("ai1", provider=_provider(replies), thinking_coalesce_ms=0)
    try:
        received = await _run_turn(kit, ai)
        events = [e for e in received if e.type.value.startswith("tool_call_")]
        assert [e.type.value for e in events] == [
            "tool_call_delta",
            "tool_call_delta",
            "tool_call_delta",
            "tool_call_start",
            "tool_call_end",
        ]
        assert [e.data["tool_calls"] for e in events[:3]] == [
            [{"id": "a", "name": "search", "arguments_chars": 5}],
            [{"id": "a", "name": "search", "arguments_chars": 15}],
            [],
        ]
        assert events[3].data["tool_calls"][0]["arguments"] == {"q": "secreté"}
        assert replies == []
        history = await kit.store.list_events("r1")
        assert sum(e.type.value == "tool_call_start" for e in history) == 1
        assert all(e.type.value != "tool_call_delta" for e in history)
    finally:
        await kit.close()
