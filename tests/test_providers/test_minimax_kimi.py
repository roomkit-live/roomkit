"""MiniMax's and Kimi's providers on an offline wire (RFC §6.4, §6.7).

What a host writes to select a vendor's dialect, on the vendor's API or on a
server of its own, and what each request then carries. The real SDK runs; only
the wire is fake, no vendor is reached.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.kimi import KimiAIProvider, KimiConfig
from roomkit.providers.minimax import MiniMaxAIProvider, MiniMaxConfig

_SERVER = "http://gpu:8000/v1"
_TOOL = AITool(name="get_weather", description="The weather.", parameters={"type": "object"})

_VENDORS: dict[str, tuple[type[Any], type[Any], str, str]] = {
    "minimax": (MiniMaxAIProvider, MiniMaxConfig, "MiniMax-M3.1-Flash", "api.minimax.io"),
    "kimi": (KimiAIProvider, KimiConfig, "kimi-k3.1-pro", "api.moonshot.ai"),
}
_BOTH = sorted(_VENDORS)


def _provider(vendor: str, *, handler: Callable[..., Any] | None = None, **config: Any) -> Any:
    cls, config_cls, model, _ = _VENDORS[vendor]
    transport = httpx.MockTransport(handler) if handler is not None else None
    settings = {"api_key": "k", "model": model, **config}
    return cls(config_cls(**settings), transport=transport)


def _stream(*deltas: dict[str, Any], finish: str = "stop") -> httpx.Response:
    frames = [{"index": 0, "delta": delta, "finish_reason": None} for delta in deltas]
    frames.append({"index": 0, "delta": {}, "finish_reason": finish})
    lines = [
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [frame],
            }
        )
        + "\n\n"
        for frame in frames
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines) + "data: [DONE]\n\n",
    )


def _call(call_id: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]
    }


# What the model answers, round by round: reasoning and a call, a call alone,
# its answer; then the next turn's answer.
_ROUNDS = [
    _stream(
        {"reasoning_content": "WHY."},
        {"content": "Let me look."},
        _call("c1"),
        finish="tool_calls",
    ),
    _stream(_call("c2"), finish="tool_calls"),
    _stream({"reasoning_content": "DONE."}, {"content": "Sunny."}),
    _stream({"content": "You are welcome."}),
]


async def _two_turns(provider_for: Callable[[Callable[..., Any]], Any]) -> list[dict[str, Any]]:
    """The requests a room sends through the provider over two turns, the first
    with its two tool rounds, as an integrator drives it."""
    requests: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _ROUNDS[len(requests) - 1]

    provider = provider_for(answer)

    async def run_tool(name: str, arguments: dict[str, Any]) -> str:
        return "sunny"

    kit = RoomKit()
    user = WebSocketChannel("user")
    kit.register_channel(user)
    kit.register_channel(
        AIChannel("assistant", provider=provider, tools=[_TOOL], tool_handler=run_tool)
    )
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "user")
    await kit.attach_channel("room-1", "assistant", category=ChannelCategory.INTELLIGENCE)
    try:
        for text in ("go", "thanks"):
            await kit.process_inbound(
                InboundMessage(channel_id="user", sender_id="me", content=TextContent(body=text)),
                room_id="room-1",
            )
    finally:
        await kit.close()
    return requests


def _reasoning(messages: list[dict[str, Any]]) -> list[tuple[str | None, str]]:
    """Each assistant message of a request: its text, and its field."""
    return [
        (m["content"], m.get("reasoning_content", "absent"))
        for m in messages
        if m["role"] == "assistant"
    ]


@pytest.mark.parametrize(
    ("vendor", "empty"),
    [("minimax", "absent"), ("kimi", " ")],
)
async def test_a_host_selects_the_dialect_by_the_class_on_a_server_of_its_own(
    vendor: str, empty: str
) -> None:
    """The end-to-end proof: a room, an AI channel, the class chosen, a generic
    server's URL. A round's reasoning goes back in the field for as long as the
    turn lasts, then no more, and no cap is sent unless one is set: reasoning
    counts toward it, and these models answer empty past a small one."""
    requests = await _two_turns(
        lambda handler: _provider(vendor, handler=handler, base_url=_SERVER)
    )

    assert len(requests) == 4
    assert "max_tokens" not in requests[0]
    assert "max_completion_tokens" not in requests[0]
    in_progress = requests[2]["messages"]
    assert _reasoning(in_progress) == [("Let me look.", "WHY."), (None, empty)]
    later = requests[3]["messages"]
    assert _reasoning(later) == [
        ("Let me look.", "absent"),
        (None, empty),
        (None, empty),
        ("Sunny.", "absent"),
    ]
    assert not any("<think>" in str(m["content"]) for m in in_progress + later)


def _completion(request: httpx.Request) -> httpx.Response:
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "Hello."},
        "finish_reason": "stop",
    }
    body = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [choice],
    }
    return httpx.Response(200, json=body)


async def _sent(vendor: str, *, config: dict[str, Any], **turn: Any) -> httpx.Request:
    """The request one generation sends through the vendor's provider."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _completion(request)

    provider = _provider(vendor, handler=answer, **config)
    try:
        await provider.generate(AIContext(messages=[AIMessage(role="user", content="Hi")], **turn))
    finally:
        await provider.close()
    return seen[0]


def _body(request: httpx.Request, *keys: str) -> dict[str, Any]:
    body = json.loads(request.content)
    return {key: body[key] for key in keys if key in body}


@pytest.mark.parametrize("vendor", _BOTH)
async def test_the_vendor_s_own_endpoint_is_the_default(vendor: str) -> None:
    """Its URL, its name in telemetry, and a cap under the field it reads."""
    request = await _sent(vendor, config={"max_tokens": 2000})

    assert (request.url.host, request.url.path) == (_VENDORS[vendor][3], "/v1/chat/completions")
    assert _body(request, "max_tokens", "max_completion_tokens") == {"max_completion_tokens": 2000}
    assert _provider(vendor).name == vendor


@pytest.mark.parametrize("vendor", _BOTH)
def test_without_the_sdk_the_error_names_the_vendor_s_extra(vendor: str) -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(ImportError, match=rf"roomkit\[{vendor}\]"),
    ):
        _provider(vendor)


_REASONING = ("reasoning_effort", "thinking", "temperature")


@pytest.mark.parametrize(
    ("vendor", "model", "turn", "sent"),
    [
        # Each cannot stop reasoning: a turn that states off gets the least it takes.
        ("kimi", "kimi-k3.1-pro", {"enable_thinking": False}, {"reasoning_effort": "low"}),
        ("kimi", "kimi-k3", {"reasoning_effort": "none"}, {"reasoning_effort": "low"}),
        ("minimax", "MiniMax-M3.1-Flash", {"thinking_budget": 0}, {"reasoning_effort": "low"}),
        # M3 can, through MiniMax's own switch.
        ("minimax", "MiniMax-M3", {"enable_thinking": False}, {"thinking": {"type": "disabled"}}),
        # A level the model does not take goes as the nearest it takes.
        ("kimi", "kimi-k3.1-pro", {"reasoning_effort": "medium"}, {"reasoning_effort": "low"}),
        ("kimi", "kimi-k3", {"reasoning_effort": "xhigh"}, {"reasoning_effort": "high"}),
        (
            "minimax",
            "MiniMax-M3.1-Flash",
            {"reasoning_effort": "minimal"},
            {"reasoning_effort": "low"},
        ),
        # A model the catalogue does not name: what the turn states goes as it is.
        (
            "kimi",
            "moonshotai/Kimi-K3",
            {"reasoning_effort": "medium"},
            {"reasoning_effort": "medium"},
        ),
        ("kimi", "moonshotai/Kimi-K3", {"enable_thinking": False}, {}),
        (
            "minimax",
            "MiniMaxAI/MiniMax-M3",
            {"reasoning_effort": "none"},
            {"reasoning_effort": "none"},
        ),
        # A turn with tools sends its effort too.
        (
            "kimi",
            "kimi-k3",
            {"reasoning_effort": "high", "tools": [_TOOL]},
            {"reasoning_effort": "high"},
        ),
        (
            "minimax",
            "MiniMax-M3.1-Flash",
            {"reasoning_effort": "high", "tools": [_TOOL]},
            {"reasoning_effort": "high"},
        ),
    ],
)
async def test_what_a_turn_states_of_its_reasoning(
    vendor: str, model: str, turn: dict[str, Any], sent: dict[str, Any]
) -> None:
    request = await _sent(vendor, config={"model": model}, **turn)

    assert _body(request, *_REASONING) == sent


@pytest.mark.parametrize(
    ("vendor", "config", "sent"),
    [
        # Kimi's K3 models fix it and ask that it be left out; MiniMax ignores it.
        ("kimi", {}, {}),
        ("minimax", {}, {}),
        ("kimi", {"supports_custom_temperature": True}, {"temperature": 0.2}),
        ("minimax", {"supports_custom_temperature": True}, {"temperature": 0.2}),
    ],
)
async def test_the_temperature_goes_where_the_server_takes_it(
    vendor: str, config: dict[str, Any], sent: dict[str, Any]
) -> None:
    request = await _sent(vendor, config=config, temperature=0.2)

    assert _body(request, "temperature") == sent


@pytest.mark.parametrize(
    ("vendor", "config", "vision"),
    [
        ("kimi", {"model": "kimi-k3"}, True),
        ("kimi", {"model": "unknown"}, False),
        ("kimi", {"model": "unknown", "base_url": _SERVER}, True),
        ("minimax", {"model": "MiniMax-M3"}, True),
        ("minimax", {"model": "unknown"}, False),
        ("minimax", {"model": "unknown", "base_url": _SERVER}, True),
    ],
)
async def test_a_server_decides_what_a_model_the_catalogue_does_not_name_reads(
    vendor: str, config: dict[str, Any], vision: bool
) -> None:
    provider = _provider(vendor, **config)
    try:
        assert provider.supports_vision is vision
    finally:
        await provider.close()


@pytest.mark.parametrize(("vendor", "takes"), [("minimax", False), ("kimi", True)])
async def test_a_response_schema_is_taken_where_the_vendor_documents_it(
    vendor: str, takes: bool
) -> None:
    default = _provider(vendor)
    stated = _provider(vendor, supports_response_schema=not takes)
    try:
        assert default.supports_response_schema is takes
        assert stated.supports_response_schema is not takes
    finally:
        await default.close()
        await stated.close()


@pytest.mark.parametrize(
    ("name", "refused"),
    [("a" * 128, False), ("_lookup-2", False), ("a" * 129, True), ("1abc", True), ("a.b", True)],
)
async def test_kimi_s_tool_names_are_checked_before_the_request(name: str, refused: bool) -> None:
    tool = AITool(name=name, description="A tool.", parameters={"type": "object"})
    if not refused:
        await _sent("kimi", config={}, tools=[tool])
        return
    with pytest.raises(ProviderError, match="kimi"):
        await _sent("kimi", config={}, tools=[tool])


async def test_behind_another_url_the_server_decides_kimi_s_tool_names() -> None:
    dotted = AITool(name="a.b", description="A tool.", parameters={"type": "object"})

    request = await _sent("kimi", config={"base_url": _SERVER}, tools=[dotted])

    assert json.loads(request.content)["tools"][0]["function"]["name"] == "a.b"


def test_minimax_m3_bills_twice_its_rates_past_512k_input_tokens() -> None:
    """MiniMax's standard tier past 512k input tokens: input and output double."""
    [m3] = [m for m in MiniMaxAIProvider.available_models() if m.id == "MiniMax-M3"]
    assert m3.pricing is not None

    short = m3.pricing.cost_for({"input_tokens": 500_000, "output_tokens": 10_000})
    long = m3.pricing.cost_for({"input_tokens": 600_000, "output_tokens": 10_000})

    assert short == pytest.approx(0.15 + 0.012)
    assert long == pytest.approx(0.36 + 0.024)
