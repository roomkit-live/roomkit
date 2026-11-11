"""A turn's budget stops its tool loop at a round boundary (RFC §6.4, RMK-320).

A token budget counts every token the provider bills for the turn, a cost
budget prices each generation at the model's catalogue rate. A turn that has
reached either ends ``budget_exceeded``: the calls its last generation asked
for do not run, and no further generation is asked for.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pytest

from roomkit.channels import FallbackRouter
from roomkit.channels._ai_loop_rules import require_schema_answer
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    ModelInfo,
    ModelPricing,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.response_schema import ResponseSchemaError
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look up an item", parameters={})
# 500 billed tokens per generation.
_USAGE = {"input_tokens": 300, "cache_read_input_tokens": 100, "output_tokens": 100}
_PRICING = ModelPricing(
    input_per_million=10.0,
    output_per_million=50.0,
    cache_read_per_million=1.0,
    long_context_threshold_tokens=1000,
    long_context_input_multiplier=2.0,
    verified=date(2026, 9, 30),
)


class _Priced(MockAIProvider):
    """A mock whose model has a catalogue price."""

    def catalog_entry(self) -> ModelInfo | None:
        return ModelInfo(id="priced", pricing=_PRICING)


def _round(n: int) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"c{n}", name="lookup", arguments={"n": n})],
        usage=dict(_USAGE),
    )


_DONE = AIResponse(content="done", finish_reason="stop", usage=dict(_USAGE))


def _script(rounds: int = 6) -> list[AIResponse]:
    return [*(_round(n) for n in range(rounds)), _DONE]


async def _turn(ch: AIChannel, metadata: dict[str, Any] | None = None) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_LOOKUP.model_dump()], **(metadata or {})},
    )
    return await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )


def _channel(provider: MockAIProvider, **options: Any) -> tuple[AIChannel, list[int]]:
    ran: list[int] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments["n"])
        return "ok"

    return AIChannel("ai1", provider=provider, tool_handler=lookup, **options), ran


async def test_a_token_budget_stops_the_turn_where_it_is_reached(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=1200)

    run = await _turn(ch)

    # 500 billed tokens a generation: the third reaches 1,200, and its call does not run.
    assert run.reason == "budget_exceeded"
    assert ran == [0, 1]
    assert len(provider.calls) == 3


async def test_a_cost_budget_stops_the_turn_at_the_catalogue_price(streaming: bool) -> None:
    provider = _Priced(ai_responses=_script(), streaming=streaming)
    per_generation = _PRICING.cost_for(_USAGE)
    ch, ran = _channel(provider, turn_budget_usd=per_generation * 2.5)

    run = await _turn(ch)

    assert run.reason == "budget_exceeded"
    assert ran == [0, 1]
    assert len(provider.calls) == 3


async def test_a_turn_under_its_budget_completes(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_script(rounds=2), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000)

    run = await _turn(ch)

    assert (run.reason, run.text, ran) == ("completed", "done", [0, 1])


async def test_the_turn_config_and_the_room_outrank_the_channel(streaming: bool) -> None:
    async def per_turn(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(turn_budget_tokens=700)

    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000, config_provider=per_turn)
    assert (await _turn(ch)).reason == "budget_exceeded" and ran == [0]

    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=100_000, config_provider=per_turn)
    run = await _turn(ch, {"turn_budget_tokens": 100_000})
    assert run.reason == "completed" and ran == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("level", ["binding", "turn"])
@pytest.mark.parametrize("key", ["turn_budget_tokens", "turn_budget_usd"])
async def test_a_null_never_lifts_the_channel_budget(
    streaming: bool, key: str, level: str
) -> None:
    """An empty field serialized as ``null``, in the room's binding metadata
    or in the turn config, keeps the channel's cap (RFC Appendix A.9)."""

    async def per_turn(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(**{key: None})

    cap = {"turn_budget_tokens": 1200, "turn_budget_usd": _PRICING.cost_for(_USAGE) * 2.5}[key]
    options: dict[str, Any] = {key: cap}
    if level == "turn":
        options["config_provider"] = per_turn
    ch, ran = _channel(_Priced(ai_responses=_script(), streaming=streaming), **options)

    run = await _turn(ch, {key: None} if level == "binding" else None)

    assert run.reason == "budget_exceeded"
    assert ran == [0, 1]


def test_a_cost_budget_needs_a_priced_model() -> None:
    with pytest.raises(ValueError, match="turn_budget_usd needs a model with a known price"):
        AIChannel("ai1", provider=MockAIProvider(), turn_budget_usd=0.05)


async def test_each_generation_is_priced_as_one_response(streaming: bool) -> None:
    """600 input tokens a generation stay under the 1,000-token long-context
    threshold: each costs its own price, and the turn the sum of them. Pricing
    the turn's summed usage as one response would apply the multiplier and
    stop the turn two generations early."""
    usage = {"input_tokens": 600}
    per_generation = _PRICING.cost_for(usage)
    assert _PRICING.cost_for({"input_tokens": 1200}) > 3.5 * per_generation
    script = [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"c{n}", name="lookup", arguments={"n": n})],
            usage=dict(usage),
        )
        for n in range(6)
    ]
    ch, ran = _channel(
        _Priced(ai_responses=script, streaming=streaming), turn_budget_usd=3.5 * per_generation
    )

    run = await _turn(ch)

    assert run.reason == "budget_exceeded"
    assert ran == [0, 1, 2]


async def test_past_its_budget_a_turn_asks_for_no_retry_of_an_empty_answer(
    streaming: bool,
) -> None:
    empty = AIResponse(content="", finish_reason="stop", usage=dict(_USAGE))
    provider = MockAIProvider(ai_responses=[_round(0), empty, _DONE], streaming=streaming)
    ch, ran = _channel(provider, turn_budget_tokens=600)

    run = await _turn(ch)

    # The empty answer brings the turn to 1,000 tokens: no retry is asked for.
    assert run.reason == "budget_exceeded"
    assert (ran, len(provider.calls)) == ([0], 2)


async def test_a_room_budget_the_model_cannot_price_fails_the_turn(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=_script(), streaming=streaming)
    ch, ran = _channel(provider)

    with pytest.raises(ValueError, match="turn_budget_usd needs a model with a known price"):
        await _turn(ch, {"turn_budget_usd": 0.05})
    with pytest.raises(ValueError, match="must be a positive number"):
        await _turn(ch, {"turn_budget_tokens": "1200"})
    assert (ran, provider.calls) == ([], [])


@pytest.mark.parametrize(
    "options",
    [{"turn_budget_tokens": -5}, {"turn_budget_tokens": True}, {"turn_budget_usd": 0}],
)
def test_a_budget_is_a_positive_number(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be a positive number"):
        AIChannel("ai1", provider=_Priced(), **options)


def test_a_schema_turn_stopped_by_its_budget_has_no_document() -> None:
    schema = {
        "type": "object",
        "properties": {"total": {"type": "number"}},
        "required": ["total"],
        "additionalProperties": False,
    }
    context = AIContext(messages=[], response_schema=schema)

    with pytest.raises(ResponseSchemaError, match="budget_exceeded"):
        require_schema_answer(context, "budget_exceeded")


@pytest.mark.parametrize(("picked", "fails"), [("unpriced", True), ("priced", False)])
async def test_a_cost_budget_is_checked_on_the_model_the_router_picks(
    picked: str, fails: bool, streaming: bool
) -> None:
    # The channel has no provider to price at construction: each turn checks
    # the one its router picks (RFC §6.4).
    model = (MockAIProvider if picked == "unpriced" else _Priced)(
        ai_responses=[_DONE], streaming=streaming
    )
    ch = AIChannel("ai1", router=FallbackRouter(model, MockAIProvider()), turn_budget_usd=1.0)

    if fails:
        with pytest.raises(ValueError, match="turn_budget_usd needs a model with a known price"):
            await _turn(ch)
    else:
        assert (await _turn(ch)).reason == "completed"


async def test_a_generation_handed_to_a_model_priced_otherwise_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Refusing(_Priced):
        async def generate(self, context: AIContext) -> AIResponse:
            raise ProviderError("overloaded", retryable=True, status_code=503)

    class _Fallback(MockAIProvider):
        @property
        def model_name(self) -> str:
            return "fallback-priced-otherwise"

    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        for _ in range(2):
            fallback = _Fallback(ai_responses=[_DONE])
            ch = AIChannel(
                "ai1", router=FallbackRouter(_Refusing(), fallback), turn_budget_usd=1.0
            )
            run = await _turn(ch)
            assert run.reason == "completed"

    warnings = [r for r in caplog.records if "priced otherwise" in r.getMessage()]
    assert len(warnings) == 1


def test_fifty_rounds_and_a_warning_at_twenty_five_by_default() -> None:
    ch = AIChannel("ai1", provider=MockAIProvider())

    assert (ch._max_tool_rounds, ch._tool_loop_warn_after) == (50, 25)
