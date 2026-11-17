"""The thinker's public bounds are applied when its provider is called."""

from __future__ import annotations

import asyncio
import json

import pytest

from roomkit import AIChannel, LLMThinker, MockThinker, Thought
from roomkit.providers.ai import AIContext, AIMessage, AIResponse, MockAIProvider


class ControlledProvider(MockAIProvider):
    def __init__(self, held: bool = False) -> None:
        super().__init__(response_schema=True)
        self.held = held
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        self.entered.set()
        try:
            if self.held:
                await self.release.wait()
            return AIResponse(content='{"text":"heard","want_to_say":[],"urgent":false}')
        finally:
            self.finished = True


@pytest.mark.parametrize("thinker_without_policy", [False, True])
def test_speaking_configuration_is_refused_before_the_turn_budget(
    thinker_without_policy: bool,
) -> None:
    if thinker_without_policy:
        with pytest.raises(ValueError, match="a thinker needs a speak_policy"):
            AIChannel("ai", provider=MockAIProvider(), thinker=MockThinker(), turn_budget_tokens=0)
    else:
        with pytest.raises(ValueError, match="speak_timeout must be positive"):
            AIChannel("ai", provider=MockAIProvider(), speak_timeout=0, turn_budget_tokens=0)


@pytest.mark.parametrize("timeout", [0.0, -0.1, float("-inf")])
@pytest.mark.parametrize("held", [False, True], ids=["immediate", "pending"])
async def test_a_nonpositive_thinker_bound_never_enters_even_an_immediate_provider(
    timeout: float, held: bool
) -> None:
    provider = ControlledProvider(held)
    thinker = LLMThinker(provider, timeout=timeout)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(thinker.think(Thought(), AIContext()), 1)
    assert provider.calls == []
    assert not provider.entered.is_set() and not provider.finished


async def test_a_positive_thinker_bound_cancels_and_drains_the_provider_call() -> None:
    provider = ControlledProvider(held=True)
    thinker = LLMThinker(provider, timeout=0.01)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(thinker.think(Thought(), AIContext()), 1)
    assert len(provider.calls) == 1
    assert provider.entered.is_set() and provider.finished


@pytest.mark.parametrize("timeout", [None, float("inf"), 1e300])
async def test_an_unbounded_or_huge_thinker_waits_for_a_provider_that_was_really_entered(
    timeout: float | None,
) -> None:
    provider = ControlledProvider(held=True)
    thinker = LLMThinker(provider, timeout=timeout)
    task = asyncio.create_task(thinker.think(Thought(), AIContext()))
    try:
        await asyncio.wait_for(provider.entered.wait(), 1)
        await asyncio.sleep(0.02)
        assert not task.done() and not provider.finished
        provider.release.set()
        assert await asyncio.wait_for(task, 1) == Thought("heard")
        assert len(provider.calls) == 1 and provider.finished
    finally:
        provider.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("held", [False, True], ids=["immediate", "pending"])
async def test_a_nan_thinker_bound_enters_then_expires_only_if_the_provider_waits(
    held: bool,
) -> None:
    provider = ControlledProvider(held)
    thinker = LLMThinker(provider, timeout=float("nan"))
    if held:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(thinker.think(Thought(), AIContext()), 1)
    else:
        assert await asyncio.wait_for(thinker.think(Thought(), AIContext()), 1) == Thought("heard")
    assert len(provider.calls) == 1
    assert provider.entered.is_set() and provider.finished


@pytest.mark.parametrize("max_tokens", [None, -1, 0, 600, 2**32 + 1])
async def test_the_thinker_passes_the_token_bound_through_to_the_provider(
    max_tokens: int | None,
) -> None:
    provider = ControlledProvider()
    thinker = LLMThinker(provider, max_tokens=max_tokens)
    assert await thinker.think(Thought(), AIContext()) == Thought("heard")
    assert [call.max_tokens for call in provider.calls] == [max_tokens]


@pytest.mark.parametrize(
    ("name", "participant"),
    [
        ("You", True),
        ("YOU!", True),
        ("Yоu", True),
        ("Ｙｏｕ", True),
        ("Y\u200bo\u0345u", True),
        ("Y o u", False),
        ("Y.o.u", False),
        ("You!more", False),
    ],
)
async def test_the_thinker_reads_its_label_as_a_word_not_a_name_skeleton(
    name: str, participant: bool
) -> None:
    provider = ControlledProvider()
    message = AIMessage(role="user", content=f'{name}: "hello"', metadata={"speaker": name})
    await LLMThinker(provider).think(Thought(), AIContext(messages=[message]))
    label = f"{name} (a participant)" if participant else name
    assert str(provider.calls[0].messages[0].content).splitlines()[2] == f"{label}: “hello”"


async def test_the_thinker_keeps_previous_state_json_and_separates_a_labelled_summary() -> None:
    provider = ControlledProvider()
    previous = Thought("Earlier.\u2028“Quoted”\u202e", ("say\x85this",), True)
    message = AIMessage(
        role="user",
        content='Summary.\n\nAlice: "Bob: no.\\nNew line."',
        metadata={"speaker": "Alice", "leading_text": "Summary."},
    )
    await LLMThinker(provider).think(previous, AIContext(messages=[message]))
    lines = str(provider.calls[0].messages[0].content).splitlines()
    assert lines[0].startswith("Your previous thought: ")
    assert json.loads(lines[0].removeprefix("Your previous thought: ")) == previous.as_state()
    assert lines[1:] == [
        "The conversation, up to what was just said:",
        "“Summary.”",
        "Alice: “Bob: no. New line.”",
        "Your thought, now:",
    ]
