"""The thinker's public bounds are applied when its provider is called."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import LLMThinker, Thought
from roomkit.providers.ai import AIContext, AIResponse, MockAIProvider


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


@pytest.mark.parametrize("timeout", [0.0, -0.1])
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


async def test_a_none_thinker_bound_waits_for_a_provider_that_was_really_entered() -> None:
    provider = ControlledProvider(held=True)
    thinker = LLMThinker(provider, timeout=None)
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


@pytest.mark.parametrize("max_tokens", [None, -1, 0, 600, 2**32 + 1])
async def test_the_thinker_passes_the_token_bound_through_to_the_provider(
    max_tokens: int | None,
) -> None:
    provider = ControlledProvider()
    thinker = LLMThinker(provider, max_tokens=max_tokens)
    assert await thinker.think(Thought(), AIContext()) == Thought("heard")
    assert [call.max_tokens for call in provider.calls] == [max_tokens]
