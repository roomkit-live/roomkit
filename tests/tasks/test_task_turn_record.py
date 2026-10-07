"""A delegated task carries its worker's turn record under ``turns`` however
the turn ended, and a failed one keeps the error its turn raised, its type
unchanged (RMK-529, RFC §23.3 step 6)."""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.enums import TaskStatus
from roomkit.providers.ai.base import AIContext, AIResponse, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult
from tests.test_framework import SimpleChannel

DOORS = pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])


class _Unavailable(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        raise ProviderError("upstream unavailable", provider="mock", status_code=503)


async def _delegated(provider: MockAIProvider, *, wait: bool) -> DelegatedTaskResult:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("worker", provider=provider))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    task = await kit.delegate("r", "worker", "Do it.", wait=wait)
    result = await task.wait(timeout=5.0)
    await kit.close()
    return result


@DOORS
async def test_a_task_carries_its_workers_turn(wait: bool) -> None:
    answer = AIResponse(content="Done.", usage={"input_tokens": 4, "output_tokens": 2})
    result = await _delegated(MockAIProvider(ai_responses=[answer]), wait=wait)

    assert result.status == TaskStatus.COMPLETED
    assert result.metadata["turns"]["worker"]["loop_end_reason"] == "completed"
    assert result.metadata["turns"]["worker"]["ai_usage"]["output_tokens"] == 2
    assert result.exception is None


@DOORS
async def test_a_failed_task_keeps_the_error_its_turn_raised(wait: bool) -> None:
    result = await _delegated(_Unavailable(), wait=wait)

    assert result.status == TaskStatus.FAILED
    assert isinstance(result.exception, ProviderError)
    assert result.exception.status_code == 503
    assert "exception" not in result.model_dump()
