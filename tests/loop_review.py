"""A Loop reviewer for tests: it judges through ``submit_review`` (RFC §19.7.4)."""

from __future__ import annotations

import json

from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

#: A review task's result that approves: an approving ``submit_review`` call's payload.
APPROVAL = json.dumps({"approved": True, "feedback": ""})


def verdict(approved: bool, feedback: str = "", *, call_id: str = "review") -> list[AIResponse]:
    """One review's answers: the ``submit_review`` call, then a closing line."""
    call = AIToolCall(
        id=call_id,
        name="submit_review",
        arguments={"approved": approved, "feedback": feedback},
    )
    return [AIResponse(content="", tool_calls=[call]), AIResponse(content="Reviewed.")]


def review_answers(*reviews: str) -> MockAIProvider:
    """A reviewer giving *reviews* in turn, round-robin: ``APPROVED`` approves,
    any other text refuses with that text as its feedback."""
    answers = [
        answer
        for at, review in enumerate(reviews, start=1)
        for answer in verdict(
            review == "APPROVED",
            "" if review == "APPROVED" else review,
            call_id=f"review-{at}",
        )
    ]
    return MockAIProvider(ai_responses=answers)
