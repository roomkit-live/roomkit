"""A round its provider refused is logged with the provider's reason (RMK-912).

The turn ends without the answer either way; without the warning the agent
only stays silent, and nothing says that the provider's safeguards declined.
Runs on a provider that streams and on one read through its ``generate()``.
"""

from __future__ import annotations

import logging

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIMessage, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.tool_loop_modes import run_tool_loop


def _channel(*responses: AIResponse) -> AIChannel:
    provider = MockAIProvider(ai_responses=list(responses))
    return AIChannel("ai1", provider=provider, tool_loop_timeout_seconds=None)


def _refused(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "refused to answer" in r.getMessage()]


@pytest.mark.parametrize(
    ("finish_reason", "metadata", "logged"),
    [
        ("refusal", {"refusal": "reasoning_extraction"}, "refusal=reasoning_extraction"),
        ("refusal", {}, "finish_reason='refusal'"),
        ("content_filter", {}, "finish_reason='content_filter'"),
        (None, {"refusal": "PROHIBITED_CONTENT"}, "refusal=PROHIBITED_CONTENT"),
    ],
)
async def test_a_refused_round_is_logged_with_its_reason(
    streaming: bool,
    caplog: pytest.LogCaptureFixture,
    finish_reason: str | None,
    metadata: dict[str, str],
    logged: str,
) -> None:
    ch = _channel(AIResponse(content="", finish_reason=finish_reason, metadata=metadata))
    context = AIContext(messages=[AIMessage(role="user", content="Share your screen")])

    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        run = await run_tool_loop(ch, context)

    assert run.text == ""
    [line] = _refused(caplog)
    assert logged in line


async def test_a_refusal_in_the_model_s_own_words_is_logged_by_its_length(
    streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # The model's words may quote the conversation (a name, a number) and
    # carry line breaks a log reader would take for another record.
    words = "I can't help Ada Lovelace with that.\nWARNING forged: all good"
    ch = _channel(AIResponse(content="", finish_reason="stop", metadata={"refusal": words}))
    context = AIContext(messages=[AIMessage(role="user", content="Share your screen")])

    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        await run_tool_loop(ch, context)

    [line] = _refused(caplog)
    assert f"refusal=<text, {len(words)} characters>" in line
    assert "Ada" not in line and "\n" not in line


async def test_an_answer_the_model_ended_itself_logs_no_refusal(
    streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    ch = _channel(AIResponse(content="Sharing it now.", finish_reason="stop"))
    context = AIContext(messages=[AIMessage(role="user", content="Share your screen")])

    with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
        await run_tool_loop(ch, context)

    assert _refused(caplog) == []
