"""Public text-only witnesses for the process-wide content logging switch."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from roomkit import content_logging_enabled, set_content_logging
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("  TrUe  ", True),
        ("enabled", False),
        ("\x1c\x1dTrUe\x1e\x1f", True),
        ("tr\x1cue", False),
    ],
)
def test_environment_is_read_once_and_the_public_setter_overrides_it(value, expected):
    env = dict(os.environ)
    env.pop("ROOMKIT_LOG_CONTENT", None)
    if value is not None:
        env["ROOMKIT_LOG_CONTENT"] = value
    code = """
import json, os
from roomkit import content_logging_enabled, set_content_logging
first = content_logging_enabled()
os.environ['ROOMKIT_LOG_CONTENT'] = 'false' if first else 'true'
second = content_logging_enabled()
set_content_logging(not first)
print(json.dumps([first, second, content_logging_enabled()]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == [expected, expected, not expected]


@pytest.mark.parametrize("answer", ["synthetic-result", "é\n" * 300 + "tail"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
async def test_public_turn_keeps_payloads_out_of_info_and_gates_debug(
    caplog, streaming, enabled, level, answer
):
    argument = "synthetic-input"
    original = content_logging_enabled()
    set_content_logging(enabled)
    caplog.set_level(level, logger="roomkit.channels.ai")
    seen = []

    async def handler(name, arguments):
        seen.append((name, dict(arguments)))
        return answer

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="t1", name="lookup", arguments={"query": argument})],
            ),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[
            AITool(
                name="lookup",
                description="Synthetic lookup",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        tool_handler=handler,
    )
    try:
        run = await respond(
            channel,
            make_event(room_id="r1", body="synthetic-request"),
            ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI),
            RoomContext(room=Room(id="r1")),
        )
        assert run.text == "done"
        assert seen == [("lookup", {"query": argument})]
        assert run.calls[0].arguments == {"query": argument}
        assert run.calls[0].result == answer
        lines = [
            record.getMessage()
            for record in caplog.records
            if record.name == "roomkit.channels.ai"
        ]
        assert "Executing tool lookup (call t1) with query" in lines
        assert any(
            line.startswith(f"Tool lookup returned {len(answer)} chars in ") for line in lines
        )
        debug = enabled and level == "DEBUG"
        assert any(argument in line for line in lines) is debug
        preview = answer.replace("\n", " ")
        if len(preview) > 500:
            preview = f"{preview[:500]}… ({len(preview)} chars)"
        assert (f"Tool lookup result: {preview}" in lines) is debug
        if level == "DEBUG" and not enabled:
            assert "Tool lookup arguments: <redacted:28 chars>" in lines
            assert f"Tool lookup result: <redacted:{len(preview)} chars>" in lines
    finally:
        set_content_logging(original)
        await channel.close()
