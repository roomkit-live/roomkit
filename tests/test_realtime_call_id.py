"""A realtime call's id is the one its provider hands on, never an argument
(RFC §12.4).

A model may write an argument under the name the service gives a call's id.
The channel keeps the two apart: the argument reaches the tool as any other,
judged as any (§9.3), and the result goes back under the call's own id.
"""

from __future__ import annotations

from typing import Any

from roomkit.voice.realtime import MockRealtimeProvider
from tests.channel_tasks import settled
from tests.test_realtime_tool_recovery import _session

TOOL: list[dict[str, Any]] = [
    {
        "name": "lookup",
        "description": "Look up a booking",
        "parameters": {
            "type": "object",
            "properties": {"call_id": {"type": "string"}, "id": {"type": "string"}},
        },
    }
]


async def test_an_argument_named_like_the_call_s_id_stays_an_argument() -> None:
    seen: list[dict[str, Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        seen.append(arguments)
        return "booked"

    provider = MockRealtimeProvider()
    kit, channel, session = await _session(provider, "rt-call-id", handler=handler, tools=TOOL)

    await provider.simulate_tool_call(
        session, "service-1", "lookup", {"call_id": "model-written", "id": "also-model"}
    )
    await settled(channel)

    assert seen == [{"call_id": "model-written", "id": "also-model"}]
    assert [call_id for _session_id, call_id, _result in provider.tool_results] == ["service-1"]
    await kit.close()
