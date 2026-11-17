"""A retained report claim must not consume another turn's public observation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.test_framework import SimpleChannel

from roomkit import (
    AIChannel,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, AIToolResultPart, MockAIProvider
from roomkit.tools import turn_report_claim


async def test_an_old_callback_does_not_suppress_the_next_turns_observer(streaming: bool) -> None:
    retained: list[Callable[[], bool]] = []
    old_claims: list[bool] = []
    reported: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        if retained:
            old_claims.append(retained[0]())
        retained.append(claim)
        return str(arguments["query"])

    responses = []
    for query in ("first", "second"):
        responses.extend(
            [
                AIResponse(
                    content="",
                    tool_calls=[
                        AIToolCall(id="c1", name="lookup", arguments={"query": query}),
                    ],
                ),
                AIResponse(content="done"),
            ]
        )
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)
    channel = AIChannel(
        "ai",
        provider=provider,
        tool_handler=handler,
        tools=[AITool(name="lookup", description="Lookup")],
    )
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "ai")

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def observed(event: ToolCallEvent, context: Any) -> None:
            reported.append(str(event.arguments["query"]))

        for message in ("one", "two"):
            await kit.process_inbound(
                InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=message)),
                room_id="r1",
            )
    # Both tools completed for the model; the observation is a separate contract.
    results = [
        part.result
        for request in (provider.calls[1], provider.calls[3])
        for message in request.messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]
    assert "first" in results and "second" in results
    assert reported == ["first", "second"]
    assert old_claims == [False]
