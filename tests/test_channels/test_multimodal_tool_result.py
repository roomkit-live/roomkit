"""A multimodal tool result survives the unified dispatcher intact.

``AIToolResultPart.result`` has accepted a content-part list (text + images,
e.g. a screenshot) since 0.50 — but the unified dispatcher's user-handler
branch coerced every result through ``str()``, flattening the list to its
Python repr (base64 included) before it could reach the provider. These
tests pin the whole path: dispatcher → tool loop → provider context.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIImagePart,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

_SCREENSHOT_TOOL = {
    "name": "take_screenshot",
    "description": "Capture the screen.",
    "parameters": {"type": "object", "properties": {}},
}

_PARTS = [
    AITextPart(text="here"),
    AIImagePart(url="data:image/png;base64,AAAA", mime_type="image/png"),
]


async def _handler(name: str, arguments: dict) -> list[AITextPart | AIImagePart]:
    return list(_PARTS)


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_SCREENSHOT_TOOL]},
    )


def _responses() -> list[AIResponse]:
    return [
        AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="t1", name="take_screenshot", arguments={})],
        ),
        AIResponse(content="done", finish_reason="stop"),
    ]


async def test_a_part_list_result_reaches_the_provider_intact() -> None:
    provider = MockAIProvider(ai_responses=_responses())
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)

    await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    final_context = provider.calls[-1]
    tool_messages = [m for m in final_context.messages if m.role == "tool"]
    assert tool_messages, "the tool round's results must be in the next context"
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    # The list of parts, not its repr: a str here means the dispatcher
    # flattened the screenshot into prose.
    assert result_part.result == _PARTS


async def test_streaming_loop_carries_the_part_list_too() -> None:
    provider = MockAIProvider(ai_responses=_responses(), streaming=True)
    ch = AIChannel("ai1", provider=provider, tool_handler=_handler)

    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    assert output.response_stream is not None
    async for _ in output.response_stream:
        pass

    final_context = provider.calls[-1]
    tool_messages = [m for m in final_context.messages if m.role == "tool"]
    assert tool_messages
    result_part = tool_messages[-1].content[0]
    assert isinstance(result_part, AIToolResultPart)
    assert result_part.result == _PARTS
