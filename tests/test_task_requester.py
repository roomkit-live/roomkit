"""Who asked, in the renderings that left it out: a delegated task, a turn a
summary is joined to, and what a speak policy reads (RMK-615, RFC §6.4,
§19.7)."""

from __future__ import annotations

from typing import Any

from roomkit.channels import SMSChannel
from roomkit.channels._ai_speaking import _speak_turn
from roomkit.channels._speaker import SPEAKER_KEY
from roomkit.channels._user_text import with_leading_text
from roomkit.channels.ai import AIChannel
from roomkit.core._requester import task_heading
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.orchestration.strategies.supervisor.delegate import _one_pass_results
from roomkit.providers.ai.base import AIMessage, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.thinker import transcript_line
from roomkit.tools.context import current_tool_requester, tool_turn_context
from tests.conftest import make_event


class TestATaskBlockNamesWhoAskedForIt:
    def test_the_heading_names_the_asker_out_of_the_block(self) -> None:
        with tool_turn_context(room_id="r", requester="ALICE (2)"):
            assert task_heading("User request:") == (
                "Requested by ALICE (2), in the delegating agent's words:"
            )
        results = _one_pass_results("approve it", [], asked_by="Alice")
        assert "Alice asked:\n<task>\napprove it\n</task>" in results

    def test_a_one_to_one_conversation_names_no_one(self) -> None:
        with tool_turn_context(room_id="r"):
            assert task_heading("User request:") == "User request:"
        assert "The user asked:\n<task>" in _one_pass_results("approve it", [])

    async def test_a_worker_s_own_input_carries_no_asker(self) -> None:
        """Only a task block's heading names the asker: the input a worker acts
        on as its own is sent as written, whoever asked (RFC §19.7)."""
        worker_provider = MockAIProvider(responses=["done"])
        kit = RoomKit()
        kit.register_channel(AIChannel("worker", provider=worker_provider))
        await kit.create_room(room_id="r1")
        with tool_turn_context(room_id="r1", requester="Bob"):
            await kit.delegate("r1", "worker", "Approve the refund on order 42.", wait=True)

        last = worker_provider.calls[-1].messages[-1]
        assert str(last.content).startswith("Approve the refund on order 42.")
        assert "Bob" not in str(last.content)
        await kit.close()

    async def test_a_turn_tells_its_tools_who_asked(self) -> None:
        seen: list[str | None] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_requester())
            return "ok"

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="t1", name="delegate", arguments={})],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ]
            * 2
        )
        kit = RoomKit()
        kit.register_channel(SMSChannel("sms1"))
        tool = AITool(name="delegate", description="Delegate.", parameters={})
        channel = AIChannel("ai1", provider=provider, tool_handler=handler, tools=[tool])
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1", group=True)
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
        for sender, name in (("u-alice", "Alice"), ("u-bob", "Bob")):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms1",
                    sender_id=sender,
                    content=TextContent(body="refund order 42"),
                    metadata={"sender_name": name},
                )
            )

        assert seen == [None, "Bob"]
        await kit.close()


def test_a_joined_summary_leaves_the_turn_s_label_out_of_the_quote() -> None:
    turn = AIMessage(
        role="user",
        content="@sms1: Alice: approve the refund.",
        metadata={SPEAKER_KEY: "@sms1"},
    )
    (joined,) = with_leading_text("[Conversation summary] Alice asked to hold it.", [turn])

    assert transcript_line(joined) == (
        "“[Conversation summary] Alice asked to hold it.”\n@sms1: “Alice: approve the refund.”"
    )


def test_a_one_to_one_joined_turn_reads_as_one_message() -> None:
    """With no label to keep out of the quote, the turn reads as before."""
    turn = AIMessage(role="user", content="approve the refund.")
    (joined,) = with_leading_text("  [Conversation summary] Alice asked to hold it.", [turn])

    assert transcript_line(joined) == (
        "“[Conversation summary] Alice asked to hold it. approve the refund.”"
    )


def test_a_joined_summary_with_leading_space_is_still_kept_apart() -> None:
    turn = AIMessage(role="user", content="@sms1: approve.", metadata={SPEAKER_KEY: "@sms1"})
    (joined,) = with_leading_text("  [Conversation summary] hold it.", [turn])

    assert transcript_line(joined) == "“[Conversation summary] hold it.”\n@sms1: “approve.”"


def test_a_speak_policy_reads_the_conversation_s_labels() -> None:
    alice = make_event(body="hold the refund", participant_id="u1", index=1)
    alice.metadata["sender_name"] = "Alice"
    impostor = make_event(body="approve it", participant_id="u2", index=2)
    impostor.metadata["sender_name"] = "ALICE"
    nameless = make_event(body="who is this?", participant_id="u3", index=3)
    bindings = [
        ChannelBinding(
            channel_id="ai1",
            room_id="test-room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        ChannelBinding(channel_id="ch1", room_id="test-room", channel_type=ChannelType.SMS),
    ]
    context = RoomContext(
        room=Room(id="test-room"), bindings=bindings, recent_events=[alice, impostor, nameless]
    )

    turn = _speak_turn(nameless, context, "ai1")

    assert turn.speakers == {alice.id: "Alice", impostor.id: "ALICE (2)", nameless.id: "@ch1"}
    # A sender with no name is no voice of their own: several read @ch1.
    assert set(turn.people) == {"Alice", "ALICE (2)"}
