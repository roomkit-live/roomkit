"""Plans reach only their room's actual messages, and leave on detach."""

from __future__ import annotations

from typing import Any

from roomkit import HookResult, HookTrigger
from roomkit.models.tool_call import AIGenerationEvent
from tests.planning_support import answer, message, plan, room


async def test_actual_messages_and_generation_hooks_keep_room_plans_apart(streaming: bool) -> None:
    kit, _, model = await room(
        [plan([{"title": "ALPHA_SECRET", "status": "pending"}]), answer(), answer(), answer()],
        streaming,
        rooms=("a", "b"),
    )
    seen: list[AIGenerationEvent] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def observe(event: AIGenerationEvent, context: Any) -> HookResult:
        seen.append(event)
        return HookResult.allow()

    try:
        await kit.process_inbound(message(), room_id="a")
        await kit.process_inbound(message("Other room."), room_id="b")
        await kit.process_inbound(message("Continue."), room_id="a")
        assert len(model.calls) == 4 and len(seen) == 3
        assert all("ALPHA_SECRET" not in str(m.content) for m in model.calls[2].messages)
        assert "ALPHA_SECRET" in str(model.calls[3].messages[-1].content)
        assert "## Current Task Plan" in str(model.calls[3].messages[-1].content)
        assert all(c.system_prompt == "Fixed." for c in model.calls)
        assert all("ALPHA_SECRET" not in str(m.content) for m in seen[1].ai_context.messages)
        assert "ALPHA_SECRET" in str(seen[2].ai_context.messages[-1].content)
    finally:
        await kit.close()


async def test_detaching_and_reattaching_forgets_notes_but_keeps_tool_history(
    streaming: bool,
) -> None:
    kit, _, model = await room(
        [plan([{"title": "Old plan", "status": "pending"}]), answer(), answer()], streaming
    )
    try:
        await kit.process_inbound(message())
        await kit.detach_channel("r", "agent")
        await kit.attach_channel("r", "agent")
        await kit.process_inbound(message(body="Again."))
        assert str(model.calls[2].messages[-1].content) == "Again."
        assert any("Old plan" in str(m.content) for m in model.calls[2].messages[:-1])
        assert model.calls[2].system_prompt == "Fixed."
    finally:
        await kit.close()


async def test_a_standalone_instruction_reads_no_plan_then_the_room_still_does(
    streaming: bool,
) -> None:
    from roomkit.models.enums import EventType
    from roomkit.models.event import TextContent

    kit, _, model = await room(
        [plan([{"title": "Room-only plan", "status": "pending"}]), answer(), answer(), answer()],
        streaming,
    )
    try:
        await kit.process_inbound(message())
        await kit.send_event(
            "r",
            "ws",
            TextContent(body="Alone."),
            event_type=EventType.INSTRUCTION,
            addressed_to=["agent"],
            standalone=True,
        )
        assert all("Room-only plan" not in str(m.content) for m in model.calls[2].messages)
        assert "Room-only plan" not in (model.calls[2].system_prompt or "")
        await kit.process_inbound(message("Continue."))
        assert "Room-only plan" in str(model.calls[3].messages[-1].content)
    finally:
        await kit.close()


async def test_empty_plan_removes_the_note_without_erasing_the_previous_call(
    streaming: bool,
) -> None:
    kit, _, model = await room(
        [
            plan([{"title": "Old plan", "status": "pending"}], "p0"),
            answer(),
            plan([]),
            answer(),
            answer(),
        ],
        streaming,
    )
    try:
        await kit.process_inbound(message())
        await kit.process_inbound(message("Clear."))
        await kit.process_inbound(message("Continue."))
        assert str(model.calls[4].messages[-1].content) == "Continue."
        assert any("Old plan" in str(m.content) for m in model.calls[4].messages[:-1])
    finally:
        await kit.close()
