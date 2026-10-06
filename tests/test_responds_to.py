"""An answer names the event it answers, and its turn is told that event (RFC §8.5)."""

from __future__ import annotations

import sqlite3
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import unanswered
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.orchestration.strategies.supervisor.delegate import _results_event
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.sqlite import _SCHEMA, _SCHEMA_VERSION, SQLiteStore
from tests.conference.test_conference_realtime import until
from tests.test_framework import AILikeChannel, SimpleChannel


class _ClockTool:
    @property
    def definition(self) -> dict[str, Any]:
        return {"name": "clock", "description": "", "parameters": {"type": "object"}}

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        return "12:00"


async def _room(kit: RoomKit, *intelligence: Any) -> None:
    kit.register_channel(SimpleChannel("sms"))
    for channel in intelligence:
        kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    for channel in intelligence:
        await kit.attach_channel("r", channel.channel_id, category=ChannelCategory.INTELLIGENCE)


async def _say(kit: RoomKit, text: str) -> RoomEvent:
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body=text)),
        room_id="r",
    )
    assert result.event is not None
    return result.event


async def _events(kit: RoomKit) -> list[RoomEvent]:
    return await kit.store.list_events("r", offset=0, limit=100)


async def test_a_buffered_answer_names_the_message_it_answers() -> None:
    kit = RoomKit()
    await _room(kit, AILikeChannel("ai"))
    asked = await _say(kit, "hello")
    events = await _events(kit)
    await kit.close()

    [answer] = [e for e in events if e.source.channel_id == "ai"]
    assert answer.responds_to == asked.id
    assert asked.responds_to is None  # a message from a person answers nothing


async def test_a_channel_that_names_what_it_answers_is_kept() -> None:
    class _Named(AILikeChannel):
        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            output = await super().on_event(event, binding, context)
            output.response_events = [
                output.response_events[0].model_copy(update={"responds_to": "earlier-event"})
            ]
            return output

    kit = RoomKit()
    await _room(kit, _Named("ai"))
    await _say(kit, "hello")
    events = await _events(kit)
    await kit.close()

    [answer] = [e for e in events if e.source.channel_id == "ai"]
    assert answer.responds_to == "earlier-event"


async def test_every_row_of_a_streamed_turn_names_the_message_it_answers() -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Je regarde.", tool_calls=[AIToolCall(id="c1", name="clock", arguments={})]
            ),
            AIResponse(content="Il est midi."),
        ]
    )
    kit = RoomKit()
    await _room(kit, AIChannel("ai", provider=provider, tools=[_ClockTool()]))
    asked = await _say(kit, "Quelle heure est-il ?")
    events = await _events(kit)
    await kit.close()

    rows = [e for e in events if e.source.channel_id == "ai"]
    kinds = {e.type for e in rows}
    assert {EventType.MESSAGE, EventType.TOOL_CALL_START, EventType.TOOL_CALL_END} <= kinds
    assert all(e.responds_to == asked.id for e in rows)


async def test_the_generation_hook_is_told_what_the_turn_answers() -> None:
    kit = RoomKit()
    await _room(kit, AIChannel("ai", provider=MockAIProvider(["ok"])))
    triggers: list[RoomEvent | None] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def seen(event: Any, ctx: Any) -> HookResult:
        triggers.append(event.trigger)
        return HookResult.allow()

    asked = await _say(kit, "hello")
    await kit.close()

    assert [t.id for t in triggers if t is not None] == [asked.id]


async def test_an_answer_to_an_instruction_names_the_instruction() -> None:
    kit = RoomKit()
    await _room(kit, AIChannel("ai", provider=MockAIProvider(["Voilà le résultat."])))
    instructions: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def seen(event: Any, ctx: Any) -> HookResult:
        instructions.append(event.id)
        return HookResult.allow()

    await kit.deliver("r", "Give the result.", instruction=True, addressed_to=["ai"])
    events = await _events(kit)
    await kit.close()

    [instruction_id] = instructions
    [answer] = [e for e in events if e.source.channel_id == "ai"]
    # The instruction is never stored: the answer names an id the timeline lacks.
    assert answer.responds_to == instruction_id
    assert all(e.id != instruction_id for e in events)


async def test_a_failed_turn_names_the_message_it_failed_to_answer() -> None:
    class _Failing(MockAIProvider):
        async def generate(self, context: AIContext) -> AIResponse:
            raise RuntimeError("provider down")

    kit = RoomKit()
    await _room(kit, AIChannel("ai", provider=_Failing()))
    errors: list[RoomEvent] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def failed(event: Any, ctx: Any) -> None:
        errors.append(event)

    asked = await _say(kit, "hello")
    await until(lambda: bool(errors), timeout=5)
    await kit.close()

    assert errors[0].responds_to == asked.id


def test_a_record_standing_for_an_answer_not_given_names_its_trigger() -> None:
    trigger = RoomEvent(
        room_id="r",
        content=TextContent(body="hi"),
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
    )
    assert unanswered(trigger, "ai", ChannelType.AI).responds_to == trigger.id


def test_a_supervisors_stand_in_names_the_event_it_stands_for() -> None:
    asked = RoomEvent(
        room_id="r",
        content=TextContent(body="plan the trip"),
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
    )
    assert _results_event(asked, "[workers' results]").responds_to == asked.id


async def test_the_answers_to_an_event_can_be_listed() -> None:
    kit = RoomKit()
    await _room(kit, AILikeChannel("ai"))
    first = await _say(kit, "first")
    await _say(kit, "second")
    answers = await kit.store.list_events(
        "r", offset=0, limit=10, event_filter=EventFilter(responds_to=first.id)
    )
    await kit.close()

    assert [(e.source.channel_id, e.responds_to) for e in answers] == [("ai", first.id)]


async def test_sqlite_keeps_and_finds_what_an_answer_answers(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "rk.db")
    kit = RoomKit(store=store)
    await _room(kit, AILikeChannel("ai"))
    asked = await _say(kit, "hello")
    answers = await store.list_events(
        "r", offset=0, limit=10, event_filter=EventFilter(responds_to=asked.id)
    )
    await kit.close()

    assert [e.responds_to for e in answers] == [asked.id]


async def test_a_v3_sqlite_file_gains_the_column_and_keeps_its_events(tmp_path) -> None:
    path = tmp_path / "v3.db"
    store = SQLiteStore(path)
    kit = RoomKit(store=store)
    await _room(kit, AILikeChannel("ai"))
    old = await _say(kit, "before the upgrade")
    await kit.close()
    # Back to v3: the events table as v3 wrote it, without the column.
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP INDEX IF EXISTS idx_events_room_responds_to;
        ALTER TABLE events DROP COLUMN responds_to;
        PRAGMA user_version=3;
    """)
    conn.close()

    store = SQLiteStore(path)
    kit = RoomKit(store=store)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(AILikeChannel("ai"))
    kept = await store.list_events("r", offset=0, limit=10)
    answers = await store.list_events(
        "r", offset=0, limit=10, event_filter=EventFilter(responds_to=old.id)
    )
    await kit.close()

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION == 4
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    finally:
        conn.close()
    assert "responds_to" in columns
    assert old.id in {e.id for e in kept}
    # The JSON of the event kept what it named; the new column was filled for
    # nothing written before it existed, so the filter reads only new rows.
    assert answers == []
    assert "responds_to TEXT" in _SCHEMA
