"""A tool call's intent: the sentence an agent's model writes with a call for
the person, carried apart from its arguments (RFC §6.4).

With ``tool_call_intents`` on, the model is shown each tool's schema with the
reserved ``_roomkit_intent`` property; the loop takes it out of the call's
arguments before anything reads them and carries it as the call's ``intent``:
on its hooks, its ephemeral START and END, its rows and its replays.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.models.steering import Cancel
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolCall,
    AIToolCallPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.tool_intent import INTENT_MAX_CHARS, normalized_intent
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tools.context import _ToolLoopContext
from tests.test_framework import SimpleChannel

INTENT = "Je recherche les fichiers texte du projet."
COMMAND = {"command": "find . -name '*.txt'"}
_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}
SHELL = AITool(name="shell", description="Run a command", parameters=_COMMAND_SCHEMA)
SHELL_CLOSED = AITool(
    name="shell_closed",
    description="Run a command",
    parameters={**_COMMAND_SCHEMA, "additionalProperties": False},
)


def _calling(*calls: tuple[str, str, dict[str, Any]]) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=i, name=n, arguments=a) for i, n, a in calls],
    )


def _with(intent: Any, **arguments: Any) -> dict[str, Any]:
    return {**COMMAND, **arguments, "_roomkit_intent": intent}


@dataclass
class _Seen:
    """What one turn showed: to the provider, the handler, the hooks, the
    realtime bus and the store."""

    provider: MockAIProvider
    received: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    before: list[ToolCallEvent] = field(default_factory=list)
    reported: list[ToolCallEvent] = field(default_factory=list)
    ephemeral: list[EphemeralEvent] = field(default_factory=list)
    rows: list[tuple[EventType, ToolCallContent]] = field(default_factory=list)

    def shown(self, name: str) -> dict[str, Any]:
        """The schema of tool *name* in the first request."""
        return next(t.parameters for t in self.provider.calls[0].tools if t.name == name)

    def payloads(self, kind: EphemeralEventType) -> list[dict[str, Any]]:
        return [call for e in self.ephemeral if e.type == kind for call in e.data["tool_calls"]]

    def row_intents(self) -> list[tuple[str, str, str | None]]:
        return [(kind.value, c.tool_id, c.intent) for kind, c in self.rows]


async def _turn(
    streaming: bool,
    *responses: AIResponse,
    tools: list[AITool] = (SHELL, SHELL_CLOSED),  # type: ignore[assignment]
    intents: bool = True,
    required: bool = False,
    only: list[str] | None = None,
    deny: bool = False,
    failing: bool = False,
) -> _Seen:
    """One message into a room whose agent answers with *responses*."""
    provider = MockAIProvider(
        ai_responses=[*responses, AIResponse(content="done")], streaming=streaming
    )
    seen = _Seen(provider)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        seen.received.append((name, dict(arguments)))
        if failing:
            raise RuntimeError("broken")
        return "a.txt"

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=provider,
            tools=list(tools),
            tool_handler=handler,
            tool_call_intents=intents,
            tool_call_intents_required=required,
            tool_call_intents_only=only,
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="before")
    async def before(event: ToolCallEvent, ctx: Any) -> HookResult:
        seen.before.append(event)
        return HookResult.block("not now") if deny else HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.reported.append(event)

    async def listen(event: EphemeralEvent) -> None:
        seen.ephemeral.append(event)

    await kit.realtime.subscribe_to_room("r1", listen)
    try:
        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
        for row in await kit.store.list_events("r1"):
            if isinstance(row.content, ToolCallContent):
                seen.rows.append((row.type, row.content))
    finally:
        # Closing waits for the async observers the turn started.
        await kit.close()
    return seen


async def test_off_nothing_is_taken_from_a_call(streaming: bool) -> None:
    """Off, nothing is taken: the schemas are the tools', the reserved key is
    an argument like any other, and no payload or row names an intent."""
    seen = await _turn(streaming, _calling(("c1", "shell", _with(INTENT))), intents=False)

    assert seen.shown("shell") == _COMMAND_SCHEMA
    assert seen.received == [("shell", _with(INTENT))]
    assert all("intent" not in p for p in seen.payloads(EphemeralEventType.TOOL_CALL_START))
    assert all("intent" not in c.model_dump() for _, c in seen.rows)


@pytest.mark.parametrize("tool", [SHELL, SHELL_CLOSED], ids=["open schema", "closed schema"])
async def test_a_call_s_intent_rides_apart_from_its_arguments(
    streaming: bool, tool: AITool
) -> None:
    seen = await _turn(streaming, _calling(("c1", tool.name, _with(f"  {INTENT}\n"))))

    shown = seen.shown(tool.name)
    assert shown["properties"]["_roomkit_intent"]["type"] == "string"
    # Every tool the turn shows, the channel's own included.
    every = seen.provider.calls[0].tools
    assert all("_roomkit_intent" in t.parameters["properties"] for t in every)
    assert shown["required"] == ["command"]
    assert tool.parameters == {
        **_COMMAND_SCHEMA,
        **({"additionalProperties": False} if tool is SHELL_CLOSED else {}),
    }
    assert seen.received == [(tool.name, COMMAND)]
    assert [(e.arguments, e.intent) for e in seen.before] == [(COMMAND, INTENT)]
    assert [(e.arguments, e.intent, e.is_error) for e in seen.reported] == [
        (COMMAND, INTENT, False)
    ]
    assert [p["intent"] for p in seen.payloads(EphemeralEventType.TOOL_CALL_START)] == [INTENT]
    assert [p["intent"] for p in seen.payloads(EphemeralEventType.TOOL_CALL_END)] == [INTENT]
    assert [(c.arguments, c.intent) for _, c in seen.rows] == [(COMMAND, INTENT)] * 2


async def test_a_required_intent_is_asked_and_a_call_without_one_still_runs(
    streaming: bool,
) -> None:
    """RMK-1067: with ``tool_call_intents_required``, every schema the model
    is shown lists the reserved property in ``required``; the requirement
    asks the model, it refuses no call: one without an intent runs and
    carries none, as roomkit-rs's."""
    seen = await _turn(
        streaming,
        _calling(("c1", "shell", _with(INTENT))),
        _calling(("c2", "shell", COMMAND)),
        required=True,
    )

    for request in seen.provider.calls:
        assert {
            t.name: t.parameters["required"]
            for t in request.tools
            if t.name in {"shell", "shell_closed"}
        } == {
            "shell": ["command", "_roomkit_intent"],
            "shell_closed": ["command", "_roomkit_intent"],
        }
    assert SHELL.parameters["required"] == ["command"]
    assert seen.received == [("shell", COMMAND), ("shell", COMMAND)]
    assert [e.intent for e in seen.before] == [INTENT, None]


async def test_only_the_tools_named_ask_for_an_intent(streaming: bool) -> None:
    """``tool_call_intents_only`` shows the property, required or not, on the
    tools it names alone; a call to another that carries it anyway runs, its
    intent taken out before its closed schema reads it."""
    seen = await _turn(
        streaming,
        _calling(("c1", "shell_closed", _with(INTENT))),
        required=True,
        only=["shell"],
    )

    tools = {t.name: t.parameters for t in seen.provider.calls[0].tools}
    assert "_roomkit_intent" in tools["shell"]["properties"]
    assert tools["shell"]["required"] == ["command", "_roomkit_intent"]
    assert "_roomkit_intent" not in tools["shell_closed"]["properties"]
    assert tools["shell_closed"]["required"] == ["command"]
    assert seen.received == [("shell_closed", COMMAND)]
    assert [e.intent for e in seen.before] == [INTENT]


async def test_a_requirement_without_intents_changes_no_schema(streaming: bool) -> None:
    """Off, ``tool_call_intents_required`` shows the tools as they are."""
    seen = await _turn(streaming, _calling(("c1", "shell", COMMAND)), intents=False, required=True)

    assert seen.shown("shell") == _COMMAND_SCHEMA


def _replayed_calls(context: Any) -> list[dict[str, Any]]:
    """The arguments of each call a request replays."""
    return [
        part.arguments
        for message in context.messages
        if message.role == "assistant" and not isinstance(message.content, str)
        for part in message.content
        if isinstance(part, AIToolCallPart)
    ]


async def test_the_next_round_reads_its_call_with_its_intent(streaming: bool) -> None:
    """The intent goes back among the call's arguments, normalized, so its
    next call keeps the shape it was shown."""
    seen = await _turn(streaming, _calling(("c1", "shell", _with(f" {INTENT} "))))

    assert _replayed_calls(seen.provider.calls[-1]) == [_with(INTENT)]


@pytest.mark.parametrize("intents", [True, False], ids=["intents on", "intents off"])
async def test_a_later_turn_reads_the_call_with_its_intent_while_asked(
    streaming: bool, intents: bool
) -> None:
    """From its rows, as the turn that made it read it; the arguments alone
    once the agent no longer asks for intents."""
    provider = MockAIProvider(
        ai_responses=[
            _calling(("c1", "shell", _with(INTENT))),
            AIResponse(content="done"),
            AIResponse(content="again"),
        ],
        streaming=streaming,
    )

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "a.txt"

    def agent(on: bool) -> AIChannel:
        return AIChannel(
            "ai1", provider=provider, tools=[SHELL], tool_handler=handler, tool_call_intents=on
        )

    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(agent(True))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    await kit.process_inbound(message)
    # The agent restarted with the option as the test sets it, its rows kept.
    gone = kit.unregister_channel("ai1")
    assert gone is not None
    await gone.close()
    kit.register_channel(agent(intents))
    await kit.process_inbound(message.model_copy(update={"content": TextContent(body="more")}))
    await kit.close()

    assert _replayed_calls(provider.calls[-1]) == [_with(INTENT) if intents else COMMAND]


async def test_a_call_a_stop_kept_from_running_ends_with_its_intent(streaming: bool) -> None:
    """A stop while the round's calls are announced: none runs, each ends
    with its intent."""
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(
            ai_responses=[_calling(("c1", "shell", _with(INTENT))), AIResponse(content="done")],
            streaming=streaming,
        ),
        tools=[SHELL],
        tool_call_intents=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    loop = _ToolLoopContext(room_id="r1")
    loop.all_context_tools = [SHELL]
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[SHELL])

    ends = []
    async for delta in channel._run_streaming_tool_loop(context, parent_loop_ctx=loop):
        if isinstance(delta, ToolCallStartMarker):
            channel.steer(Cancel(reason="user stopped"))
        elif isinstance(delta, ToolCallEndMarker):
            ends.append((delta.intent, delta.outcome))
    await kit.close()

    assert ends == [(INTENT, "cancelled")]


async def test_a_tool_that_declares_the_property_keeps_it_as_its_argument(
    streaming: bool,
) -> None:
    owner = AITool(
        name="label",
        description="Label a file",
        parameters={
            "type": "object",
            "properties": {"_roomkit_intent": {"type": "string", "enum": ["keep", "drop"]}},
        },
    )
    seen = await _turn(
        streaming, _calling(("c1", "label", {"_roomkit_intent": "keep"})), tools=[owner]
    )

    assert seen.shown("label") == owner.parameters
    assert seen.received == [("label", {"_roomkit_intent": "keep"})]
    assert [c.intent for _, c in seen.rows] == [None, None]


MISSPELLED = "roomkit_intent"


@pytest.mark.parametrize("tool", [SHELL, SHELL_CLOSED], ids=["open schema", "closed schema"])
async def test_the_property_written_without_its_underscore_is_taken_as_the_intent(
    streaming: bool, tool: AITool
) -> None:
    """A model that drops the leading underscore writes ``roomkit_intent``:
    the loop takes it as the property, so a closed schema does not refuse the
    call and the person still reads what it does."""
    seen = await _turn(streaming, _calling(("c1", tool.name, {**COMMAND, MISSPELLED: INTENT})))

    assert seen.received == [(tool.name, COMMAND)]
    assert [(e.arguments, e.intent) for e in seen.before] == [(COMMAND, INTENT)]
    assert [(c.arguments, c.intent) for _, c in seen.rows] == [(COMMAND, INTENT)] * 2


async def test_the_property_wins_over_its_misspelling_and_neither_reaches_the_tool(
    streaming: bool,
) -> None:
    arguments = {**_with(INTENT), MISSPELLED: "Autre chose."}
    seen = await _turn(streaming, _calling(("c1", "shell_closed", arguments)))

    assert seen.received == [("shell_closed", COMMAND)]
    assert [c.intent for _, c in seen.rows] == [INTENT, INTENT]


async def test_a_tool_that_declares_the_misspelled_name_keeps_it_as_its_argument(
    streaming: bool,
) -> None:
    owner = AITool(
        name="note",
        description="Write a note",
        parameters={"type": "object", "properties": {MISSPELLED: {"type": "string"}}},
    )
    seen = await _turn(streaming, _calling(("c1", "note", {MISSPELLED: "keep"})), tools=[owner])

    assert seen.received == [("note", {MISSPELLED: "keep"})]
    assert [c.intent for _, c in seen.rows] == [None, None]


async def test_off_the_misspelled_name_is_an_argument_like_any_other(streaming: bool) -> None:
    arguments = {**COMMAND, MISSPELLED: INTENT}
    seen = await _turn(streaming, _calling(("c1", "shell", arguments)), intents=False)

    assert seen.received == [("shell", arguments)]


@pytest.mark.parametrize(
    ("value", "intent"),
    [
        (INTENT, INTENT),
        ("  Je  cherche\n\tles fichiers ", "Je cherche les fichiers"),
        ("Je\x00cherche\x1b[31m", "Je cherche [31m"),
        ("\u202eJe cherche\u2066", "Je cherche"),
        ("é" * (INTENT_MAX_CHARS + 5), "é" * INTENT_MAX_CHARS),
        ("a" * (INTENT_MAX_CHARS - 1) + " b", "a" * (INTENT_MAX_CHARS - 1)),
        ("   ", None),
        ("", None),
        (None, None),
        (42, None),
        (["Je cherche"], None),
        ({"text": "Je cherche"}, None),
    ],
    ids=[
        "plain",
        "whitespace",
        "control characters",
        "bidi controls",
        "too long",
        "cut before a space",
        "blank",
        "empty",
        "null",
        "number",
        "list",
        "object",
    ],
)
def test_an_intent_is_normalized_or_none(value: Any, intent: str | None) -> None:
    assert normalized_intent(value) == intent


async def test_an_unusable_intent_never_fails_the_call(streaming: bool) -> None:
    seen = await _turn(streaming, _calling(("c1", "shell_closed", _with({"nested": True}))))

    assert seen.received == [("shell_closed", COMMAND)]
    assert [(c.intent, c.outcome) for kind, c in seen.rows if kind == EventType.TOOL_CALL_END] == [
        (None, "served")
    ]


@pytest.mark.parametrize(
    ("call", "outcome", "kwargs"),
    [
        (_with(INTENT), "refused", {"deny": True}),
        (_with(INTENT, command=3), "refused", {}),
        (_with(INTENT), "failed", {"failing": True}),
    ],
    ids=["BEFORE_TOOL_USE denies", "invalid arguments", "handler raises"],
)
async def test_a_call_refused_or_failed_keeps_its_intent(
    streaming: bool, call: dict[str, Any], outcome: str, kwargs: dict[str, Any]
) -> None:
    """Its report, its END payload and its end row carry it; its outcome is
    the engine's, never inferred from the intent."""
    seen = await _turn(streaming, _calling(("c1", "shell_closed", call)), **kwargs)

    assert [(e.intent, e.is_error) for e in seen.reported] == [(INTENT, True)]
    assert [p["intent"] for p in seen.payloads(EphemeralEventType.TOOL_CALL_END)] == [INTENT]
    assert [(c.intent, c.outcome) for kind, c in seen.rows if kind == EventType.TOOL_CALL_END] == [
        (INTENT, outcome)
    ]
    assert "_roomkit_intent" not in str([c.arguments for _, c in seen.rows])


async def test_two_calls_under_one_id_keep_their_own_intents(streaming: bool) -> None:
    """A round's second call under an id another holds is a call of its own
    (RFC §12.4): refused, with its own intent, matched by place, not by id."""
    seen = await _turn(
        streaming,
        _calling(("c1", "shell", _with("Je liste.")), ("c1", "shell", _with("Je compte."))),
    )

    assert [p["intent"] for p in seen.payloads(EphemeralEventType.TOOL_CALL_START)] == [
        "Je liste.",
        "Je compte.",
    ]
    assert [p["intent"] for p in seen.payloads(EphemeralEventType.TOOL_CALL_END)] == [
        "Je liste.",
        "Je compte.",
    ]
    assert sorted((c.intent, c.outcome) for k, c in seen.rows if k == EventType.TOOL_CALL_END) == [
        ("Je compte.", "refused"),
        ("Je liste.", "served"),
    ]


async def test_a_call_cut_while_it_runs_ends_with_its_intent(streaming: bool) -> None:
    """A turn cancelled while the handler runs closes the call from its start,
    intent included."""
    started = asyncio.Event()

    async def slow(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        await asyncio.sleep(30)
        return "late"

    provider = MockAIProvider(
        ai_responses=[_calling(("c1", "shell", _with(INTENT))), AIResponse(content="done")],
        streaming=streaming,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1", provider=provider, tools=[SHELL], tool_handler=slow, tool_call_intents=True
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    published: list[EphemeralEvent] = []

    async def listen(event: EphemeralEvent) -> None:
        published.append(event)

    await kit.realtime.subscribe_to_room("r1", listen)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(started.wait(), 5)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    rows = [
        (e.type, e.content.intent, e.content.outcome)
        for e in await kit.store.list_events("r1")
        if isinstance(e.content, ToolCallContent)
    ]
    assert rows == [
        (EventType.TOOL_CALL_START, INTENT, None),
        (EventType.TOOL_CALL_END, INTENT, "cancelled"),
    ]
    await kit.close()
    # The live END the cut round publishes for the call it aborted.
    ended = [e for e in published if e.type == EphemeralEventType.TOOL_CALL_END]
    assert [(c["intent"], c["status"]) for e in ended for c in e.data["tool_calls"]] == [
        (INTENT, "failed")
    ]


def test_a_row_without_an_intent_has_no_intent_key() -> None:
    """Read from a row that has none, written without one, and its fields
    still in the serialization schema an integrator's OpenAPI reads."""
    row = ToolCallContent.model_validate(
        {"type": "tool_call", "tool_name": "shell", "tool_id": "c1", "arguments": COMMAND}
    )

    assert row.intent is None
    assert "intent" not in row.model_dump()
    with_one = row.model_copy(update={"intent": INTENT})
    assert ToolCallContent.model_validate_json(with_one.model_dump_json()).intent == INTENT
    schema = ToolCallContent.model_json_schema(mode="serialization")
    assert {"tool_name", "arguments", "intent"} <= set(schema["properties"])
