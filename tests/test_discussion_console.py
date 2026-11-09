"""The discussion console: what it shows of a room a discussion holds, and what it sends.

The view is pure state; the screen runs without a terminal (a pipe for input,
a dummy output) on a real kit, with scripted agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from roomkit import Agent, ChannelDeclaration, Discussion, InMemoryTurnBackend, RoomKit
from roomkit.console import AgentCard, DiscussionConsole
from roomkit.console._discussion_screen import STYLE
from roomkit.console._discussion_view import AGENT_STYLES, DiscussionView
from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tasks.background import BACKGROUND_TOOL
from roomkit.turns import TurnExecutor
from tests.test_framework import SimpleChannel

CARDS = (AgentCard("dev", role="Developer", model="m-1"), AgentCard("sre", description="Ops"))


def _event(change: SpeakQueueChange, *agents: str, **fields: object) -> SpeakQueueEvent:
    queue = {
        "speaking": None,
        "queue": (),
        "listening": frozenset(),
        "asked": (),
        "waiting": False,
        "over": False,
        **fields,
    }
    return SpeakQueueEvent(
        room_id="r1",
        queue=SpeakQueue(**queue),  # type: ignore[arg-type]
        change=change,
        channel_ids=agents,
    )


def _panel(view: DiscussionView) -> str:
    return "".join(text for _style, text in view.agent_fragments())


def _room(view: DiscussionView) -> str:
    return "\n".join(line for _style, line in view.lines)


# -- The view --


def test_the_panel_shows_each_agents_state_from_the_queue() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.queue_changed(
        _event(
            SpeakQueueChange.TURN_GIVEN,
            "dev",
            speaking="dev",
            queue=("sre",),
            listening=frozenset(),
        )
    )
    panel = _panel(view)
    assert "@dev  speaking" in panel and "@sre  next" in panel
    assert "Developer" in panel and "m-1" in panel and "Ops" in panel
    assert view.status_text() == "speaking @dev · next @sre"

    view.queue_changed(
        _event(SpeakQueueChange.LISTENING, "sre", listening=frozenset({"sre"}), waiting=True)
    )
    assert "@sre  listening" in _panel(view)
    assert view.status_text() == "nobody speaking · waiting for a person"
    assert "listening only: @sre" in _room(view)


def test_a_turn_that_delivered_nothing_reads_as_nothing_to_add() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.queue_changed(_event(SpeakQueueChange.TURN_GIVEN, "dev", speaking="dev"))
    view.agent("dev", ["sre"], "@sre can you check?")
    view.queue_changed(_event(SpeakQueueChange.TURN_ENDED, "dev"))
    view.queue_changed(_event(SpeakQueueChange.TURN_GIVEN, "sre", speaking="sre"))
    view.queue_changed(_event(SpeakQueueChange.TURN_ENDED, "sre"))

    room = _room(view)
    assert "@dev · m-1 → @sre" in room
    assert "@sre has nothing to add" in room and "@dev has nothing to add" not in room
    assert view.said["dev"] == 1


def test_an_agent_asking_the_person_is_called_out_once() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.queue_changed(_event(SpeakQueueChange.TURN_ENDED, "sre", asked=(("sre", "ops"),)))
    view.queue_changed(_event(SpeakQueueChange.WAITING, asked=(("sre", "ops"),), waiting=True))

    assert _room(view).count(">>> @sre is asking you") == 1
    assert "@sre  asked @ops" in _panel(view)


def test_people_and_tools_are_shown_and_counted() -> None:
    view = DiscussionView(CARDS, you="ops")
    view.person("ops", None, "what happened?", mine=True)
    view.person("Alice", ["dev"], "@dev and you?")
    view.person("ops", None, "I am ops too")
    view.tool("dev", "query_logs", {"level": "ERROR"}, "line one\nline two")

    room = _room(view)
    assert "@ops (you) → the room" in room and "Alice → @dev" in room
    # A name is no proof: only the console says which messages are yours.
    assert room.count("(you)") == 1 and "ops → the room" in room
    assert "@dev ⚙ query_logs(level='ERROR') → line one | line two" in room
    assert view.used["dev"] == 1


def test_the_palette_has_a_color_for_each_agent_style() -> None:
    defined = {name for name, _ in STYLE.style_rules}
    assert {f"agent{i}" for i in range(AGENT_STYLES)} <= defined


# -- The console on a room --


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield pipe


def _agent(handle: str, *answers: str, role: str | None = None) -> Agent:
    provider = MockAIProvider(ai_responses=[AIResponse(content=a) for a in answers])
    return Agent(handle, provider=provider, role=role, description=f"{handle} agent")


async def _discussion_room(kit: RoomKit) -> None:
    kit.register_channel(SimpleChannel("ops"))
    agents = [_agent("dev", "@sre please check", role="Developer"), _agent("sre", "all clear")]
    await kit.create_room(room_id="r1", orchestration=Discussion(agents))
    await kit.attach_channel("r1", "ops")


async def _until(predicate: object, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.01)


async def test_what_the_person_types_goes_to_the_room_and_the_answers_come_back(
    pipe: PipeInput,
) -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    console = DiscussionConsole(kit, "r1", channel_id="ops", title="test room")
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)

    pipe.send_text("@dev status?\r")
    await _until(lambda: "all clear" in _room(console.view))

    room = _room(console.view)
    assert "@ops (you) → @dev" in room
    assert "@dev · mock → @sre" in room and "@sre · mock → the room" in room
    cards = {c.handle: c for c in console.view.cards}
    assert cards["dev"].role == "Developer" and cards["sre"].description == "sre agent"

    pipe.send_text("/listen @sre\r")
    await _until(lambda: "listening only: @sre" in _room(console.view))
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.listening == frozenset({"sre"})

    pipe.send_text("/quit\r")
    await asyncio.wait_for(running, 5)
    # Closed, it leaves nothing behind in the room's hooks.
    for engine in (kit.hook_engine, kit.turn_executor.hooks):
        assert not any(
            h.name.startswith("discussion_console_") for h in engine._room_hooks.get("r1", [])
        )
    await kit.close()


async def test_host_commands_and_unknown_ones(pipe: PipeInput) -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    seen: list[str] = []

    async def mission(rest: str) -> None:
        seen.append(rest)

    console = DiscussionConsole(kit, "r1", channel_id="ops", commands={"mission": mission})
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)
    assert "/mission" in _room(console.view)

    pipe.send_text("/mission now\r")
    pipe.send_text("/nope\r")
    await _until(lambda: "unknown command /nope" in _room(console.view))
    assert seen == ["now"]

    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


async def test_logs_go_to_the_file_while_the_screen_is_up(pipe: PipeInput, tmp_path: Path) -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    log_file = tmp_path / "console.log"
    root = logging.getLogger()
    before = list(root.handlers)
    console = DiscussionConsole(kit, "r1", channel_id="ops", log_file=log_file)
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)

    logging.getLogger("roomkit.test").warning("kept out of the screen")
    console.exit()
    await asyncio.wait_for(running, 5)

    assert "kept out of the screen" in log_file.read_text()
    assert root.handlers == before
    await kit.close()


# -- What an agent does, wherever it runs (RMK-1612) --


def _agent_with_a_tool() -> Agent:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="c1", name="query_logs", arguments={"level": "ERROR"})],
            ),
            AIResponse(content="all clear"),
        ]
    )

    async def query_logs(name: str, arguments: dict[str, object]) -> str:
        return "line one\nline two"

    return Agent(
        "dev",
        provider=provider,
        tools=[AITool(name="query_logs", description="Reads the logs.", parameters={})],
        tool_handler=query_logs,
    )


@pytest.mark.parametrize("apart", [False, True], ids=["on_the_kit", "on_another_executor"])
async def test_the_console_shows_the_tools_an_agent_calls(pipe: PipeInput, apart: bool) -> None:
    backend = InMemoryTurnBackend()
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("ops"))
    agent = _agent_with_a_tool()
    executor = None
    if apart:
        kit.register_channel(ChannelDeclaration(channel_id="dev"))
        executor = TurnExecutor(
            lambda _channel: agent, hosts={"dev"}, executor_id="elsewhere", store=kit.store
        )
        await backend.serve(executor)
    await kit.create_room(
        room_id="r1", orchestration=Discussion(agents=["dev"] if apart else [agent])
    )
    await kit.attach_channel("r1", "ops")
    console = DiscussionConsole(kit, "r1", channel_id="ops")
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)

    pipe.send_text("status?\r")
    view = console.view
    await _until(lambda: "all clear" in _room(view) and view.used["dev"] > 0)

    assert "@dev ⚙ query_logs(level='ERROR') → line one | line two" in _room(view)
    assert view.used["dev"] == 1
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()
    if executor is not None:
        await executor.close()


async def test_a_background_tool_call_makes_the_task_the_calling_agents(pipe: PipeInput) -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    console = DiscussionConsole(kit, "r1", channel_id="ops")
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)
    view = console.view

    async def tool_ended(result: str) -> None:
        call = {"id": "c1", "name": BACKGROUND_TOOL, "result": result, "status": "completed"}
        await kit.realtime.publish_to_room(
            "r1",
            EphemeralEvent(
                room_id="r1",
                type=EphemeralEventType.TOOL_CALL_END,
                user_id="dev",
                channel_id="dev",
                data={"tool_calls": [call]},
            ),
        )

    await tool_ended(json.dumps({"task_id": "task-abc123"}))
    await _until(lambda: view.used["dev"] == 1)
    view.task_started("task-abc123", "worker", "go through the tickets")
    assert view.working["task-abc123"] == "dev"

    console.exit()
    await asyncio.wait_for(running, 5)
    witnessed = asyncio.Event()

    async def witness(_event: EphemeralEvent) -> None:
        witnessed.set()

    await kit.realtime.subscribe_to_room("r1", witness)
    await tool_ended("after the console closed")
    await asyncio.wait_for(witnessed.wait(), 5)
    await asyncio.sleep(0)
    assert view.used["dev"] == 1, "closed, the console follows nothing"
    await kit.close()


async def test_a_turn_that_failed_reads_as_failed(pipe: PipeInput) -> None:
    kit = RoomKit(turn_backend=InMemoryTurnBackend())
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(ChannelDeclaration(channel_id="dev"))
    await kit.create_room(room_id="r1", orchestration=Discussion(agents=["dev"]))
    await kit.attach_channel("r1", "ops")
    console = DiscussionConsole(kit, "r1", channel_id="ops")
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)

    pipe.send_text("status?\r")
    view = console.view
    # The speaking turn is over too: what its end shows has been shown.
    await _until(
        lambda: "failed" in _room(view) and view.queue is not None and view.queue.speaking is None
    )

    room = _room(view)
    assert "@dev failed: No executor serves channel 'dev'" in room
    assert "nothing to add" not in room
    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


# -- Agents addressed by handle (RMK-764) --

DEV_ID, SRE_ID = "agent:8e41f0d2", "agent:c5a3e7b1"
"""Channel ids no message can name: the agents are addressed by handle."""

HANDLED = (
    AgentCard("Dev", role="Developer", model="m-1", channel_id=DEV_ID),
    AgentCard("SRE", channel_id=SRE_ID),
)


def test_the_view_names_agents_by_handle_and_follows_them_by_channel_id() -> None:
    view = DiscussionView(HANDLED, you="ops")
    view.queue_changed(
        _event(SpeakQueueChange.TURN_GIVEN, DEV_ID, speaking=DEV_ID, queue=(SRE_ID,))
    )
    view.agent(DEV_ID, [SRE_ID], "@sre can you check?")
    view.tool(DEV_ID, "query_logs", {}, "ok")
    view.queue_changed(_event(SpeakQueueChange.TURN_ENDED, DEV_ID, queue=(SRE_ID,)))
    view.queue_changed(_event(SpeakQueueChange.TURN_GIVEN, SRE_ID, speaking=SRE_ID))
    view.queue_changed(_event(SpeakQueueChange.TURN_ENDED, SRE_ID, asked=((SRE_ID, "ops"),)))
    view.queue_changed(
        _event(
            SpeakQueueChange.LISTENING,
            DEV_ID,
            listening=frozenset({DEV_ID}),
            asked=((SRE_ID, "ops"),),
        )
    )
    view.task_started("task-abc123", DEV_ID, "go through the tickets")

    room, panel = _room(view), _panel(view)
    assert "@Dev · m-1 → @SRE" in room and "@Dev ⚙ query_logs" in room
    assert "@SRE has nothing to add" in room and ">>> @SRE is asking you" in room
    assert "listening only: @Dev" in room and "started: @Dev" in room
    assert "@Dev  listening" in panel and "@SRE  asked @ops" in panel
    assert view.said[DEV_ID] == 1 and view.used[DEV_ID] == 1
    assert "agent:" not in room + panel


def test_a_card_without_a_channel_id_is_its_handle() -> None:
    card = AgentCard("dev")
    assert card.channel == "dev"
    view = DiscussionView((card,), you="ops")
    view.queue_changed(_event(SpeakQueueChange.TURN_GIVEN, "dev", speaking="dev"))
    assert view.status_text() == "speaking @dev"


async def test_the_console_reads_and_shows_the_discussions_handles(pipe: PipeInput) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    dev = Agent(DEV_ID, provider=MockAIProvider(ai_responses=[]), name="Dev", role="Developer")
    sre = Agent(SRE_ID, provider=MockAIProvider(ai_responses=[]), name="SRE")
    await kit.create_room(room_id="r1", orchestration=Discussion([dev, sre]))
    await kit.attach_channel("r1", "ops")
    console = DiscussionConsole(kit, "r1", channel_id="ops")
    running = asyncio.create_task(console.run())
    await _until(lambda: console._view is not None)

    assert [(c.handle, c.channel) for c in console.view.cards] == [
        ("Dev", DEV_ID),
        ("SRE", SRE_ID),
    ]
    assert "@Dev @SRE" in _room(console.view) and "agent:" not in _room(console.view)
    pipe.send_text("/listen @dev\r")
    await _until(lambda: "listening only: @Dev" in _room(console.view))
    queue = kit.speak_queue("r1")
    assert queue is not None and queue.listening == frozenset({DEV_ID})
    assert "‖ @Dev  listening" in _panel(console.view)

    console.exit()
    await asyncio.wait_for(running, 5)
    await kit.close()


def test_a_posted_result_by_another_agent_names_it_by_handle() -> None:
    view = DiscussionView(HANDLED, you="ops")
    view.agent(DEV_ID, [], "Findings.", by=SRE_ID)
    assert "@Dev · a task's result, by @SRE → nobody" in _room(view)


@pytest.mark.parametrize(
    ("typed", "named"),
    [
        ("@dev", [DEV_ID]),
        ("dev", [DEV_ID]),
        ("@DEV:", [DEV_ID]),
        ("@dev,@sre", [DEV_ID, SRE_ID]),
        ("@all", [DEV_ID, SRE_ID]),
        ("@agent:8e41f0d2", []),
        ("@nobody", []),
    ],
)
def test_a_command_names_agents_as_the_discussion_reads_names(
    typed: str, named: list[str]
) -> None:
    console = DiscussionConsole(RoomKit(), "r1", channel_id="ops")
    console._view = DiscussionView(HANDLED, you="ops")
    assert console._named_agents(typed) == named
