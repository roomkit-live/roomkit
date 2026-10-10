"""The discussion console: what it shows of a room a discussion holds, and what it sends.

The view is pure state; the screen runs without a terminal (a pipe for input,
a dummy output) on a real kit, with scripted agents.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from roomkit import Agent, Discussion, RoomKit
from roomkit.console import AgentCard, DiscussionConsole
from roomkit.console._discussion_screen import STYLE
from roomkit.console._discussion_view import AGENT_STYLES, DiscussionView
from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
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
    assert not any(
        h.name.startswith("discussion_console_") for h in kit.hook_engine._room_hooks["r1"]
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
