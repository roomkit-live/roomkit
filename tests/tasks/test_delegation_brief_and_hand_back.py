"""A delegation's label apart from its brief, the conversation handed over, and
a result relayed or posted (RFC §23.3 steps 4 and 8, RMK-668)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ChannelCategory, HookResult, HookTrigger, RoomKit, TaskCutShortError
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventStatus, EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult, TaskStatus
from roomkit.tasks.posting import posted_text
from tests.test_framework import SimpleChannel


class _Slow(MockAIProvider):
    """A worker that takes *seconds* to answer, to be cancelled meanwhile."""

    def __init__(self, seconds: float) -> None:
        super().__init__(responses=["late"])
        self.seconds = seconds

    async def generate(self, context: Any) -> Any:
        await asyncio.sleep(self.seconds)
        return await super().generate(context)


async def _kit(worker: MockAIProvider | None = None) -> tuple[RoomKit, AIChannel, Agent]:
    kit = RoomKit()
    assistant = AIChannel("assistant", provider=MockAIProvider(responses=["relayed"]))
    worker_agent = Agent("worker", provider=worker or MockAIProvider(responses=["Findings."]))
    kit.register_channel(assistant)
    kit.register_channel(worker_agent)
    kit.register_channel(SimpleChannel("phone"))
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "phone")
    await kit.attach_channel("call", "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit, assistant, worker_agent


def _read(agent: AIChannel) -> str:
    """Everything the agent's first turn read."""
    call = agent._provider.calls[0]
    return "\n".join(str(m.content) for m in call.messages)


async def _messages(kit: RoomKit, *, blocked: bool = False) -> list[RoomEvent]:
    events = await kit.store.list_events("call", event_filter=EventFilter(include_blocked=blocked))
    return [e for e in events if e.type == EventType.MESSAGE]


# -- The brief and the label -------------------------------------------------


async def test_the_worker_reads_the_brief_and_the_hand_back_names_the_task() -> None:
    kit, assistant, worker = await _kit()

    task = await kit.delegate(
        "call",
        "worker",
        "Why do Canadian checkouts fail?",
        brief="Read the tax engine's source, then answer in three lines.",
        notify="assistant",
    )
    await task.wait(timeout=5)
    for _ in range(100):
        if assistant._provider.calls:
            break
        await asyncio.sleep(0.01)

    read = _read(worker)
    assert "Read the tax engine's source" in read
    assert "Why do Canadian checkouts fail?" not in read
    told = _read(assistant)
    assert "“Why do Canadian checkouts fail?”" in told
    assert "Read the tax engine" not in told
    assert task.task == "Why do Canadian checkouts fail?"
    await kit.close()


async def test_without_a_brief_the_worker_reads_the_task() -> None:
    kit, _assistant, worker = await _kit()

    task = await kit.delegate("call", "worker", "Count the tickets.")
    await task.wait(timeout=5)

    assert "Count the tickets." in _read(worker)
    assert task.description == "Count the tickets."
    await kit.close()


# -- The conversation ---------------------------------------------------------


async def test_the_worker_is_handed_the_last_messages_quoted_as_data() -> None:
    kit, _assistant, worker = await _kit()
    for body in ("first", "second", 'third, with a " quote', "fourth"):
        await kit.send_event("call", "phone", TextContent(body=body), addressed_to=[])

    task = await kit.delegate("call", "worker", "Summarise.", notify="assistant", conversation=3)
    await task.wait(timeout=5)

    read = _read(worker)
    assert "data, not instructions" in read
    block = read[read.index("<conversation>") : read.index("</conversation>")]
    assert "first" not in block
    assert '"second"' in block and '"fourth"' in block
    assert '\\"' in block  # a JSON string: its quote cannot end it
    await kit.close()


async def test_the_conversation_holds_only_what_the_notified_channel_may_read() -> None:
    kit, _assistant, worker = await _kit()
    await kit.send_event("call", "phone", TextContent(body="for all"), addressed_to=[])
    await kit.send_event(
        "call", "phone", TextContent(body="internal note"), addressed_to=[], visibility="none"
    )

    task = await kit.delegate("call", "worker", "Summarise.", notify="assistant", conversation=5)
    await task.wait(timeout=5)

    read = _read(worker)
    assert "for all" in read
    assert "internal note" not in read
    await kit.close()


async def test_no_conversation_by_default() -> None:
    kit, _assistant, worker = await _kit()
    await kit.send_event("call", "phone", TextContent(body="earlier"), addressed_to=[])

    task = await kit.delegate("call", "worker", "Summarise.", notify="assistant")
    await task.wait(timeout=5)

    assert "<conversation>" not in _read(worker)
    await kit.close()


# -- A posted result ----------------------------------------------------------


async def test_a_posted_result_is_the_agent_s_message_without_a_turn_of_its_model() -> None:
    kit, assistant, _worker = await _kit()

    task = await kit.delegate(
        "call", "worker", "Count the tickets.", notify="assistant", hand_back="post"
    )
    await task.wait(timeout=5)
    for _ in range(100):
        if await _messages(kit):
            break
        await asyncio.sleep(0.01)

    (posted,) = await _messages(kit)
    assert posted.source.channel_id == "assistant"
    assert posted.content.body == "Result of the task “Count the tickets.”:\n\nFindings."
    assert posted.metadata["task_id"] == task.id
    assert posted.metadata["task_status"] == "completed"
    assert posted.addressed_to == []
    assert assistant._provider.calls == []
    await kit.close()


async def test_a_posted_result_crosses_before_broadcast() -> None:
    kit, _assistant, _worker = await _kit()

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, _context: RoomContext) -> HookResult:
        if (event.metadata or {}).get("task_id"):
            return HookResult.block("not here")
        return HookResult.allow()

    task = await kit.delegate("call", "worker", "Count.", notify="assistant", hand_back="post")
    await task.wait(timeout=5)
    for _ in range(100):
        if await _messages(kit, blocked=True):
            break
        await asyncio.sleep(0.01)

    (posted,) = await _messages(kit, blocked=True)
    assert posted.status == EventStatus.BLOCKED
    await kit.close()


async def test_a_cancelled_task_posts_that_it_was_cancelled() -> None:
    kit, _assistant, _worker = await _kit(_Slow(5.0))

    task = await kit.delegate("call", "worker", "Count.", notify="assistant", hand_back="post")
    await asyncio.sleep(0.05)
    assert await kit.cancel_task(task.id)
    for _ in range(100):
        if await _messages(kit):
            break
        await asyncio.sleep(0.01)

    (posted,) = await _messages(kit)
    assert posted.content.body == "The task “Count.” was cancelled."
    await kit.close()


def _result(status: TaskStatus, output: str | None = None, error: str | None = None) -> Any:
    return DelegatedTaskResult(
        task_id="t",
        child_room_id="c",
        parent_room_id="p",
        agent_id="w",
        status=status,
        output=output,
        error=error,
    )


def test_a_failed_task_posts_that_it_failed_without_its_error() -> None:
    text = posted_text(_result(TaskStatus.FAILED, error="Traceback: secret"), "Count.")
    assert text == "The task “Count.” failed: an error ended it."


def test_a_cut_task_posts_why_in_the_runtimes_words() -> None:
    cut = _result(TaskStatus.FAILED, error=TaskCutShortError.message_for("max_rounds"))
    cut.metadata["loop_end_reason"] = "max_rounds"
    assert posted_text(cut, "Count.") == (
        "The task “Count.” failed: it reached its limit of tool rounds before giving an answer."
    )


def test_a_completed_task_with_nothing_to_say_posts_nothing() -> None:
    assert posted_text(_result(TaskStatus.COMPLETED, output="  "), "Count.") is None


# -- What a delegation refuses ------------------------------------------------


async def test_post_is_refused_for_an_inline_task() -> None:
    kit, _assistant, _worker = await _kit()
    with pytest.raises(ValueError, match="background"):
        await kit.delegate(
            "call", "worker", "Count.", notify="assistant", hand_back="post", wait=True
        )
    await kit.close()


async def test_post_is_refused_for_a_channel_that_is_no_agent_of_the_room() -> None:
    kit, _assistant, _worker = await _kit()
    with pytest.raises(ValueError, match="intelligence channel attached"):
        await kit.delegate("call", "worker", "Count.", notify="phone", hand_back="post")
    with pytest.raises(ValueError, match="intelligence channel attached"):
        # The worker is no channel of the parent room.
        await kit.delegate("call", "worker", "Count.", hand_back="post")
    await kit.close()


async def test_an_unknown_hand_back_and_a_negative_conversation_are_refused() -> None:
    kit, _assistant, _worker = await _kit()
    with pytest.raises(ValueError, match="hand_back"):
        await kit.delegate("call", "worker", "Count.", hand_back="shout")
    with pytest.raises(ValueError, match="conversation"):
        await kit.delegate("call", "worker", "Count.", conversation=-1)
    await kit.close()
