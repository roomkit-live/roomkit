"""Tests for the Loop orchestration strategy (orchestration/strategies/loop.py).

Extends existing test_strategy_loop.py to cover uncovered lines:
loop execution, reviewer strategies, async delivery, edge cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.loop import (
    Loop,
    _execute_loop,
    _review_parallel,
    _review_sequential,
    _run_loop,
    _run_reviewers,
)
from roomkit.orchestration.strategies.supervisor import WorkerStrategy
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult

# -- Helpers ------------------------------------------------------------------


class _NoopLock:
    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_agent(
    channel_id: str,
    responses: list[str] | None = None,
    role: str | None = None,
) -> Agent:
    return Agent(
        channel_id=channel_id,
        provider=MockAIProvider(responses=responses or ["ok"]),
        role=role,
    )


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.channels = {}
    kit.register_channel = MagicMock()
    kit.deliver = AsyncMock()
    return kit


def _make_event(
    body: str = "Hello",
    channel_id: str = "user-ch",
    channel_type: ChannelType = ChannelType.SMS,
    room_id: str = "r1",
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=TextContent(body=body),
    )


def _make_binding(room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id="user-ch",
        room_id=room_id,
        channel_type=ChannelType.SMS,
    )


def _make_context(room_id: str = "r1") -> RoomContext:
    return RoomContext(room=Room(id=room_id))


def _delegated_task_with_output(output: str) -> MagicMock:
    task = MagicMock()
    task.result = DelegatedTaskResult(
        task_id="t1",
        child_room_id="child-1",
        parent_room_id="r1",
        agent_id="agent-1",
        output=output,
    )
    return task


def _delegated_task_empty() -> MagicMock:
    task = MagicMock()
    task.result = None
    return task


# -- Tests: Constructor -------------------------------------------------------


class TestLoopConstructor:
    def test_both_reviewer_and_reviewers_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Loop(
                agent=_make_agent("w"),
                reviewers=[_make_agent("r1")],
                reviewer=_make_agent("r2"),
            )

    def test_no_reviewer_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one reviewer"):
            Loop(agent=_make_agent("w"))

    def test_single_reviewer_convenience(self) -> None:
        reviewer = _make_agent("r")
        loop = Loop(agent=_make_agent("w"), reviewer=reviewer)
        assert len(loop._reviewers) == 1
        assert loop._reviewers[0] is reviewer

    def test_multiple_reviewers(self) -> None:
        reviewers = [_make_agent("r1"), _make_agent("r2")]
        loop = Loop(agent=_make_agent("w"), reviewers=reviewers)
        assert len(loop._reviewers) == 2

    def test_strategy_parsed(self) -> None:
        loop = Loop(
            agent=_make_agent("w"),
            reviewer=_make_agent("r"),
            strategy="parallel",
        )
        assert loop._strategy == WorkerStrategy.PARALLEL

    def test_agents_empty_when_async_delivery(self) -> None:
        loop = Loop(
            agent=_make_agent("w"),
            reviewer=_make_agent("r"),
            async_delivery=True,
        )
        assert loop.agents() == []

    def test_agents_returns_producer(self) -> None:
        producer = _make_agent("w")
        loop = Loop(agent=producer, reviewer=_make_agent("r"))
        assert loop.agents() == [producer]


# -- Tests: Install -----------------------------------------------------------


class TestLoopInstall:
    async def test_install_registers_reviewers(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        kit.register_channel.assert_called_once_with(editor)

    async def test_install_wraps_on_event(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        original = writer.on_event
        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        assert writer.on_event is not original

    async def test_install_sets_initial_state(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor, max_iterations=4)
        await loop.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.context["_loop_max_iterations"] == 4
        assert state.context["_loop_iteration"] == 0
        assert state.context["_loop_approved"] is False

    async def test_install_async_registers_producer(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor, async_delivery=True)
        await loop.install(kit, "r1")

        # Producer registered when async and not already in channels
        calls = kit.register_channel.call_args_list
        channel_ids = [c[0][0].channel_id for c in calls]
        assert "writer" in channel_ids


# -- Tests: Wrapped on_event -------------------------------------------------


class TestWrappedOnEvent:
    async def test_different_room_delegates_to_original(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        # Event from a different room should go to original on_event
        event = _make_event(room_id="r2")
        ctx = _make_context(room_id="r2")
        result = await writer.on_event(event, _make_binding("r2"), ctx)
        # Should have gone through original (AIChannel.on_event) which produces output
        assert isinstance(result, ChannelOutput)

    async def test_event_from_producer_returns_empty(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        event = _make_event(channel_id="writer", channel_type=ChannelType.AI)
        result = await writer.on_event(event, _make_binding(), _make_context())
        assert not result.responded

    async def test_ai_event_goes_to_original(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        event = _make_event(channel_id="other-ai", channel_type=ChannelType.AI)
        result = await writer.on_event(event, _make_binding(), _make_context())
        assert isinstance(result, ChannelOutput)

    async def test_system_event_goes_to_original(self) -> None:
        writer = _make_agent("writer")
        editor = _make_agent("editor")
        kit = _make_mock_kit(Room(id="r1"))

        loop = Loop(agent=writer, reviewer=editor)
        await loop.install(kit, "r1")

        event = _make_event(channel_id="system", channel_type=ChannelType.SYSTEM)
        result = await writer.on_event(event, _make_binding(), _make_context())
        assert isinstance(result, ChannelOutput)


# -- Tests: _run_loop ---------------------------------------------------------


class TestRunLoop:
    async def test_run_loop_returns_output(self) -> None:
        """Full loop: producer → reviewer (APPROVED) → returns output."""
        kit = _make_mock_kit(Room(id="r1"))
        producer = _make_agent("writer")
        reviewer = _make_agent("editor")

        producer_task = _delegated_task_with_output("Great content")
        reviewer_task = _delegated_task_with_output("APPROVED")
        kit.delegate = AsyncMock(side_effect=[producer_task, reviewer_task])

        event = _make_event(body="Write something")
        result = await _run_loop(
            kit=kit,
            room_id="r1",
            producer=producer,
            reviewers=[reviewer],
            strategy=None,
            event=event,
            max_iterations=3,
        )

        assert result.responded
        assert len(result.response_events) == 1
        assert "Great content" in result.response_events[0].content.body

    async def test_run_loop_empty_content_returns_empty(self) -> None:
        """Non-text event should return empty output."""
        from roomkit.models.event import MediaContent

        kit = _make_mock_kit(Room(id="r1"))
        producer = _make_agent("writer")
        reviewer = _make_agent("editor")

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )
        result = await _run_loop(
            kit=kit,
            room_id="r1",
            producer=producer,
            reviewers=[reviewer],
            strategy=None,
            event=event,
            max_iterations=3,
        )

        assert not result.responded


# -- Tests: _execute_loop ---------------------------------------------------


class TestExecuteLoop:
    async def test_producer_empty_output_breaks(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_empty())

        result = await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[_make_agent("editor")],
            strategy=None,
            task_desc="Write something",
            max_iterations=3,
        )

        assert result["approved"] is False
        assert result["output"] == ""

    async def test_all_reviewers_approve(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        call_count = 0

        async def mock_delegate(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _delegated_task_with_output("My content")
            return _delegated_task_with_output("APPROVED")

        kit.delegate = AsyncMock(side_effect=mock_delegate)

        result = await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[_make_agent("editor")],
            strategy=None,
            task_desc="Write",
            max_iterations=3,
        )

        assert result["approved"] is True
        assert result["output"] == "My content"
        assert result["iteration"] == 1

    async def test_revision_cycle(self) -> None:
        """Reviewer rejects first, approves second."""
        kit = _make_mock_kit(Room(id="r1"))
        call_seq = [
            _delegated_task_with_output("Draft 1"),  # producer iter 1
            _delegated_task_with_output("Needs work"),  # reviewer iter 1 (reject)
            _delegated_task_with_output("Draft 2"),  # producer iter 2
            _delegated_task_with_output("APPROVED"),  # reviewer iter 2 (approve)
        ]
        kit.delegate = AsyncMock(side_effect=call_seq)

        result = await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[_make_agent("editor")],
            strategy=None,
            task_desc="Write",
            max_iterations=5,
        )

        assert result["approved"] is True
        assert result["iteration"] == 2
        assert result["output"] == "Draft 2"

    async def test_max_iterations_reached(self) -> None:
        """When max iterations hit, returns last output as not approved."""
        kit = _make_mock_kit(Room(id="r1"))

        async def always_reject(*args: Any, **kwargs: Any) -> MagicMock:
            # Alternate: producer output, then reject
            if kwargs.get("wait"):
                return _delegated_task_with_output("Some output")
            return _delegated_task_with_output("Some output")

        # Producer always returns content, reviewer never approves
        call_seq = []
        for _ in range(3):
            call_seq.append(_delegated_task_with_output("Content"))
            call_seq.append(_delegated_task_with_output("Needs more work"))
        kit.delegate = AsyncMock(side_effect=call_seq)

        result = await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[_make_agent("editor")],
            strategy=None,
            task_desc="Write",
            max_iterations=3,
        )

        assert result["approved"] is False
        assert result["iteration"] == 3

    async def test_updates_room_state(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("Content"),
                _delegated_task_with_output("APPROVED"),
            ]
        )

        await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[_make_agent("editor")],
            strategy=None,
            task_desc="Write",
            max_iterations=3,
        )

        # State should be updated on the room
        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.context["_loop_approved"] is True


# -- Tests: _run_reviewers ---------------------------------------------------


class TestRunReviewers:
    async def test_sequential_single_reviewer(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        reviewer = _make_agent("editor", role="Editor")
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("APPROVED"))

        results = await _run_reviewers(kit, "r1", [reviewer], None, "content")

        assert len(results) == 1
        assert results[0]["approved"] is True
        assert results[0]["reviewer"] == "Editor"

    async def test_sequential_multiple_reviewers(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        r1 = _make_agent("r1", role="Security")
        r2 = _make_agent("r2", role="Style")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("Fix SQL injection"),
                _delegated_task_with_output("APPROVED"),
            ]
        )

        results = await _run_reviewers(kit, "r1", [r1, r2], WorkerStrategy.SEQUENTIAL, "content")

        assert len(results) == 2
        assert results[0]["approved"] is False
        assert results[1]["approved"] is True

    async def test_parallel_reviewers(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        r1 = _make_agent("r1", role="Security")
        r2 = _make_agent("r2", role="Style")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("APPROVED"),
                _delegated_task_with_output("APPROVED"),
            ]
        )

        results = await _run_reviewers(kit, "r1", [r1, r2], WorkerStrategy.PARALLEL, "content")

        assert len(results) == 2
        assert all(r["approved"] for r in results)


# -- Tests: _review_sequential -----------------------------------------------


class TestReviewSequential:
    async def test_feedback_chains_to_next_reviewer(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        r1 = _make_agent("r1", role="First")
        r2 = _make_agent("r2", role="Second")

        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("Fix this issue"),
                _delegated_task_with_output("APPROVED"),
            ]
        )

        _results = await _review_sequential(kit, "r1", [r1, r2], "Review this")

        # The second reviewer's call should include the first reviewer's feedback
        second_call_args = kit.delegate.call_args_list[1]
        task_input = second_call_args[0][2]  # positional arg for task desc
        assert "First" in task_input
        assert "Fix this issue" in task_input

    async def test_empty_output_not_approved(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        reviewer = _make_agent("r1")
        kit.delegate = AsyncMock(return_value=_delegated_task_empty())

        results = await _review_sequential(kit, "r1", [reviewer], "Review this")
        assert results[0]["approved"] is False

    async def test_reviewer_uses_role_for_name(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        reviewer = _make_agent("r1", role="QA Lead")
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("APPROVED"))

        results = await _review_sequential(kit, "r1", [reviewer], "Review")
        assert results[0]["reviewer"] == "QA Lead"

    async def test_reviewer_without_role_uses_channel_id(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        reviewer = _make_agent("r1")
        reviewer.role = None
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("APPROVED"))

        results = await _review_sequential(kit, "r1", [reviewer], "Review")
        assert results[0]["reviewer"] == "r1"


# -- Tests: _review_parallel -------------------------------------------------


class TestReviewParallel:
    async def test_all_reviewers_run_concurrently(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        r1 = _make_agent("r1", role="A")
        r2 = _make_agent("r2", role="B")

        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("APPROVED"),
                _delegated_task_with_output("Fix something"),
            ]
        )

        results = await _review_parallel(kit, "r1", [r1, r2], "Review this")

        assert len(results) == 2
        assert results[0]["approved"] is True
        assert results[1]["approved"] is False

    async def test_parallel_empty_result(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        reviewer = _make_agent("r1")
        kit.delegate = AsyncMock(return_value=_delegated_task_empty())

        results = await _review_parallel(kit, "r1", [reviewer], "Review")
        assert results[0]["approved"] is False
        assert results[0]["feedback"] == ""


# -- Tests: Feedback combination in _execute_loop ----------------------------


class TestFeedbackCombination:
    async def test_multiple_reviewer_feedback_combined(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        r1 = _make_agent("r1", role="Security")
        r2 = _make_agent("r2", role="Style")

        # Iteration 1: producer output, then both reviewers reject
        # Iteration 2: producer revised, then both approve
        call_seq = [
            _delegated_task_with_output("Draft 1"),  # producer
            _delegated_task_with_output("SQL injection"),  # r1 rejects
            _delegated_task_with_output("Bad naming"),  # r2 rejects
            _delegated_task_with_output("Draft 2"),  # producer revised
            _delegated_task_with_output("APPROVED"),  # r1 approves
            _delegated_task_with_output("APPROVED"),  # r2 approves
        ]
        kit.delegate = AsyncMock(side_effect=call_seq)

        result = await _execute_loop(
            kit=kit,
            room_id="r1",
            producer=_make_agent("writer"),
            reviewers=[r1, r2],
            strategy=None,  # Sequential
            task_desc="Write code",
            max_iterations=5,
        )

        assert result["approved"] is True
        assert result["iteration"] == 2

        # The revision input (call 4 = index 3) should contain both feedbacks
        revision_call = kit.delegate.call_args_list[3]
        revision_input = revision_call[0][2]
        assert "Security" in revision_input
        assert "Style" in revision_input
