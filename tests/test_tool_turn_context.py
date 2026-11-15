"""Tests for ``tool_turn_context``: a test describes the turn a handler runs under.

A handler called directly, outside any tool loop, reads ``None`` from every
accessor of ``roomkit.tools``. ``tool_turn_context`` installs the context a
tool loop would, from public arguments, and restores the previous one on the
way out. The realtime path installs its per-call context through the same
installer, so both are held to the same restoring rule here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager
from unittest.mock import MagicMock

import pytest

from roomkit import RoomKit, TextContent
from roomkit.channels._realtime_tool_calls import RealtimeToolCall
from roomkit.channels._realtime_tool_executor import serving_tool_call
from roomkit.models.room import Room
from roomkit.providers.ai.base import AITool
from roomkit.tools import (
    ToolCallContext,
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_chain_depth,
    current_tool_requester,
    current_tool_room,
    current_tool_room_id,
    tool_turn_context,
)
from roomkit.tools.context import _ToolLoopContext
from tests.test_framework import SimpleChannel


def _readers() -> tuple[object, ...]:
    return (
        current_tool_room_id(),
        current_tool_room(),
        current_tool_actor_id(),
        current_tool_allowed_names(),
        current_tool_call(),
        current_response_metadata(),
    )


_OUTSIDE = (None, None, None, None, None, None)


def _public(call: ToolCallContext) -> AbstractContextManager[None]:
    return tool_turn_context(room_id="r1", actor_id="alice", call=call)


def _realtime(call: ToolCallContext) -> AbstractContextManager[None]:
    realtime_call = RealtimeToolCall(
        session=MagicMock(), call_id=call.tool_call_id, name="lookup", arguments={}
    )
    loop_ctx = _ToolLoopContext(room_id="r1", actor_id="alice", has_turn=False)
    return serving_tool_call(realtime_call, call.channel_id, loop_ctx)


_INSTALLERS: dict[str, Callable[[ToolCallContext], AbstractContextManager[None]]] = {
    "public": _public,
    "realtime": _realtime,
}


class TestTheTurnItDescribes:
    def test_the_accessors_answer_the_turn_inside_and_none_outside(self) -> None:
        tools = [AITool(name="lookup", description="Look up"), AITool(name="pay", description="")]
        call = ToolCallContext(room_id="r1", tool_call_id="tc1", channel_id="ai")

        with tool_turn_context(room_id="r1", actor_id="alice", tools=tools, call=call):
            assert current_tool_room_id() == "r1"
            assert current_tool_actor_id() == "alice"
            assert current_tool_allowed_names() == {"lookup", "pay"}
            assert current_tool_call() is call
            assert current_tool_room() is None

        assert _readers() == _OUTSIDE

    def test_a_room_names_the_turns_room_id(self) -> None:
        room = Room(id="billing", organization_id="acme")

        with tool_turn_context(room=room):
            assert current_tool_room() is room
            assert current_tool_room_id() == "billing"

    def test_a_room_and_the_same_room_id_agree(self) -> None:
        room = Room(id="billing")

        with tool_turn_context(room=room, room_id="billing"):
            assert current_tool_room_id() == "billing"

    def test_a_room_and_another_room_id_are_refused(self) -> None:
        with (
            pytest.raises(ValueError, match="name different rooms"),
            tool_turn_context(room=Room(id="billing"), room_id="support"),
        ):
            pass
        assert _readers() == _OUTSIDE

    def test_a_call_record_of_another_room_is_refused(self) -> None:
        with (
            pytest.raises(ValueError, match="is not the turn's room"),
            tool_turn_context(room_id="r1", call=ToolCallContext(room_id="other")),
        ):
            pass
        with (
            pytest.raises(ValueError, match="is not the turn's room"),
            tool_turn_context(call=ToolCallContext(room_id="r1")),
        ):
            pass
        assert _readers() == _OUTSIDE

    def test_a_call_record_of_the_turns_room_is_installed(self) -> None:
        room = Room(id="billing")
        call = ToolCallContext(room_id="billing", tool_call_id="tc1")

        with tool_turn_context(room=room, call=call):
            assert current_tool_call() is call

    def test_no_toolset_is_not_an_empty_one(self) -> None:
        with tool_turn_context(room_id="r1"):
            assert current_tool_allowed_names() is None
        with tool_turn_context(room_id="r1", tools=[]):
            assert current_tool_allowed_names() == set()

    def test_a_turn_without_author_reads_none(self) -> None:
        with tool_turn_context(room_id="r1"):
            assert current_tool_room_id() == "r1"
            assert current_tool_actor_id() is None
            assert current_tool_call() is None

    @pytest.mark.parametrize("depth", [-1, 0, 1, 3, 2**32 - 1, 2**32, 2**100])
    def test_the_turn_carries_its_chain_depth(self, depth: int) -> None:
        with tool_turn_context(room_id="r1", chain_depth=depth):
            assert current_tool_chain_depth() == depth
        assert current_tool_chain_depth() == 0

    def test_a_requester_is_explicit_and_an_inner_default_clears_it(self) -> None:
        assert current_tool_requester() is None
        with tool_turn_context(requester="Bob"):
            assert current_tool_requester() == "Bob"
            with tool_turn_context():
                assert current_tool_requester() is None
            assert current_tool_requester() == "Bob"
        assert current_tool_requester() is None


class TestWhatTheHandlerWritesBack:
    async def test_the_response_record_is_the_turns_one(self) -> None:
        async def handler(name: str, arguments: dict[str, object]) -> str:
            record = current_response_metadata()
            assert record is not None
            record["sources"] = ["doc-1"]
            return "ok"

        with tool_turn_context(room_id="r1"):
            record = current_response_metadata()
            await handler("lookup", {})
            assert record is not None
            assert record["sources"] == ["doc-1"]
            assert current_response_metadata() is record

    async def test_the_structured_copy_lands_on_the_call_record_passed(self) -> None:
        call = ToolCallContext(room_id="r1", tool_call_id="tc1")

        async def handler(name: str, arguments: dict[str, object]) -> str:
            ctx = current_tool_call()
            assert ctx is not None
            ctx.structured_content = {"page": 3}
            return "ok"

        with tool_turn_context(room_id="r1", call=call):
            await handler("lookup", {})
        assert call.structured_content == {"page": 3}

    async def test_tasks_started_inside_inherit_the_turn(self) -> None:
        async def actor() -> str | None:
            await asyncio.sleep(0)
            return current_tool_actor_id()

        with tool_turn_context(room_id="r1", actor_id="alice"):
            assert await asyncio.gather(actor(), actor()) == ["alice", "alice"]


class TestRestoring:
    def test_each_scope_has_a_fresh_response_record(self) -> None:
        with tool_turn_context():
            outer = current_response_metadata()
            assert outer is not None
            outer["source"] = "outer"
            with tool_turn_context():
                inner = current_response_metadata()
                assert inner is not None
                assert inner is not outer
                assert dict(inner) == {}
                inner["source"] = "inner"
            assert current_response_metadata() is outer
            assert outer["source"] == "outer"
        assert current_response_metadata() is None

    def test_a_nested_turn_gives_the_outer_one_back(self) -> None:
        outer_call = ToolCallContext(room_id="outer", tool_call_id="outer")
        with tool_turn_context(room_id="outer", actor_id="alice", call=outer_call):
            with tool_turn_context(room_id="inner", actor_id="bob"):
                assert (current_tool_room_id(), current_tool_actor_id()) == ("inner", "bob")
                assert current_tool_call() is None
            assert (current_tool_room_id(), current_tool_actor_id()) == ("outer", "alice")
            assert current_tool_call() is outer_call

    @pytest.mark.parametrize("installer", list(_INSTALLERS), ids=list(_INSTALLERS))
    def test_both_installers_restore_on_an_exception(self, installer: str) -> None:
        call = ToolCallContext(room_id="r1", tool_call_id="tc1", channel_id="voice")

        with (
            pytest.raises(RuntimeError, match="handler failed"),
            _INSTALLERS[installer](call),
        ):
            assert current_tool_room_id() == "r1"
            assert current_tool_actor_id() == "alice"
            installed = current_tool_call()
            assert installed is not None
            assert installed.tool_call_id == "tc1"
            raise RuntimeError("handler failed")

        assert _readers() == _OUTSIDE


class TestLiveDepth:
    async def test_an_expired_inner_scope_still_masks_the_live_outer_in_its_child(self) -> None:
        kit = RoomKit()
        resume = asyncio.Event()

        async def child() -> int:
            await resume.wait()
            await kit.deliver("r1", "child")
            return current_tool_chain_depth()

        await kit.create_room(room_id="r1")
        kit.register_channel(SimpleChannel("out"))
        await kit.attach_channel("r1", "out")
        task: asyncio.Task[int] | None = None
        try:
            with tool_turn_context(room_id="r1", chain_depth=3):
                with tool_turn_context(room_id="r1", chain_depth=1):
                    task = asyncio.create_task(child())
                resume.set()
                assert await task == 1
                assert current_tool_chain_depth() == 3
            [delivered] = [
                event
                for event in await kit.store.list_events("r1")
                if isinstance(event.content, TextContent)
            ]
            assert delivered.content.body == "child"
            assert delivered.chain_depth == 0
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await kit.close()

    async def test_a_child_keeps_readers_but_stops_lending_depth_after_exit(self) -> None:
        kit = RoomKit()
        resume = asyncio.Event()

        async def child() -> tuple[int, str | None]:
            await resume.wait()
            await kit.deliver("r1", "after scope")
            return current_tool_chain_depth(), current_tool_actor_id()

        await kit.create_room(room_id="r1")
        kit.register_channel(SimpleChannel("out"))
        await kit.attach_channel("r1", "out")
        task: asyncio.Task[tuple[int, str | None]] | None = None
        try:
            with tool_turn_context(room_id="r1", actor_id="alice", chain_depth=3):
                await kit.deliver("r1", "in scope")
                task = asyncio.create_task(child())
            resume.set()
            assert await task == (3, "alice")
            depths = {
                event.content.body: event.chain_depth
                for event in await kit.store.list_events("r1")
                if isinstance(event.content, TextContent)
            }
            assert depths == {"in scope": 3, "after scope": 0}
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await kit.close()

    async def test_an_inner_scope_lends_its_own_depth_then_restores_the_outer(self) -> None:
        kit = RoomKit()
        await kit.create_room(room_id="r1")
        kit.register_channel(SimpleChannel("out"))
        await kit.attach_channel("r1", "out")
        try:
            with tool_turn_context(room_id="r1", chain_depth=3):
                await kit.deliver("r1", "outer")
                with tool_turn_context(room_id="r1", chain_depth=1):
                    await kit.deliver("r1", "inner")
                await kit.deliver("r1", "restored")
            depths = {
                event.content.body: event.chain_depth
                for event in await kit.store.list_events("r1")
                if isinstance(event.content, TextContent)
            }
            assert depths == {"outer": 3, "inner": 1, "restored": 3}
        finally:
            await kit.close()
