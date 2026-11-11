"""Public named-hook and local-executor contracts (RFC §10.6.6)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import HookExecution, HookRegistration, HookTrigger, RoomKit
from tests.test_framework import SimpleChannel


async def noop(event: Any, context: Any) -> None:
    pass


def registration(name: str, trigger: HookTrigger = HookTrigger.ON_TOOL_CALL) -> HookRegistration:
    return HookRegistration(name=name, trigger=trigger, execution=HookExecution.ASYNC, fn=noop)


async def test_named_definitions_and_selection_are_separate_from_owner_hooks() -> None:
    async with RoomKit() as kit:
        engine = kit.turn_executor.hooks
        engine.define_room_hook(registration("audit"))
        engine.define_room_hook(registration("audit", HookTrigger.ON_AI_RESPONSE))
        assert engine.defined_room_hooks() == {"audit"}
        assert engine.defines_room_hooks([])
        assert not engine.defines_room_hooks(["missing"])
        assert engine.room_hook_names("room") == []
        engine.use_room_hooks("room", ["missing", "audit", "audit"])
        assert engine.room_hook_names("room") == ["audit", "audit"]
        assert kit.hook_engine.room_hook_names("room") == []
        assert not kit.hook_engine.undefine_room_hook("audit")
        assert kit.hook_engine.defined_room_hooks() == set()
        assert engine.remove_room_hook("room", "audit")
        assert engine.room_hook_names("room") == []
        assert engine.defines_room_hooks(["audit"])
        assert not engine.remove_room_hook("room", "missing")
        assert engine.undefine_room_hook("audit")
        assert not engine.undefine_room_hook("audit")
        assert not engine.has_hooks()


async def test_executor_and_owner_global_names_have_separate_removal_scopes() -> None:
    async with RoomKit() as kit:
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="shared")(noop)
        executor = kit.turn_executor
        executor.hook(HookTrigger.ON_TOOL_CALL, name="shared")(noop)
        assert executor.hooks.remove_global_hook("shared")
        assert kit.hook_engine.has_hooks(HookTrigger.BEFORE_BROADCAST)
        assert kit.hook_engine.remove_global_hook("shared")

        executor.hook(HookTrigger.ON_TOOL_CALL, name="only-executor")(noop)
        assert not kit.hook_engine.remove_global_hook("only-executor")
        kit.hook(HookTrigger.BEFORE_BROADCAST, name="only-executor")(noop)
        assert kit.hook_engine.remove_global_hook("only-executor")
        assert executor.hooks.has_hooks(HookTrigger.ON_TOOL_CALL)


async def test_an_owner_hook_given_to_executor_can_share_a_named_definition() -> None:
    async with RoomKit() as kit:
        engine = kit.turn_executor.hooks
        engine.define_room_hook(registration("audit"))
        await kit.enable_room_hook("room", "audit")
        engine.add_owner_room_hook("room", registration("audit"))
        assert engine.room_hook_names("room") == ["audit"]
        assert kit.hook_engine.room_hook_names("room") == []
        assert not await kit.remove_room_hook("room", "audit")
        assert engine.remove_room_hook("room", "audit")
        assert engine.room_hook_names("room") == ["audit"]
        assert engine.remove_room_hook("room", "audit")
        assert engine.room_hook_names("room") == []
        # Direct removal affects the engine only; the owner's name stays enabled.
        assert await kit.disable_room_hook("room", "audit")


async def test_local_executor_queries_follow_registration_and_close() -> None:
    kit = RoomKit()
    executor = kit.turn_executor
    assert executor.hosted is None
    assert executor.organizations == {"*"}
    assert executor.serves(None) and executor.serves("acme")
    assert not executor.declares("agent")
    kit.register_channel(SimpleChannel("agent"))
    assert executor.hosts_channel("agent") and executor.declares("agent")
    assert executor.takes("agent")
    assert not executor.takes("agent", ["audit"])
    executor.room_hook("audit", HookTrigger.ON_TOOL_CALL)(noop)
    assert executor.takes("agent", ["audit"], "acme")
    await kit.close()
    assert executor.closed
    assert executor.hosts_channel("agent")
    assert not executor.declares("agent")
    assert not executor.takes("agent")


async def test_refused_redefinition_keeps_the_original_and_default_options() -> None:
    async with RoomKit() as kit:
        engine = kit.turn_executor.hooks
        engine.define_room_hook(registration("audit"))
        invalid = registration("audit")
        invalid.fail_closed = True
        with pytest.raises(ValueError):
            engine.define_room_hook(invalid)
        assert engine.defined_room_hooks() == {"audit"}
        assert engine.has_hooks(HookTrigger.ON_TOOL_CALL)
        assert not engine.has_hooks(HookTrigger.BEFORE_BROADCAST)


async def test_add_owner_room_hook_returns_the_registered_name() -> None:
    async with RoomKit() as kit:
        engine = kit.turn_executor.hooks
        assert engine.add_owner_room_hook("room", registration("audit")).name == "audit"
        generated = engine.add_owner_room_hook("room", registration(""))
        assert generated.name == "noop"
        assert engine.remove_room_hook("room", generated.name)
