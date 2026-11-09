"""Injection eligibility belongs to the SYNC trigger, not its decision (§9.4)."""

from __future__ import annotations

import logging

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, VerificationPolicy
from roomkit.core.hooks import TURN_TRIGGERS, HookEngine, HookRegistration
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity
from roomkit.models.room import Room
from roomkit.models.task import Task
from tests.test_injected_events import _hello, _notice, _notices, _room


@pytest.mark.parametrize(
    "trigger",
    [
        HookTrigger.BEFORE_BROADCAST,
        HookTrigger.BEFORE_DELIVER,
        HookTrigger.ON_VERIFICATION_REQUIRED,
        HookTrigger.ON_TRANSCRIPTION,
        HookTrigger.BEFORE_TTS,
        HookTrigger.BEFORE_TOOL_USE,
        HookTrigger.ON_TOOL_CALL,
        HookTrigger.AFTER_TOOL_ROUND,
        HookTrigger.ON_USER_INPUT_REQUIRED,
        HookTrigger.BEFORE_AI_GENERATION,
        HookTrigger.BEFORE_BRIDGE_AUDIO,
        HookTrigger.BEFORE_BRIDGE_VIDEO,
        HookTrigger.ON_SPEAK_DECISION,
        HookTrigger.ON_VISION_RESULT,
    ],
)
@pytest.mark.parametrize("action", ["allow", "modify", "block"])
async def test_only_broadcast_keeps_injections_in_a_sync_chain(
    trigger: HookTrigger, action: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Engine contract, independent of each trigger's typed caller payload."""
    engine = HookEngine(turns=trigger in TURN_TRIGGERS)
    event = _notice("cause")
    task = Task(id="review", room_id="r1", title="Review")

    async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult(
            action=action,
            event=event.model_copy(update={"content": TextContent(body="changed")})
            if action == "modify"
            else None,
            reason="held" if action == "block" else None,
            injected_events=[InjectedEvent(event=_notice(), target_channel_ids=["ws"])],
            tasks=[task],
            metadata={"kept": True},
        )

    engine.register(HookRegistration(trigger, HookExecution.SYNC, notify, name="notify"))
    with caplog.at_level(logging.WARNING, logger="roomkit.hooks"):
        result = await engine.run_sync_hooks("r1", trigger, event, RoomContext(room=Room(id="r1")))
    assert len(result.injected_events) == (1 if trigger == HookTrigger.BEFORE_BROADCAST else 0)
    assert result.allowed == (action != "block")
    assert result.tasks == [task]
    assert result.metadata == {"kept": True}
    if action == "modify":
        assert result.event.content == TextContent(body="changed")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == (0 if trigger == HookTrigger.BEFORE_BROADCAST else 1)
    if warnings:
        assert (
            "notify" in warnings[0] and trigger.value in warnings[0] and "dropped" in warnings[0]
        )


async def test_verification_drops_injections_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real verification gate preserves its decision, drops and warns."""
    kit, ws, _ = await _room(
        identity_resolver=MockIdentityResolver(mapping={"u": Identity(id="alice")}),
        verification_policy=VerificationPolicy(),
    )

    @kit.hook(HookTrigger.ON_VERIFICATION_REQUIRED, name="verify_notice")
    async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow_unverified().model_copy(
            update={"injected_events": [InjectedEvent(event=_notice(), target_channel_ids=["ws"])]}
        )

    try:
        with caplog.at_level(logging.WARNING, logger="roomkit.hooks"):
            await _hello(kit)
        assert await _notices(kit) == []
        assert [e.content.body for e in ws.delivered] == ["hello", "AI reply"]
        assert any(
            "verify_notice" in r.getMessage() and "on_verification_required" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )
    finally:
        await kit.close()
