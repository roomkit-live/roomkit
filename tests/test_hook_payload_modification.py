"""Sync hooks may rewrite payloads that are not RoomEvents (RFC §9).

HookResult.event used to be typed as a RoomEvent, so a hook on any of the eight
sync triggers that pass something else — a string to BEFORE_TTS, a frame to the
bridge triggers, their own event types to the tool and generation triggers —
raised when it tried to return a modified value. The engine logged the error and
carried on with the original, which meant a redaction hook silently published
what it existed to suppress.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit.core.hooks import HookEngine
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room


def _context() -> RoomContext:
    return RoomContext(room=Room(id="r"), bindings=[])


def _register(engine: HookEngine, trigger: HookTrigger, fn: object) -> None:
    from roomkit.core.hooks import HookRegistration

    engine.register(
        HookRegistration(
            trigger=trigger,
            execution=HookExecution.SYNC,
            fn=fn,  # type: ignore[arg-type]
            name="test-hook",
        )
    )


class TestModification:
    async def test_a_string_payload_can_be_rewritten(self) -> None:
        engine = HookEngine()

        async def redact(text: object, ctx: object) -> HookResult:
            return HookResult(action="modify", event="le numéro est ****")

        _register(engine, HookTrigger.BEFORE_TTS, redact)

        result = await engine.run_sync_hooks(
            "r",
            HookTrigger.BEFORE_TTS,
            "le numéro est 4111",
            _context(),
            skip_event_filter=True,
        )

        assert result.allowed
        assert result.event == "le numéro est ****"

    async def test_an_arbitrary_payload_can_be_rewritten(self) -> None:
        engine = HookEngine()

        class Transcription:
            def __init__(self, text: str) -> None:
                self.text = text

        async def redact(payload: object, ctx: object) -> HookResult:
            return HookResult(action="modify", event=Transcription("masqué"))

        _register(engine, HookTrigger.ON_TRANSCRIPTION, redact)

        result = await engine.run_sync_hooks(
            "r",
            HookTrigger.ON_TRANSCRIPTION,
            Transcription("4111"),
            _context(),
            skip_event_filter=True,
        )

        assert result.event.text == "masqué"

    async def test_modify_still_requires_a_payload(self) -> None:
        with pytest.raises(ValueError, match="requires 'event'"):
            HookResult(action="modify")


class TestFailClosed:
    async def test_a_raising_hook_blocks_on_a_content_trigger(self) -> None:
        """Carrying on would publish exactly what the hook existed to suppress."""
        engine = HookEngine()

        async def boom(text: object, ctx: object) -> HookResult:
            raise RuntimeError("detection service down")

        _register(engine, HookTrigger.BEFORE_TTS, boom)

        result = await engine.run_sync_hooks(
            "r",
            HookTrigger.BEFORE_TTS,
            "le numéro est 4111",
            _context(),
            skip_event_filter=True,
        )

        assert result.allowed is False
        assert result.reason is not None and "detection service down" in result.reason

    async def test_a_raising_hook_stays_non_fatal_elsewhere(self) -> None:
        """A broken hook must not be able to take a room down."""
        engine = HookEngine()

        async def boom(event: object, ctx: object) -> HookResult:
            raise RuntimeError("oops")

        _register(engine, HookTrigger.BEFORE_BROADCAST, boom)

        event = RoomEvent(
            room_id="r",
            source=EventSource(channel_id="c", channel_type="sms"),  # type: ignore[arg-type]
            content=TextContent(body="bonjour"),
        )
        result = await engine.run_sync_hooks("r", HookTrigger.BEFORE_BROADCAST, event, _context())

        assert result.allowed is True
        assert result.hook_errors


class TestFailClosedCoversEveryFailure:
    """A partial rule is no rule: an engine that blocks on exceptions but allows
    on timeouts leaks through the timeout.
    """

    async def test_a_timeout_blocks_on_a_content_trigger(self) -> None:
        engine = HookEngine()

        async def slow(text: object, ctx: object) -> HookResult:
            await asyncio.sleep(1)
            return HookResult.allow()

        from roomkit.core.hooks import HookRegistration

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_TTS,
                execution=HookExecution.SYNC,
                fn=slow,  # type: ignore[arg-type]
                name="slow-hook",
                timeout=0.01,
            )
        )

        result = await engine.run_sync_hooks(
            "r", HookTrigger.BEFORE_TTS, "numéro 4111", _context(), skip_event_filter=True
        )

        assert result.allowed is False

    async def test_an_invalid_return_blocks_on_a_content_trigger(self) -> None:
        engine = HookEngine()

        async def wrong(text: object, ctx: object) -> str:
            return "je ne suis pas un HookResult"

        _register(engine, HookTrigger.BEFORE_TTS, wrong)

        result = await engine.run_sync_hooks(
            "r", HookTrigger.BEFORE_TTS, "numéro 4111", _context(), skip_event_filter=True
        )

        assert result.allowed is False

    async def test_an_unusable_modify_payload_blocks(self) -> None:
        """The consumer would ignore a payload of the wrong type and carry on
        with the original — publishing what the hook meant to replace.
        """
        engine = HookEngine()

        async def wrong_type(text: object, ctx: object) -> HookResult:
            return HookResult(action="modify", event={"redacted": True})

        _register(engine, HookTrigger.BEFORE_TTS, wrong_type)

        result = await engine.run_sync_hooks(
            "r", HookTrigger.BEFORE_TTS, "numéro 4111", _context(), skip_event_filter=True
        )

        assert result.allowed is False


class TestFalsyModification:
    async def test_redacting_to_empty_is_a_modification(self) -> None:
        """Truthiness testing would hand the next hook the original secret."""
        engine = HookEngine()
        seen: list[str] = []

        async def blank(text: object, ctx: object) -> HookResult:
            return HookResult(action="modify", event="")

        async def observer(text: object, ctx: object) -> HookResult:
            seen.append(str(text))
            return HookResult.allow()

        _register(engine, HookTrigger.BEFORE_TTS, blank)
        _register(engine, HookTrigger.BEFORE_TTS, observer)

        result = await engine.run_sync_hooks(
            "r", HookTrigger.BEFORE_TTS, "numéro 4111", _context(), skip_event_filter=True
        )

        assert seen == [""]
        assert result.event == ""


class TestModifyConstructor:
    def test_modify_accepts_the_trigger_payload(self) -> None:
        assert HookResult.modify("texte réécrit").event == "texte réécrit"
