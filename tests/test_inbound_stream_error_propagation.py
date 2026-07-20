"""A response-stream failure must reach the caller via ``InboundResult.error``.

Companion to ``test_inbound_error_surfacing`` (which proves ON_ERROR fires).
A headless one-shot caller (e.g. an isolated RoomKit used for a single AI
call) has no streaming target to render an error card, so before this contract
the failure fired ON_ERROR and then vanished — ``process_inbound`` returned a
result with no signal, and the caller saw an empty response. The failure is
now also returned on ``InboundResult.error`` (with its cause chain intact) so
the caller can classify + react. ``ProviderError`` — an expected transient — is
logged as one WARNING line without a traceback; any other exception keeps its
full traceback.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIResponse, ProviderError, StreamEvent
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


def _provider_error_from_connect() -> ProviderError:
    """A ProviderError wrapping a connect failure, mirroring the real chain
    (ConnectError -> APIConnectionError -> ProviderError)."""
    cause = ConnectionError("[Errno -2] Name or service not known")
    err = ProviderError("connection error")
    err.__cause__ = cause
    return err


class _StreamRaisingProvider(MockAIProvider):
    """Structured-streaming provider that raises before yielding anything."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(streaming=True)
        self._exc = exc

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        raise self._exc
        yield  # pragma: no cover - keep this an async generator


async def _run_headless_turn(ai: AIChannel) -> tuple[InboundResult, list[RoomEvent]]:
    """One inbound turn on a room whose only transport is a plain (non-streaming)
    channel — the no-target branch, as PostProcessKit uses it. Returns the
    InboundResult and any ON_ERROR events."""
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_error",
        )
    )

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)  # ON_ERROR runs after the room lock is released
    await kit.close()
    return result, errors


async def test_provider_error_propagates_and_still_fires_on_error() -> None:
    """A ProviderError raised while consuming the stream is BOTH returned on
    InboundResult.error (cause chain intact) AND delivered to ON_ERROR."""
    exc = _provider_error_from_connect()
    result, errors = await _run_headless_turn(
        AIChannel("ai1", provider=_StreamRaisingProvider(exc))
    )

    assert result.error is exc
    assert isinstance(result.error.__cause__, ConnectionError)
    assert len(errors) == 1  # ON_ERROR still fires (interactive contract intact)


async def test_non_provider_error_also_propagates() -> None:
    """The contract is exception-agnostic: any stream failure reaches the caller."""
    exc = RuntimeError("unexpected boom")
    result, _errors = await _run_headless_turn(
        AIChannel("ai1", provider=_StreamRaisingProvider(exc))
    )

    assert result.error is exc


async def test_successful_turn_has_no_error() -> None:
    """A stream that completes leaves InboundResult.error as None."""
    ai = AIChannel("ai1", provider=MockAIProvider(responses=["all good"], streaming=True))
    result, errors = await _run_headless_turn(ai)

    assert result.error is None
    assert errors == []


async def test_headless_provider_error_logged_at_debug_no_traceback(caplog) -> None:
    """With no streaming target the error is returned to the caller, which owns
    logging — the framework line drops to DEBUG (no traceback) so it doesn't
    duplicate the caller's WARNING for the same incident."""
    with caplog.at_level(logging.DEBUG, logger="roomkit.framework"):
        await _run_headless_turn(
            AIChannel("ai1", provider=_StreamRaisingProvider(_provider_error_from_connect()))
        )

    stream_records = [r for r in caplog.records if "stream consumption (no targets)" in r.message]
    assert len(stream_records) == 1
    record = stream_records[0]
    assert record.levelno == logging.DEBUG
    assert record.exc_info is None  # no traceback attached


async def test_unexpected_error_keeps_traceback(caplog) -> None:
    """A non-ProviderError is an unexpected defect — keep the full traceback."""
    with caplog.at_level(logging.ERROR, logger="roomkit.framework"):
        await _run_headless_turn(
            AIChannel("ai1", provider=_StreamRaisingProvider(RuntimeError("bug")))
        )

    stream_records = [r for r in caplog.records if "stream consumption (no targets)" in r.message]
    assert len(stream_records) == 1
    record = stream_records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None  # traceback preserved for diagnosis


# ── Non-streaming generation path (the LUG-204 twin) ──────────────────────


class _GenerateRaisingProvider(MockAIProvider):
    """Non-streaming provider whose generate() raises (no streaming support)."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(streaming=False)
        self._exc = exc

    async def generate(self, context: AIContext) -> AIResponse:
        raise self._exc


async def test_non_streaming_provider_error_propagates_with_cause() -> None:
    """A non-streaming AI failure reaches InboundResult.error (cause intact),
    symmetric with the streaming path — a headless caller reading the store is
    no longer misled by an empty answer with no signal."""
    cause = ConnectionError("[Errno -2] Name or service not known")
    exc = ProviderError("provider request failed", provider="mock")
    exc.__cause__ = cause
    result, errors = await _run_headless_turn(
        AIChannel("ai1", provider=_GenerateRaisingProvider(exc))
    )

    assert result.error is exc
    assert isinstance(result.error.__cause__, ConnectionError)
    assert len(errors) == 1  # ON_ERROR card still fires


async def test_non_streaming_provider_error_logs_warning_no_traceback(caplog) -> None:
    """The router logs a non-streaming ProviderError as WARNING without a stack,
    matching the streaming path's _log_stream_failure."""
    exc = ProviderError("connection refused", provider="mock")
    with caplog.at_level(logging.WARNING, logger="roomkit.event_router"):
        await _run_headless_turn(AIChannel("ai1", provider=_GenerateRaisingProvider(exc)))

    target_records = [r for r in caplog.records if "Processing target" in r.message]
    assert len(target_records) == 1
    assert target_records[0].levelno == logging.WARNING
    assert target_records[0].exc_info is None


# ── regenerate_response surfaces the same error ───────────────────────────


async def test_regenerate_surfaces_stream_error() -> None:
    """regenerate_response returns the failure on InboundResult.error instead of
    a success-looking result (it used to discard the stream error)."""
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    ai = AIChannel("ai1", provider=_StreamRaisingProvider(_provider_error_from_connect()))
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    # First inbound persists the transport message regenerate re-runs on (the AI
    # fails, which is the already-tested inbound path).
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    result = await kit.regenerate_response("r1")
    await kit.close()

    assert result is not None
    assert result.error is not None


async def test_regenerate_non_streaming_failure_fires_on_error() -> None:
    """A non-streaming provider failure during regenerate fires ON_ERROR (parity
    with the inbound path) so the host renders an error card, not just surfaces
    it on InboundResult.error."""
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    ai = AIChannel(
        "ai1", provider=_GenerateRaisingProvider(ProviderError("boom", provider="mock"))
    )
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_error",
        )
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)
    before = len(errors)  # the inbound failure already fired one

    result = await kit.regenerate_response("r1")
    await asyncio.sleep(0.05)
    await kit.close()

    assert result is not None and result.error is not None
    assert len(errors) == before + 1  # regenerate fired its own ON_ERROR card
