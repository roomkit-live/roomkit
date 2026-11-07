"""A response-stream failure must reach the caller via ``InboundResult.error``.

Companion to ``test_inbound_error_surfacing`` (which proves ON_ERROR fires).
A headless one-shot caller (e.g. an isolated RoomKit used for a single AI
call) has no streaming target to render an error card, so before this contract
the failure fired ON_ERROR and then vanished — ``process_inbound`` returned a
result with no signal, and the caller saw an empty response. The failure is
now also returned on ``InboundResult.error`` (with its cause chain intact) so
the caller can classify + react. A ``ProviderError`` is logged once, without a
traceback, by the stream's consumer: at its own level (WARNING, ERROR for a 404
or a 5xx) unless the caller receives the error, then at DEBUG. Any other
exception keeps its full traceback.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.core.mixins import verification as verification_mixin
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.identity import Identity
from roomkit.models.verification import VerificationPolicy
from roomkit.orchestration.strategies import Discussion
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    ProviderError,
    StreamEvent,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.orchestration.test_strategy_discussion import _settle
from tests.test_framework import SimpleChannel
from tests.turn_errors import same_error


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
    channel — the no-target branch, as a one-shot AI call uses it. Returns the
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
    await kit.hook_engine.drain_observers()  # ON_ERROR runs after the room lock is released
    await kit.close()
    return result, errors


async def test_provider_error_propagates_and_still_fires_on_error() -> None:
    """A ProviderError raised while consuming the stream is BOTH returned on
    InboundResult.error (cause chain intact) AND delivered to ON_ERROR."""
    exc = _provider_error_from_connect()
    result, errors = await _run_headless_turn(
        AIChannel("ai1", provider=_StreamRaisingProvider(exc))
    )

    assert same_error(result.error, exc)
    assert isinstance(result.error.__cause__, ConnectionError)
    assert len(errors) == 1  # ON_ERROR still fires (interactive contract intact)
    # Named by the exception's type, as every intelligence ON_ERROR is.
    assert errors[0].metadata["error_type"] == "ProviderError"


async def test_non_provider_error_also_propagates() -> None:
    """The contract is exception-agnostic: any stream failure reaches the caller."""
    exc = RuntimeError("unexpected boom")
    result, _errors = await _run_headless_turn(
        AIChannel("ai1", provider=_StreamRaisingProvider(exc))
    )

    assert same_error(result.error, exc)


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

    assert same_error(result.error, exc)
    assert isinstance(result.error.__cause__, ConnectionError)
    assert len(errors) == 1  # ON_ERROR card still fires


class _StreamingTransport(SimpleChannel):
    """A transport that takes the live text of a streamed answer."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[object],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for _chunk in text_stream:
            pass
        return ChannelOutput.empty()


async def _run_targeted_turn(ai: AIChannel) -> InboundResult:
    """One inbound turn on a room where a second transport takes the live stream."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_StreamingTransport("ws1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    result = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await kit.close()
    return result


def _raising_provider(streams: bool, exc: Exception) -> MockAIProvider:
    return _StreamRaisingProvider(exc) if streams else _GenerateRaisingProvider(exc)


def _roomkit_records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("roomkit") and r.levelno >= level]


@pytest.mark.parametrize("streams", [True, False])
async def test_a_turn_failing_before_any_round_is_one_warning_for_a_streaming_target(
    caplog, streams: bool
) -> None:
    """One incident, one roomkit line: the stream's consumer logs it, the loop
    does not log it a second time, whatever the provider streams."""
    exc = ProviderError("connection refused", provider="mock")
    with caplog.at_level(logging.DEBUG):
        result = await _run_targeted_turn(
            AIChannel("ai1", provider=_raising_provider(streams, exc))
        )

    assert same_error(result.error, exc)
    warnings = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert "connection refused" in warnings[0].getMessage()
    assert warnings[0].exc_info is None


@pytest.mark.parametrize("streams", [True, False])
async def test_a_turn_failing_before_any_round_is_no_roomkit_warning_when_headless(
    caplog, streams: bool
) -> None:
    """A headless caller gets the error back and owns its log: roomkit's own
    line stays at DEBUG, whatever the provider streams."""
    exc = ProviderError("connection refused", provider="mock")
    with caplog.at_level(logging.DEBUG):
        result, _errors = await _run_headless_turn(
            AIChannel("ai1", provider=_raising_provider(streams, exc))
        )

    assert same_error(result.error, exc)
    assert _roomkit_records(caplog, logging.WARNING) == []


@pytest.mark.parametrize("streams", [True, False])
@pytest.mark.parametrize("status", [404, 503])
async def test_a_missing_model_or_a_server_fault_is_an_error_for_a_streaming_target(
    caplog, status: int, streams: bool
) -> None:
    exc = ProviderError("model gone", provider="mock", status_code=status)
    with caplog.at_level(logging.DEBUG):
        await _run_targeted_turn(AIChannel("ai1", provider=_raising_provider(streams, exc)))

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].exc_info is None
    assert "provider=mock" in records[0].getMessage()
    assert f"status={status}" in records[0].getMessage()


@pytest.mark.parametrize("streams", [True, False])
async def test_a_send_event_caller_that_never_gets_the_failure_sees_it_logged(
    caplog, streams: bool
) -> None:
    """``send_event`` hands back the stored event, never the turn's failure: with
    no streaming target the framework's line is the only record, at the
    failure's own level, not the DEBUG kept for a caller that receives it."""
    exc = ProviderError("model gone", provider="mock", status_code=404)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=_raising_provider(streams, exc)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    with caplog.at_level(logging.DEBUG):
        await kit.send_event("r1", "sms1", TextContent(body="go"))
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].exc_info is None


@pytest.mark.parametrize("streams", [True, False])
async def test_a_verification_release_s_failed_turn_is_logged_at_its_level(
    caplog, monkeypatch: pytest.MonkeyPatch, streams: bool
) -> None:
    """A held message's release reads no result (RFC §11.7.6, RMK-1500): the
    failed turn of the agent it runs is logged at the failure's own level, as
    a send_event's, not the DEBUG kept for a caller that receives the error."""
    monkeypatch.setattr(verification_mixin, "_SCRYPT_N", 2**4)
    exc = ProviderError("model gone", provider="mock", status_code=404)
    kit = RoomKit(
        identity_resolver=MockIdentityResolver(mapping={"+15550000001": Identity(id="id-alice")}),
        verification_policy=VerificationPolicy(),
    )
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=_raising_provider(streams, exc)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    held = await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="+15550000001", content=TextContent(body="go"))
    )
    request, secret = await kit.request_verification(
        "id-alice", room_id="r1", channel_id="sms1", window_start=held.event.created_at
    )
    with caplog.at_level(logging.DEBUG):
        await kit.complete_verification(request.id, secret)
        await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].exc_info is None
    assert "stream consumption (no targets)" in records[0].getMessage()


@pytest.mark.parametrize("streams", [True, False])
async def test_a_deferred_call_sees_the_failure_logged_at_its_level(caplog, streams: bool) -> None:
    """A call that defers its delivery returns at its commit: its stream is
    read in the background, so with no streaming target the framework's line
    is at the failure's own level, not the DEBUG kept for a caller that
    waits for the error."""
    exc = ProviderError("model gone", provider="mock", status_code=404)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=_raising_provider(streams, exc)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    with caplog.at_level(logging.DEBUG):
        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go")),
            defer_delivery=True,
        )
        assert result.delivery is not None
        await result.delivery.wait()
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "stream consumption (no targets)" in records[0].getMessage()


class _BreakingTransport(_StreamingTransport):
    """A streaming transport that raises an error of its own when the
    response it reads fails."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[object],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        try:
            async for _chunk in text_stream:
                pass
        except ProviderError as exc:
            raise RuntimeError("the socket broke") from exc
        return ChannelOutput.empty()


async def test_a_streaming_channel_raising_its_own_error_is_one_line_more(caplog) -> None:
    """A streaming channel that raises an error of its own on a failed
    response has that error logged, an unexpected one, beside the turn's
    line: the turn's failure stays the response's."""
    exc = ProviderError("connection refused", provider="mock")
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_BreakingTransport("ws1"))
    kit.register_channel(AIChannel("ai1", provider=_StreamRaisingProvider(exc)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ws1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    with caplog.at_level(logging.DEBUG):
        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
        )
    await kit.close()

    assert same_error(result.error, exc)
    records = _roomkit_records(caplog, logging.WARNING)
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.ERROR,
            "streaming channel ws1 handling the failed response of ai1 in room r1 failed",
        ),
        (
            logging.WARNING,
            "streaming delivery of ai1 to ws1 for room r1 failed (provider=mock, status=None): "
            "mock: connection refused",
        ),
    ]


class _FailsAfterARound(MockAIProvider):
    """A model that calls a tool, then fails on the next round."""

    def __init__(self, exc: Exception, *, streams: bool) -> None:
        super().__init__(
            ai_responses=[
                AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="look", arguments={})])
            ],
            streaming=streams,
        )
        self._exc = exc

    async def generate(self, context: AIContext) -> AIResponse:
        if self.calls:
            raise self._exc
        return await super().generate(context)


def _agent_with_a_tool(provider: MockAIProvider) -> AIChannel:
    async def look(_name: str, _arguments: dict[str, object]) -> str:
        return "seen"

    return AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name="look", description="Look.", parameters={"type": "object"})],
        tool_handler=look,
    )


@pytest.mark.parametrize("streams", [True, False])
async def test_a_turn_failing_after_a_round_is_one_line_too(caplog, streams: bool) -> None:
    """Once a round ran, the turn ends ``error`` and its consumer logs it: the
    loop adds no line (and no traceback) of its own."""
    exc = ProviderError("connection reset", provider="mock")
    agent = _agent_with_a_tool(_FailsAfterARound(exc, streams=streams))
    with caplog.at_level(logging.DEBUG):
        result = await _run_targeted_turn(agent)

    assert same_error(result.error, exc)
    warnings = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert warnings[0].exc_info is None


@pytest.mark.parametrize("streams", [True, False])
async def test_a_headless_turn_failing_after_a_round_is_no_roomkit_warning(
    caplog, streams: bool
) -> None:
    exc = ProviderError("connection reset", provider="mock")
    agent = _agent_with_a_tool(_FailsAfterARound(exc, streams=streams))
    with caplog.at_level(logging.DEBUG):
        result, _errors = await _run_headless_turn(agent)

    assert same_error(result.error, exc)
    assert _roomkit_records(caplog, logging.WARNING) == []


async def test_a_delegated_turn_failure_is_one_line_at_its_level(caplog) -> None:
    """The delegation receives the child's failure and logs it, once, by its
    status: read as the task's trace, the child's stream has no line of its
    own."""
    exc = ProviderError("upstream down", provider="mock", status_code=503)
    kit = RoomKit()
    kit.register_channel(AIChannel("w1", provider=_GenerateRaisingProvider(exc)))
    await kit.create_room(room_id="r1")
    with caplog.at_level(logging.DEBUG):
        await kit.delegate(room_id="r1", agent_id="w1", task="go", wait=True)
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].exc_info is None
    assert "status=503" in records[0].getMessage()


async def test_a_background_task_failure_is_one_line_at_its_level(caplog) -> None:
    """A delegated task run in the background logs its turn's failure once,
    by its status, as the inline one does."""
    exc = ProviderError("upstream down", provider="mock", status_code=503)
    kit = RoomKit()
    kit.register_channel(AIChannel("w1", provider=_GenerateRaisingProvider(exc)))
    await kit.create_room(room_id="r1")
    with caplog.at_level(logging.DEBUG):
        task = await kit.delegate(room_id="r1", agent_id="w1", task="go")
        await task.wait()
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert records[0].getMessage().startswith(f"Task {task.id} failed (provider=mock, status=503)")


async def test_a_delegated_turn_failure_in_a_shared_room_is_one_line_at_its_level(caplog) -> None:
    """A task whose child room a transport is shared into reads its worker's
    stream through the room: that stream's line stays at DEBUG, for the task
    to log the failure once, by its status."""
    exc = ProviderError("upstream down", provider="mock", status_code=503)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("email-out"))
    kit.register_channel(AIChannel("w1", provider=_GenerateRaisingProvider(exc)))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "email-out")
    with caplog.at_level(logging.DEBUG):
        task = await kit.delegate(
            room_id="r1", agent_id="w1", task="go", wait=True, share_channels=["email-out"]
        )
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert (
        records[0]
        .getMessage()
        .startswith(f"Inline task {task.id} failed (provider=mock, status=503)")
    )


async def test_another_agent_s_streamed_failure_in_a_task_room_is_its_turn_s_line(caplog) -> None:
    """Another agent shared into a task's room, whose stream fails with a
    503: the room's reader logs its own turn's at DEBUG, the task's room
    path once at its level, as another agent's when the room's two streams
    leave whose it was untold; its chained turn on the worker's answer is
    no turn of the task's, logged at its level."""
    exc = ProviderError("B's upstream 503", provider="b", status_code=503)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("email-out"))
    kit.register_channel(AIChannel("b", provider=_StreamRaisingProvider(exc)))
    kit.register_channel(AIChannel("w1", provider=MockAIProvider(responses=["Found it."])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "email-out")
    await kit.attach_channel("r1", "b", category=ChannelCategory.INTELLIGENCE)
    with caplog.at_level(logging.DEBUG):
        task = await kit.delegate(
            room_id="r1", agent_id="w1", task="go", wait=True, share_channels=["b", "email-out"]
        )
    await kit.close()

    assert task.result is not None and task.result.output == "Found it."
    room = task.child_room_id
    records = _roomkit_records(caplog, logging.WARNING)
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (
            logging.ERROR,
            f"stream consumption (no targets) of b for room {room} failed "
            "(provider=b, status=503): B's upstream 503",
        ),
        (
            logging.ERROR,
            f"In {room}, another agent's turn failed (provider=b, status=503): B's upstream 503",
        ),
    ]


async def test_a_discussion_turn_s_failure_is_logged_at_its_level(caplog) -> None:
    """A discussion's turn is the discussion's, not the message's caller's:
    its stream's failure is logged at its own level."""
    exc = ProviderError("model gone", provider="mock", status_code=404)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    agent = AIChannel("ai1", provider=_GenerateRaisingProvider(exc))
    await kit.create_room(room_id="r1", orchestration=Discussion([agent]))
    await kit.attach_channel("r1", "sms1")
    with caplog.at_level(logging.DEBUG):
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="u1",
                content=TextContent(body="go"),
                addressed_to=["ai1"],
            )
        )
        await _settle(kit)
    await kit.close()

    records = _roomkit_records(caplog, logging.WARNING)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "stream consumption (no targets) of ai1" in records[0].getMessage()


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
    await kit.hook_engine.drain_observers()
    before = len(errors)  # the inbound failure already fired one

    result = await kit.regenerate_response("r1")
    await kit.hook_engine.drain_observers()
    await kit.close()

    assert result is not None and result.error is not None
    assert len(errors) == before + 1  # regenerate fired its own ON_ERROR card
    assert errors[-1].metadata["error_type"] == "ProviderError"


class _RaisingIntelligence(SimpleChannel):
    """An intelligence channel that fails before any stream exists: its
    failure reaches the broadcast's errors, not a stream."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        raise RuntimeError(self.channel_id)


async def test_regenerate_fires_on_error_for_each_failed_agent() -> None:
    """As the inbound path: one ON_ERROR card per intelligence channel whose
    failure the broadcast reports, the first one on InboundResult.error."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_RaisingIntelligence("ai1"))
    kit.register_channel(_RaisingIntelligence("ai2"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "ai2", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_errors",
        )
    )
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await kit.hook_engine.drain_observers()
    inbound = sorted(event.source.channel_id for event in errors)
    errors.clear()

    result = await kit.regenerate_response("r1")
    await kit.hook_engine.drain_observers()
    await kit.close()

    assert inbound == ["ai1", "ai2"]
    assert sorted(event.source.channel_id for event in errors) == ["ai1", "ai2"]
    assert result is not None and str(result.error) == "ai1"
