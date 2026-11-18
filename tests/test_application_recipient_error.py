"""A transport implemented by the application can refuse before a provider send."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from roomkit import (
    ChannelBinding,
    ChannelOutput,
    DeliveryResult,
    FrameworkEvent,
    InboundMessage,
    NoRecipientError,
    ProviderDeliveryError,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.channels import WebSocketChannel
from roomkit.core import circuit_breaker
from roomkit.models import RetryPolicy


@pytest.mark.parametrize("reported_by_provider", [False, True], ids=["before-send", "provider"])
async def test_application_recipient_refusal_and_provider_answer_have_distinct_breaker_effects(
    reported_by_provider: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = NoRecipientError("app", "customer")
    assert error.channel_id == "app" and error.recipient_key == "customer"
    assert str(error) == error.code == "no_recipient"
    assert error.retryable is False
    assert error.provider_result.model_dump() == {
        "success": False,
        "provider_message_id": None,
        "error": "no_recipient",
        "metadata": {
            "code": "no_recipient",
            "retryable": False,
            "channel_id": "app",
            "recipient_key": "customer",
        },
    }

    class ApplicationChannel(WebSocketChannel):
        attempts = 0

        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            self.attempts += 1
            if self.attempts > 6:
                return ChannelOutput.empty()
            if reported_by_provider:
                raise ProviderDeliveryError(error.provider_result)
            raise NoRecipientError("app", "customer")

    kit = RoomKit()
    application = ApplicationChannel("app")
    kit.register_channel(WebSocketChannel("source"))
    kit.register_channel(application)
    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "source")
    await kit.attach_channel(
        "support", "app", retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.001)
    )
    try:
        results = []
        for i in range(7):
            message = InboundMessage(
                channel_id="source", sender_id="alice", content=TextContent(body=str(i))
            )
            result = await kit.process_inbound(message, room_id="support")
            results.append(result.delivery_results["app"])
        failures = 5 if reported_by_provider else 6
        assert application.attempts == (5 if reported_by_provider else 7)
        for delivery in results[:failures]:
            assert delivery.error is not None
            assert delivery.error.code == delivery.error.message == "no_recipient"
            assert delivery.error.retryable is False
            assert delivery.provider_result == error.provider_result
        if reported_by_provider:
            assert all(d.error and d.error.message == "circuit_breaker_open" for d in results[5:])
        else:
            assert results[-1].status == "sent"
            warnings = [
                r for r in caplog.records if "the binding has no customer" in r.getMessage()
            ]
            assert len(warnings) == 6
            assert all(r.levelno == logging.WARNING and r.exc_info is None for r in warnings)
    finally:
        await kit.close()


async def test_missing_recipient_releases_the_probe_for_the_next_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(circuit_breaker, "time", SimpleNamespace(monotonic=lambda: now))
    transitions: list[str] = []

    async def note_transition(event: FrameworkEvent) -> None:
        transitions.append(event.type)

    class RecoveringChannel(WebSocketChannel):
        attempts = 0

        async def deliver(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            self.attempts += 1
            if self.attempts <= 5:
                raise ProviderDeliveryError(NoRecipientError("app", "customer").provider_result)
            if self.attempts == 6:
                raise NoRecipientError("app", "customer")
            return ChannelOutput.empty()

    kit = RoomKit()
    application = RecoveringChannel("app")
    kit.register_channel(WebSocketChannel("source"))
    kit.register_channel(application)
    for name in ("circuit_breaker_opened", "circuit_breaker_closed"):
        kit.on(name)(note_transition)
    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "source")
    await kit.attach_channel("support", "app")

    async def send(text: str) -> DeliveryResult:
        message = InboundMessage(
            channel_id="source", sender_id="alice", content=TextContent(body=text)
        )
        result = await kit.process_inbound(message, room_id="support")
        return result.delivery_results["app"]

    try:
        for i in range(5):
            assert (await send(str(i))).status == "failed"

        now += 60
        refused = await send("missing recipient")
        assert refused.provider_result == NoRecipientError("app", "customer").provider_result
        await asyncio.sleep(0)
        assert transitions == ["circuit_breaker_opened"]

        assert (await send("deliver immediately")).status == "sent"
        await asyncio.sleep(0)
        assert transitions == ["circuit_breaker_opened", "circuit_breaker_closed"]
        assert application.attempts == 7
    finally:
        await kit.close()
