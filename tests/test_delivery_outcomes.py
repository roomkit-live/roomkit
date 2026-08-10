"""Delivery outcomes are reported, and a failure is remembered (RFC §5.13, §10.1 step 18).

`RoomEvent.delivery_results` was declared and written nowhere; `DeliveryResult`
was declared and constructed nowhere; `InboundResult` had no such field at all.
Outcomes existed only as live framework events, so an integrator who was not
subscribed at that instant could never learn afterwards whether a message
reached its channels.

Only a failing set is persisted. A set that all succeeded is its own record —
paying a write per event to say "everything worked" spends the whole message
volume on a question nobody asks.
"""

from __future__ import annotations

import pytest

from roomkit import RoomKit, TransportChannel
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput, RetryPolicy
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent, TextContent
from tests.test_framework import SimpleChannel


class BrokenChannel(SimpleChannel):
    """A transport whose provider refuses, the way a real one does."""

    def __init__(self, channel_id: str, error: Exception) -> None:
        super().__init__(channel_id)
        self._error = error

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        raise self._error


class _RejectedError(Exception):
    code = "550"
    retryable = False


class ResultProvider:
    """Provider whose structured result is controlled by the test."""

    def __init__(self, *results: ProviderResult) -> None:
        self.results = list(results)
        self.calls = 0

    async def send(self, event: RoomEvent, *, to: str) -> ProviderResult:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


async def _room(*channels: Channel) -> RoomKit:
    kit = RoomKit()
    for channel in channels:
        kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    for channel in channels:
        await kit.attach_channel("r1", channel.channel_id)
    return kit


def _msg(channel_id: str) -> InboundMessage:
    return InboundMessage(channel_id=channel_id, sender_id="u1", content=TextContent(body="hello"))


class TestTheCallerIsToldWhatHappened:
    async def test_a_successful_delivery_is_reported(self) -> None:
        kit = await _room(SimpleChannel("in"), SimpleChannel("out"))

        result = await kit.process_inbound(_msg("in"))

        assert result.delivery_results["out"].status == "sent"
        assert result.delivery_results["out"].error is None

    async def test_a_failure_says_why_and_whether_to_retry(self) -> None:
        kit = await _room(SimpleChannel("in"), BrokenChannel("out", _RejectedError("recipient")))

        result = await kit.process_inbound(_msg("in"))

        failed = result.delivery_results["out"]
        assert failed.status == "failed"
        assert failed.error is not None
        assert failed.error.code == "550"
        assert failed.error.retryable is False
        assert "recipient" in failed.error.message

    async def test_a_negative_provider_result_is_not_reported_as_sent(self) -> None:
        provider_result = ProviderResult(
            success=False,
            provider_message_id="provider-rejection-1",
            error="recipient_rejected",
            metadata={"code": "550", "retryable": False},
        )
        provider = ResultProvider(provider_result)
        transport = TransportChannel("out", ChannelType.SMS, provider=provider)
        kit = await _room(SimpleChannel("in"), transport)

        result = await kit.process_inbound(_msg("in"))

        failed = result.delivery_results["out"]
        assert failed.status == "failed"
        assert failed.provider_message_id == "provider-rejection-1"
        assert failed.provider_result == provider_result
        assert failed.error is not None
        assert failed.error.code == "550"
        assert failed.error.retryable is False

    async def test_a_successful_provider_result_is_preserved(self) -> None:
        provider_result = ProviderResult(
            success=True,
            provider_message_id="provider-message-1",
            metadata={"accepted": True},
        )
        transport = TransportChannel(
            "out", ChannelType.SMS, provider=ResultProvider(provider_result)
        )
        kit = await _room(SimpleChannel("in"), transport)

        result = await kit.process_inbound(_msg("in"))

        sent = result.delivery_results["out"]
        assert sent.status == "sent"
        assert sent.provider_message_id == "provider-message-1"
        assert sent.provider_result == provider_result

    async def test_a_negative_provider_result_uses_the_retry_path(self) -> None:
        provider = ResultProvider(
            ProviderResult(success=False, error="timeout"),
            ProviderResult(success=True, provider_message_id="accepted-after-retry"),
        )
        transport = TransportChannel("out", ChannelType.SMS, provider=provider)
        kit = RoomKit()
        kit.register_channel(SimpleChannel("in"))
        kit.register_channel(transport)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "in")
        await kit.attach_channel(
            "r1",
            "out",
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.001),
        )

        result = await kit.process_inbound(_msg("in"))

        assert provider.calls == 2
        assert result.delivery_results["out"].status == "sent"
        assert result.delivery_results["out"].provider_message_id == "accepted-after-retry"

    async def test_an_unmarked_failure_reports_as_retryable(self) -> None:
        kit = await _room(SimpleChannel("in"), BrokenChannel("out", RuntimeError("boom")))

        result = await kit.process_inbound(_msg("in"))

        failed = result.delivery_results["out"]
        assert failed.error is not None
        assert failed.error.code == "RuntimeError"
        assert failed.error.retryable is True

    async def test_the_map_covers_every_channel_in_the_set(self) -> None:
        kit = await _room(
            SimpleChannel("in"), SimpleChannel("ok"), BrokenChannel("bad", RuntimeError("no"))
        )

        result = await kit.process_inbound(_msg("in"))

        assert result.delivery_results["ok"].status == "sent"
        assert result.delivery_results["bad"].status == "failed"


class TestOnlyAFailingSetIsRemembered:
    async def test_a_clean_delivery_writes_nothing(self) -> None:
        kit = await _room(SimpleChannel("in"), SimpleChannel("out"))

        result = await kit.process_inbound(_msg("in"))
        stored = await kit.store.get_event(result.event.id)

        assert stored.delivery_results == {}

    async def test_a_failure_survives_the_call(self) -> None:
        """The question an operator asks hours later, once the live event is gone."""
        kit = await _room(SimpleChannel("in"), BrokenChannel("out", _RejectedError("recipient")))

        result = await kit.process_inbound(_msg("in"))
        stored = await kit.store.get_event(result.event.id)

        assert stored.delivery_results["out"]["status"] == "failed"
        assert stored.delivery_results["out"]["error"]["code"] == "550"

    async def test_the_record_keeps_the_successes_too(self) -> None:
        """A list of casualties with no denominator answers half the question."""
        kit = await _room(
            SimpleChannel("in"), SimpleChannel("ok"), BrokenChannel("bad", RuntimeError("no"))
        )

        result = await kit.process_inbound(_msg("in"))
        stored = await kit.store.get_event(result.event.id)

        assert stored.delivery_results["ok"]["status"] == "sent"
        assert stored.delivery_results["bad"]["status"] == "failed"


@pytest.mark.parametrize("channel_count", [1, 3])
async def test_the_source_channel_is_not_in_its_own_delivery_set(channel_count: int) -> None:
    others = [SimpleChannel(f"out{i}") for i in range(channel_count)]
    kit = await _room(SimpleChannel("in"), *others)

    result = await kit.process_inbound(_msg("in"))

    assert "in" not in result.delivery_results
    assert len(result.delivery_results) == channel_count
