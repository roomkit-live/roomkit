"""Tests for the Buzz (Nostr relay) provider and source parser.

These tests carry no ``buzzkit`` dependency: the parser is pure and the provider
tests drive a fake source, so the whole delivery + inbound path is covered
without a live relay.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels import BuzzChannel
from roomkit.core.framework import RoomKit
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.providers.buzz import BuzzConfig, BuzzProvider, MockBuzzProvider
from roomkit.sources.buzz import default_message_parser, parse_buzz_event

# =============================================================================
# Helpers
# =============================================================================


def _event(**overrides: Any) -> dict[str, Any]:
    """A minimal Nostr kind-9 event dict."""
    event = {
        "id": "abc123",
        "pubkey": "sender_pubkey_hex",
        "kind": 9,
        "content": "hello world",
        "tags": [["h", "chan-uuid"]],
        "created_at": 1_700_000_000,
    }
    event.update(overrides)
    return event


def _make_source(*, result: dict[str, Any] | None = None, client: bool = True) -> MagicMock:
    """Fake BuzzRelaySource exposing a client with an async send_message."""
    source = MagicMock()
    if client:
        c = MagicMock()
        c.send_message = AsyncMock(
            return_value=result
            if result is not None
            else {"accepted": True, "event_id": "evt1", "message": ""}
        )
        source.client = c
    else:
        source.client = None
    return source


def _make_event(content: Any) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="buzz-main", channel_type=ChannelType.BUZZ),
        content=content,
    )


# =============================================================================
# Config
# =============================================================================


class TestBuzzConfig:
    def test_requires_fields(self) -> None:
        with pytest.raises(ValueError):
            BuzzConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        cfg = BuzzConfig(relay_url="wss://relay", private_key="nsec1secret")
        assert cfg.relay_url == "wss://relay"
        assert cfg.private_key.get_secret_value() == "nsec1secret"
        assert cfg.ignore_own is True


# =============================================================================
# Inbound parsing
# =============================================================================


class TestParseBuzzEvent:
    def test_basic(self) -> None:
        msg = parse_buzz_event(_event(), "buzz-main")
        assert msg is not None
        assert msg.channel_id == "buzz-main"
        assert msg.sender_id == "sender_pubkey_hex"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "hello world"
        assert msg.external_id == "abc123"
        assert msg.idempotency_key == "abc123"
        assert msg.metadata["buzz_channel_id"] == "chan-uuid"
        assert msg.metadata["nostr_kind"] == 9

    def test_drops_own_event(self) -> None:
        assert parse_buzz_event(_event(pubkey="me"), "buzz-main", own_pubkey="me") is None

    def test_keeps_own_when_not_ignoring(self) -> None:
        msg = parse_buzz_event(_event(pubkey="me"), "buzz-main", own_pubkey="me", ignore_own=False)
        assert msg is not None

    def test_drops_empty_content(self) -> None:
        assert parse_buzz_event(_event(content=""), "buzz-main") is None

    def test_default_parser_binds_channel_and_policy(self) -> None:
        parser = default_message_parser("buzz-x", ignore_own=True)
        assert parser(_event(pubkey="me"), "me") is None
        msg = parser(_event(), "me")
        assert msg is not None
        assert msg.channel_id == "buzz-x"


# =============================================================================
# Mock provider
# =============================================================================


class TestMockBuzzProvider:
    async def test_records_sends(self) -> None:
        provider = MockBuzzProvider()
        result = await provider.send(_make_event(TextContent(body="hi")), to="chan-uuid")
        assert result.success
        assert provider.sent == [{"event": provider.sent[0]["event"], "to": "chan-uuid"}]


# =============================================================================
# Outbound provider (via shared client)
# =============================================================================


class TestBuzzProvider:
    async def test_send_text(self) -> None:
        source = _make_source()
        provider = BuzzProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi there")), to="chan-uuid")
        assert result.success
        assert result.provider_message_id == "evt1"
        source.client.send_message.assert_awaited_once_with("chan-uuid", "hi there")

    async def test_empty_message(self) -> None:
        provider = BuzzProvider(_make_source())
        result = await provider.send(_make_event(TextContent(body="")), to="chan-uuid")
        assert not result.success
        assert result.error == "empty_message"

    async def test_not_ready_guard(self) -> None:
        provider = BuzzProvider(_make_source(client=False))
        result = await provider.send(_make_event(TextContent(body="hi")), to="chan-uuid")
        assert not result.success
        assert result.error == "buzz_not_ready"

    async def test_relay_rejection(self) -> None:
        source = _make_source(result={"accepted": False, "event_id": "x", "message": "nope"})
        provider = BuzzProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi")), to="chan-uuid")
        assert not result.success

    async def test_send_failure_becomes_result(self) -> None:
        source = _make_source()
        source.client.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        provider = BuzzProvider(source)
        result = await provider.send(_make_event(TextContent(body="hi")), to="chan-uuid")
        assert not result.success
        assert "boom" in (result.error or "")


# =============================================================================
# Inbound pipeline integration
# =============================================================================


class TestBuzzInboundIntegration:
    async def test_parsed_message_flows_through_pipeline(self) -> None:
        kit = RoomKit()
        kit.register_channel(BuzzChannel("buzz-main", provider=MockBuzzProvider()))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "buzz-main")

        inbound = parse_buzz_event(_event(content="ping"), "buzz-main")
        assert inbound is not None
        result = await kit.process_inbound(inbound)

        assert not result.blocked
        assert result.event is not None
        assert result.event.content.body == "ping"
        assert result.event.source.channel_type == ChannelType.BUZZ
