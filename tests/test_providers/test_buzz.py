"""Tests for the Buzz (Nostr relay) provider and source parser.

These tests carry no ``buzzkit`` dependency: the parser is pure and the provider
tests drive a fake source, so the whole delivery + inbound path is covered
without a live relay.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels import BuzzChannel
from roomkit.core.framework import RoomKit
from roomkit.models.enums import ChannelType
from roomkit.models.event import ChannelData, EventSource, RoomEvent, TextContent
from roomkit.providers.buzz import BuzzConfig, BuzzProvider, MockBuzzProvider
from roomkit.sources.buzz import (
    KIND_DELETION,
    KIND_REACTION,
    KIND_STREAM_MESSAGE,
    default_message_parser,
    parse_buzz_event,
    parse_buzz_reaction,
)

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
        ok = (
            result if result is not None else {"accepted": True, "event_id": "evt1", "message": ""}
        )
        c.send_message = AsyncMock(return_value=ok)
        c.react = AsyncMock(return_value=ok)
        c.remove_reaction = AsyncMock(return_value=ok)
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

    def test_no_thread_tags_means_no_thread(self) -> None:
        msg = parse_buzz_event(_event(), "buzz-main")
        assert msg is not None
        assert msg.thread_id is None
        assert "nostr_reply_to" not in msg.metadata

    def test_direct_reply_thread_root(self) -> None:
        # A direct reply carries a single "reply"-marked e-tag: it IS the root.
        tags = [["h", "chan-uuid"], ["e", "root_id", "", "reply"]]
        msg = parse_buzz_event(_event(tags=tags), "buzz-main")
        assert msg is not None
        assert msg.thread_id == "root_id"
        assert msg.metadata["nostr_reply_to"] == "root_id"

    def test_nested_reply_resolves_to_root(self) -> None:
        tags = [
            ["h", "chan-uuid"],
            ["e", "root_id", "", "root"],
            ["e", "parent_id", "", "reply"],
        ]
        msg = parse_buzz_event(_event(tags=tags), "buzz-main")
        assert msg is not None
        assert msg.thread_id == "root_id"
        assert msg.metadata["nostr_reply_to"] == "parent_id"


class TestParseBuzzReaction:
    def test_reaction_add(self) -> None:
        ev = _event(kind=KIND_REACTION, content="🔥", tags=[["e", "target_id"]])
        data = parse_buzz_reaction(ev)
        assert data == {
            "action": "add",
            "emoji": "🔥",
            "user_id": "sender_pubkey_hex",
            "target_event_id": "target_id",
            "reaction_event_id": "abc123",
        }

    def test_deletion_becomes_remove(self) -> None:
        ev = _event(kind=KIND_DELETION, content="", tags=[["e", "reaction_id"]])
        data = parse_buzz_reaction(ev)
        assert data == {
            "action": "remove",
            "user_id": "sender_pubkey_hex",
            "reaction_event_id": "reaction_id",
        }

    def test_other_kinds_and_missing_target_are_none(self) -> None:
        assert parse_buzz_reaction(_event(kind=KIND_STREAM_MESSAGE)) is None
        assert parse_buzz_reaction(_event(kind=KIND_REACTION, tags=[["h", "c"]])) is None
        # A threaded chat message carries e-tags too — it is not a reaction.
        threaded = _event(kind=KIND_STREAM_MESSAGE, tags=[["e", "root_id", "", "reply"]])
        assert parse_buzz_reaction(threaded) is None


# =============================================================================
# Mock provider
# =============================================================================


class TestMockBuzzProvider:
    async def test_records_sends(self) -> None:
        provider = MockBuzzProvider()
        result = await provider.send(_make_event(TextContent(body="hi")), to="chan-uuid")
        assert result.success
        assert provider.sent == [{"event": provider.sent[0]["event"], "to": "chan-uuid"}]

    async def test_records_reactions(self) -> None:
        provider = MockBuzzProvider()
        assert (await provider.send_reaction("evt1", "👍")).success
        assert (await provider.remove_reaction("react1")).success
        assert provider.reactions == [
            {"action": "add", "target": "evt1", "emoji": "👍"},
            {"action": "remove", "target": "react1"},
        ]


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

    async def test_threaded_send_passes_reply_to(self) -> None:
        source = _make_source()
        provider = BuzzProvider(source)
        event = _make_event(TextContent(body="in thread")).model_copy(
            update={"channel_data": ChannelData(thread_id="root_nostr_id")}
        )
        result = await provider.send(event, to="chan-uuid")
        assert result.success
        source.client.send_message.assert_awaited_once_with(
            "chan-uuid", "in thread", reply_to="root_nostr_id"
        )

    async def test_send_reaction(self) -> None:
        source = _make_source()
        provider = BuzzProvider(source)
        result = await provider.send_reaction("target_evt", "🔥")
        assert result.success
        assert result.provider_message_id == "evt1"
        source.client.react.assert_awaited_once_with("target_evt", "🔥")

    async def test_remove_reaction(self) -> None:
        source = _make_source()
        provider = BuzzProvider(source)
        result = await provider.remove_reaction("reaction_evt")
        assert result.success
        source.client.remove_reaction.assert_awaited_once_with("reaction_evt")

    async def test_reaction_not_ready_and_failure(self) -> None:
        provider = BuzzProvider(_make_source(client=False))
        assert (await provider.send_reaction("evt", "👍")).error == "buzz_not_ready"
        source = _make_source()
        source.client.react = AsyncMock(side_effect=RuntimeError("down"))
        result = await BuzzProvider(source).send_reaction("evt", "👍")
        assert not result.success
        assert "down" in (result.error or "")


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


# =============================================================================
# Source lifecycle (fake client, no buzzkit)
# =============================================================================


class FakeBuzzClient:
    """Duck-typed ``buzzkit.BuzzClient`` driving the source's relay loop."""

    def __init__(self, relay_url: str, secret: str, *, auth_tag: Any = None) -> None:
        self.pubkey_hex = "bot_pubkey"
        self.npub = "npub_bot"
        self.close_code: int | None = None
        self.events: list[dict[str, Any]] = []
        self.raise_after: Exception | None = None
        self.subscribe_kinds: Any = "unset"
        self.joined: list[str] = []
        self.left: list[str] = []
        self.released = asyncio.Event()

    async def connect(self) -> None: ...

    async def join_channel(self, channel_id: str) -> dict[str, Any]:
        self.joined.append(channel_id)
        return {"accepted": True, "message": ""}

    async def leave_channel(self, channel_id: str) -> dict[str, Any]:
        self.left.append(channel_id)
        return {"accepted": True, "message": ""}

    async def publish_presence(self, status: str = "online") -> dict[str, Any]:
        return {}

    async def subscribe_channel(self, channel_id: str, *, kinds: Any = None) -> Any:
        self.subscribe_kinds = kinds
        for event in self.events:
            yield event
        if self.raise_after is not None:
            exc, self.raise_after = self.raise_after, None
            raise exc
        await self.released.wait()

    async def close(self) -> None:
        self.released.set()


@pytest.fixture
def buzz_source(monkeypatch):
    """Factory building a BuzzRelaySource wired to a FakeBuzzClient."""
    import roomkit.sources.buzz as buzz_module

    monkeypatch.setattr(buzz_module, "HAS_BUZZKIT", True)
    monkeypatch.setattr(buzz_module, "BuzzClient", FakeBuzzClient)
    monkeypatch.setattr(buzz_module, "_INITIAL_BACKOFF", 0.01)

    def factory(**kwargs):
        config = kwargs.pop(
            "config", BuzzConfig(relay_url="wss://relay", private_key="nsec1secret")
        )
        return buzz_module.BuzzRelaySource(
            config, "buzz-main", relay_channel_id="chan-uuid", **kwargs
        )

    return factory


async def _run_source_once(source: Any) -> list[Any]:
    """Run start() until the fake's events are consumed, then stop it."""
    emitted: list[Any] = []

    async def emit(message: Any) -> Any:
        emitted.append(message)

    task = asyncio.create_task(source.start(emit))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if source.client.subscribe_kinds != "unset":
            break
    await asyncio.sleep(0.05)
    await source.stop()
    await asyncio.wait_for(task, timeout=2)
    return emitted


class TestBuzzRelaySourceLifecycle:
    def test_on_event_widens_default_kinds(self, buzz_source) -> None:
        assert buzz_source()._kinds is None
        with_cb = buzz_source(on_event=lambda data: None)
        assert with_cb._kinds == [KIND_STREAM_MESSAGE, KIND_REACTION, KIND_DELETION]
        explicit = buzz_source(kinds=[48100], on_event=lambda data: None)
        assert explicit._kinds == [48100]

    async def test_reactions_go_to_callback_not_pipeline(self, buzz_source) -> None:
        seen: list[dict[str, Any]] = []
        source = buzz_source(on_event=seen.append)
        source.client.events = [
            _event(kind=KIND_REACTION, content="🔥", tags=[["e", "target_id"]]),
            _event(id="msg1", kind=9, content="hello"),
            _event(
                id="own_react",
                kind=KIND_REACTION,
                pubkey="bot_pubkey",
                content="👍",
                tags=[["e", "target_id"]],
            ),
        ]
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["msg1"]
        assert len(seen) == 1
        assert seen[0]["action"] == "add"
        assert seen[0]["emoji"] == "🔥"
        assert seen[0]["channel_id"] == "chan-uuid"

    async def test_async_callback_and_callback_errors_are_contained(self, buzz_source) -> None:
        calls: list[str] = []

        async def boom(data: dict[str, Any]) -> None:
            calls.append(data["action"])
            raise RuntimeError("handler crash")

        source = buzz_source(on_event=boom)
        source.client.events = [
            _event(kind=KIND_REACTION, content="🎉", tags=[["e", "t1"]]),
            _event(id="msg2", kind=9, content="still flowing"),
        ]
        emitted = await _run_source_once(source)
        assert calls == ["add"]
        assert [m.external_id for m in emitted] == ["msg2"]

    async def test_graceful_restart_is_not_an_error(self, buzz_source, caplog) -> None:
        source = buzz_source()
        source.client.raise_after = ConnectionError("gone")
        source.client.close_code = 1012
        with caplog.at_level("INFO", logger="roomkit.sources.buzz"):
            await _run_source_once(source)
        health = await source.healthcheck()
        assert health.error is None
        assert "relay restarting" in caplog.text
        assert "Buzz source buzz-main error" not in caplog.text

    async def test_unexpected_drop_logs_error(self, buzz_source, caplog) -> None:
        source = buzz_source()
        source.client.raise_after = ConnectionError("gone")
        with caplog.at_level("INFO", logger="roomkit.sources.buzz"):
            await _run_source_once(source)
        assert "Buzz source buzz-main error" in caplog.text

    async def test_leave_on_stop_opt_in(self, buzz_source) -> None:
        source = buzz_source()
        await source.stop()
        assert source.client.left == []
        config = BuzzConfig(relay_url="wss://relay", private_key="nsec1x", leave_on_stop=True)
        leaving = buzz_source(config=config)
        await leaving.stop()
        assert leaving.client.left == ["chan-uuid"]


class TestHuddleAnnouncementParser:
    def _announcement(self, **overrides) -> dict:
        event = {
            "id": "evt48100",
            "kind": 48100,
            "pubkey": "creator_pubkey",
            "created_at": 1_000,
            "content": '{"ephemeral_channel_id": "huddle-uuid"}',
            "tags": [["h", "parent-uuid"]],
        }
        event.update(overrides)
        return event

    def test_emits_message_with_huddle_id(self) -> None:
        from roomkit.sources.buzz import huddle_announcement_parser

        parser = huddle_announcement_parser("buzz-events")
        msg = parser(self._announcement(), None)
        assert msg is not None
        assert msg.channel_id == "buzz-events"
        assert msg.metadata["ephemeral_channel_id"] == "huddle-uuid"
        assert msg.idempotency_key == "evt48100"

    def test_drops_other_kinds_and_malformed_content(self) -> None:
        from roomkit.sources.buzz import huddle_announcement_parser

        parser = huddle_announcement_parser("buzz-events")
        assert parser(self._announcement(kind=9), None) is None
        assert parser(self._announcement(content="not json"), None) is None
        assert parser(self._announcement(content="{}"), None) is None

    def test_started_after_drops_replayed_history(self) -> None:
        from roomkit.sources.buzz import huddle_announcement_parser

        parser = huddle_announcement_parser("buzz-events", started_after=2_000)
        assert parser(self._announcement(created_at=1_999), None) is None
        assert parser(self._announcement(created_at=2_000), None) is not None
