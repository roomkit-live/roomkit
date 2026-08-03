"""Tests for the Buzz (Nostr relay) provider and source parser.

These tests carry no ``buzzkit`` dependency: the parser is pure and the provider
tests drive a fake source, so the whole delivery + inbound path is covered
without a live relay.
"""

from __future__ import annotations

import asyncio
import time
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
        assert cfg.owner_pubkey is None
        assert cfg.obey_owner_commands is True

    def test_owner_pubkey_is_normalized_and_validated(self) -> None:
        cfg = BuzzConfig(
            relay_url="wss://relay", private_key="nsec1x", owner_pubkey=" " + "A" * 64 + " "
        )
        assert cfg.owner_pubkey == "a" * 64
        for bad in ("a" * 63, "z" * 64, "npub1abc"):
            with pytest.raises(ValueError):
                BuzzConfig(relay_url="wss://relay", private_key="nsec1x", owner_pubkey=bad)


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

    #: Class-level so the fixture factory can arm a verified owner before the
    #: source constructs its own client instance.
    verified_owner: str | None = None

    def __init__(self, relay_url: str, secret: str, *, auth_tag: Any = None) -> None:
        self.pubkey_hex = "bot_pubkey"
        self.npub = "npub_bot"
        self.verified_owner_hex: str | None = type(self).verified_owner
        self.close_code: int | None = None
        self.events: list[dict[str, Any]] = []
        self.raise_after: Exception | None = None
        self.subscribe_kinds: Any = "unset"
        self.joined: list[str] = []
        self.left: list[str] = []
        self.presence: list[str] = []
        self.presence_error: Exception | None = None  # raised once, then cleared
        self.released = asyncio.Event()

    async def connect(self) -> None: ...

    async def join_channel(self, channel_id: str) -> dict[str, Any]:
        self.joined.append(channel_id)
        return {"accepted": True, "message": ""}

    async def leave_channel(self, channel_id: str) -> dict[str, Any]:
        self.left.append(channel_id)
        return {"accepted": True, "message": ""}

    async def publish_presence(self, status: str = "online") -> dict[str, Any]:
        if self.presence_error is not None:
            exc, self.presence_error = self.presence_error, None
            raise exc
        self.presence.append(status)
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


def _stand_in_parse_owner_command(event: dict[str, Any], agent_pubkey_hex: str) -> str | None:
    """Faithful stand-in for ``buzzkit.parse_owner_command`` (tested there)."""
    if event.get("kind") != 9:
        return None
    content = str(event.get("content", "") or "").strip()
    if not content.startswith("!") or content[1:] not in ("shutdown", "cancel", "rotate"):
        return None
    tags = event.get("tags") or []
    mentioned = any(
        isinstance(t, list) and len(t) >= 2 and t[:2] == ["p", agent_pubkey_hex] for t in tags
    )
    return content[1:] if mentioned else None


@pytest.fixture
def buzz_source(monkeypatch):
    """Factory building a BuzzRelaySource wired to a FakeBuzzClient."""
    import roomkit.sources.buzz as buzz_module

    monkeypatch.setattr(buzz_module, "HAS_BUZZKIT", True)
    monkeypatch.setattr(buzz_module, "BuzzClient", FakeBuzzClient)
    monkeypatch.setattr(buzz_module, "parse_owner_command", _stand_in_parse_owner_command)
    monkeypatch.setattr(buzz_module, "_INITIAL_BACKOFF", 0.01)
    monkeypatch.setattr(FakeBuzzClient, "verified_owner", None)

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

    async def test_presence_failure_retries_next_beat(
        self, buzz_source, monkeypatch, caplog
    ) -> None:
        """One failed heartbeat must not silently kill the presence loop."""
        import roomkit.sources.buzz as buzz_module

        monkeypatch.setattr(buzz_module, "_PRESENCE_INTERVAL", 0.01)
        source = buzz_source()
        source.client.presence_error = ConnectionError("beat lost")
        with caplog.at_level("WARNING", logger="roomkit.sources.buzz"):
            await _run_source_once(source)
        # The first beat failed, later beats landed ("online"), then stop()
        # published "offline" — the loop survived the failure.
        assert "online" in source.client.presence
        assert "retrying next beat" in caplog.text

    async def test_stop_publishes_offline(self, buzz_source) -> None:
        """A deliberate stop flips presence to offline instead of lapsing by TTL."""
        source = buzz_source()
        await _run_source_once(source)
        assert source.client.presence[0] == "online"
        assert source.client.presence[-1] == "offline"

    async def test_no_offline_when_presence_disabled(self, buzz_source) -> None:
        config = BuzzConfig(relay_url="wss://relay", private_key="nsec1x", announce_presence=False)
        source = buzz_source(config=config)
        await _run_source_once(source)
        assert source.client.presence == []


OWNER = "a" * 64


def _owner_event(
    content: str, *, pubkey: str = OWNER, id: str = "cmd1", created_at: int | None = None
) -> dict[str, Any]:
    # Fresh by default: commands older than the source's start are stale
    # (replay protection) and deliberately ignored.
    return _event(
        id=id,
        kind=9,
        pubkey=pubkey,
        content=content,
        tags=[["p", "bot_pubkey"]],
        created_at=created_at if created_at is not None else int(time.time()) + 5,
    )


class TestOwnerCommands:
    def _owned(self, buzz_source, **kwargs):
        config = BuzzConfig(relay_url="wss://relay", private_key="nsec1x", owner_pubkey=OWNER)
        return buzz_source(config=config, **kwargs)

    async def test_owner_shutdown_stops_the_source_and_never_reaches_the_pipeline(
        self, buzz_source
    ) -> None:
        source = self._owned(buzz_source)
        source.client.events = [_owner_event("!shutdown")]
        emitted: list[Any] = []

        async def emit(message: Any) -> Any:
            emitted.append(message)

        # No external stop(): the source must terminate ITSELF on the command.
        await asyncio.wait_for(source.start(emit), timeout=2)
        assert emitted == []
        assert source.client.presence[-1] == "offline"

    async def test_non_owner_shutdown_is_a_regular_message(self, buzz_source) -> None:
        source = self._owned(buzz_source)
        source.client.events = [_owner_event("!shutdown", pubkey="c" * 64)]
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["cmd1"]

    async def test_commands_are_inert_without_a_provable_owner(self, buzz_source) -> None:
        source = buzz_source()  # no auth_tag, no owner_pubkey
        source.client.events = [_owner_event("!shutdown")]
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["cmd1"]

    async def test_commands_are_inert_when_disabled(self, buzz_source) -> None:
        config = BuzzConfig(
            relay_url="wss://relay",
            private_key="nsec1x",
            owner_pubkey=OWNER,
            obey_owner_commands=False,
        )
        source = buzz_source(config=config)
        source.client.events = [_owner_event("!shutdown")]
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["cmd1"]

    async def test_verified_auth_tag_owner_wins_over_config(
        self, buzz_source, monkeypatch
    ) -> None:
        monkeypatch.setattr(FakeBuzzClient, "verified_owner", "b" * 64)
        source = self._owned(buzz_source)  # config says OWNER, attestation says b*64
        source.client.events = [_owner_event("!shutdown")]  # from OWNER: not the proven owner
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["cmd1"]

    async def test_callback_owns_every_command_including_shutdown(self, buzz_source) -> None:
        seen: list[str] = []

        async def on_cmd(command: str, event: dict[str, Any]) -> None:
            seen.append(command)

        source = self._owned(buzz_source, on_owner_command=on_cmd)
        source.client.events = [
            _owner_event("!cancel", id="c1"),
            _owner_event("!rotate", id="c2"),
            _owner_event("!shutdown", id="c3"),
            _event(id="msg1", kind=9, content="hello"),
        ]
        # The callback replaces the self-stop, so the source keeps running
        # until _run_source_once stops it — and the plain message still flows.
        emitted = await _run_source_once(source)
        assert seen == ["cancel", "rotate", "shutdown"]
        assert [m.external_id for m in emitted] == ["msg1"]

    async def test_cancel_and_rotate_are_consumed_without_callback(self, buzz_source) -> None:
        source = self._owned(buzz_source)
        source.client.events = [
            _owner_event("!cancel", id="c1"),
            _event(id="msg1", kind=9, content="still here"),
        ]
        emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["msg1"]

    async def test_replayed_stale_shutdown_is_ignored(self, buzz_source, caplog) -> None:
        """A !shutdown replayed from relay history must not kill a fresh start.

        The relay replays recent stored events on every (re)subscribe; an
        owner command issued before this start is consumed without action —
        neither obeyed nor forwarded to the pipeline (live-found bug: every
        restart died instantly until the command aged out of replay).
        """
        source = self._owned(buzz_source)
        source.client.events = [
            _owner_event("!shutdown", id="old", created_at=int(time.time()) - 300),
            _event(id="msg1", kind=9, content="still alive"),
        ]
        with caplog.at_level("INFO", logger="roomkit.sources.buzz"):
            emitted = await _run_source_once(source)
        assert [m.external_id for m in emitted] == ["msg1"]
        assert "stale owner command" in caplog.text
        assert source.client.presence[-1] == "offline"  # stopped by the test, not the command

    async def test_command_issued_after_start_is_honored_on_replay(self, buzz_source) -> None:
        """A !shutdown issued while disconnected still stops the agent when
        the reconnect replays it: created_at >= the source's start time."""
        source = self._owned(buzz_source)
        source.client.events = [_owner_event("!shutdown", created_at=int(time.time()) + 30)]
        emitted: list[Any] = []

        async def emit(message: Any) -> Any:
            emitted.append(message)

        await asyncio.wait_for(source.start(emit), timeout=2)
        assert emitted == []


class TestBuzzConfigFromEnv:
    def test_reads_the_reserved_triplet(self, monkeypatch) -> None:
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1fromenv")
        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.env")
        monkeypatch.setenv("BUZZ_AUTH_TAG", '["auth","o","",""]')
        cfg = BuzzConfig.from_env(owner_pubkey="b" * 64)
        assert cfg.private_key.get_secret_value() == "nsec1fromenv"
        assert cfg.relay_url == "wss://relay.env"
        assert cfg.auth_tag == '["auth","o","",""]'
        assert cfg.owner_pubkey == "b" * 64

    def test_nostr_private_key_alias(self, monkeypatch) -> None:
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("NOSTR_PRIVATE_KEY", "nsec1alias")
        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.env")
        monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
        cfg = BuzzConfig.from_env()
        assert cfg.private_key.get_secret_value() == "nsec1alias"
        assert cfg.auth_tag is None

    def test_identity_is_fail_closed(self, monkeypatch) -> None:
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "")
        monkeypatch.delenv("NOSTR_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.env")
        with pytest.raises(ValueError, match="identityless"):
            BuzzConfig.from_env()
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.delenv("BUZZ_RELAY_URL")
        with pytest.raises(ValueError, match="BUZZ_RELAY_URL"):
            BuzzConfig.from_env()


class TestBuzzAgent:
    def _agent(self, buzz_source, **agent_kwargs):
        from roomkit import RoomKit
        from roomkit.providers.buzz import BuzzAgent

        config = BuzzConfig(relay_url="wss://relay", private_key="nsec1x", owner_pubkey=OWNER)
        source = buzz_source(config=config)
        kit = RoomKit()
        return kit, source, BuzzAgent(kit, [source], **agent_kwargs)

    def test_constructor_validates(self, buzz_source) -> None:
        from roomkit import RoomKit
        from roomkit.providers.buzz import BuzzAgent

        with pytest.raises(ValueError, match="at least one source"):
            BuzzAgent(RoomKit(), [])
        source = buzz_source()
        with pytest.raises(ValueError, match="positive"):
            BuzzAgent(RoomKit(), [source], exit_after_inactivity=0)

    async def test_owner_shutdown_closes_everything(self, buzz_source) -> None:
        from roomkit.providers.buzz import BuzzAgentStopCause

        kit, source, agent = self._agent(buzz_source)
        source.client.events = [_owner_event("!shutdown")]
        cause = await asyncio.wait_for(agent.run(), timeout=2)
        assert cause is BuzzAgentStopCause.OWNER_SHUTDOWN
        assert kit._closed is True
        assert source.client.presence[-1] == "offline"
        with pytest.raises(RuntimeError, match="single-shot"):
            await agent.run()

    async def test_inactivity_reaps_a_quiet_agent(self, buzz_source, monkeypatch) -> None:
        import roomkit.providers.buzz.agent as agent_module
        from roomkit.providers.buzz import BuzzAgentStopCause

        monkeypatch.setattr(agent_module, "_INACTIVITY_CHECK_CAP", 0.02)
        kit, source, agent = self._agent(buzz_source, exit_after_inactivity=0.1)
        cause = await asyncio.wait_for(agent.run(), timeout=2)
        assert cause is BuzzAgentStopCause.INACTIVITY
        assert kit._closed is True

    async def test_cancel_and_rotate_pass_through_and_shutdown_stays_ours(
        self, buzz_source
    ) -> None:
        from roomkit.providers.buzz import BuzzAgentStopCause

        seen: list[str] = []
        kit, source, agent = self._agent(buzz_source, on_owner_command=lambda c, e: seen.append(c))
        source.client.events = [
            _owner_event("!cancel", id="c1"),
            _owner_event("!rotate", id="c2"),
            _owner_event("!shutdown", id="c3"),
        ]
        cause = await asyncio.wait_for(agent.run(), timeout=2)
        assert seen == ["cancel", "rotate"]
        assert cause is BuzzAgentStopCause.OWNER_SHUTDOWN

    async def test_takes_over_an_existing_callback_with_a_warning(
        self, buzz_source, caplog
    ) -> None:
        kit, source, agent = self._agent(buzz_source)
        source.on_owner_command = lambda c, e: None
        source.client.events = [_owner_event("!shutdown")]
        with caplog.at_level("WARNING", logger="roomkit.providers.buzz.agent"):
            await asyncio.wait_for(agent.run(), timeout=2)
        assert "takes over on_owner_command" in caplog.text


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
