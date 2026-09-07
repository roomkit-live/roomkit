"""ACP usage provenance through the transport and framework hook boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

import acp
import pytest
from acp.schema import (
    ConfigOptionUpdate,
    Cost,
    PromptResponse,
    Usage,
    UsageUpdate,
)

from roomkit import ACPChannel
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.memory import InMemoryRealtime
from tests.conftest import make_event
from tests.test_channels.test_acp import (
    TestEndOfTurnReport as _EndOfTurnReport,
)
from tests.test_channels.test_acp import (
    _binding,
    _context,
    _FakeACPConnection,
    _FakeTransport,
    _model_option,
    _prompt,
)


class _UsageTransport(_FakeTransport):
    @property
    def provides_usage_metadata(self) -> bool:
        return True


def _channel(tmp_path: Any, *, emit_updates: bool = False) -> tuple[ACPChannel, Any, Any]:
    connection = _FakeACPConnection(None, emit_updates=emit_updates)
    transport = _UsageTransport(connection)
    channel = ACPChannel("acp-agent", transport=transport, cwd=tmp_path)
    connection.client = channel._client
    return channel, connection, transport


def _envelope(session: str = "original-session") -> dict[str, Any]:
    return {
        "usage_protocol": 1,
        "session_id": session,
        "session_epoch": "epoch-1",
        "node_id": "node-1",
        "agent_id": "adapter-1",
        "adapter_info": '{"name":"test-adapter","version":"1.0"}',
        "result_id": "result-2",
        "turn_id": "turn-2",
        "generation": 0,
        "replayed": True,
        "usage_report": {
            "report_id": "report-1",
            "observed_at_ms": 1234,
            "source": "session/update",
            "scope": "session",
            "source_result_id": "result-1",
            "update": {
                "sessionUpdate": "usage_update",
                "used": 0,
                "size": 100,
                "cost": {"amount": 0, "currency": "USD"},
            },
        },
    }


def _update(**kwargs: Any) -> UsageUpdate:
    return UsageUpdate(session_update="usage_update", used=0, size=100, **kwargs)


class TestUsageProvenance:
    async def test_model_snapshots_and_no_carry_into_next_prompt(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        notices: list[Any] = []

        async def capture(event: Any) -> None:
            if event.data.get("type") == "acp_usage":
                notices.append(event.data)

        await realtime.subscribe_to_room("room-1", capture)

        async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            await connection.client.session_update(
                session_id, _update(cost=Cost(amount=0, currency="USD"))
            )
            await connection.client.session_update(
                session_id,
                ConfigOptionUpdate(
                    session_update="config_option_update", config_options=[_model_option("sonnet")]
                ),
            )
            return PromptResponse(
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        connection.prompt = prompt
        first = make_event(body="first")
        await _prompt(channel, first, _context())
        observation = reports[0].usage_metadata
        assert observation["session_id"] == "session-1"
        assert observation["event_id"] == first.id
        assert observation["adapter_info"]["name"] == "fake-agent"
        assert observation["prompt"] == {
            "source": "session/prompt",
            "scope": "unspecified",
            "model_at_start": "opus",
            "stop_reason": "end_turn",
        }
        assert observation["usage_report"]["model_at_observation"] == "opus"
        assert "source_result_id" not in observation["usage_report"]
        assert reports[0].usage["cost"] == reports[0].usage["input_tokens"] == 0
        assert (
            notices[0]["usage_metadata"]["usage_report"]["report_id"]
            == observation["usage_report"]["report_id"]
        )

        # A notification between prompts remains observable but cannot become
        # the next prompt's report, even though the session is unchanged.
        await connection.client.session_update(
            "session-1", _update(cost=Cost(amount=5, currency="EUR"))
        )

        async def empty(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            return PromptResponse(stop_reason="end_turn")

        connection.prompt = empty
        await _prompt(channel, make_event(body="second"), _context())
        assert reports[1].usage == {}
        assert "usage_report" not in reports[1].usage_metadata
        assert reports[1].usage_metadata["prompt"]["model_at_start"] == "sonnet"
        assert observation["usage_report"]["update"]["cost"]["amount"] == 0
        await channel.close()
        await realtime.close()

    @pytest.mark.parametrize("with_report", [False, True])
    async def test_recovered_snapshot_is_authoritative_and_repeatable(
        self, tmp_path: Any, with_report: bool
    ) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)
        envelope = _envelope()
        if not with_report:
            envelope.pop("usage_report")

        async def recovered(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            await connection.client.session_update(
                session_id, _update(cost=Cost(amount=99, currency="EUR"))
            )
            return PromptResponse(
                stop_reason="cancelled",
                usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
                field_meta={"roomkit.live/usage": envelope},
            )

        connection.prompt = recovered
        for body in ("recover", "replay"):
            await _prompt(channel, make_event(body=body), _context())
        for event in reports:
            meta = event.usage_metadata
            assert meta["session_id"] == "original-session"
            assert meta["session_epoch"] == "epoch-1"
            assert meta["adapter_info"] == envelope["adapter_info"]
            assert meta["generation"] == 0
            assert meta["result_id"] == "result-2"
            assert meta["prompt"] == {
                "source": "session/prompt",
                "scope": "unspecified",
                "stop_reason": "cancelled",
            }
            assert event.usage["input_tokens"] == 1
            if with_report:
                assert meta["usage_report"] == envelope["usage_report"]
                assert meta["usage_report"]["source_result_id"] != meta["result_id"]
                assert event.usage["cost"] == 0
                assert event.usage["currency"] == "USD"
            else:
                assert "usage_report" not in meta
                assert "cost" not in event.usage
        envelope["session_id"] = "mutated"
        assert reports[0].usage_metadata["session_id"] == "original-session"
        await channel.close()

    async def test_wrong_session_notification_does_not_poison_active_turn(
        self, tmp_path: Any
    ) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)

        async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            await connection.client.session_update(
                session_id, _update(field_meta={"roomkit.live/usage": _envelope("old-session")})
            )
            return PromptResponse(stop_reason="end_turn")

        connection.prompt = prompt
        await _prompt(channel, make_event(body="new session"), _context())
        assert reports[0].usage == {}
        assert "usage_report" not in reports[0].usage_metadata
        assert reports[0].usage_metadata["session_id"] == "session-1"
        await channel.close()

    async def test_late_durable_report_preserves_origin_without_claiming_prompt(
        self, tmp_path: Any
    ) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)
        envelope = _envelope("session-1")

        async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            await connection.client.session_update(
                session_id, _update(field_meta={"roomkit.live/usage": envelope})
            )
            return PromptResponse(stop_reason="end_turn")

        connection.prompt = prompt
        await _prompt(channel, make_event(body="new prompt"), _context())
        meta = reports[0].usage_metadata
        assert meta["usage_report"]["source_result_id"] == "result-1"
        assert meta["usage_report"]["report_id"] == "report-1"
        assert "result_id" not in meta
        assert meta["session_epoch"] == "epoch-1"
        await channel.close()

    async def test_same_native_id_after_reset_keeps_distinct_epochs(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)
        envelope = _envelope("reused-native-id")

        async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            return PromptResponse(
                stop_reason="end_turn", field_meta={"roomkit.live/usage": envelope}
            )

        connection.prompt = prompt
        await _prompt(channel, make_event(body="first"), _context())
        await channel.close_session("room-1")
        envelope["session_epoch"] = "epoch-2"
        await _prompt(channel, make_event(body="second"), _context())
        assert [e.usage_metadata["session_epoch"] for e in reports] == ["epoch-1", "epoch-2"]
        assert all(e.usage_metadata["session_id"] == "reused-native-id" for e in reports)
        await channel.close()

    @pytest.mark.parametrize("failure", ["error", "abandon"])
    async def test_interrupted_usage_remains_observable_without_response(
        self, tmp_path: Any, failure: str
    ) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        reports = _EndOfTurnReport._capture(channel)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        notices: list[Any] = []

        async def capture(event: Any) -> None:
            if event.data.get("type") == "acp_usage":
                notices.append(event.data)

        await realtime.subscribe_to_room("room-1", capture)

        async def interrupted(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
            await connection.client.session_update(session_id, _update())
            if failure == "error":
                raise RuntimeError("lost agent")
            await connection.client.session_update(
                session_id, acp.update_agent_message_text("partial")
            )
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        connection.prompt = interrupted
        output = await channel.on_event(make_event(body="go"), _binding(), _context())
        if failure == "error":
            with pytest.raises(ProviderError):
                _ = [chunk async for chunk in output.response_stream]
        else:
            assert await anext(output.response_stream) == "partial"
            await output.response_stream.aclose()
        await asyncio.sleep(0)
        assert reports == []
        assert notices[0]["usage_metadata"]["session_id"] == "session-1"
        assert "cost" not in notices[0]["usage_metadata"]["usage_report"]["update"]
        await channel.close()
        await realtime.close()
