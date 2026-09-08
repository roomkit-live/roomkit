"""Effective per-session configuration survives handoffs and partial updates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_providers.test_gemini_realtime import _build_fake_genai
from tests.test_realtime_skills import _registry_with_skill


def tool(name: str, kind: str = "string") -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": kind}},
            "required": ["value"],
        },
    }


async def test_handoff_replaces_tool_names_and_schemas() -> None:
    provider = MockRealtimeProvider()
    provider.reconfigure = AsyncMock()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[tool("old"), tool("same")],
    )
    active = await channel.start_session("r", "p", object())
    other = await channel.start_session("r2", "p2", object())
    try:
        await channel.reconfigure_session(active, tools=[tool("new"), tool("same", "integer")])
        for name, args, allowed in [
            ("new", {"value": "ok"}, True),
            ("old", {"value": "ok"}, False),
            ("same", {"value": 7}, True),
            ("same", {"value": "bad"}, False),
        ]:
            _, denial, _ = await channel._authorize_realtime_tool(name, args, "call", None, active)
            assert (denial is None) is allowed
        assert channel._is_declared_realtime_tool("old", other)
        assert not channel._is_declared_realtime_tool("new", other)
        assert channel._known_tool_names(active.id) == {"new", "same"}
        assert channel._tool_param_types("same", active.id) == {"value": "integer"}
    finally:
        await channel.close()


async def test_empty_session_catalogue_does_not_fall_back_to_defaults() -> None:
    channel = RealtimeVoiceChannel(
        "rt",
        provider=MockRealtimeProvider(),
        transport=MockRealtimeTransport(),
        tools=[tool("old")],
    )
    active = await channel.start_session("r", "p", object(), metadata={"tools": []})
    try:
        # Empty catalogues keep the documented dynamic-handler mode, without
        # resurrecting defaults in schema validation or textual recovery.
        assert channel._tool_parameters("old", active) is None
        assert channel._known_tool_names(active.id) == set()
        assert channel._is_declared_realtime_tool("dynamic", active)
    finally:
        await channel.close()


async def test_failed_handoff_preserves_catalogue() -> None:
    provider = MockRealtimeProvider()
    provider.reconfigure = AsyncMock(side_effect=ConnectionError("update failed"))
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[tool("old")],
        tool_search=True,
    )
    active = await channel.start_session("r", "p", object())
    try:
        with pytest.raises(ConnectionError):
            await channel.reconfigure_session(active, tools=[tool("new")])
        assert channel._known_tool_names(active.id) == {"old"}
        support = channel._tool_search_support
        assert support is not None
        result, _ = await support.handle_tool_call("list_tools", {}, active.id)
        assert "old" in result and "new" not in result
    finally:
        await channel.close()


async def test_tool_search_follows_each_sessions_catalogue_and_keeps_skills(
    tmp_path: Path,
) -> None:
    provider = MockRealtimeProvider()
    provider.reconfigure = AsyncMock()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[tool("default")],
        tool_search=True,
        skills=_registry_with_skill(tmp_path),
    )
    active = await channel.start_session("r", "p", object(), metadata={"tools": [tool("weather")]})
    other = await channel.start_session("r2", "p2", object())
    try:
        support = channel._tool_search_support
        assert support is not None
        result, _ = await support.handle_tool_call("find_tools", {"query": "weather"}, active.id)
        assert "weather" in result
        await channel.reconfigure_session(active, tools=[tool("calendar")])
        exposed = provider.reconfigure.await_args.kwargs["tools"]
        assert {"find_tools", "list_tools", "activate_skill"} <= {t["name"] for t in exposed}
        result, _ = await support.handle_tool_call("list_tools", {}, active.id)
        assert "calendar" in result and "weather" not in result
        result, _ = await support.handle_tool_call("list_tools", {}, other.id)
        assert "default" in result and "calendar" not in result
    finally:
        await channel.close()


async def test_gemini_partial_updates_preserve_vad_and_options() -> None:
    with patch.dict(sys.modules, _build_fake_genai()):
        from roomkit.providers.gemini.realtime import GeminiLiveProvider
        from roomkit.voice.base import VoiceSession

        provider = GeminiLiveProvider(api_key="test")
        provider._receive_loop = AsyncMock()
        cm = AsyncMock()
        provider._client.aio.live.connect = MagicMock(return_value=cm)
        active = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="rt")
        options = {"language": "fr-FR", "max_output_tokens": 512, "no_interruption": True}
        await provider.connect(active, server_vad=False, voice="Kore", provider_config=options)
        provider._reconnect = AsyncMock()
        try:
            options["language"] = "en-US"
            await provider.reconfigure(active, system_prompt="Updated")
            await provider.reconfigure(active, provider_config={"max_output_tokens": 256})
            state = provider._sessions[active.id]
            cfg = state.live_config
            assert cfg.realtime_input_config.automatic_activity_detection.disabled is True
            assert cfg.realtime_input_config.activity_handling == "NO_INTERRUPTION"
            assert cfg.speech_config.language_code == "fr-FR"
            assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
            assert cfg.max_output_tokens == 256
            assert state.system_prompt == "Updated"
            await provider.reconfigure(
                active, provider_config={"language": None, "no_interruption": False}
            )
            cfg = state.live_config
            assert not hasattr(cfg.speech_config, "language_code")
            assert not hasattr(cfg.realtime_input_config, "activity_handling")
            assert cfg.realtime_input_config.automatic_activity_detection.disabled is True
        finally:
            await provider.disconnect(active)
