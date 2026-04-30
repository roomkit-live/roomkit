"""Tests for partial-reconfigure field preservation in GeminiLiveProvider.

When ``reconfigure`` is called with only some fields (e.g. only
``system_prompt`` after a skill activation), the others must be
preserved from the session's current effective config — otherwise
``_build_config`` (which treats ``None`` as "absent") silently drops
the tools/voice/temperature, leaving the model with no functions to
call.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.gemini.realtime import (
    GeminiLiveProvider,
    _GeminiSessionState,
)
from roomkit.voice.base import VoiceSession, VoiceSessionState


def _make_session() -> VoiceSession:
    from uuid import uuid4

    return VoiceSession(
        id=uuid4().hex,
        room_id="room-1",
        participant_id="user-1",
        channel_id="rt-gemini",
        state=VoiceSessionState.ACTIVE,
    )


def _populate_session_state(
    provider: GeminiLiveProvider,
    session: VoiceSession,
    *,
    system_prompt: str | None = "Initial prompt.",
    voice: str | None = "Aoede",
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = 0.7,
) -> _GeminiSessionState:
    """Inject a session state as if connect() had run, without real I/O."""
    if tools is None:
        tools = [
            {
                "name": "lookup_phone",
                "description": "Find a contact",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    state = _GeminiSessionState(
        session=session,
        live_session=MagicMock(),  # truthy is enough — _reconnect is patched
        ctxmgr=MagicMock(),
        live_config=MagicMock(),
        started_at=time.monotonic(),
        system_prompt=system_prompt,
        voice=voice,
        tools=tools,
        temperature=temperature,
    )
    provider._sessions[session.id] = state
    return state


async def _noop_receive_loop(session: VoiceSession) -> None:
    """Drop-in for _receive_loop that exits immediately."""
    return None


@pytest.fixture
def provider() -> GeminiLiveProvider:
    p = GeminiLiveProvider(api_key="dummy")
    # Stub network-touching parts. _receive_loop must also be neutered —
    # reconfigure spawns a fresh receive task at the end and the real
    # implementation would block on the MagicMock live_session.
    p._reconnect = AsyncMock()  # type: ignore[method-assign]
    p._receive_loop = _noop_receive_loop  # type: ignore[method-assign]
    return p


class TestReconfigurePreservation:
    async def test_partial_prompt_only_preserves_tools_voice_temperature(
        self, provider: GeminiLiveProvider
    ) -> None:
        """system_prompt-only reconfigure must NOT wipe tools/voice/temperature."""
        captured: list[dict[str, Any]] = []
        original_build = provider._build_config

        def spy_build(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return original_build(**kwargs)

        provider._build_config = spy_build  # type: ignore[method-assign]

        session = _make_session()
        original_tools = [
            {
                "name": "tool_a",
                "description": "first",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        _populate_session_state(
            provider,
            session,
            system_prompt="Old prompt.",
            voice="Aoede",
            tools=original_tools,
            temperature=0.7,
        )

        await provider.reconfigure(session, system_prompt="New prompt.")

        # _build_config got called with the preserved values, NOT None.
        assert len(captured) == 1
        kw = captured[0]
        assert kw["system_prompt"] == "New prompt."
        assert kw["voice"] == "Aoede"
        assert kw["tools"] == original_tools
        assert kw["temperature"] == 0.7

    async def test_partial_tools_only_preserves_prompt_voice_temperature(
        self, provider: GeminiLiveProvider
    ) -> None:
        """tools-only reconfigure (e.g. Tool Search) must preserve everything else."""
        captured: list[dict[str, Any]] = []
        original_build = provider._build_config

        def spy_build(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return original_build(**kwargs)

        provider._build_config = spy_build  # type: ignore[method-assign]

        session = _make_session()
        _populate_session_state(
            provider,
            session,
            system_prompt="Persistent prompt.",
            voice="Charon",
            temperature=0.5,
        )

        new_tools = [
            {
                "name": "newly_revealed",
                "description": "found via search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        await provider.reconfigure(session, tools=new_tools)

        kw = captured[0]
        assert kw["tools"] == new_tools
        assert kw["system_prompt"] == "Persistent prompt."
        assert kw["voice"] == "Charon"
        assert kw["temperature"] == 0.5

    async def test_explicit_empty_list_clears_tools(self, provider: GeminiLiveProvider) -> None:
        """Passing tools=[] is "clear", not "preserve"; only None preserves."""
        captured: list[dict[str, Any]] = []
        original_build = provider._build_config

        def spy_build(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return original_build(**kwargs)

        provider._build_config = spy_build  # type: ignore[method-assign]

        session = _make_session()
        _populate_session_state(
            provider,
            session,
            tools=[
                {
                    "name": "old_tool",
                    "description": "to be cleared",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )

        await provider.reconfigure(session, tools=[])

        kw = captured[0]
        assert kw["tools"] == []  # explicit empty propagates

    async def test_state_updated_for_next_reconfigure(self, provider: GeminiLiveProvider) -> None:
        """State must remember the effective values, so a chain of partials
        keeps building on the latest config — not on the original.
        """
        original_build = provider._build_config

        captured: list[dict[str, Any]] = []

        def spy_build(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return original_build(**kwargs)

        provider._build_config = spy_build  # type: ignore[method-assign]

        session = _make_session()
        state = _populate_session_state(
            provider,
            session,
            system_prompt="P0",
            voice="V0",
            tools=[],
            temperature=0.7,
        )

        # First reconfigure: change prompt only.
        await provider.reconfigure(session, system_prompt="P1")
        assert state.system_prompt == "P1"
        assert state.voice == "V0"  # preserved

        # Second reconfigure: change voice only — must NOT revert to P0.
        await provider.reconfigure(session, voice="V1")
        assert state.system_prompt == "P1"  # carried forward, not P0
        assert state.voice == "V1"
        # The build_config call for the second reconfigure must reflect this.
        assert captured[1]["system_prompt"] == "P1"
        assert captured[1]["voice"] == "V1"

    async def test_no_session_is_no_op(self, provider: GeminiLiveProvider) -> None:
        """reconfigure for an unknown session returns silently."""
        provider._build_config = MagicMock()  # type: ignore[method-assign]
        session = _make_session()  # never registered with provider
        await provider.reconfigure(session, system_prompt="anything")
        provider._build_config.assert_not_called()
