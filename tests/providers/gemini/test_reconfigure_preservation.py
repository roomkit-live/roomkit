"""Tests for partial-reconfigure field preservation in GeminiLiveProvider.

When ``reconfigure`` is called with only some fields (e.g. only
``system_prompt`` after a skill activation), the others must be
preserved from the session's current effective config — otherwise
``_build_config`` (which treats ``None`` as "absent") silently drops
the tools/voice/temperature, leaving the model with no functions to
call.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
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


class TestReconfigureCancellation:
    async def test_a_cancelled_caller_gets_its_cancellation_after_the_switch(
        self, provider: GeminiLiveProvider
    ) -> None:
        """RMK-288: the caller's cancellation, landing while reconfigure waits
        for the old receive task, reaches the caller once the session is
        switched, never halfway."""
        session = _make_session()
        state = _populate_session_state(provider, session)
        release = asyncio.Event()

        async def old_receive_loop() -> None:
            try:
                await asyncio.sleep(999)
            finally:
                # Its teardown outlasts the caller's own cancellation.
                await release.wait()

        state.receive_task = asyncio.create_task(old_receive_loop())
        await asyncio.sleep(0)
        caller = asyncio.create_task(provider.reconfigure(session, system_prompt="New."))
        await asyncio.sleep(0)
        old_loop = state.receive_task
        caller.cancel()
        await asyncio.wait({caller}, timeout=0.05)
        assert not caller.done()  # the switch finishes first
        release.set()
        await asyncio.wait({caller, old_loop}, timeout=1.0)

        # Cancelled, and the session is whole: reconnected under its new
        # config, with a receive loop of its own (RMK-288)
        assert caller.cancelled()
        provider._reconnect.assert_awaited_once()  # type: ignore[attr-defined]
        assert state.system_prompt == "New."
        assert state.receive_task is not None and state.receive_task is not old_loop


class TestReconfigureResumption:
    """RMK-288: gemini-3.8-live resumes a session under its original
    instruction, so a session with nothing in it reconnects fresh."""

    async def _handle_at_reconnect(
        self, provider: GeminiLiveProvider, state: _GeminiSessionState
    ) -> str | None:
        seen: list[str | None] = []
        provider._reconnect = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda session: seen.append(state.resumption_handle)
        )
        await provider.reconfigure(state.session, system_prompt="You are Bill.")
        return seen[0]

    async def test_a_session_with_no_conversation_reconnects_fresh(
        self, provider: GeminiLiveProvider
    ) -> None:
        state = _populate_session_state(provider, _make_session())
        state.resumption_handle = "handle-1"

        assert await self._handle_at_reconnect(provider, state) is None

    async def test_a_session_with_a_conversation_resumes(
        self, provider: GeminiLiveProvider
    ) -> None:
        state = _populate_session_state(provider, _make_session())
        state.resumption_handle = "handle-1"
        state.has_conversation = True

        assert await self._handle_at_reconnect(provider, state) == "handle-1"

    @pytest.mark.parametrize(
        ("model", "carried"), [("gemini-3.8-live", True), ("gemini-3.1-flash-live-preview", False)]
    )
    async def test_the_next_injection_carries_what_resumption_kept_out(
        self, provider: GeminiLiveProvider, model: str, carried: bool
    ) -> None:
        provider._model = model
        state = _populate_session_state(provider, _make_session())
        state.resumption_handle = "handle-1"
        state.has_conversation = True
        state.live_session = AsyncMock()

        await provider.reconfigure(state.session, system_prompt="You are Bill.")
        await provider.inject_text(state.session, "Greet the caller.", role="system")
        await provider.inject_text(state.session, "And again.", role="system")

        sent = [str(c) for c in state.live_session.send_client_content.call_args_list]
        assert ("You are Bill." in sent[0]) is carried
        assert "You are Bill." not in sent[1]  # carried once

    async def test_the_text_never_reads_as_more_of_the_instructions_it_carries(
        self, provider: GeminiLiveProvider
    ) -> None:
        """The instruction a resumption left unapplied is set apart in a block
        of its own before the text it rides with (RFC §12.4, RMK-591)."""
        provider._model = "gemini-3.8-live"
        state = _populate_session_state(provider, _make_session())
        state.resumption_handle = "handle-1"
        state.has_conversation = True
        state.live_session = AsyncMock()

        await provider.reconfigure(state.session, system_prompt="You are Bill.")
        await provider.inject_text(state.session, "Marie · sms: “hello”", role="user")

        [call] = state.live_session.send_client_content.call_args_list
        assert call.kwargs["turns"].parts[0].text == (
            "Your instructions have been replaced. From now on, follow only the ones in "
            "this block:\n<instructions>\nYou are Bill.\n</instructions>\n\n"
            "Marie · sms: “hello”"
        )

    async def test_a_fresh_session_carries_no_instructions(
        self, provider: GeminiLiveProvider
    ) -> None:
        provider._model = "gemini-3.8-live"
        state = _populate_session_state(provider, _make_session())
        state.resumption_handle = "handle-1"
        state.live_session = AsyncMock()

        await provider.reconfigure(state.session, system_prompt="You are Bill.")

        assert state.pending_instructions is None

    @pytest.mark.parametrize("kind", ["user_text", "model_turn", "tool_call"])
    async def test_what_counts_as_a_conversation(
        self, provider: GeminiLiveProvider, kind: str
    ) -> None:
        session = _make_session()
        state = _populate_session_state(provider, session)
        state.live_session = AsyncMock()
        if kind == "user_text":
            await provider.inject_text(session, "Hello", role="user")
        elif kind == "model_turn":
            content = SimpleNamespace(model_turn=SimpleNamespace(parts=[]))
            await provider._on_server_content(session, state, content)
        else:
            await provider._on_tool_call(session, state, SimpleNamespace(function_calls=[]))

        assert state.has_conversation is True

    async def test_audio_alone_is_no_conversation(self, provider: GeminiLiveProvider) -> None:
        session = _make_session()
        state = _populate_session_state(provider, session)
        state.live_session = AsyncMock()

        await provider.send_audio(session, b"\x00\x00" * 160)

        assert state.has_conversation is False
