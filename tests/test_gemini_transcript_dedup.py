"""Gemini Live: a re-emitted final transcription must not surface twice.

Gemini re-sends a finished utterance after the provider's buffer already
flushed it at a lifecycle boundary (speech end, model turn). Unfiltered,
every re-emission reached the channel as a second identical final — the
chat rendered duplicate user bubbles and the phantom "user speech" falsely
interrupted the assistant's streaming reply. The guard drops consecutive
identical finals per role and is lifted when new speech (ACTIVITY_START)
or a new response (model turn) genuinely begins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from roomkit.providers.gemini.realtime import _GeminiSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def provider():
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider(api_key="test-key")


@pytest.fixture
def session(provider):
    session = VoiceSession(
        id="sess-1", room_id="room-1", participant_id="user-1", channel_id="voice"
    )
    session.state = VoiceSessionState.ACTIVE
    provider._sessions[session.id] = _GeminiSessionState(session=session, started_at=0.0)
    return session


def _finals(calls: list[tuple[str, str, bool]]) -> list[tuple[str, str]]:
    return [(text, role) for text, role, final in calls if final]


@pytest.fixture
def calls(provider):
    received: list[tuple[str, str, bool]] = []
    provider.on_transcription(
        lambda sess, text, role, is_final: received.append((text, role, is_final))
    )
    return received


async def _activity(provider, session, vtype: str) -> None:
    state = provider._sessions[session.id]
    await provider._on_voice_activity(session, state, SimpleNamespace(voice_activity_type=vtype))


async def test_reemitted_finished_chunk_after_flush_is_dropped(provider, session, calls):
    await _activity(provider, session, "ACTIVITY_START")
    await provider._handle_transcription_chunk(session, "Salut, comment ça va ?", "user", False)
    await _activity(provider, session, "ACTIVITY_END")  # flush → final #1
    # Gemini re-sends the finished utterance after the flush.
    await provider._handle_transcription_chunk(session, "Salut, comment ça va ?", "user", True)

    assert _finals(calls) == [("Salut, comment ça va ?", "user")]


async def test_double_flush_emits_once(provider, session, calls):
    await provider._handle_transcription_chunk(session, "Une tarte", "assistant", True)
    # Barge-in + turn_complete both flush the assistant buffer; a re-buffered
    # identical transcript must not surface again.
    await provider._handle_transcription_chunk(session, "Une tarte", "assistant", True)

    assert _finals(calls) == [("Une tarte", "assistant")]


async def test_repeating_the_same_words_next_utterance_still_emits(provider, session, calls):
    await _activity(provider, session, "ACTIVITY_START")
    await provider._handle_transcription_chunk(session, "Oui", "user", True)
    # New utterance lifts the guard: saying the same thing again is legitimate.
    await _activity(provider, session, "ACTIVITY_START")
    await provider._handle_transcription_chunk(session, "Oui", "user", True)

    assert _finals(calls) == [("Oui", "user"), ("Oui", "user")]


async def test_identical_assistant_reply_next_turn_still_emits(provider, session, calls):
    state = provider._sessions[session.id]
    await provider._handle_transcription_chunk(session, "Bonjour !", "assistant", True)
    # New model turn lifts the assistant guard.
    await provider._on_server_content(
        session, state, SimpleNamespace(model_turn=object(), turn_complete=None)
    )
    await provider._handle_transcription_chunk(session, "Bonjour !", "assistant", True)

    assert _finals(calls) == [("Bonjour !", "assistant"), ("Bonjour !", "assistant")]


async def test_partials_are_untouched_by_the_guard(provider, session, calls):
    await provider._handle_transcription_chunk(session, "Sal", "user", False)
    await provider._handle_transcription_chunk(session, "Sal", "user", False)

    partials = [(text, final) for text, _, final in calls]
    assert partials == [("Sal", False), ("Sal", False)]


async def test_user_final_is_emitted_before_the_replys_first_chunk(provider, session, calls):
    """Captured wire case: no VAD events, and one server message carries both
    the reply's first transcript chunk and model_turn. The user final must go
    out before ANY assistant transcription — emitted after, it reads
    downstream as new user speech (phantom barge-in, duplicated user entry).
    """
    state = provider._sessions[session.id]
    await provider._handle_transcription_chunk(session, "Salut, comment ça va ?", "user", False)
    await provider._on_server_content(
        session,
        state,
        SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="Hey, salut", finished=False),
            model_turn=object(),
            interrupted=None,
            turn_complete=None,
        ),
    )

    assert calls == [
        ("Salut, comment ça va ?", "user", False),
        ("Salut, comment ça va ?", "user", True),
        ("Hey, salut", "assistant", False),
    ]
    assert state.response_started is True


async def test_tool_call_flushes_the_user_final_first(provider, session):
    """A tool round has no model_turn yet — the function_call itself is the
    model acting on the utterance, so the user final must precede it (the
    field capture: partials → tool line → late final → duplicated entry)."""
    state = provider._sessions[session.id]
    timeline: list[tuple[str, ...]] = []
    provider.on_transcription(
        lambda sess, text, role, is_final: timeline.append(("tx", text, role, str(is_final)))
    )
    provider.on_tool_call(lambda sess, call_id, name, args: timeline.append(("tool", name)))

    await provider._handle_transcription_chunk(session, "Cherche dans Luge", "user", False)
    await provider._on_tool_call(
        session,
        state,
        SimpleNamespace(function_calls=[SimpleNamespace(name="luge_cli", id="fc_1", args=None)]),
    )

    assert timeline == [
        ("tx", "Cherche dans Luge", "user", "False"),
        ("tx", "Cherche dans Luge", "user", "True"),
        ("tool", "luge_cli"),
    ]


async def test_reply_chunk_ahead_of_model_turn_still_flushes_the_user_first(
    provider, session, calls
):
    """Same inversion when the first output transcript arrives in a message
    of its own, before model_turn."""
    state = provider._sessions[session.id]
    await provider._handle_transcription_chunk(session, "Oui", "user", False)
    await provider._on_server_content(
        session,
        state,
        SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="D'accord", finished=False),
            model_turn=None,
            interrupted=None,
            turn_complete=None,
        ),
    )

    assert calls == [
        ("Oui", "user", False),
        ("Oui", "user", True),
        ("D'accord", "assistant", False),
    ]
